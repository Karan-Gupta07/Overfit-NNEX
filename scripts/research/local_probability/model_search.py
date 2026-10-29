"""Encoder-side weight search with estimated sizes; only finalists are coded.

Planes share no weights and their inputs never depend on another plane's
network, so a file's size is the header plus a per-plane sum of model bytes
and stream bits. Each plane therefore picks its own weight candidate. Stream
bits are estimated with the float model on the training pixels; on development
images this tracked the integer coder within 0.04%, with a near-constant
per-model offset, so real encodes are needed only for the finalists.
"""

from __future__ import annotations

import math

import neural_bitstream as coder
import numpy as np
import torch
from quantize_model import _quantized, _round

UNIFORM = (16, 12, 10, 8, 7, 6)
WEIGHTS = ("hidden.0.weight", "hidden.2.weight", "head.weight")
# Grid codes: b is one b-bit grid for the tensor; COLUMN + b and ROW + b give
# every column or row its own b-bit grid. GPTQ + code reaches that grid by
# error-feedback rounding (see gptq_round) instead of rounding to nearest.
COLUMN, ROW, GPTQ = 100, 200, 1000
_SLICED = (12, 11, 10, 9, 8, 7, 6, 5)
# The guide stays exact: it defines the network inputs.
GRIDS = {
    **dict.fromkeys(
        WEIGHTS,
        (16, *_SLICED, *(COLUMN + b for b in _SLICED), *(ROW + b for b in _SLICED)),
    ),
    **dict.fromkeys(
        ("affine.weight", "hidden.0.bias", "hidden.2.bias", "head.bias"),
        (16, 14, 13, 12, 11, 10, 9, 8),
    ),
}
CHUNK = 262144


def grid_label(code: int) -> str:
    if code >= GPTQ:
        return "g" + grid_label(code - GPTQ)
    if code < COLUMN:
        return str(code)
    return ("c" if code < ROW else "r") + str(code % 100)


_TENSOR_BYTES: dict[tuple[bytes, tuple[int, ...]], int] = {}


def _rice_bits(z: np.ndarray) -> int:
    # The Rice cost is convex in k: each step saves sum(ceil((z >> k) / 2)),
    # which never grows with k, and costs len(z).
    cost = coder._rice_cost(z, 0)
    for k in range(1, coder.MAX_RICE + 1):
        following = coder._rice_cost(z, k)
        if following >= cost:
            break
        cost = following
    return cost


def tensor_bytes(raw: bytes, shape: tuple[int, ...] = ()) -> int:
    """len(neural_bitstream._encode_tensor(raw, shape)) without building the bits."""
    key = (raw, shape)
    if key in _TENSOR_BYTES:
        return _TENSOR_BYTES[key]
    values = np.frombuffer(raw, "<f2").astype(np.float64)
    _shift, q = coder._grid(values)
    body = (_rice_bits(coder._zigzag(q)) + 7) // 8
    size = min(3 + len(coder._varint(body)) + body, 1 + len(raw))
    if coder.sliceable(shape):
        for kind in (coder.COLUMNS, coder.ROWS):
            shifts, slices = coder.slice_grids(values, shape, kind)
            coded = slices[~coder.zero_slices(slices, True)]
            bits = coder.SLICE_SHIFT_BITS * len(shifts) + _rice_bits(coder._zigzag(coded).ravel())
            body = (bits + 7) // 8
            size = min(size, 2 + len(coder._varint(body)) + body)
    _TENSOR_BYTES[key] = size
    return size


SHAPES = {
    "affine.weight": lambda d, h, r: (1, d),
    "hidden.0.weight": lambda d, h, r: (h, d),
    "hidden.0.bias": lambda d, h, r: (h,),
    "hidden.2.weight": lambda d, h, r: (h, h),
    "hidden.2.bias": lambda d, h, r: (h,),
    "head.weight": lambda d, h, r: (r, h),
    "head.bias": lambda d, h, r: (r,),
}


# The first layer each tensor feeds: 1 is the second layer, 2 the head, 3 the
# affine prediction (which only the mixture means read).
STAGES = {
    "hidden.0.weight": 0,
    "hidden.0.bias": 0,
    "hidden.2.weight": 1,
    "hidden.2.bias": 1,
    "head.weight": 2,
    "head.bias": 2,
    "affine.weight": 3,
}


def state_shapes(state: dict[str, bytes]) -> dict[str, tuple[int, ...]]:
    d, h, r = (len(state[name]) // 2 for name in ("affine.weight", "hidden.0.bias", "head.bias"))
    return {name: shape(d, h, r) for name, shape in SHAPES.items()}


def plane_bytes(state: dict[str, bytes]) -> int:
    shapes = state_shapes(state)
    return sum(tensor_bytes(state[name], shapes.get(name, ())) for name in coder.NAMES)


def _round_slices(raw: bytes, bits: int, shape: tuple[int, ...], columns: bool) -> bytes:
    matrix = np.frombuffer(raw, "<f2").reshape(shape)
    if columns:
        matrix = matrix.T
    rows = [_round(row.tobytes(), bits) for row in np.ascontiguousarray(matrix)]
    out = np.frombuffer(b"".join(rows), "<f2").reshape(matrix.shape)
    return np.ascontiguousarray(out.T if columns else out).tobytes()


def _round_code(raw: bytes, code: int, shape: tuple[int, ...]) -> bytes:
    if code < 16:
        return _round(raw, code)
    if code < COLUMN:
        return raw
    return _round_slices(raw, code % 100, shape, code < ROW)


def round_state(state: dict[str, bytes], bits) -> dict[str, bytes]:
    """bits: one precision for the hidden/head weights, or a per-tensor grid code."""
    if isinstance(bits, int):
        return {
            name: _round(raw, bits) if _quantized(name) and bits < 16 else raw
            for name, raw in state.items()
        }
    if "guide_coeff" in bits:
        raise ValueError("the guide defines the inputs and stays exact")
    shapes = state_shapes(state) if any(code >= COLUMN for code in bits.values()) else {}
    return {
        name: _round_code(raw, bits.get(name, 16), shapes.get(name, ()))
        for name, raw in state.items()
    }


# Matrices that also try GPTQ roundings, and the bias each one feeds. A first
# layer tries only those (and 16 bits): rounding to nearest never won one in
# 45 searched development planes.
REFINED = (*WEIGHTS, "affine.weight")
GPTQ_ONLY = ("hidden.0.weight",)
BIASES = {
    "hidden.0.weight": "hidden.0.bias",
    "hidden.2.weight": "hidden.2.bias",
    "head.weight": "head.bias",
}
# Ridge added to each curvature, relative to its mean diagonal.
DAMPING = 0.01


def _grid_shift(maximum, bits: float) -> np.ndarray:
    """quantize_model._round's shift for each maximum (0 for a zero maximum)."""
    maximum = np.asarray(maximum, dtype=np.float64)
    with np.errstate(divide="ignore"):
        shift = np.floor(np.log2((2.0 ** (bits - 1) - 1) / maximum))
    return np.where(maximum > 0, shift, 0.0)


def gptq_round(raw: bytes, shape: tuple[int, int], codes, curvature: np.ndarray, bias=None):
    """Error-feedback (GPTQ) rounding of a weight matrix to every grid in codes.

    Columns are rounded one at a time, most curved first, and each column's
    rounding error is spread over the columns still unrounded through the
    Cholesky factor of the inverse curvature, which minimizes the curvature-
    weighted output error (Frantar et al., 2022). curvature is (d, d) for every
    row or (rows, d, d) per row. With bias (binary16 bytes), the curvature
    covers a last, constant input of 1 and the bias, left unrounded, absorbs
    error as well. Grids keep their rounding-to-nearest shifts, except that a
    column grid takes its shift from the column as it is when rounded. Returns
    (matrices, compensated biases or None).
    """
    w = np.frombuffer(raw, "<f2").astype(np.float64).reshape(shape)
    rows, d = shape
    active = np.flatnonzero(np.any(w != 0, axis=0))
    m = len(active)
    if m == 0:
        return [_round_code(raw, code, shape) for code in codes], None
    if bias is not None:
        w = np.hstack([w, np.frombuffer(bias, "<f2").astype(np.float64)[:, None]])
        active = np.append(active, d)
    hessian = curvature if curvature.ndim == 3 else curvature[None]
    hessian = hessian[:, active][:, :, active]
    diagonal = np.diagonal(hessian, axis1=1, axis2=2)
    ridge = DAMPING * diagonal.mean(axis=1) + 1e-12
    hessian = hessian + ridge[:, None, None] * np.eye(len(active))
    order = np.argsort(-diagonal[:, :m].sum(axis=0), kind="stable")
    if bias is not None:
        order = np.append(order, m)
    hessian = hessian[:, order][:, :, order]
    columns = active[order]
    inverse = np.linalg.inv(hessian)
    upper = np.linalg.cholesky((inverse + inverse.transpose(0, 2, 1)) / 2).transpose(0, 2, 1)
    pivots = np.diagonal(upper, axis1=1, axis2=2)
    bits = np.array([code % 100 for code in codes], dtype=np.float64)
    per_column = np.array([COLUMN <= code < ROW for code in codes])
    shifts = np.zeros((len(codes), rows))
    for v, code in enumerate(codes):
        if code < COLUMN:
            shifts[v] = _grid_shift(np.abs(w[:, :d]).max(), bits[v])
        elif code >= ROW:
            shifts[v] = _grid_shift(np.abs(w[:, :d]).max(axis=1), bits[v])
    work = np.repeat(w[None][:, :, columns], len(codes), axis=0)
    for i in range(m):
        column = work[:, :, i]
        if per_column.any():
            shifts[per_column] = _grid_shift(
                np.abs(column[per_column]).max(axis=1), bits[per_column]
            )[:, None]
        scale = np.exp2(shifts)
        rounded = (np.rint(column * scale) / scale).astype("<f2").astype(np.float64)
        error = (column - rounded) / pivots[:, i]
        work[:, :, i] = rounded
        work[:, :, i + 1 :] -= error[:, :, None] * upper[None, :, i, i + 1 :]
    out = np.zeros((len(codes), rows, w.shape[1]))
    out[:, :, columns] = work
    matrices = [matrix[:, :d].astype("<f2").tobytes() for matrix in out]
    if bias is None:
        return matrices, None
    return matrices, [matrix[:, d].astype("<f2").tobytes() for matrix in out]


def _weighted_gram(inputs: torch.Tensor, weights: torch.Tensor, bias: bool) -> np.ndarray:
    """(rows, d, d) sums of weights[:, r] * x x^T, with a constant input appended if bias."""
    if inputs.device.type == "mps":
        import metal_training

        return metal_training.weighted_gram(inputs, weights, bias).cpu().double().numpy()
    n, d = inputs.shape
    rows = weights.shape[1]
    gram = torch.zeros(rows * d, d, device=inputs.device)
    chunk = max(1024, (1 << 25) // (rows * d))
    for start in range(0, n, chunk):
        x, w = inputs[start : start + chunk], weights[start : start + chunk]
        gram += (w[:, :, None] * x[:, None, :]).reshape(len(x), rows * d).T @ x
    gram = gram.reshape(rows, d, d).cpu().double().numpy()
    if not bias:
        return gram
    out = np.empty((rows, d + 1, d + 1))
    out[:, :d, :d] = gram
    out[:, :d, d] = out[:, d, :d] = (weights.T @ inputs).cpu().double().numpy()
    out[:, d, d] = weights.sum(0).cpu().double().numpy()
    return out


class Estimator:
    """Float bits of each plane over the training pixels already on the device.

    On MPS one fused kernel launch scores a whole batch of candidates.
    """

    def __init__(self, x, y, a, dimensions, mixtures, loss):
        self.x, self.y, self.a = x, y, a
        self.loss = loss
        self.dimensions = dimensions
        self.k = mixtures
        self.cache: dict[tuple[int, bytes], float] = {}
        self._fused: dict[int, object] = {}

    def _tensor(self, raw: bytes, shape):
        values = np.frombuffer(raw, "<f2").astype(np.float32).reshape(shape)
        return torch.from_numpy(values).to(self.x.device)

    @staticmethod
    def _key(g: int, state: dict[str, bytes]) -> tuple[int, bytes]:
        return g, b"".join(state[name] for name in coder.NAMES[1:])

    def _evaluator(self, hidden: int):
        if self.x.device.type != "mps":
            return None
        import metal_training

        if not metal_training.supported(self.x.shape[2], hidden, self.k):
            return None
        if hidden not in self._fused:
            self._fused[hidden] = metal_training.FusedEvaluator(
                self.x, self.y, self.a, hidden, self.k, self.dimensions
            )
        return self._fused[hidden]

    def many(self, requests: list[tuple[int, dict[str, bytes]]], hidden: int) -> list[float]:
        """bits() for every (plane, state) request, scoring uncached ones together."""
        evaluator = self._evaluator(hidden)
        if evaluator is None:
            return [self.bits(g, state, hidden) for g, state in requests]
        missing: dict[tuple[int, bytes], tuple[int, dict[str, bytes]]] = {}
        for g, state in requests:
            key = self._key(g, state)
            if key not in self.cache:
                missing[key] = (g, state)
        if missing:
            blocks = np.stack([self._pack(g, state, hidden) for g, state in missing.values()])
            self._store(missing, evaluator.bits([g for g, _state in missing.values()], blocks))
        return [self.cache[self._key(g, state)] for g, state in requests]

    def many_after(self, g: int, base: dict[str, bytes], states, hidden: int) -> list[float]:
        """many() for states of plane g, reusing base's outputs of the layers they share.

        The bits are many()'s: the fused evaluator computes the shared layers
        exactly as a full pass would.
        """
        evaluator = self._evaluator(hidden)
        stages = {STAGES[name] for state in states for name in STAGES if state[name] != base[name]}
        if evaluator is None or not stages or 0 in stages or (3 in stages and len(stages) > 1):
            return self.many([(g, state) for state in states], hidden)
        missing = {}
        for state in states:
            key = self._key(g, state)
            if key not in self.cache:
                missing[key] = (g, state)
        if missing:
            base_key = self._key(g, base)
            if evaluator.outputs_key != base_key:
                evaluator.activations(g, self._pack(g, base, hidden), base_key)
            blocks = np.stack([self._pack(g, state, hidden) for _g, state in missing.values()])
            self._store(missing, evaluator.bits_from(min(stages), blocks))
        return [self.cache[self._key(g, state)] for state in states]

    def _pack(self, g: int, state: dict[str, bytes], hidden: int) -> np.ndarray:
        import metal_training

        d, r = self.x.shape[2], 3 * self.k
        return metal_training.pack(
            {
                name: np.frombuffer(state[name], "<f2")
                .astype(np.float32)
                .reshape(shape(self.dimensions[g], hidden, r))
                for name, shape in SHAPES.items()
            },
            d,
            hidden,
            self.k,
        )

    def _store(self, missing, values) -> None:
        for key, value in zip(missing, values, strict=True):
            if not math.isfinite(value):
                raise ValueError("invalid estimate")
            self.cache[key] = value

    @torch.no_grad()
    def bits(self, g: int, state: dict[str, bytes], hidden: int) -> float:
        key = self._key(g, state)
        if key in self.cache:
            return self.cache[key]
        d, k = self.dimensions[g], self.k
        affine = self._tensor(state["affine.weight"], (1, d))
        w0 = self._tensor(state["hidden.0.weight"], (hidden, d))
        b0 = self._tensor(state["hidden.0.bias"], (hidden,))
        w1 = self._tensor(state["hidden.2.weight"], (hidden, hidden))
        b1 = self._tensor(state["hidden.2.bias"], (hidden,))
        wh = self._tensor(state["head.weight"], (3 * k, hidden))
        bh = self._tensor(state["head.bias"], (3 * k,))
        total = torch.zeros((), dtype=torch.float32, device=self.x.device)
        for start in range(0, self.x.shape[1], CHUNK):
            xs = self.x[g, start : start + CHUNK, :d].float()
            z = torch.relu(torch.addmm(b0, xs, w0.T))
            z = torch.relu(torch.addmm(b1, z, w1.T))
            raw = torch.addmm(bh, z, wh.T)
            means = raw[:, k : 2 * k] + xs @ affine.T
            scales = torch.exp(raw[:, 2 * k :].clamp(-4.0, 7.0))
            lost = self.loss(
                raw[:, :k],
                means,
                scales,
                self.y[g, start : start + CHUNK],
                self.a[g, start : start + CHUNK],
            )
            total += lost.sum()
        value = float(total.cpu())
        if not math.isfinite(value):
            raise ValueError("invalid estimate")
        self.cache[key] = value
        return value

    def _forward(self, g: int, state: dict[str, bytes], hidden: int, upto: str, stride: int = 1):
        """The input of upto's layer at every stride-th sample (no gradients)."""
        d = self.dimensions[g]
        xs = self.x[g, ::stride, :d].float()
        if upto in ("hidden.0.weight", "affine.weight"):
            return xs
        w0 = self._tensor(state["hidden.0.weight"], (hidden, d))
        b0 = self._tensor(state["hidden.0.bias"], (hidden,))
        z = torch.relu(torch.addmm(b0, xs, w0.T))
        if upto == "head.weight":
            w1 = self._tensor(state["hidden.2.weight"], (hidden, hidden))
            b1 = self._tensor(state["hidden.2.bias"], (hidden,))
            z = torch.relu(torch.addmm(b1, z, w1.T))
        return z

    def output_gradients(
        self, g: int, state: dict[str, bytes], hidden: int
    ) -> dict[str, torch.Tensor]:
        """Per-sample squared loss gradients of each matrix's outputs (n, rows)."""
        fused = self._fused_gradients(g, state, hidden)
        if fused is not None:
            return fused
        d, k = self.dimensions[g], self.k
        xs = self.x[g, :, :d].float()
        affine = self._tensor(state["affine.weight"], (1, d))
        w0 = self._tensor(state["hidden.0.weight"], (hidden, d))
        b0 = self._tensor(state["hidden.0.bias"], (hidden,))
        w1 = self._tensor(state["hidden.2.weight"], (hidden, hidden))
        b1 = self._tensor(state["hidden.2.bias"], (hidden,))
        wh = self._tensor(state["head.weight"], (3 * k, hidden))
        bh = self._tensor(state["head.bias"], (3 * k,))
        with torch.enable_grad():
            first = torch.addmm(b0, xs, w0.T).requires_grad_()
            second = torch.addmm(b1, torch.relu(first), w1.T)
            second.retain_grad()
            raw = torch.addmm(bh, torch.relu(second), wh.T)
            raw.retain_grad()
            linear = (xs @ affine.T).requires_grad_()
            means = raw[:, k : 2 * k] + linear
            scales = torch.exp(raw[:, 2 * k :].clamp(-4.0, 7.0))
            self.loss(raw[:, :k], means, scales, self.y[g], self.a[g]).sum().backward()
        return {
            "hidden.0.weight": first.grad.square(),
            "affine.weight": linear.grad.square(),
            "hidden.2.weight": second.grad.square(),
            "head.weight": raw.grad.square(),
        }

    # Rows per fused forward-backward launch in _fused_gradients.
    GRADIENT_ROWS = 1 << 16

    @torch.no_grad()
    def _fused_gradients(self, g: int, state: dict[str, bytes], hidden: int):
        """output_gradients from the trainer's forward-backward kernel, or None off MPS.

        The kernel writes every sample's gradients at the first and second
        pre-activations (ReLU masks applied), the affine output and the head
        outputs, which are what autograd gives; no weight gradient is reduced.
        """
        if self.x.device.type != "mps":
            return None
        import metal_training

        d, k, r = self.x.shape[2], self.k, 3 * self.k
        if not metal_training.supported(d, hidden, k):
            return None
        lib = metal_training.library(d, hidden, k)
        pads = metal_training.padded(d, hidden, k)
        block = metal_training.pack(
            {
                name: np.frombuffer(state[name], "<f2")
                .astype(np.float32)
                .reshape(shape(self.dimensions[g], hidden, r))
                for name, shape in SHAPES.items()
            },
            d,
            hidden,
            k,
        )
        device = self.x.device
        params = torch.from_numpy(block[None]).to(device)
        x = self.x[g : g + 1].to(torch.float16).contiguous()
        y, a = self.y[g : g + 1].contiguous(), self.a[g : g + 1].contiguous()
        dims = torch.tensor([self.dimensions[g]], dtype=torch.int32, device=device)
        n = x.shape[1]
        group = metal_training.GROUP
        parts: dict[str, list[torch.Tensor]] = {name: [] for name in REFINED}
        for start in range(0, n, self.GRADIENT_ROWS):
            batch = min(self.GRADIENT_ROWS, n - start)
            rows = metal_training._pad(batch)
            index = torch.arange(start, start + batch, dtype=torch.int32, device=device)
            xb = torch.empty(1, rows, pads["DA"], device=device, dtype=torch.float16)
            e1, z1, d2, z2, dr = (
                torch.empty(1, rows, pads[width], device=device)
                for width in ("HA", "HA", "HB", "HA", "RA")
            )
            bits = torch.empty(1, batch, device=device)
            bpad = -(-batch // group) * group
            lib.forward_backward_float(
                params,
                x,
                y,
                a,
                index,
                dims,
                xb,
                e1,
                z1,
                d2,
                z2,
                dr,
                bits,
                n,
                batch,
                bpad,
                1.0 / math.log(2),
                threads=bpad,
                group_size=group,
            )
            parts["hidden.0.weight"].append(e1[0, :batch, :hidden].square())
            parts["affine.weight"].append(e1[0, :batch, hidden : hidden + 1].square())
            parts["hidden.2.weight"].append(d2[0, :batch, :hidden].square())
            parts["head.weight"].append(dr[0, :batch, :r].square())
        return {name: torch.cat(chunks) for name, chunks in parts.items()}

    # GPTQ curvatures use evenly strided rows, at most this many: half of the
    # estimator's 2^18 coded 0.0006% larger in total on the 12 development
    # crops (no crop by more than 0.16%) and 0.005% on the 12 natives.
    CURVATURE_SAMPLES = 1 << 17

    @torch.no_grad()
    def curvature(self, g, state, hidden, name, gradients, bias: bool) -> np.ndarray:
        """Per-row empirical Fisher of the matrix name: its squared output
        gradients weight the Gram matrix of its inputs under state."""
        stride = -(-self.x.shape[1] // self.CURVATURE_SAMPLES)
        inputs = self._forward(g, state, hidden, name, stride)
        return _weighted_gram(inputs, gradients[name][::stride], bias)


def _costs(est: Estimator, g: int, states, hidden: int, scale: float, base=None) -> list[float]:
    if base is None:
        bits = est.many([(g, state) for state in states], hidden)
    else:
        bits = est.many_after(g, base, states, hidden)
    return [b * scale / 8 + plane_bytes(state) for b, state in zip(bits, states, strict=True)]


def search_plane(est, g, sources, hidden, scale):
    """Best (estimated bytes, label, state) for one plane.

    sources maps a label to binary16 tensors; "float" is the trained model and
    the rest are quantization-aware snapshots already on their grids. Every
    source tries the uniform grids, then per-tensor coordinate descent starts
    from the best uniform result; each tensor's grids are scored together.
    A REFINED matrix also tries the GPTQ rounding to each grid against the
    current best state, with its bias compensated; a chosen compensated bias
    is what that bias's own grids then round.
    """
    labelled = [
        (f"{label}-q{bits}", round_state(state, bits))
        for label, state in sources.items()
        for bits in UNIFORM
    ]
    costs = _costs(est, g, [state for _label, state in labelled], hidden, scale)
    rows = [(cost, label, state) for cost, (label, state) in zip(costs, labelled, strict=True)]
    best_cost, best_label, best_state = min(rows, key=lambda row: row[0])
    origin = sources[best_label.rsplit("-q", 1)[0]]
    start = int(best_label.rsplit("-q", 1)[1])
    initial = {name: start if name in WEIGHTS else 16 for name in GRIDS}
    choice = dict(initial)
    shapes = state_shapes(origin)
    base = dict(origin)
    improved = True
    sweeps = 0
    while improved and sweeps < 2:
        improved = False
        sweeps += 1
        # Gradients are refreshed once a sweep, layer inputs at every matrix;
        # fresh gradients at every matrix were no better on development crops.
        gradients = None
        for name, grids in GRIDS.items():
            bias = BIASES.get(name)
            codes = list(grids)
            variants = [_round_code(base[name], code, shapes[name]) for code in codes]
            biases = [origin[bias] if bias else None] * len(codes)
            if name in REFINED:
                if gradients is None:
                    gradients = est.output_gradients(g, best_state, hidden)
                refined = [code for code in grids if code != 16]
                curvature = est.curvature(g, best_state, hidden, name, gradients, bias is not None)
                matrices, compensated = gptq_round(
                    origin[name], shapes[name], refined, curvature, origin[bias] if bias else None
                )
                if name in GPTQ_ONLY:
                    codes, variants, biases = [16], variants[:1], biases[:1]
                codes += [GPTQ + code for code in refined]
                variants += matrices
                biases += compensated or [origin[bias] if bias else None] * len(refined)
            trials = [
                (code, variant, bias_base)
                for code, variant, bias_base in zip(codes, variants, biases, strict=True)
                if code != choice[name]
            ]
            candidates = []
            for _code, variant, bias_base in trials:
                candidate = {**best_state, name: variant}
                if bias:
                    candidate[bias] = _round_code(bias_base, choice[bias], shapes[bias])
                candidates.append(candidate)
            costs = _costs(est, g, candidates, hidden, scale, best_state)
            # The first strictly better grid wins ties, as a sequential scan would.
            for cost, (code, _variant, bias_base), candidate in zip(
                costs, trials, candidates, strict=True
            ):
                if cost < best_cost - 1e-9:
                    best_cost, best_state, improved = cost, candidate, True
                    choice = {**choice, name: code}
                    if bias:
                        base[bias] = bias_base
    if choice != initial:
        best_label = (
            best_label.rsplit("-q", 1)[0]
            + "-t"
            + ".".join(grid_label(choice[name]) for name in GRIDS)
        )
    return best_cost, best_label, best_state, rows


def search(est, header, sources_by_plane, scale):
    """Selected model payload, its estimated file bytes, labels, and the best uniform model."""
    hidden = header[2]
    planes, labels, costs, total = [], [], [], 0.0
    uniform_rows: dict[str, float] = {}
    uniform_states: dict[str, list] = {}
    for g, sources in enumerate(sources_by_plane):
        cost, label, state, rows = search_plane(est, g, sources, hidden, scale)
        planes.append(state)
        labels.append(label)
        costs.append(cost)
        total += cost
        for row_cost, row_label, row_state in rows:
            if row_label.startswith("float-"):
                uniform_rows[row_label] = uniform_rows.get(row_label, 0.0) + row_cost
                uniform_states.setdefault(row_label, []).append(row_state)
    payload = coder.encode_model(header, planes)
    fixed = len(payload) - sum(plane_bytes(state) for state in planes) + coder.HEADER_SIZE
    uniform = min(uniform_rows, key=uniform_rows.get)
    return {
        "payload": payload,
        "estimated_bytes": total + fixed,
        "labels": labels,
        "plane_estimated_bytes": costs,
        "uniform_label": uniform,
        "uniform_payload": coder.encode_model(header, uniform_states[uniform]),
        "uniform_estimated_bytes": uniform_rows[uniform] + fixed,
    }
