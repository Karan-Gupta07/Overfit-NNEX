"""Image-local probability experiment. Estimates, not a codec or bitstream."""

import argparse
import copy
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cpu_training
import imagecodecs
import metal_training
import model_search
import neural_bitstream as coder
import numpy as np
import predictor_probe as linear
import torch
from input_image import MAX_TRAIN_PIXELS, read_rgb
from torch import nn

from semanticcodec.format.container import compress, encode_cbor


class ProbabilityNet(nn.Module):
    def __init__(self, dimensions, hidden, mixtures, initial):
        super().__init__()
        self.mixtures = mixtures
        self.affine = nn.Linear(dimensions, 1, bias=False)
        self.hidden = nn.Sequential(
            nn.Linear(dimensions, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.head = nn.Linear(hidden, 3 * mixtures)
        with torch.no_grad():
            self.affine.weight.copy_(torch.from_numpy((initial * 32)[None].astype(np.float32)))
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[2 * mixtures :] = 0.5
            self.head.bias[mixtures : 2 * mixtures] = torch.linspace(-0.25, 0.25, mixtures)

    def forward(self, x):
        k = self.mixtures
        raw = self.head(self.hidden(x))
        logits = raw[:, :k]
        means = raw[:, k : 2 * k] + self.affine(x)
        scales = torch.exp(raw[:, 2 * k :].clamp(-4.0, 7.0))
        return logits, means, scales


def mixture_bits(logits, mean, scale, target, anchor):
    """Bits per sample; leading dimensions are batch dimensions, mixtures are last."""
    upper = (target[..., None] + 0.5 - mean) / scale
    lower = (target[..., None] - 0.5 - mean) / scale
    # Stable logistic CDF difference, avoiding cancellation for tiny bins.
    logprob = (
        torch.nn.functional.logsigmoid(upper)
        + torch.nn.functional.logsigmoid(-lower)
        + torch.log(-torch.expm1(-(upper - lower)).clamp(max=-1e-12))
    )
    logmix = torch.log_softmax(logits, dim=-1)
    lp = torch.logsumexp(logmix + logprob, dim=-1)
    top = (255.5 - anchor[..., None] - mean) / scale
    bottom = (-0.5 - anchor[..., None] - mean) / scale
    mass = (torch.sigmoid(top) - torch.sigmoid(bottom)).clamp_min(1e-10)
    normal = torch.logsumexp(logmix + torch.log(mass), dim=-1)
    return (normal - lp) / math.log(2)


def _log_sigmoid_difference(upper, lower, width):
    """log(s(upper) - s(lower)) for upper - lower = width > 0, s the integer coder's logistic.

    s is exactly 0 or 1 beyond +-CODER_SATURATION; EMPTY where both ends saturate alike.
    """
    logsigmoid = torch.nn.functional.logsigmoid
    limit = metal_training.CODER_SATURATION
    high, low = upper >= limit, lower <= -limit
    both = (
        logsigmoid(upper) + logsigmoid(-lower) + torch.log(-torch.expm1(-width).clamp(max=-1e-12))
    )
    value = torch.where(
        high,
        torch.where(low, torch.zeros_like(both), logsigmoid(-lower)),
        torch.where(low, logsigmoid(upper), both),
    )
    empty = (lower >= limit) | (upper <= -limit)
    return torch.where(empty, torch.full_like(value, metal_training.EMPTY), value)


def coded_bits(logits, mean, scale, target, anchor):
    """mixture_bits as the integer coder spends them (metal_training.CODER_SATURATION)."""
    inv = 1 / scale
    t, low = target[..., None], anchor[..., None]
    logmix = torch.log_softmax(logits, dim=-1)
    lp = torch.logsumexp(
        logmix + _log_sigmoid_difference((t + 0.5 - mean) * inv, (t - 0.5 - mean) * inv, inv),
        dim=-1,
    )
    edges = (255.5 - low - mean) * inv, (-0.5 - low - mean) * inv
    normal = torch.logsumexp(logmix + _log_sigmoid_difference(*edges, 256 * inv), dim=-1)
    # -log2 of one count plus (TOTAL - 256) counts times the share exp(-d).
    d = normal - lp
    total = metal_training.CODER_TOTAL
    keep = torch.full_like(d, math.log1p(-256 / total))
    bits = (d - torch.logaddexp(d - math.log(total), keep)) / math.log(2)
    return torch.where(normal < metal_training.EMPTY / 2, torch.full_like(bits, 8.0), bits)


def nll(model, x, target, anchor):
    return mixture_bits(*model(x), target, anchor)


class GroupedNet(nn.Module):
    """Independent per-plane ProbabilityNets trained in one batched step.

    Inputs are zero-padded to the widest plane; padded weight rows start at zero
    and receive zero gradient, so each plane matches a separately trained net.
    """

    def __init__(self, nets, dimensions):
        super().__init__()
        width = max(dimensions)
        self.mixtures = nets[0].mixtures

        def stacked(get, pad=False):
            parts = [get(net).detach() for net in nets]
            if pad:
                parts = [
                    nn.functional.pad(p, (0, 0, 0, width - d))
                    for p, d in zip(parts, dimensions, strict=True)
                ]
            return nn.Parameter(torch.stack(parts))

        self.affine = stacked(lambda n: n.affine.weight.T, pad=True)
        self.w1 = stacked(lambda n: n.hidden[0].weight.T, pad=True)
        self.b1 = stacked(lambda n: n.hidden[0].bias[None])
        self.w2 = stacked(lambda n: n.hidden[2].weight.T)
        self.b2 = stacked(lambda n: n.hidden[2].bias[None])
        self.w3 = stacked(lambda n: n.head.weight.T)
        self.b3 = stacked(lambda n: n.head.bias[None])

    def forward(self, x):
        k = self.mixtures
        z = torch.relu(torch.baddbmm(self.b1, x, self.w1))
        z = torch.relu(torch.baddbmm(self.b2, z, self.w2))
        raw = torch.baddbmm(self.b3, z, self.w3)
        means = raw[..., k : 2 * k] + torch.bmm(x, self.affine)
        return raw[..., :k], means, torch.exp(raw[..., 2 * k :].clamp(-4.0, 7.0))

    @torch.no_grad()
    def export(self, nets, dimensions):
        for g, (net, d) in enumerate(zip(nets, dimensions, strict=True)):
            net.affine.weight.copy_(self.affine[g, :d].T)
            net.hidden[0].weight.copy_(self.w1[g, :d].T)
            net.hidden[0].bias.copy_(self.b1[g, 0])
            net.hidden[2].weight.copy_(self.w2[g].T)
            net.hidden[2].bias.copy_(self.b2[g, 0])
            net.head.weight.copy_(self.w3[g].T)
            net.head.bias.copy_(self.b3[g, 0])


def clip_per_plane(parameters, max_norm):
    """clip_grad_norm_ applied separately to each plane (leading dimension)."""
    parameters = list(parameters)
    total = sum(p.grad.pow(2).flatten(1).sum(1) for p in parameters).sqrt()
    factor = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for p in parameters:
        p.grad.mul_(factor.view(-1, *[1] * (p.dim() - 1)))


GUIDE_CHUNK = 1 << 19


def guide_fit(xf, target, pixels):
    """Robust IRLS linear predictor used for the guide and the affine initialization.

    Normal equations accumulate over row chunks, so large images never build a
    weighted copy of the whole feature matrix.
    """
    k = xf.shape[1]
    co = np.zeros(k)
    for _ in range(4):
        matrix, rhs = np.zeros((k, k)), np.zeros(k)
        for start in range(0, len(xf), GUIDE_CHUNK):
            xc = xf[start : start + GUIDE_CHUNK].astype(np.float64, copy=False)
            tc = target[start : start + GUIDE_CHUNK]
            importance = 1 / np.maximum(1, np.abs(tc - xc @ co))
            matrix += xc.T @ (xc * importance[:, None])
            rhs += xc.T @ (tc * importance)
        matrix.flat[:: k + 1] += pixels * 0.002
        co = np.linalg.solve(matrix, rhs)
    return co


# Must match far_taps in nnproto_integer.cpp, as (dy, dx).
FAR = [
    (0, -3),
    (-3, 0),
    (-1, -3),
    (-1, 3),
    (-3, -1),
    (-3, 1),
    (-2, -3),
    (-2, 3),
    (-3, -2),
    (-3, 2),
    (-3, -3),
    (-3, 3),
]


# Must match fit_errors in nnproto_integer.cpp, as (dy, dx).
FIT_ERRORS = [(0, -1), (-1, 0), (-1, -1), (-1, 1)]
PRIOR_PLANES = {1: (), 0: (1,), 2: (1, 0)}


def spread_feature(spread):
    """8 log2(1 + spread / 16), linear between powers of two, as in nnproto_integer.cpp."""
    v = spread.astype(np.int64) + 16
    e = np.frexp(v.astype(np.float64))[1] - 1
    return 8 * (e - 4) + (((v - (1 << e)) * 8) >> e)


def local_planes(rgb, source=None, feature_set=3):
    """The feature set's least-squares (fit, spread) planes per window and channel,
    for the whole image.

    source names a file of coder.local_fits output for these pixels and this
    feature set; otherwise the integer backend computes them.
    """
    h, w = rgb.shape[:2]
    count = coder.FITS[feature_set]
    if source is None:
        planes = coder.local_fits(rgb, feature_set)
    else:
        planes = np.fromfile(source, "<i2")
        if planes.size != 6 * h * w * count:
            raise ValueError("local fit size")
        planes = planes.reshape(2 * count, 3, h, w)
    return {
        channel: [(planes[2 * s, g], planes[2 * s + 1, g]) for s in range(count)]
        for g, channel in enumerate((1, 0, 2))
    }


def causal_shift(a, dy, dx):
    """Shift with zeros outside the canvas; never clamps onto undecoded samples."""
    h, w = a.shape
    out = np.zeros_like(a)
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    out[y0:y1, x0:x1] = a[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return out


def plane_inputs(
    rgb,
    channel,
    co_base,
    prior_errors,
    causal_context,
    xf=None,
    anchor=None,
    feature_set=0,
    local=None,
):
    """Exact float features for one plane, given the transmitted binary16 guide."""
    h, w = rgb.shape[:2]
    if xf is None:
        xf, anchor = linear.features(rgb, channel)
    if feature_set in coder.FITS and local is None:
        local = local_planes(rgb, feature_set=feature_set)
    target, base_raw, base_error = _base_prediction(rgb, channel, co_base, xf, anchor)
    columns = list(
        _context_columns(
            rgb,
            channel,
            anchor,
            base_raw,
            base_error,
            prior_errors,
            causal_context,
            feature_set,
            local=local,
        )
    )
    if columns:
        xf = np.concatenate([xf, np.stack(columns, axis=-1).reshape(h * w, -1)], axis=1)
    return xf.astype(np.float64, copy=False), target, anchor, base_error


def _base_prediction(rgb, channel, co_base, xf, anchor):
    h, w = rgb.shape[:2]
    target = (rgb[..., channel].astype(np.int16) - anchor).ravel().astype(np.float64)
    # Integer features times binary16 weights add up exactly, so chunks change no bit.
    base_raw = np.empty(len(xf))
    for start in range(0, len(xf), GUIDE_CHUNK):
        rows = slice(start, start + GUIDE_CHUNK)
        np.matmul(xf[rows].astype(np.float64, copy=False), co_base, out=base_raw[rows])
    base_prediction = np.clip(np.floor(base_raw + anchor.ravel() + 0.5), 0, 255)
    base_error = (
        (rgb[..., channel].astype(np.int16).ravel() - base_prediction)
        .reshape(h, w)
        .astype(np.int16)
    )
    return target, base_raw, base_error


def _context_columns(
    rgb,
    channel,
    anchor,
    base_raw,
    base_error,
    prior_errors,
    causal_context,
    feature_set,
    origin=(0, 0),
    local=None,
):
    """Feature columns after the base features, one (h, w) array at a time, in order.

    origin is the crop's position in the image; parity features use image coordinates.
    local holds the crop of local_planes for feature sets 3 and 4.
    """
    h, w = rgb.shape[:2]
    if causal_context:
        for delta in linear.CAUSAL[:6]:
            e = linear.shift(base_error, *delta)
            yield e
            yield np.abs(e)
        for previous in prior_errors.values():
            for delta in linear.GUIDE:
                yield linear.shift(previous, *delta)
        if channel != 1:
            yield (rgb[..., 1] % 4).astype(np.float64) * 8
            yield rgb[..., 1].astype(float) / 8
        if channel == 2:
            yield (rgb[..., 0] % 4).astype(np.float64) * 8
        yield (base_raw - np.floor(base_raw)).reshape(h, w) * 32
    if feature_set:
        if feature_set not in (1, 2, 3, 4) or not causal_context:
            raise ValueError("feature sets 1 to 4 require causal context")
        if channel == 1:
            plane = rgb[..., 1].astype(np.int16)
            for d in FAR:
                yield causal_shift(plane, *d) - anchor
        else:
            diff = rgb[..., channel].astype(np.int16) - rgb[..., 1].astype(np.int16)
            for d in FAR:
                yield causal_shift(diff, *d)
    if feature_set >= 2:
        # Level drives sensor noise; parity sees demosaicing and scanner patterns.
        own = rgb[..., channel].astype(np.int16)
        yield anchor if channel == 1 else causal_shift(own, 0, -1)
        rows, columns = np.indices((h, w))
        rows, columns = rows + origin[0], columns + origin[1]
        yield (rows % 2) * 32
        yield (columns % 2) * 32
        yield ((rows + columns) % 2) * 32
    for s in range(coder.FITS.get(feature_set, 0)):
        fit, spread = local[channel][s]
        yield fit / 8 - anchor
        yield spread_feature(spread)
        error = rgb[..., channel] - fit / 8
        for d in FIT_ERRORS:
            yield causal_shift(error, *d)
        for prior in PRIOR_PLANES[channel]:
            yield rgb[..., prior] - local[prior][s][0] / 8


def learning_rate_factor(schedule, total_steps):
    if schedule == "constant":
        return lambda step: 1.0
    if schedule != "cosine":
        raise ValueError("unknown learning-rate schedule")
    warmup = max(1, total_steps // 20)

    def factor(step):
        progress = min(step, total_steps) / max(1, total_steps)
        return min(1.0, (step + 1) / warmup) * (
            0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
        )

    return factor


TILE = 256
# Feature reach is at most four samples (far taps, errors of causal errors).
TILE_MARGIN = 6


def training_tiles(h, w, budget=MAX_TRAIN_PIXELS):
    """Deterministic, evenly spread training crops whose interiors total <= budget.

    Entries are (row slice, column slice, interior mask or None). Margins make
    every interior feature identical to the corresponding whole-image feature.
    """
    if h * w <= budget:
        return [(slice(0, h), slice(0, w), None)]
    rows, cols = max(1, h // TILE), max(1, w // TILE)
    count = min(rows * cols, max(1, budget // (TILE * TILE)))
    picks = np.unique(np.round(np.linspace(0, rows * cols - 1, count)).astype(int))
    tiles = []
    for index in picks.tolist():
        ty, tx = divmod(index, cols)
        y0, x0 = ty * TILE, tx * TILE
        y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
        cy0, cx0 = max(0, y0 - TILE_MARGIN), max(0, x0 - TILE_MARGIN)
        cy1, cx1 = min(h, y1 + TILE_MARGIN), min(w, x1 + TILE_MARGIN)
        mask = np.zeros((cy1 - cy0, cx1 - cx0), bool)
        mask[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0] = True
        tiles.append((slice(cy0, cy1), slice(cx0, cx1), mask))
    return tiles


def _join(parts):
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def plane_bases(rgb, channel, tiles):
    """Base features, guide fit, and base predictions of one plane's tiles.

    None of it depends on another plane, so planes may prepare concurrently.
    """
    bases = []
    for ys, xs, mask in tiles:
        crop = rgb[ys, xs]
        # int16, not float64: at 2400 by 2400 the three planes' rows otherwise take
        # 3 GB per worker, which pushed two parallel workers into memory compression.
        xf, anchor = linear.features(crop, channel, np.int16)
        target = (crop[..., channel].astype(np.int16) - anchor).ravel().astype(np.float64)
        keep = slice(None) if mask is None else mask.ravel()
        bases.append((crop, xf, anchor, target, keep, (ys.start or 0, xs.start or 0)))
    co = guide_fit(
        _join([b[1][b[4]] for b in bases]),
        _join([b[3][b[4]] for b in bases]),
        sum(len(b[3][b[4]]) for b in bases),
    )
    co_base = co.astype("<f2").astype(np.float64)
    predictions = [_base_prediction(b[0], channel, co_base, b[1], b[2]) for b in bases]
    return bases, co, co_base, predictions


# Threads that write feature columns; numpy releases the GIL in each column's loops.
COLUMN_WRITERS = 8
# A tile without an interior mask is written in row bands of at least
# BAND_MIN_ROWS rows, about BANDS of them, each in its own thread. A band reads
# TILE_MARGIN rows of context on both sides, so its rows equal the tile's.
BANDS = 16
BAND_MIN_ROWS = 64
# Rows per float32 to binary16 conversion block.
ROW_BLOCK = 1 << 15


def _band(unit, bases, tiles):
    """A work unit's context rows (lo, hi) of its tile, and the selection of its rows."""
    index, y0, y1, _start = unit
    h, w = bases[index][0].shape[:2]
    if tiles[index][2] is not None:
        return 0, h, bases[index][4]
    lo, hi = max(0, y0 - TILE_MARGIN), min(h, y1 + TILE_MARGIN)
    return lo, hi, slice((y0 - lo) * w, (y1 - lo) * w)


def tile_inputs(
    rgb, channel, args, prior_errors, tiles, extra=None, local=None, prepared=None, binary16=None
):
    """Training rows from crop interiors; prior_errors holds per-tile error lists.

    Rows are float32 features already divided by 32, written column by column so
    large images never hold a float64 copy of the full feature matrix. Least-squares
    fits reach far beyond a tile margin, so local (from local_planes) is always for
    the whole image. prepared is this plane's plane_bases, if already computed.
    binary16, if given, is a width: rows then come back row-major in binary16,
    zero padded to that width. Each float32 rounds to nearest even, as the
    device conversion does, and no float32 copy of the whole matrix is made.
    """
    feature_set = getattr(args, "feature_set", 0)
    if feature_set in coder.FITS and local is None:
        local = local_planes(rgb, getattr(args, "local_fits", None), feature_set)
    bases, co, co_base, predictions = prepared or plane_bases(rgb, channel, tiles)
    if extra is not None and (len(tiles) != 1 or binary16 is not None):
        raise ValueError("diagnostic extra inputs need a single float32 training tile")
    original_dimensions = bases[0][1].shape[1]
    units, row = [], 0
    for index, (crop, _xf, _anchor, target, keep, _origin) in enumerate(bases):
        h, w = crop.shape[:2]
        step = h
        if extra is None and tiles[index][2] is None:
            step = max(BAND_MIN_ROWS, -(-h // BANDS))
        units.extend((index, y0, min(h, y0 + step), row + y0 * w) for y0 in range(0, h, step))
        row += len(target[keep])

    def columns(unit, lo, hi):
        index = unit[0]
        crop, _xf, anchor, _target, _keep, origin = bases[index]
        _target, base_raw, base_error = predictions[index]
        ys, xs, _mask = tiles[index]
        band, w = slice(lo, hi), crop.shape[1]
        crop_local = None
        if local is not None:
            crop_local = {
                c: [(fit[ys, xs][band], spread[ys, xs][band]) for fit, spread in fits]
                for c, fits in local.items()
            }
        return _context_columns(
            crop[band],
            channel,
            anchor[band],
            base_raw[lo * w : hi * w],
            base_error[band],
            {c: e[index][band] for c, e in prior_errors.items()},
            args.causal_context,
            feature_set,
            (origin[0] + lo, origin[1]),
            crop_local,
        )

    # The column count depends only on the plane and the feature set.
    width = original_dimensions + sum(1 for _ in columns(units[0], 0, 1))
    added = None
    if extra is not None:
        added = extra(predictions[0][2]) if callable(extra) else extra
    total = width + (0 if added is None else added.shape[1])
    if binary16 is not None:
        if binary16 < width:
            raise ValueError("binary16 width below the feature count")
        out = np.empty((row, binary16), np.float16)
    else:
        # Column-major: every column below is one contiguous write.
        out = np.empty((row, total), np.float32, order="F")

    def write(unit):
        lo, hi, select = _band(unit, bases, tiles)
        xf, w = bases[unit[0]][1], bases[unit[0]][0].shape[1]
        base = xf[lo * w : hi * w][select]
        start, n = unit[3], len(base)
        rows = out[start : start + n] if binary16 is None else np.empty((n, width), np.float32, "F")
        # Each quotient is float64, then rounded to float32 on output.
        for column in range(original_dimensions):
            np.divide(base[:, column], 32, out=rows[:, column])
        for column, values in enumerate(columns(unit, lo, hi), start=original_dimensions):
            np.divide(values.ravel()[select], 32, out=rows[:, column])
        if binary16 is not None:
            for s in range(0, n, ROW_BLOCK):
                block = out[start + s : start + min(n, s + ROW_BLOCK)]
                block[:, :width] = rows[s : s + ROW_BLOCK]
                block[:, width:] = 0

    with ThreadPoolExecutor(COLUMN_WRITERS) as pool:
        list(pool.map(write, units))
    if added is not None:
        out[:, width:] = added[bases[0][4]] / 32
    targets = [p[0][b[4]] for p, b in zip(predictions, bases, strict=True)]
    anchors = [b[2].ravel()[b[4]] for b in bases]
    errors = [p[2] for p in predictions]
    co = np.pad(co, (0, total - original_dimensions))
    return out, _join(targets), _join(anchors), co, co_base, errors


def fit(rgb, channel, args, device, prior_errors, extra=None, tiles=None):
    """Train one plane. prior_errors maps earlier channels to per-tile error lists."""
    h, w = rgb.shape[:2]
    tiles = tiles or [(slice(0, h), slice(0, w), None)]
    xf, target, anchor, co, co_base, base_error = tile_inputs(
        rgb, channel, args, prior_errors, tiles, extra
    )
    x = torch.from_numpy(np.ascontiguousarray(xf)).to(device)
    y = torch.from_numpy(target.astype(np.float32)).to(device)
    a = torch.from_numpy(anchor.ravel().astype(np.float32)).to(device)
    model = ProbabilityNet(x.shape[1], args.hidden, args.mixtures, co).to(device)
    model.register_buffer("guide_coeff", torch.from_numpy(co_base.astype(np.float32)).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    count = len(x)
    batch = args.batch
    total_steps = args.epochs * math.ceil(count / batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_factor(getattr(args, "schedule", "constant"), total_steps)
    )
    for epoch in range(args.epochs):
        order = torch.randperm(count, device=device)
        for indices in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = nll(model, x[indices], y[indices], a[indices]).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training objective")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                loss = float(nll(model, x[:65536], y[:65536], a[:65536]).mean().cpu())
            print(f"channel {channel} epoch {epoch + 1}: {loss:.5f} sample bits", flush=True)
    tensors, report = finish(model, x, y, a, device, channel)
    return tensors, report, base_error


def finish(model, x, y, a, device, channel, evaluate=True):
    """Round weights to binary16 in place; estimate bits with only the in-band weights.

    evaluate=False skips the estimate (conditional_bits None).
    """
    tensors = []
    with torch.no_grad():
        for name, value in model.state_dict().items():
            quant = value.detach().cpu().numpy().astype("<f2")
            tensors.append([name, list(quant.shape), quant.tobytes()])
            value.copy_(torch.from_numpy(quant.astype(np.float32)).to(device))
    report = {"channel": channel, "parameters": sum(v.numel() for v in model.parameters())}
    if not evaluate:
        return tensors, {**report, "conditional_bits": None, "conditional_bytes": None}
    bits = 0.0
    count = len(x)
    with torch.no_grad():
        for start in range(0, count, 32768):
            losses = nll(
                model,
                x[start : start + 32768].float(),
                y[start : start + 32768],
                a[start : start + 32768],
            )
            if not torch.isfinite(losses).all():
                raise ValueError("invalid evaluation objective")
            bits += (
                float(losses.double().sum().cpu()) if device == "cpu" else float(losses.sum().cpu())
            )
    return tensors, {**report, "conditional_bits": bits, "conditional_bytes": bits / 8}


def wait_for_local_fits(args):
    """args.local_ready, if set, blocks until the args.local_fits file is complete."""
    ready = getattr(args, "local_ready", None)
    if ready is not None:
        ready()


def features_ready(args):
    """Calls args.features_ready, if set, once this model's features exist."""
    ready = getattr(args, "features_ready", None)
    if ready is not None:
        ready()


def fit_planes(rgb, args, device, tiles, extras=None):
    """Train G, R, B together. Planes share no weights or optimizer state.

    Plane inputs depend only on pixels and transmitted guides, never on another
    plane's trained network, so joint training changes nothing in the format.
    """
    mark = time.perf_counter()
    planes, targets, anchors, prior, dimensions = [], [], [], {}, []
    feature_set = getattr(args, "feature_set", 0)
    expected = [coder.feature_dimensions(c, args.causal_context, feature_set)[0] for c in (1, 0, 2)]
    fused = getattr(args, "fused", False) and fused_supported(device, expected, args)
    x = None
    with ThreadPoolExecutor(3) as pool:
        prepared = list(pool.map(lambda c: plane_bases(rgb, c, tiles), (1, 0, 2)))
    local = None
    if feature_set in coder.FITS:
        wait_for_local_fits(args)
        local = local_planes(rgb, getattr(args, "local_fits", None), feature_set)
    for g, channel in enumerate((1, 0, 2)):
        xf, target, anchor, co, co_base, errors = tile_inputs(
            rgb,
            channel,
            args,
            prior,
            tiles,
            local=local,
            prepared=prepared[g],
            binary16=max(expected) if fused else None,
        )
        prepared[g] = None
        if len(co) != expected[g]:
            raise ValueError("feature dimensions differ from the bitstream layout")
        prior[channel] = errors
        planes.append((channel, co, co_base))
        dimensions.append(len(co))
        # Upload each plane as soon as it exists; host copies of all three are large.
        if fused:
            if x is None:
                x = torch.empty(3, len(xf), max(expected), device=device, dtype=torch.float16)
            x[g].copy_(torch.from_numpy(xf))
        else:
            if x is None:
                x = torch.zeros(3, len(xf), max(expected), device=device)
            # The column-major rows go up as their contiguous transpose.
            x[g, :, : xf.shape[1]] = torch.from_numpy(xf.T).to(device).T
        del xf
        targets.append(target.astype(np.float32))
        anchors.append(anchor.ravel().astype(np.float32))
    count = len(targets[0])
    y = torch.from_numpy(np.stack(targets)).to(device)
    a = torch.from_numpy(np.stack(anchors)).to(device)
    del targets, anchors
    nets = []
    for (_, co, co_base), d in zip(planes, dimensions, strict=True):
        net = ProbabilityNet(d, args.hidden, args.mixtures, co).to(device)
        net.register_buffer("guide_coeff", torch.from_numpy(co_base.astype(np.float32)).to(device))
        nets.append(net)
    model = GroupedNet(nets, dimensions).to(device)
    print(f"features and upload: {time.perf_counter() - mark:.2f} s", flush=True)
    features_ready(args)
    mark = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch = args.batch
    total_steps = args.epochs * math.ceil(count / batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_factor(getattr(args, "schedule", "constant"), total_steps)
    )

    if fused:
        del model, optimizer
        extras = {} if extras is None else extras
        extras["header"] = (
            rgb.shape[1],
            rgb.shape[0],
            args.hidden,
            args.mixtures,
            args.causal_context,
            getattr(args, "feature_set", 0),
        )
        extras["pixels"] = rgb.shape[0] * rgb.shape[1]
        return _fit_fused(args, x, y, a, nets, dimensions, planes, prior, count, mark, extras)

    def objective(net, xb, yb, ab):
        return mixture_bits(*net(xb), yb, ab).mean(1).sum()

    if getattr(args, "compile", False):
        # Fused kernels round differently, so compiled training is a different
        # (not byte-identical) encoder setting.
        objective = torch.compile(objective)
    for epoch in range(args.epochs):
        order = torch.randperm(count, device=device)
        for indices in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, x[:, indices], y[:, indices], a[:, indices])
            loss.backward()
            clip_per_plane(model.parameters(), 10)
            optimizer.step()
            scheduler.step()
        if not torch.isfinite(loss):
            raise ValueError("non-finite training objective")
        if epoch % 10 == 9:
            with torch.no_grad():
                probe = mixture_bits(*model(x[:, :65536]), y[:, :65536], a[:, :65536]).mean(1)
            print(f"epoch {epoch + 1}: {probe.cpu().tolist()} sample bits", flush=True)
    model.export(nets, dimensions)
    print(f"optimization: {time.perf_counter() - mark:.2f} s", flush=True)
    del model, optimizer
    results = []
    for g, (channel, *_rest) in enumerate(planes):
        xs = x[g, :, : dimensions[g]].contiguous()
        results.append(finish(nets[g], xs, y[g], a[g], device, channel))
    return results, prior


def fused_supported(device, dimensions, args):
    """The fused trainer runs on Metal, and its PyTorch copy (cpu_training) elsewhere."""
    supported = metal_training.supported(max(dimensions), args.hidden, args.mixtures)
    return device in ("mps", "cpu", "cuda") and supported


def grouped_trainer(nets, dimensions, x, y, a, lr, max_norm=10.0):
    if x.device.type == "mps":
        return metal_training.FusedGroupedTrainer(nets, dimensions, x, y, a, lr, max_norm)
    return cpu_training.TorchGroupedTrainer(nets, dimensions, x, y, a, lr, max_norm)


def evaluator(x, y, a, hidden, mixtures, dimensions):
    if x.device.type == "mps":
        return metal_training.FusedEvaluator(x, y, a, hidden, mixtures, dimensions)
    return cpu_training.TorchEvaluator(x, y, a, hidden, mixtures, dimensions, loss=coded_bits)


class Dropped(Exception):
    """A width race stopped this model's training early."""


def _train_fused(fused, count, batch, epochs, lr, schedule, bits=None, checkpoints=(), prune=None):
    """checkpoints: (epoch, callback) pairs; a callback returning False raises Dropped.

    Each callback gets each plane's mean training bits per sample over the last
    eighth (at least one) of the epochs before its checkpoint. prune(epoch), if
    given, runs before every epoch and consumes no random numbers.
    """
    factor = learning_rate_factor(schedule, epochs * math.ceil(count / batch))
    step = 0
    seen, samples = [None] * len(checkpoints), [0] * len(checkpoints)
    windows = [max(1, at // 8) for at, _ in checkpoints]
    for epoch in range(epochs):
        if prune is not None:
            prune(epoch)
        averaging = [i for i, (at, _) in enumerate(checkpoints) if 1 <= at - epoch <= windows[i]]
        order = torch.randperm(count, device=fused.x.device).to(torch.int32)
        for indices in order.split(batch):
            lost = fused.step(indices, lr * factor(step), bits=bits)
            if averaging:
                # Before the next step reuses the bits buffer: the queue is in order.
                total = lost.sum(1)
                for i in averaging:
                    seen[i] = total if seen[i] is None else seen[i] + total
                    samples[i] += lost.shape[1]
            step += 1
        if not torch.isfinite(lost).all():
            raise ValueError("non-finite training objective")
        if epoch % 10 == 9:
            print(f"epoch {epoch + 1}: {lost.mean(1).cpu().tolist()} batch bits", flush=True)
        for i, (at, callback) in enumerate(checkpoints):
            if epoch + 1 == at and not callback((seen[i] / samples[i]).cpu().tolist()):
                raise Dropped


def _states(nets):
    return [
        {
            name: value.detach().cpu().numpy().astype("<f2").tobytes()
            for name, value in net.state_dict().items()
        }
        for net in nets
    ]


# First-layer input pruning in fused training. At these fractions of the float
# epochs each plane drops up to PRUNE_FRACTION of its live input columns, those
# whose removal costs fewest estimated bytes, if under PRUNE_BYTES; the
# constant input folds into the bias first. Pruned weights stay exactly zero,
# and the model coder stores a zero column in a few bits. A column's affine
# weight stays live (pruning it too cost 0.33% on the development crops).
# The estimate is taken before the network adapts, so it overstates the cost:
# on the development crops a 100-byte limit coded 0.23% smaller than the old
# 15 bytes, 25 to 60 bytes about 0.1% smaller, and no limit 5.1% larger.
PRUNE_AT = ((1, 4), (3, 8), (1, 2), (5, 8))
PRUNE_FRACTION = 0.5
PRUNE_BYTES = 100.0
# Scores use at most this many evenly strided training rows.
PRUNE_SAMPLES = 1 << 20


class Pruner:
    def __init__(self, fused, epochs: int, pixels: int):
        self.fused = fused
        self.epochs = {max(1, epochs * a // b) for a, b in PRUNE_AT}
        self.last = max(self.epochs)
        self.pixels = pixels
        self.pruned: list[set[int]] = [set() for _ in range(fused.g)]
        self.evaluator = None

    def __call__(self, epoch: int) -> None:
        if epoch not in self.epochs:
            return
        fused, h = self.fused, self.fused.h
        if self.evaluator is None:
            # Input 0 is the guide's constant feature, 1 / 32 after scaling.
            fused.fold(0, 1 / 32)
            for plane in self.pruned:
                plane.add(0)
            stride = -(-fused.n // PRUNE_SAMPLES)
            x, y, a = (v[:, ::stride] if stride > 1 else v for v in (fused.x, fused.y, fused.a))
            self.evaluator = evaluator(x, y, a, h, fused.k, fused.dimensions)
            self.scale = self.pixels / x.shape[1]
        offset = fused.lay["w1a"][0]
        params = fused.params.cpu().numpy()
        planes, blocks, live = [], [], []
        for g in range(fused.g):
            columns = [j for j in range(fused.dimensions[g]) if j not in self.pruned[g]]
            planes.append(g)
            blocks.append(params[g])
            for j in columns:
                block = params[g].copy()
                block[offset + j * (h + 1) : offset + j * (h + 1) + h] = 0.0
                planes.append(g)
                blocks.append(block)
            live.append(columns)
        bits = self.evaluator.bits(planes, np.stack(blocks))
        chosen, position = [], 0
        for g, columns in enumerate(live):
            costs = sorted(
                ((bits[position + 1 + i] - bits[position]) * self.scale / 8, j)
                for i, j in enumerate(columns)
            )
            position += 1 + len(columns)
            limit = int(len(columns) * PRUNE_FRACTION)
            chosen.append([j for cost, j in costs[:limit] if cost < PRUNE_BYTES])
            self.pruned[g].update(chosen[-1])
        fused.prune(chosen)
        if epoch == self.last:
            self.evaluator = None


RACE_BITS = 10
# Weight estimates on large images use evenly strided training rows; candidate
# rankings need far fewer samples than training. On the 12 development natives
# 2^19 rows instead of 2^21 changed the total by -0.001% (no image by more than
# 0.013%) and halved the search. With GPTQ candidates, 2^18 instead of 2^19
# cost 0.024% on three development natives and took their weight search from
# 15 s to 10 s; a 512 by 512 crop has 2^18 rows. (Subsampling the guide fit the
# same way cost 0.16% on a 2400 by 2400 native, so the guide sees every row.)
SEARCH_SAMPLES = 1 << 18


def _estimator(x, y, a, dimensions, mixtures, pixels):
    """Estimator over at most about SEARCH_SAMPLES rows, and its bits-to-image scale."""
    stride = -(-x.shape[1] // SEARCH_SAMPLES)
    if stride > 1:
        x, y, a = (v[:, ::stride].contiguous() for v in (x, y, a))
    estimator = model_search.Estimator(x, y, a, dimensions, mixtures, coded_bits)
    return estimator, pixels / x.shape[1]


# Races score each width by its planes' mean training bits over the epochs
# just before the checkpoint, not by an estimate of the weights at the
# checkpoint: at a 32nd of training on the development crops, the two widths'
# ratio of such estimates moved by up to 15% between runs whose learning rates
# differed by a millionth, and the ratio of averaged bits by at most 1.5%.
def race_score(fused, nets, mean_bits, pixels, absorb=None):
    """Estimated plane bytes: mean_bits per sample of each plane, plus its RACE_BITS weights.

    With absorb, also the score that counts only that share of the bits (see mixing_absorb).
    """
    snapshot = [copy.deepcopy(net) for net in nets]
    fused.export(snapshot)
    states = [model_search.round_state(state, RACE_BITS) for state in _states(snapshot)]
    stream = sum(b * pixels / 8 for b in mean_bits)
    weights = sum(model_search.plane_bytes(state) for state in states)
    if absorb is None:
        return stream + weights
    return stream + weights, stream * absorb + weights


def _fit_fused(args, x, y, a, nets, dimensions, planes, prior, count, mark, extras):
    """fit_planes with the fused step (Metal, or its CPU copy): same objective, different rounding.

    With args.search, quantization-aware snapshots continue from the trained
    model (one chain through args.qat_bits) and model_search picks each
    plane's weights by estimated size. With args.race, training reports its
    race_score at args.race.checkpoint(epochs) and stops (Dropped) unless
    args.race.report says to continue, and with args.mixing_absorb again at
    args.race.late_checkpoint(epochs) (args.race.report_late); reports consume
    no random numbers, so a model that continues is the one trained without a
    race. With args.estimate, the score also goes to args.estimate, at the
    race's first checkpoint or else at epoch args.estimate_at.
    """
    fused = grouped_trainer(nets, dimensions, x, y, a, args.lr, getattr(args, "max_norm", 10.0))
    schedule = getattr(args, "schedule", "constant")
    race = getattr(args, "race", None)
    estimate = getattr(args, "estimate", None)
    checkpoints = []
    if race is not None or estimate is not None:

        def report(mean_bits):
            absorb = getattr(args, "mixing_absorb", None)
            score = race_score(fused, nets, mean_bits, extras["pixels"], absorb)
            score, absorbed = score if absorb is not None else (score, None)
            if estimate is not None:
                estimate(score)
            if race is None:
                return True
            return race.report(args.hidden, getattr(args, "feature_set", 0), score, absorbed)

        at = race.checkpoint(args.epochs) if race is not None else args.estimate_at
        checkpoints.append((at, report))
    absorb = getattr(args, "mixing_absorb", None)
    late = None if race is None or absorb is None else race.late_checkpoint(args.epochs)
    if late is not None:

        def report_late(mean_bits):
            score, absorbed = race_score(fused, nets, mean_bits, extras["pixels"], absorb)
            return race.report_late(args.hidden, getattr(args, "feature_set", 0), score, absorbed)

        checkpoints.append((late, report_late))
    pruner = Pruner(fused, args.epochs, extras["pixels"]) if getattr(args, "prune", False) else None
    _train_fused(
        fused,
        count,
        args.batch,
        args.epochs,
        args.lr,
        schedule,
        checkpoints=checkpoints,
        prune=pruner,
    )
    if pruner is not None:
        extras["pruned_inputs"] = [len(plane) for plane in pruner.pruned]
    fused.export(nets)
    print(f"optimization: {time.perf_counter() - mark:.2f} s", flush=True)
    # The weight search makes its own estimates; the float model's are only a report.
    search = getattr(args, "search", False)
    results = []
    for g, (channel, *_rest) in enumerate(planes):
        results.append(
            finish(
                nets[g],
                x[g, :, : dimensions[g]],
                y[g],
                a[g],
                x.device.type,
                channel,
                evaluate=not search,
            )
        )
    if search:
        mark = time.perf_counter()
        sources = [{"float": {name: raw for name, _shape, raw in state}} for state, _ in results]
        snapshot = [copy.deepcopy(net) for net in nets]
        for bits in getattr(args, "qat_bits", ()):
            _train_fused(fused, count, args.batch, args.qat_epochs, args.qat_lr, "cosine", bits)
            fused.export(snapshot, fused.quantized(bits))
            for g, state in enumerate(_states(snapshot)):
                sources[g][f"qat{bits}"] = model_search.round_state(state, bits)
        estimator, scale = _estimator(x, y, a, dimensions, args.mixtures, extras["pixels"])
        extras["search"] = model_search.search(estimator, extras["header"], sources, scale)
        absorb = getattr(args, "mixing_absorb", None)
        if absorb is not None:
            # A context model mixed in absorbs most of what coarser weights cost the network.
            mixing = model_search.search(estimator, extras["header"], sources, scale * absorb)
            # Its estimate with every stream bit counted, as for a file of the network alone.
            fixed = len(mixing["payload"]) + coder.HEADER_SIZE
            mixing["network_estimated_bytes"] = (mixing["estimated_bytes"] - fixed) / absorb + fixed
            extras["mixing_search"] = mixing
        print(f"weight search: {time.perf_counter() - mark:.2f} s", flush=True)
    return results, prior


def resolve_device(name: str) -> str:
    """auto: Metal, else a CUDA GPU (NVIDIA, or AMD under ROCm), else the CPU."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(path, args):
    start = time.perf_counter()
    torch.manual_seed(getattr(args, "seed", 20261023))
    rgb = read_rgb(path, max_pixels=coder.MAX_PIXELS)
    device = resolve_device(args.device)
    baseline = None
    if getattr(args, "baseline", True):
        baseline = imagecodecs.jpegxl_encode(rgb, lossless=True, effort=9, numthreads=1)
        assert np.array_equal(imagecodecs.jpegxl_decode(baseline, index=0, numthreads=1), rgb)
    tiles = training_tiles(*rgb.shape[:2], getattr(args, "train_pixels", MAX_TRAIN_PIXELS))
    training_pixels = sum(
        rgb.shape[0] * rgb.shape[1] if mask is None else int(mask.sum()) for *_, mask in tiles
    )
    tensors = []
    planes = []
    extras: dict = {}
    if getattr(args, "sequential", False):
        wait_for_local_fits(args)
        features_ready(args)
        prior_errors = {}
        for channel in (1, 0, 2):
            state, report, error = fit(rgb, channel, args, device, prior_errors, tiles=tiles)
            prior_errors[channel] = error
            tensors.append(state)
            planes.append(report)
    else:
        for state, report in fit_planes(rgb, args, device, tiles, extras)[0]:
            tensors.append(state)
            planes.append(report)
    for report in planes:
        print(path.name, report, flush=True)
    feature_set = getattr(args, "feature_set", 0)
    header = (rgb.shape[1], rgb.shape[0], args.hidden, args.mixtures, args.causal_context)
    payload = coder.encode_model(
        (*header, feature_set), [{name: raw for name, _shape, raw in state} for state in tensors]
    )
    stem = args.out.parent / (args.out.stem + "-" + path.stem)
    v3_path = stem.with_name(stem.name + "-model.nnm")
    v3_path.write_bytes(payload)
    search = extras.get("search")
    if search is not None:
        selected_path = stem.with_name(stem.name + "-selected.nnm")
        selected_path.write_bytes(search["payload"])
        uniform_path = stem.with_name(stem.name + "-uniform.nnm")
        uniform_path.write_bytes(search["uniform_payload"])
        search = {
            "selected_file": str(selected_path),
            "uniform_file": str(uniform_path),
            **{k: v for k, v in search.items() if not k.endswith("payload")},
        }
        mixing = extras.get("mixing_search")
        if mixing is not None:
            mixing_path = stem.with_name(stem.name + "-mixing.nnm")
            mixing_path.write_bytes(mixing["payload"])
            search["mixing_file"] = str(mixing_path)
            search["mixing_labels"] = mixing["labels"]
            search["mixing_estimated_bytes"] = mixing["network_estimated_bytes"]
            search["mixing_absorbed_bytes"] = mixing["estimated_bytes"]
    model_path = None
    if feature_set == 0:
        metadata = compress(encode_cbor([*header, tensors]))
        model_path = stem.with_name(stem.name + "-model.cbor.zst")
        model_path.write_bytes(metadata)
    whole = training_pixels == rgb.shape[0] * rgb.shape[1]
    total = None
    if all(p["conditional_bytes"] is not None for p in planes):
        total = sum(p["conditional_bytes"] for p in planes) + len(payload) + coder.HEADER_SIZE
    return {
        "image": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline_jxl_e9": None if baseline is None else len(baseline),
        "training_pixels": training_pixels,
        "training_tiles": len(tiles),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "mixtures": args.mixtures,
        "causal_context": args.causal_context,
        "schedule": getattr(args, "schedule", "constant"),
        "lr": args.lr,
        "batch": args.batch,
        "feature_set": feature_set,
        "training": "sequential" if getattr(args, "sequential", False) else "grouped",
        "compiled": bool(getattr(args, "compile", False)),
        "fused": bool(getattr(args, "fused", False)),
        "search": search,
        "pruned_inputs": extras.get("pruned_inputs"),
        "model_bytes": len(payload),
        "planes": planes,
        "model_file": None if model_path is None else str(model_path),
        "model_v3_file": str(v3_path),
        "estimated_bytes": total if whole else None,
        "ratio": total / len(baseline) if whole and baseline and total is not None else None,
        "device": device,
        "seconds": time.perf_counter() - start,
        "warning": (
            "Rounded per-image weights included. This record is an entropy estimate only "
            "(over training pixels when tiled); use neural_bitstream.py for actual coding "
            "and independent decoding."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--mixtures", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--causal-context", action="store_true")
    parser.add_argument("--feature-set", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sequential", action="store_true", help="historical per-plane loop")
    parser.add_argument("--compile", action="store_true", help="torch.compile the grouped loss")
    parser.add_argument("--fused", action="store_true", help="fused Metal training step on MPS")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = []
    for path in args.images:
        r = run(path, args)
        rows.append(r)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(r), flush=True)
