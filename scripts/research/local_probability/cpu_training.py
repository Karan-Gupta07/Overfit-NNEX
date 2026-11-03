"""The fused trainer's step and evaluation on the CPU, in plain PyTorch.

TorchGroupedTrainer keeps FusedGroupedTrainer's parameter blocks, its Adam
with per-plane clipping, its pruning mask and its post-training grid, so the
width race, quantization-aware snapshots, input pruning and weight search run
unchanged without Metal. The arithmetic rounds differently from the Metal
kernels, so its models are close to the Metal trainer's, not identical.
Training only: the decoder never runs this code.
"""

from __future__ import annotations

import math

import metal_training
import numpy as np
import torch
import torch.nn.functional as F


class TorchGroupedTrainer(metal_training.FusedGroupedTrainer):
    """FusedGroupedTrainer whose step runs in PyTorch on any device.

    The objective is neural_probability_probe.mixture_bits, as in the fused
    kernel. Its gradients are written out by hand: at batch 4096 a step costs
    a few dozen small tensor operations, and autograd's several hundred took
    about twice as long on the CPU.
    """

    def __init__(self, nets, dimensions, x, y, a, lr: float, max_norm: float = 10.0):
        super().__init__(nets, dimensions, x, y, a, lr, max_norm, half_scratch=False)

    def _compile(self):
        return None

    def _stacked(self, params: torch.Tensor, name: str) -> torch.Tensor:
        offset, shape = self.lay[name]
        return params[:, offset : offset + shape[0] * shape[1]].view(self.g, *shape)

    def gradients(self, index: torch.Tensor, params: torch.Tensor | None = None):
        """FusedGroupedTrainer.gradients: (w1a, w2b, w3b) gradient blocks and per-sample bits."""
        grads, bits = self._gradients(index, self.params if params is None else params)
        d, h, r = self.d, self.h, 3 * self.k
        g1, g2, g3 = (self._stacked(grads, name) for name in ("w1a", "w2b", "w3b"))
        return g1[:, : d + 1, : h + 1], g2[:, : h + 1, :h], g3[:, : h + 1, :r], bits

    @torch.no_grad()
    def _gradients(self, index: torch.Tensor, params: torch.Tensor):
        """Gradient (params layout) of the planes' summed mean bits, and each sample's bits."""
        d, h, k = self.d, self.h, self.k
        w1a, w2b, w3b = (self._stacked(params, name) for name in ("w1a", "w2b", "w3b"))
        # Inputs past a plane's own count are zero, as are their weights' gradients.
        x = self.x.index_select(1, index).float()
        target = self.y.index_select(1, index)[..., None]
        anchor = self.a.index_select(1, index)[..., None]
        first = torch.bmm(x, w1a[:, :d])
        z1 = torch.relu(first[..., :h] + w1a[:, d : d + 1, :h])
        z2 = torch.relu(torch.baddbmm(w2b[:, h : h + 1], z1, w2b[:, :h]))
        raw = torch.baddbmm(w3b[:, h : h + 1], z2, w3b[:, :h])
        log_scale = raw[..., 2 * k :]
        mean = raw[..., k : 2 * k] + first[..., h : h + 1]
        # As mixture_bits rounds them.
        scale = torch.exp(log_scale.clamp(-4.0, 7.0))
        upper, lower = (target + 0.5 - mean) / scale, (target - 0.5 - mean) / scale
        top, bottom = (255.5 - anchor - mean) / scale, (-0.5 - anchor - mean) / scale
        inv = upper - lower
        logmix = torch.log_softmax(raw[..., :k], -1)
        joint = (
            logmix
            + F.logsigmoid(upper)
            + F.logsigmoid(-lower)
            + torch.log(-torch.expm1(-inv).clamp(max=-1e-12))
        )
        lp = torch.logsumexp(joint, -1, keepdim=True)
        s_top, s_bottom = torch.sigmoid(top), torch.sigmoid(bottom)
        mass = s_top - s_bottom
        normed = logmix + torch.log(mass.clamp_min(1e-10))
        normal = torch.logsumexp(normed, -1, keepdim=True)
        lost = (normal - lp)[..., 0] / math.log(2)
        # Posterior weights of the bin's and the range's mixtures, times d loss / d bits.
        per = 1.0 / (index.numel() * math.log(2))
        w = torch.exp(joint - lp).mul_(per)
        v = torch.exp(normed - normal).mul_(per)
        v_mass = torch.where(mass >= 1e-10, v / mass, 0.0)
        d_top, d_bottom = s_top * (1 - s_top), s_bottom * (1 - s_bottom)
        s_upper, s_lower = torch.sigmoid(upper), torch.sigmoid(lower)
        d_mean = inv * (v_mass * (d_bottom - d_top) - w * (s_upper + s_lower - 1))
        d_log_scale = v_mass * (bottom * d_bottom - top * d_top) - w * (
            upper * (s_upper - 1) + lower * s_lower - inv / torch.expm1(inv)
        )
        d_log_scale = torch.where((log_scale >= -4.0) & (log_scale <= 7.0), d_log_scale, 0.0)
        d_raw = torch.cat((v - w, d_mean, d_log_scale), -1)
        grads = torch.zeros_like(params)
        g1, g2, g3 = (self._stacked(grads, name) for name in ("w1a", "w2b", "w3b"))
        g3[:, :h] = torch.bmm(z2.transpose(1, 2), d_raw)
        g3[:, h] = d_raw.sum(1)
        d_z = torch.bmm(d_raw, w3b[:, :h].transpose(1, 2)).mul_(z2 > 0)
        g2[:, :h] = torch.bmm(z1.transpose(1, 2), d_z)
        g2[:, h] = d_z.sum(1)
        d_z = torch.bmm(d_z, w2b[:, :h].transpose(1, 2)).mul_(z1 > 0)
        g1[:, :d] = torch.bmm(x.transpose(1, 2), torch.cat((d_z, d_mean.sum(-1, keepdim=True)), -1))
        g1[:, d, :h] = d_z.sum(1)
        return grads, lost

    @torch.no_grad()
    def quantized(self, bits: int) -> torch.Tensor:
        """Parameters with hidden/head weights on the post-training precision grid.

        As the quantize kernel: biases (last rows) and the affine column pass
        through, and each plane's tensor scales by a power of two.
        """
        limit = float(2 ** (bits - 1) - 1)
        out = self.params.clone()
        for name, (rows, cols) in (
            ("w1a", (self.d, self.h)),
            ("w2b", (self.h, self.h)),
            ("w3b", (self.h, 3 * self.k)),
        ):
            region = self._stacked(out, name)[:, :rows, :cols]
            top = region.abs().amax((1, 2), keepdim=True)
            live = top > 0
            scale = torch.where(live, torch.exp2(torch.floor(torch.log2(limit / top))), 1.0)
            region.copy_(torch.where(live, torch.round(region * scale) / scale, region))
        return out

    def step(self, index: torch.Tensor, lr: float, bits: int | None = None):
        """One Adam step; with `bits`, a straight-through quantization-aware step."""
        forward = self.params if bits is None else self.quantized(bits)
        grads, lost = self._gradients(index, forward)
        self.steps += 1
        bc1 = 1 - 0.9**self.steps
        bc2 = 1 - 0.999**self.steps
        with torch.no_grad():
            # Pruned weights stay zero; their gradients still count in the clipping norm.
            norm = grads.square().sum(1, keepdim=True).sqrt()
            grads.mul_((self.max_norm / (norm + 1e-6)).clamp(max=1.0))
            live = self.keep != 0
            self.m.copy_(torch.where(live, 0.9 * self.m + 0.1 * grads, self.m))
            self.v.copy_(torch.where(live, 0.999 * self.v + 0.001 * grads * grads, self.v))
            update = (lr / bc1) * self.m / (self.v.sqrt() / math.sqrt(bc2) + 1e-8)
            self.params.sub_(torch.where(live, update, 0.0))
        return lost


class TorchEvaluator:
    """FusedEvaluator.bits on any PyTorch device: summed bits of candidate weight sets."""

    # Candidates scored together and samples per pass, bounding the scratch.
    GROUP = 32
    CHUNK = 1 << 13

    def __init__(self, x, y, a, hidden: int, mixtures: int, dimensions=None, *, loss):
        """loss(logits, means, scales, target, anchor) gives each sample's coded bits."""
        self.g, self.n, self.d = x.shape
        self.h, self.k = hidden, mixtures
        self.lay = metal_training.layout(self.d, hidden, mixtures)
        self.x = x.to(torch.float16).contiguous()
        self.y = y.contiguous()
        self.a = a.contiguous()
        self.loss = loss

    def _blocks(self, params: torch.Tensor, name: str) -> torch.Tensor:
        offset, shape = self.lay[name]
        return params[:, offset : offset + shape[0] * shape[1]].view(len(params), *shape)

    @torch.no_grad()
    def bits(self, planes: list[int], blocks: np.ndarray) -> list[float]:
        """Total bits for candidate i: weights blocks[i] (pack layout) on plane planes[i]."""
        d, h, k = self.d, self.h, self.k
        params = torch.from_numpy(np.ascontiguousarray(blocks, np.float32)).to(self.x.device)
        out = [0.0] * len(planes)
        for g in sorted(set(planes)):
            chosen = [i for i, plane in enumerate(planes) if plane == g]
            for start in range(0, len(chosen), self.GROUP):
                group = chosen[start : start + self.GROUP]
                p = params[group]
                w1a, w2b, w3b = (self._blocks(p, name) for name in ("w1a", "w2b", "w3b"))
                total = torch.zeros(len(group), dtype=torch.float64, device=p.device)
                for at in range(0, self.n, self.CHUNK):
                    x = self.x[g, at : at + self.CHUNK].float()
                    z = torch.relu(
                        torch.baddbmm(
                            w1a[:, d : d + 1, :h], x.expand(len(group), -1, -1), w1a[:, :d, :h]
                        )
                    )
                    z = torch.relu(torch.baddbmm(w2b[:, h : h + 1], z, w2b[:, :h]))
                    raw = torch.baddbmm(w3b[:, h : h + 1], z, w3b[:, :h])
                    means = raw[..., k : 2 * k] + torch.matmul(x, w1a[:, :d, h : h + 1])
                    scales = torch.exp(raw[..., 2 * k :].clamp(-4.0, 7.0))
                    lost = self.loss(
                        raw[..., :k],
                        means,
                        scales,
                        self.y[g, at : at + self.CHUNK],
                        self.a[g, at : at + self.CHUNK],
                    )
                    total += lost.sum(1).double()
                for i, value in zip(group, total.tolist(), strict=True):
                    out[i] = value
        return out
