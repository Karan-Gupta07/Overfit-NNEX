"""Float prototype: causal-window least-squares predictions as extra network inputs.

Estimates only. A decodable version needs a deterministic integer solver.
"""

from __future__ import annotations

import numpy as np
import predictor_probe as linear
import torch

STRIPE = 32


def _box(cumulative, y0, y1, x0, x1):
    """Sum over rows [y0, y1) and columns [x0, x1) per pixel; inputs are index arrays."""
    return (
        cumulative[y1[:, None], x1[None, :]]
        - cumulative[y0[:, None], x1[None, :]]
        - cumulative[y1[:, None], x0[None, :]]
        + cumulative[y0[:, None], x0[None, :]]
    )


def causal_window_sums(maps: torch.Tensor, radius: int, first: int) -> torch.Tensor:
    """Window sums for rows first..end of maps: rows y-radius..y-1 (x±radius) and row y left."""
    h, w, c = maps.shape
    cumulative = torch.zeros(h + 1, w + 1, c, dtype=maps.dtype)
    cumulative[1:, 1:] = maps.cumsum(0).cumsum(1)
    ys = torch.arange(first, h)
    xs = torch.arange(w)
    above = _box(
        cumulative,
        (ys - radius).clamp(min=0),
        ys,
        (xs - radius).clamp(min=0),
        (xs + radius + 1).clamp(max=w),
    )
    left = _box(cumulative, ys, ys + 1, (xs - radius).clamp(min=0), xs)
    return above + left


def plane_columns(rgb: np.ndarray, channel: int, taps: int):
    """Same target as the trainer, with causal own-plane taps and decoded-plane guides."""
    green = rgb[..., 1].astype(np.float64)
    if channel == 1:
        a = green
        west, north, nw = (linear.shift(a, *d) for d in ((0, -1), (-1, 0), (-1, -1)))
        anchor = np.maximum(
            np.minimum(west, north), np.minimum(np.maximum(west, north), west + north - nw)
        )
        columns = [linear.shift(a, *d) - anchor for d in linear.CAUSAL[:taps]]
        return a - anchor, columns
    diff = rgb[..., channel].astype(np.float64) - green
    columns = [linear.shift(diff, *d) for d in linear.CAUSAL[:taps]]
    columns += [linear.shift(green, *d) - green for d in linear.GUIDE[1:5]]
    if channel == 2:
        rg = rgb[..., 0].astype(np.float64) - green
        columns += [linear.shift(rg, *d) for d in linear.GUIDE[:5]]
    return diff, columns


def local_least_squares(target, columns, radius=8, ridge=0.1):
    """Per-pixel ridge LS fit on a causal window; returns prediction and window fit RMS."""
    h, w = target.shape
    f_all = torch.from_numpy(np.stack(columns, axis=-1))
    t_all = torch.from_numpy(target)[..., None]
    k = f_all.shape[-1]
    prediction = np.zeros((h, w))
    rms = np.zeros((h, w))
    eye = torch.eye(k, dtype=f_all.dtype)
    for start in range(0, h, STRIPE):
        first = max(0, start - radius)
        stop = min(h, start + STRIPE)
        f = f_all[first:stop]
        t = t_all[first:stop]
        rows = stop - first
        outer = (f[..., :, None] * f[..., None, :]).reshape(rows, w, k * k)
        maps = torch.cat([outer, f * t, t * t, torch.ones(rows, w, 1, dtype=f.dtype)], dim=-1)
        sums = causal_window_sums(maps, radius, start - first)
        gram = sums[..., : k * k].reshape(-1, w, k, k)
        rhs = sums[..., k * k : k * k + k]
        tt = sums[..., k * k + k]
        count = sums[..., k * k + k + 1]
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
        system = gram + (ridge * scale)[..., None, None] * eye
        beta = torch.linalg.solve(system.reshape(-1, k, k), rhs.reshape(-1, k, 1)).reshape(-1, w, k)
        here = f[start - first :]
        prediction[start:stop] = (here * beta).sum(-1).numpy()
        residual = (tt - (beta * rhs).sum(-1)).clamp_min(0) / count.clamp_min(1)
        rms[start:stop] = residual.sqrt().numpy()
    return prediction, rms


def extra_features(rgb: np.ndarray, channel: int, radius=8, ridge=0.1, taps=12):
    target, columns = plane_columns(rgb, channel, taps)
    prediction, rms = local_least_squares(target, columns, radius, ridge)
    return np.stack([prediction, rms], axis=-1).reshape(-1, 2), target
