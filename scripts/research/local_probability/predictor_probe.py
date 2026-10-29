"""Development-only lower bounds for in-band, image-trained exact prediction.

No result here is a real codec file. Build and verify an entropy coder before
promoting any apparent gain. No third-party predictor implementation is used.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import imagecodecs
import numpy as np

from semanticcodec.format.container import compress, encode_cbor

CAUSAL = [
    (0, -1),
    (-1, 0),
    (-1, -1),
    (-1, 1),
    (0, -2),
    (-2, 0),
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (-2, -2),
    (-2, 2),
]
GUIDE = [(0, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (0, 1), (1, 0), (1, -1), (1, 1)]
Q = 4096
TRANSFORM = "rgb"


def shift(a, dy, dx):
    h, w = a.shape
    yy = np.clip(np.arange(h) + dy, 0, h - 1)
    xx = np.clip(np.arange(w) + dx, 0, w - 1)
    # Missing causal samples are zero, not the undecoded current sample.
    if (dy, dx) in CAUSAL:
        out = np.zeros_like(a)
        y0 = max(0, -dy)
        y1 = min(h, h - dy)
        x0 = max(0, -dx)
        x1 = min(w, w - dx)
        out[y0:y1, x0:x1] = a[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        return out
    return a[yy[:, None], xx]


def features(rgb, channel):
    a = rgb[..., channel].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    west, north, nw = shift(a, 0, -1), shift(a, -1, 0), shift(a, -1, -1)
    anchor = (
        np.maximum(np.minimum(west, north), np.minimum(np.maximum(west, north), west + north - nw))
        if channel == 1
        else green
    )
    if TRANSFORM == "ycocg" and channel != 1:
        anchor = np.zeros_like(a)
    columns = [np.ones_like(a)]
    if channel == 1:
        columns += [shift(a, *d) - anchor for d in CAUSAL]
        columns += [np.clip(west + north - nw, 0, 255) - anchor]
    else:
        diff = a - green if TRANSFORM == "rgb" else a
        columns += [shift(diff, *d) for d in CAUSAL]
        # Previously decoded full green plane, including future positions.
        columns += [
            (shift(green, *d) - green if TRANSFORM == "rgb" else shift(green, *d))
            for d in (GUIDE[1:] if TRANSFORM == "rgb" else GUIDE)
        ]
        if channel == 2:
            rg = rgb[..., 0].astype(np.int16) - (green if TRANSFORM == "rgb" else 0)
            columns += [shift(rg, *d) for d in GUIDE]
    return np.stack(columns, axis=-1).reshape(-1, len(columns)).astype(np.float64), anchor


def blocksum(a, tile):
    h, w = a.shape[:2]
    tail = a.shape[2:]
    hh = (h + tile - 1) // tile
    ww = (w + tile - 1) // tile
    padded = np.zeros((hh * tile, ww * tile, *tail), dtype=a.dtype)
    padded[:h, :w] = a
    shape = (hh, tile, ww, tile, *tail)
    return padded.reshape(shape).sum(axis=(1, 3))


def expand(a, h, w, tile):
    return np.repeat(np.repeat(a, tile, axis=0), tile, axis=1)[:h, :w]


def fit_plane(rgb, channel, classes=8, tile=8, iterations=3):
    h, w = rgb.shape[:2]
    a = rgb[..., channel].astype(np.int16)
    signed = TRANSFORM == "ycocg" and channel != 1
    alphabet = 512 if signed else 256
    x, anchor = features(rgb, channel)
    target = (a - anchor).ravel().astype(np.float64)
    activity = np.abs(a - shift(a, 0, -1)) + np.abs(a - shift(a, -1, 0))
    block_activity = blocksum(activity.astype(np.float64), tile) / (tile * tile)
    edges = np.quantile(block_activity, np.arange(1, classes) / classes)
    labels = np.searchsorted(edges, block_activity, side="right").astype(np.uint8)
    weights = np.zeros((classes, x.shape[1]), np.int32)
    for _iteration in range(iterations):
        full = expand(labels, h, w, tile).ravel()
        for k in range(classes):
            selected = full == k
            if selected.sum() < x.shape[1] * 2:
                continue
            xx = x[selected]
            yy = target[selected]
            # Iteratively reweighted least squares, approximating an L1 fit.
            co = weights[k].astype(float) / Q
            for _ in range(3):
                error = yy - xx @ co
                importance = 1 / np.maximum(1, np.abs(error))
                matrix = xx.T @ (xx * importance[:, None])
                matrix.flat[:: len(co) + 1] += max(1, float(selected.sum()) * 0.002)
                rhs = xx.T @ (yy * importance)
                co = np.linalg.solve(matrix, rhs)
            weights[k] = np.round(np.clip(co, -128, 128) * Q).astype(np.int32)
        candidate = np.floor((x @ weights.T + Q // 2) / Q) + anchor.reshape(-1, 1)
        candidate = np.clip(candidate, -255 if signed else 0, 255).astype(np.int16)
        # Coded residual is modular; choosing predictors does not change pixels.
        err = ((a.ravel()[:, None] - candidate + alphabet // 2) % alphabet - alphabet // 2).astype(
            np.int16
        )
        magnitude = np.abs(err).astype(np.float64)
        # Summed absolute error approximates Laplacian rate within each block.
        cost = blocksum(magnitude.reshape(h, w, classes), tile)
        labels = cost.argmin(axis=2).astype(np.uint8)
    full = expand(labels, h, w, tile).ravel()
    prediction = candidate[np.arange(h * w), full].reshape(h, w)
    residual = ((a - prediction + alphabet // 2) % alphabet - alphabet // 2).astype(np.int16)
    reconstruction = (prediction + residual + (256 if signed else 0)) % alphabet - (
        256 if signed else 0
    )
    assert np.array_equal(reconstruction, a)
    magnitude = np.abs(residual)
    context_activity = (
        shift(magnitude, 0, -1) + shift(magnitude, -1, 0) + shift(magnitude, -1, -1)
    ) / 3
    scale = np.minimum(15, np.floor(np.log2(context_activity + 1) * 2)).astype(np.int16)
    contexts = (expand(labels, h, w, tile).astype(np.int32) * 16 + scale).ravel()
    symbols = np.where(residual >= 0, 2 * residual, -2 * residual - 1).astype(np.uint16)
    counts = np.bincount(contexts * alphabet + symbols.ravel(), minlength=classes * 16 * alphabet)
    counts = counts.reshape(classes * 16, alphabet)
    totals = counts.sum(axis=1)
    nz = counts > 0
    probs = counts / np.maximum(totals, 1)[:, None]
    entropy_bits = float(-(counts[nz] * np.log2(probs[nz])).sum())
    return {
        "channel": channel,
        "weights": weights,
        "labels": labels,
        "counts": counts.astype("<u4"),
        "entropy_bits": entropy_bits,
        "residual": residual,
        "contexts": contexts,
        "symbols": symbols,
        "features": x,
        "prediction": prediction,
        "mae": float(np.abs(residual).mean()),
    }


def run(p, out, classes=8, tile=8, iterations=3):
    from input_image import read_rgb

    rgb = read_rgb(p)
    start = time.perf_counter()
    baseline = imagecodecs.jpegxl_encode(rgb, lossless=True, effort=9, numthreads=1)
    assert np.array_equal(imagecodecs.jpegxl_decode(baseline, index=0, numthreads=1), rgb)
    if TRANSFORM == "ycocg":
        r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
        co = r - b
        t = b + (co // 2)
        cg = g - t
        y = t + (cg // 2)
        work = np.stack([co, y, cg], axis=2)
        t2 = y - cg // 2
        b2 = t2 - co // 2
        inverse = np.stack([co + b2, cg + t2, b2], axis=2)
        assert np.array_equal(inverse, rgb)
    else:
        work = rgb
    planes = []
    summary = []
    for channel in (1, 0, 2):
        record = fit_plane(work, channel, classes, tile, iterations)
        planes.append(
            [
                record["weights"].shape[1],
                record["weights"].astype("<i4").tobytes(),
                record["labels"].tobytes(),
                record["counts"].tobytes(),
            ]
        )
        summary.append(
            {"channel": channel, "mae": record["mae"], "entropy_bytes": record["entropy_bits"] / 8}
        )
        # Data can support a subsequent actual coder without retraining.
        np.savez(
            out.parent / f"{p.stem}-{TRANSFORM}-plane{channel}.npz",
            weights=record["weights"],
            labels=record["labels"],
            counts=record["counts"],
            residual=record["residual"],
            contexts=record["contexts"],
            symbols=record["symbols"],
        )
        print(p.name, summary[-1], flush=True)
    metadata = compress(encode_cbor([rgb.shape[1], rgb.shape[0], classes, tile, planes]))
    entropy_bytes = sum(s["entropy_bytes"] for s in summary)
    total_lower_bound = entropy_bytes + len(metadata) + 16
    result = {
        "image": p.name,
        "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "classes": classes,
        "tile": tile,
        "iterations": iterations,
        "transform": TRANSFORM,
        "baseline_jxl_e9": len(baseline),
        "planes": summary,
        "model_bytes": len(metadata),
        "entropy_bytes": entropy_bytes,
        "optimistic_total_bytes": total_lower_bound,
        "ratio": total_lower_bound / len(baseline),
        "seconds": time.perf_counter() - start,
        "warning": (
            "Static empirical-entropy estimate, not a codec bitstream or a compression claim."
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--transform", choices=["rgb", "ycocg"], default="rgb")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    TRANSFORM = args.transform
    rows = []
    for p in args.images:
        r = run(p, args.out, args.classes, args.tile, args.iterations)
        rows.append(r)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(r), flush=True)
