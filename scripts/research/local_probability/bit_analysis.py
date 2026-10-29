"""Where the bits go in an NNEX v2 file: per-plane estimates and adaptivity headroom.

Development diagnostics only. Float estimates approximate the integer coder; all
"oracle" numbers ignore side information unless stated and are upper bounds.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import neural_bitstream as coder
import neural_probability_probe as trainer
import numpy as np
import predictor_probe as linear
import torch
from input_image import read_rgb

from semanticcodec.format.container import decode_cbor, decompress

LN2 = math.log(2)


def load_models(metadata: bytes):
    body = decode_cbor(decompress(metadata, max_output=coder.MAX_MODEL_BYTES))
    _w, _h, hidden, mixtures, context, planes = body
    models = {}
    for channel, tensors in zip((1, 0, 2), planes, strict=True):
        state = {
            name: torch.from_numpy(np.frombuffer(raw, "<f2").astype(np.float32).reshape(shape))
            for name, shape, raw in tensors
        }
        dimensions = state["affine.weight"].shape[1]
        model = trainer.ProbabilityNet(dimensions, hidden, mixtures, np.zeros(dimensions))
        model.register_buffer("guide_coeff", torch.zeros(state["guide_coeff"].shape))
        model.load_state_dict(state)
        models[channel] = model.eval()
    return models, context


def mixture_bits(logits, means, scales, y, a):
    upper = (y[:, None] + 0.5 - means) / scales
    lower = (y[:, None] - 0.5 - means) / scales
    logprob = (
        torch.nn.functional.logsigmoid(upper)
        + torch.nn.functional.logsigmoid(-lower)
        + torch.log(-torch.expm1(-(upper - lower)).clamp(max=-1e-12))
    )
    logmix = torch.log_softmax(logits, dim=1)
    lp = torch.logsumexp(logmix + logprob, dim=1)
    top = (255.5 - a[:, None] - means) / scales
    bottom = (-0.5 - a[:, None] - means) / scales
    mass = (torch.sigmoid(top) - torch.sigmoid(bottom)).clamp_min(1e-10)
    normal = torch.logsumexp(logmix + torch.log(mass), dim=1)
    return (normal - lp) / LN2


def blocksum(values, h, w, tile):
    return linear.blocksum(values.reshape(h, w, *values.shape[1:]), tile)


def analyse_plane(outputs, y, a, h, w, rng):
    logits, means, scales = outputs
    weights = torch.softmax(logits, dim=1)
    sigma = (weights * scales).sum(dim=1)
    centre = (weights * means).sum(dim=1)
    base = mixture_bits(logits, means, scales, y, a)
    total = float(base.double().sum())
    report = {"bits": total, "bits_per_sample": total / (h * w)}

    # One global scale multiplier: miscalibration of the static model.
    factors = [2 ** (k / 16) for k in range(-8, 9)]
    global_bits = [
        float(mixture_bits(logits, means, scales * f, y, a).double().sum()) for f in factors
    ]
    best = int(np.argmin(global_bits))
    report["global_scale"] = {"factor": factors[best], "saving_bits": total - global_bits[best]}

    # Oracle per-block scale/shift selection, an upper bound on block adaptivity.
    grid = [(2 ** (k / 8), s) for k in range(-4, 5) for s in (-0.5, 0.0, 0.5)]
    per_candidate = np.stack(
        [
            mixture_bits(logits, means + s * sigma[:, None], scales * f, y, a).double().numpy()
            for f, s in grid
        ],
        axis=-1,
    )
    for tile in (16, 32, 64):
        sums = blocksum(per_candidate, h, w, tile)
        blocks = sums.shape[0] * sums.shape[1]
        oracle = float(sums.min(axis=-1).sum())
        side = blocks * math.log2(len(grid))
        report[f"oracle_block{tile}"] = {
            "saving_bits": total - oracle,
            "saving_after_side_info_bits": total - oracle - side,
        }

    # Causal, row-delayed adaptive correction a decoder could reproduce.
    error = (y - centre).reshape(h, w)
    neighbours = (
        torch.stack(
            [
                torch.from_numpy(linear.shift(error.numpy(), *d)).abs()
                for d in ((0, -1), (-1, 0), (-1, -1), (-1, 1))
            ]
        )
        .mean(dim=0)
        .reshape(-1)
    )
    log_sigma = torch.log2(sigma.clamp_min(1e-3))
    ratio = torch.log2((neighbours + 0.25) / sigma.clamp_min(1e-3))
    scale_bucket = (log_sigma * 2 + 6).floor().clamp(0, 19).long()
    ratio_bucket = (ratio * 1.5 + 4).floor().clamp(0, 7).long()
    context = scale_bucket * 8 + ratio_bucket
    for lr in (0.02, 0.05):
        params = torch.zeros(160, 2, requires_grad=True)
        accum = torch.full((160, 2), 1e-3)
        bits = 0.0
        for row in range(h):
            sl = slice(row * w, (row + 1) * w)
            c = context[sl]
            p = params[c]
            row_bits = mixture_bits(
                logits[sl],
                means[sl] + (p[:, 1:2] * sigma[sl, None]),
                scales[sl] * torch.exp(p[:, 0:1]),
                y[sl],
                a[sl],
            )
            bits += float(row_bits.detach().double().sum())
            (grad,) = torch.autograd.grad(row_bits.sum(), params)
            with torch.no_grad():
                accum += grad * grad
                params -= lr * grad / accum.sqrt()
        report[f"adaptive_rowdelay_lr{lr}"] = {"saving_bits": total - bits}

    # Spatial non-stationarity of the cost.
    per_pixel = base.double().numpy().reshape(h, w)
    tiles = linear.blocksum(per_pixel, 32) / (32 * 32)
    report["block32_bits_per_sample_quantiles"] = [
        float(q) for q in np.quantile(tiles, [0.05, 0.25, 0.5, 0.75, 0.95])
    ]
    report["mean_mixture_sigma_quantiles"] = [
        float(q) for q in np.quantile(sigma.numpy(), [0.05, 0.25, 0.5, 0.75, 0.95])
    ]
    del rng
    return report


def run(image: Path, coded: Path) -> dict:
    rgb = read_rgb(image)
    h, w = rgb.shape[:2]
    blob = coded.read_bytes()
    if not blob.startswith(coder.MAGIC):
        raise ValueError("expected integer NNEX v2")
    model_size, stream_size = struct.unpack_from("<2I", blob, len(coder.MAGIC))
    metadata = blob[coder.HEADER_SIZE : coder.HEADER_SIZE + model_size]
    models, context = load_models(metadata)
    prior = {}
    planes = []
    rng = np.random.default_rng(0)
    torch.set_grad_enabled(False)
    for channel in (1, 0, 2):
        model = models[channel]
        co_base = model.guide_coeff.double().numpy()
        xf, target, anchor, base_error = trainer.plane_inputs(rgb, channel, co_base, prior, context)
        prior[channel] = base_error
        x = torch.from_numpy((xf / 32).astype(np.float32))
        y = torch.from_numpy(target.astype(np.float32))
        a = torch.from_numpy(anchor.ravel().astype(np.float32))
        chunks = [model(x[s : s + 65536]) for s in range(0, len(x), 65536)]
        outputs = tuple(torch.cat([c[i] for c in chunks]) for i in range(3))
        torch.set_grad_enabled(True)
        report = analyse_plane(outputs, y, a, h, w, rng)
        torch.set_grad_enabled(False)
        report["channel"] = channel
        planes.append(report)
        print(image.name, json.dumps(report), flush=True)
    estimated = sum(p["bits"] for p in planes) / 8
    return {
        "image": image.name,
        "coded": str(coded),
        "complete_file_bytes": len(blob),
        "model_bytes": model_size,
        "stream_bytes": stream_size,
        "header_bytes": coder.HEADER_SIZE,
        "estimated_stream_bytes": estimated,
        "planes": planes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="image.png=file.coded")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    for pair in args.pairs:
        image, coded = pair.split("=", 1)
        rows.append(run(Path(image), Path(coded)))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
