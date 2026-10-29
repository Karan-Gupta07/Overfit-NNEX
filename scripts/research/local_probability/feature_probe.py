"""Development estimate: does a causal local-LS input reduce conditional bits?

Trains the same width/schedule with and without the extra inputs. Entropy
estimates with binary16 weights only; nothing here is decodable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import neural_probability_probe as trainer
import numpy as np
import torch
from input_image import read_rgb
from local_ls_probe import extra_features

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


def causal_shift(a, dy, dx):
    h, w = a.shape
    out = np.zeros_like(a)
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    out[y0:y1, x0:x1] = a[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return out


def far_features(rgb, channel):
    green = rgb[..., 1].astype(np.float64)
    if channel == 1:
        west, north, nw = (causal_shift(green, *d) for d in ((0, -1), (-1, 0), (-1, -1)))
        anchor = np.maximum(
            np.minimum(west, north), np.minimum(np.maximum(west, north), west + north - nw)
        )
        columns = [causal_shift(green, *d) - anchor for d in FAR]
    else:
        diff = rgb[..., channel].astype(np.float64) - green
        columns = [causal_shift(diff, *d) for d in FAR]
    return np.stack(columns, axis=-1).reshape(-1, len(columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["none", "ls"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--taps", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    settings = argparse.Namespace(
        epochs=args.epochs,
        hidden=16,
        mixtures=3,
        batch=8192,
        lr=0.003,
        schedule="cosine",
        causal_context=True,
    )
    rows = []
    for image in args.images:
        rgb = read_rgb(image)
        for variant in args.variants:
            torch.manual_seed(20261023)
            prior = {}
            planes = []
            for channel in (1, 0, 2):
                extra = None
                parts = []
                if "ls" in variant.split("+"):
                    ls, _ = extra_features(rgb, channel, radius=args.radius, taps=args.taps)
                    parts += [ls[:, :1], np.log2(ls[:, 1:] + 1) * 8]
                if "far" in variant.split("+"):
                    parts.append(far_features(rgb, channel))
                if parts:
                    extra = np.concatenate(parts, axis=1)
                if "err" in variant.split("+"):
                    fixed = extra

                    def extra(base_error, fixed=fixed):
                        columns = []
                        for d in trainer.linear.CAUSAL[6:]:
                            e = causal_shift(base_error.astype(np.float64), *d).ravel()
                            columns += [e, np.abs(e)]
                        found = np.stack(columns, axis=1)
                        return found if fixed is None else np.concatenate([fixed, found], axis=1)

                _tensors, report, error = trainer.fit(rgb, channel, settings, device, prior, extra)
                prior[channel] = error
                planes.append(report["conditional_bytes"])
            rows.append(
                {"image": image.name, "variant": variant, "planes": planes, "total": sum(planes)}
            )
            print(json.dumps(rows[-1]), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
