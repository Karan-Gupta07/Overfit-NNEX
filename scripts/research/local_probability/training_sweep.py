"""Development sweep of encoder-only training settings, measured as actual files.

Every row trains, quantizes, encodes, and independently decodes real integer NNEX
v3 candidates. Use only on already-known development images. Configurations
without the g- prefix were recorded with --sequential.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import neural_probability_probe as trainer
import torch
from quantize_model import run as precision_search

BASE = {
    "epochs": 200,
    "lr": 0.001,
    "schedule": "constant",
    "batch": 8192,
    "mixtures": 3,
    "hidden": 16,
    "feature_set": 0,
}
CONFIGS = {
    "base": {},
    "cos1": {"schedule": "cosine"},
    "cos3": {"schedule": "cosine", "lr": 0.003},
    "cos3-e400": {"schedule": "cosine", "lr": 0.003, "epochs": 400},
    "cos3-b2048": {"schedule": "cosine", "lr": 0.003, "batch": 2048},
    "cos3-k4": {"schedule": "cosine", "lr": 0.003, "mixtures": 4},
    "cos3-h24": {"schedule": "cosine", "lr": 0.003, "hidden": 24},
    "cos3-h12": {"schedule": "cosine", "lr": 0.003, "hidden": 12},
    "k4": {"schedule": "cosine", "lr": 0.003, "mixtures": 4},
    "far-k4": {"schedule": "cosine", "lr": 0.003, "mixtures": 4, "feature_set": 1},
    "far-k4-h12": {
        "schedule": "cosine",
        "lr": 0.003,
        "mixtures": 4,
        "feature_set": 1,
        "hidden": 12,
    },
    "far-k4-h24": {
        "schedule": "cosine",
        "lr": 0.003,
        "mixtures": 4,
        "feature_set": 1,
        "hidden": 24,
    },
    "far-k4-e400": {
        "schedule": "cosine",
        "lr": 0.003,
        "mixtures": 4,
        "feature_set": 1,
        "epochs": 400,
    },
    "far-k4-e800": {
        "schedule": "cosine",
        "lr": 0.003,
        "mixtures": 4,
        "feature_set": 1,
        "epochs": 800,
    },
    "far-k4-e400-lr6": {
        "schedule": "cosine",
        "lr": 0.006,
        "mixtures": 4,
        "feature_set": 1,
        "epochs": 400,
    },
}
for _lr, _epochs in itertools.product((0.006, 0.01, 0.015, 0.02, 0.03), (400, 800)):
    CONFIGS[f"g-lr{round(_lr * 1000)}-e{_epochs}"] = {
        "schedule": "cosine",
        "lr": _lr,
        "mixtures": 4,
        "feature_set": 1,
        "epochs": _epochs,
    }

CONFIGS["g-lr15-e400-h24"] = {**CONFIGS["g-lr15-e400"], "hidden": 24}
CONFIGS["g-lr15-e400-fs0"] = {**CONFIGS["g-lr15-e400"], "feature_set": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--sequential", action="store_true", help="historical per-plane loop")
    args = parser.parse_args()
    torch.set_num_threads(4)
    settings = {**BASE, **CONFIGS[args.config]}
    rows = []
    for image in args.images:
        start = time.perf_counter()
        work = args.work / args.config / image.stem
        work.mkdir(parents=True, exist_ok=True)
        namespace = argparse.Namespace(
            **settings,
            device=args.device,
            sequential=args.sequential,
            causal_context=True,
            out=work / "model.json",
        )
        trained = trainer.run(image, namespace)
        choices = precision_search(image, Path(trained["model_v3_file"]), work / "precision.json")
        rows.append(
            {
                "image": image.name,
                "config": args.config,
                "settings": settings,
                "planes": [
                    {"channel": p["channel"], "conditional_bytes": p["conditional_bytes"]}
                    for p in trained["planes"]
                ],
                "binary16_model_bytes": trained["model_bytes"],
                "selected": choices["selected"],
                "candidates": choices["candidates"],
                "seconds": time.perf_counter() - start,
            }
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
