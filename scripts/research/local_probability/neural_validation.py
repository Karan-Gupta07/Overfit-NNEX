"""Rerun the fixed SIPI cohort as a regression, never relabel it as fresh."""

import argparse
import json
from pathlib import Path

import neural_bitstream as coder
import neural_probability_probe as nnprobe
import torch

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/corpora/sipi-color-holdout/manifest.json").read_text())
    names = manifest["images"][args.start : args.start + args.limit]
    if args.start < 0 or args.limit <= 0 or not names:
        parser.error("invalid cohort selection")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(
        epochs=200,
        hidden=32,
        mixtures=3,
        batch=8192,
        lr=0.001,
        device="auto",
        causal_context=True,
        sequential=True,
        out=args.out,
    )
    torch.set_num_threads(4)
    receipt = {
        "complete": False,
        "selection": names,
        "training": vars(train_args).copy(),
        "scope": "Fixed SIPI regression after the original fresh architecture experiment",
        "warning": "Actual integer v2 files; subsequent cohort uses are not fresh holdouts.",
        "images": [],
    }
    receipt["training"]["out"] = str(args.out)
    for record in names:
        path = ROOT / "data/corpora/sipi-color-holdout" / record["image"]
        trained = nnprobe.run(path, train_args)
        bitstream = ROOT / "data/probe" / (path.stem + ".nnexp")
        row = coder.run(path, Path(trained["model_file"]), bitstream)
        if row["source_sha256"] != record["png_sha256"]:
            raise AssertionError("changed holdout image")
        if row["pixel_sha256"] != record["pixel_sha256"]:
            raise AssertionError("changed source samples")
        row["training"] = trained
        receipt["images"].append(row)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            "VERIFIED", path.name, row["complete_file_bytes"], "vs best", row["vs_best"], flush=True
        )
    receipt["complete"] = True
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
