"""Rerun the frozen aerial protocol settings. The cohort is now known.

Reruns are regressions, not fresh validation. The current encoder writes NNEX v3
and extends the precision search, so sizes differ from the original v2 receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import compress as experiment
import neural_bitstream as coder
from fetch_sipi_aerial import COHORT
from input_image import read_rgb

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/experiments/2026-10-23-neural-aerial-protocol.json"
# Pinned because the compress.py defaults changed after this protocol froze.
AERIAL = {
    "widths": (16, 32),
    "feature_sets": (0,),
    "mixtures": 3,
    "epochs": 200,
    "schedule": "constant",
    "lr": 0.001,
    "sequential": True,
    "envelope": 3,
    "fused_training": False,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = ROOT / "data/corpora/sipi-aerial-512"
    manifest = json.loads((corpus / "manifest.json").read_text())
    if [row["image"] for row in manifest["images"]] != [name + ".png" for name in COHORT.names]:
        raise ValueError("changed fixed cohort")
    rows = manifest["images"][args.start : args.start + args.limit]
    if args.start < 0 or args.limit <= 0 or not rows:
        parser.error("invalid selection")
    receipt = {
        "complete": False,
        "scope": "Fresh, frozen adaptive-encoder validation; all twelve native color aerials",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "protocol": json.loads(PROTOCOL.read_text()),
        "manifest_sha256": hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
        "selection": [row["image"] for row in rows],
        "native_backend_sha256": hashlib.sha256(coder.INTEGER_BINARY.read_bytes()).hexdigest(),
        "images": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = corpus / row["image"]
        pixels = read_rgb(path)
        if pixels.shape != (512, 512, 3):
            raise ValueError("unexpected native source size")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != row["png_sha256"]
            or hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]
        ):
            raise ValueError("changed source")
        baselines = coder.measure_baselines(pixels)
        blob, report = experiment.compress(path, **AERIAL)
        output = ROOT / "data/probe" / f"aerial-{path.stem}.coded"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(blob)
        report.update(
            baselines=baselines,
            file_sha256=hashlib.sha256(blob).hexdigest(),
            source_pixel_sha256=row["pixel_sha256"],
        )
        receipt["images"].append(report)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            "FRESH VERIFIED",
            path.name,
            len(blob),
            "vs",
            min(baselines.values()),
            "selection",
            report["selected"],
            flush=True,
        )
    receipt["complete"] = True
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
