"""Execute the frozen Tecnick protocol. Do not tune using these cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import compress as experiment
import imagecodecs
import neural_bitstream as coder
import numpy as np
from input_image import read_rgb

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/experiments/2026-10-24-tecnick-protocol.json"
CORPORA = ROOT / "data/corpora"
COHORTS = ("tecnick-crop512", "tecnick-2400")


def learned_settings(protocol: dict, key: str) -> dict:
    settings = protocol[key]
    return {
        "widths": tuple(settings["hidden_widths"]),
        "feature_sets": tuple(settings["feature_sets"]),
        "mixtures": settings["mixtures"],
        "epochs": settings["epochs"],
        "schedule": settings["schedule"],
        "lr": settings["learning_rate"],
        "sequential": settings["training"] == "sequential",
        # The protocol froze NNEX v3; v4 only splits the arithmetic stream.
        "envelope": 3,
        # ...and PyTorch training with the full precision search.
        "fused_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    manifest_path = CORPORA / "tecnick-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["archive_sha256"] != protocol["archive_sha256"]:
        raise ValueError("changed fixed cohort")
    rows = manifest["cohorts"][args.cohort][args.start : args.start + args.limit]
    if args.start < 0 or args.limit <= 0 or not rows:
        parser.error("invalid selection")
    current = learned_settings(protocol, "learned")
    reference = (
        learned_settings(protocol, "prior_round_reference")
        if args.cohort in protocol["prior_round_reference"]["cohorts"]
        else None
    )
    receipt = {
        "complete": False,
        "scope": f"Fresh, frozen validation on {args.cohort}",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "protocol": protocol,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selection": [row["image"] for row in rows],
        "native_backend_sha256": hashlib.sha256(coder.INTEGER_BINARY.read_bytes()).hexdigest(),
        "images": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = CORPORA / args.cohort / row["image"]
        pixels = read_rgb(path, max_pixels=coder.MAX_PIXELS)
        if list(pixels.shape) != row["shape"]:
            raise ValueError("unexpected source size")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != row["png_sha256"]
            or hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]
        ):
            raise ValueError("changed source")
        start = time.perf_counter()
        baselines = coder.measure_baselines(pixels)
        strongest = imagecodecs.jpegxl_encode(pixels, lossless=True, effort=10, numthreads=1)
        if not np.array_equal(imagecodecs.jpegxl_decode(strongest, numthreads=1), pixels):
            raise AssertionError("JPEG XL effort 10 baseline mismatch")
        baselines["jxl-lossless-e10"] = len(strongest)
        baseline_seconds = time.perf_counter() - start
        blob, report = experiment.compress(path, **current)
        start = time.perf_counter()
        experiment.decompress(blob)
        decode_seconds = time.perf_counter() - start
        output = ROOT / "data/probe" / f"{args.cohort}-{path.stem}.coded"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(blob)
        report.update(
            baselines=baselines,
            baseline_seconds=baseline_seconds,
            decode_seconds=decode_seconds,
            file_sha256=hashlib.sha256(blob).hexdigest(),
            source_pixel_sha256=row["pixel_sha256"],
        )
        if reference is not None:
            prior, prior_report = experiment.compress(path, conventional=False, **reference)
            report["prior_round_reference"] = {
                "complete_file_bytes": len(prior),
                "selected": prior_report["selected"],
                "file_sha256": hashlib.sha256(prior).hexdigest(),
                "seconds": prior_report["seconds"],
            }
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
