"""Measure the current default encoder on the fresh Tecnick cohort.

The cohort is fixed by fetch_tecnick_fresh.py. Do not tune settings on it: run
this only on committed code, and report every image it measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import compress as experiment
import imagecodecs
import neural_bitstream as coder
import numpy as np
from input_image import read_rgb

ROOT = Path(__file__).resolve().parents[3]
CORPORA = ROOT / "data/corpora"
MANIFEST = CORPORA / "tecnick-fresh-manifest.json"
COHORTS = ("tecnick-fresh-crop512", "tecnick-fresh-2400")


def baselines(pixels: np.ndarray) -> dict[str, int]:
    sizes = coder.measure_baselines(pixels)
    strongest = imagecodecs.jpegxl_encode(pixels, lossless=True, effort=10, numthreads=1)
    if not np.array_equal(imagecodecs.jpegxl_decode(strongest, numthreads=1), pixels):
        raise AssertionError("JPEG XL effort 10 baseline mismatch")
    sizes["jxl-lossless-e10"] = len(strongest)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, default=ROOT / "data/probe/fresh")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    receipt = {
        "complete": False,
        "scope": f"Current default encoder on fresh {args.cohort}",
        "commit": commit,
        "tracked_changes": bool(dirty),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "native_backend_sha256": hashlib.sha256(coder.INTEGER_BINARY.read_bytes()).hexdigest(),
        "images": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.save.mkdir(parents=True, exist_ok=True)
    for row in manifest["cohorts"][args.cohort]:
        path = CORPORA / args.cohort / row["image"]
        pixels = read_rgb(path, max_pixels=coder.MAX_PIXELS)
        if (
            list(pixels.shape) != row["shape"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["png_sha256"]
            or hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]
        ):
            raise ValueError(f"changed source: {row['image']}")
        start = time.perf_counter()
        sizes = baselines(pixels)
        baseline_seconds = time.perf_counter() - start
        blob, report = experiment.compress(path)
        start = time.perf_counter()
        decoded = experiment.decompress(blob)
        decode_seconds = time.perf_counter() - start
        if not np.array_equal(np.asarray(decoded)[..., :3], pixels):
            raise AssertionError(f"decode mismatch: {row['image']}")
        (args.save / f"{args.cohort}-{path.stem}.coded").write_bytes(blob)
        report.update(
            baselines=sizes,
            best_baseline=min(sizes, key=sizes.get),
            baseline_seconds=baseline_seconds,
            decode_seconds=decode_seconds,
            exact_decode=True,
            file_sha256=hashlib.sha256(blob).hexdigest(),
            source_pixel_sha256=row["pixel_sha256"],
            samples=int(pixels.size),
        )
        receipt["images"].append(report)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(path.name, len(blob), "best baseline", min(sizes.values()), flush=True)
    receipt["complete"] = True
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
