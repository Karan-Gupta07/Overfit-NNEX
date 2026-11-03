"""Measure the current default encoder on LPCB (see fetch_lpcb.py).

Encodes the given PPM images in order, one complete file each, and decodes
every file back to its source samples; --files keeps the coded files. Resumes
from --out. Workers sharing a --claims directory take each image once: a worker
skips any image another has claimed. Run on committed code. LPCB is not held
out for v5: its NASA images (and five other images, for the mixing screen) were
used to develop and tune context mixing (MIX_* and SCREEN_* in compress.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import compress as experiment
import neural_bitstream as coder
import numpy as np
from input_image import read_raster

ROOT = Path(__file__).resolve().parents[3]
LPCB = ROOT / "data/corpora/lpcb"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def children_maxrss_mb() -> float | None:
    """Peak resident memory of the largest finished child, or None on Windows."""
    if os.name == "nt":
        return None
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--claims", type=Path, help="directory shared by parallel workers")
    parser.add_argument("--files", type=Path, help="directory for the coded files")
    parser.add_argument("images", nargs="+", help="PPM names under data/corpora/lpcb/ppm")
    args = parser.parse_args()
    if args.out.exists():
        receipt = json.loads(args.out.read_text())
    else:
        receipt = {
            "scope": "Current default encoder on LPCB",
            "commit": git("rev-parse", "HEAD"),
            "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
            "manifest_sha256": hashlib.sha256((LPCB / "manifest.json").read_bytes()).hexdigest(),
            "native_backend_sha256": hashlib.sha256(coder.INTEGER_BINARY.read_bytes()).hexdigest(),
            "images": [],
        }
    done = {row["image"] for row in receipt["images"]}
    for directory in (args.claims, args.files):
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
    for name in args.images:
        path = LPCB / "ppm" / name
        if path.name in done:
            continue
        if args.claims:
            try:
                os.close(os.open(args.claims / path.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        pixels = read_raster(
            path,
            max_pixels=experiment.MAX_STRIPS * coder.MAX_PIXELS,
            max_bytes=experiment.MAX_SOURCE_BYTES,
        )
        started_at = time.time()
        start = time.perf_counter()
        blob, report = experiment.compress(path)
        encode_wall = time.perf_counter() - start
        start = time.perf_counter()
        decoded = experiment.decompress(blob)
        decode_seconds = time.perf_counter() - start
        peak = children_maxrss_mb()
        if not np.array_equal(np.asarray(decoded)[..., :3], pixels[..., :3]):
            raise AssertionError(f"decode mismatch: {path.name}")
        if args.files:
            (args.files / f"{path.stem}.coded").write_bytes(blob)
        receipt["images"].append(
            {
                "image": path.name,
                "source_md5": hashlib.md5(path.read_bytes()).hexdigest(),
                "width": int(pixels.shape[1]),
                "height": int(pixels.shape[0]),
                "strips": report["selected"].get("count", 1)
                if report["selected"]["format"] == "strips"
                else 1,
                "bytes": len(blob),
                "encode_wall": encode_wall,
                "decode_seconds": decode_seconds,
                "exact": True,
                "selected": report.get("selected"),
                "strip_reports": report.get("strips"),
                "phase_seconds": report.get("phase_seconds"),
                "shared_values": report.get("shared_values"),
                "mixing": report.get("mixing"),
                "children_maxrss_mb": peak,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "started_at": started_at,
                "finished_at": time.time(),
            }
        )
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
        print(path.name, len(blob), f"{encode_wall:.1f}s", f"dec {decode_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
