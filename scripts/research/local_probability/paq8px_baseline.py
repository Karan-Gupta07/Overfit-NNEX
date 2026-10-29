"""Supplementary paq8px baseline on the Tecnick cohorts, with exact round trips.

Post-hoc and outside the frozen protocol: it adds a reference compressor and
never changes the learned encoder or its results. Input is a binary PPM of the
exact RGB samples, the format used by the Lossless Photo Compression Benchmark.
The compress phase keeps each archive under tmp/paq8px so a later verify phase
can decompress it and compare against a PPM rebuilt from the source image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from input_image import read_rgb

ROOT = Path(__file__).resolve().parents[3]
CORPORA = ROOT / "data/corpora"
BINARY = ROOT / "data/research-local-probability/paq8px-v217"
WORK = ROOT / "tmp/paq8px"
MANIFESTS = {
    "tecnick-crop512": "tecnick-manifest.json",
    "tecnick-2400": "tecnick-manifest.json",
    "tecnick-fresh-crop512": "tecnick-fresh-manifest.json",
    "tecnick-fresh-2400": "tecnick-fresh-manifest.json",
}


def ppm(pixels: np.ndarray) -> bytes:
    h, w, _ = pixels.shape
    return f"P6\n{w} {h}\n255\n".encode() + pixels.tobytes()


def run(command: list[str], cwd: Path) -> tuple[float, int]:
    start = time.perf_counter()
    timed = ["/usr/bin/time", "-l", *command]
    proc = subprocess.run(timed, cwd=cwd, capture_output=True, check=False)
    seconds = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError((proc.stdout + proc.stderr).decode(errors="replace")[-2000:])
    peak = re.search(rb"(\d+)\s+maximum resident set size", proc.stderr)
    return seconds, int(peak.group(1)) if peak else -1


def source_ppm(cohort: str, row: dict) -> bytes:
    pixels = read_rgb(CORPORA / cohort / row["image"], max_pixels=16_777_216)
    if hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]:
        raise ValueError("changed source")
    return ppm(pixels)


def compress(cohort: str, row: dict, level: str) -> dict:
    stem = Path(row["image"]).stem
    work = WORK / cohort / stem
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    source = source_ppm(cohort, row)
    (work / f"{stem}.ppm").write_bytes(source)
    archive = work / f"{stem}.paq8px"
    seconds, peak = run([str(BINARY), level, f"{stem}.ppm", archive.name], work)
    (work / f"{stem}.ppm").unlink()
    return {
        "image": row["image"],
        "ppm_bytes": len(source),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "compress_seconds": round(seconds, 1),
        "compress_peak_rss_bytes": peak,
    }


def verify(cohort: str, row: dict) -> dict:
    stem = Path(row["image"]).stem
    work = WORK / cohort / stem
    archive = work / f"{stem}.paq8px"
    restored = work / "out" / f"{stem}.ppm"
    restored.parent.mkdir(exist_ok=True)
    restored.unlink(missing_ok=True)
    seconds, peak = run([str(BINARY), "-d", archive.name, f"out/{stem}.ppm"], work)
    exact = restored.read_bytes() == source_ppm(cohort, row)
    restored.unlink()
    if not exact:
        raise AssertionError(f"paq8px round trip mismatch for {row['image']}")
    return {
        "image": row["image"],
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "exact_rgb": True,
        "decompress_seconds": round(seconds, 1),
        "decompress_peak_rss_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compress", "verify"), required=True)
    parser.add_argument("--cohort", choices=tuple(MANIFESTS), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--level", default="-9L")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((CORPORA / MANIFESTS[args.cohort]).read_text())
    rows = manifest["cohorts"][args.cohort][args.start : args.start + args.limit]
    if not rows:
        parser.error("invalid selection")
    receipt = {
        "complete": False,
        "scope": "Supplementary post-hoc paq8px baseline; not part of the frozen protocol",
        "phase": args.phase,
        "cohort": args.cohort,
        "level": args.level if args.phase == "compress" else None,
        "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        "source": "https://github.com/hxim/paq8px commit 553024fcb4a5a4dfa16204d0313fd4ee43e5a8ac "
        "(v217), two macOS min() overload casts, clang -mcpu=apple-m1 -O3 -flto",
        "machine": platform.machine(),
        "images": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if args.phase == "compress":
            result = compress(args.cohort, row, args.level)
        else:
            result = verify(args.cohort, row)
        receipt["images"].append(result)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print("PAQ8PX", args.phase.upper(), row["image"], result["archive_bytes"], flush=True)
    receipt["complete"] = True
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
