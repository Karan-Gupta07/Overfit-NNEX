"""Decode timing: selected NNEX files vs native JPEG XL, same pixels.

Supplementary to the frozen Tecnick protocol: a single quiet pass, not an
isolated latency benchmark. NNEX timing includes model parsing, temporary
files, the native subprocess, and the pixel checksum; JPEG XL decodes in process.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import compress as experiment
import imagecodecs
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def timed(function, repeats: int) -> tuple[float, object]:
    best, value = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coded", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--jxl-threads", type=int, default=1, help="also time JPEG XL decoding with N threads"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.coded:
        blob = path.read_bytes()
        nnex_seconds, rgba = timed(lambda blob=blob: experiment.decompress(blob), args.repeats)
        rgb = np.ascontiguousarray(rgba[..., :3])
        jxl = imagecodecs.jpegxl_encode(rgb, lossless=True, effort=9, numthreads=1)
        jxl_seconds, back = timed(
            lambda jxl=jxl: imagecodecs.jpegxl_decode(jxl, numthreads=1), args.repeats
        )
        if not np.array_equal(back, rgb):
            raise AssertionError("JPEG XL timing round-trip mismatch")
        rows.append(
            {
                "file": path.name,
                "pixels": rgb.shape[0] * rgb.shape[1],
                "nnex_decode_seconds": round(nnex_seconds, 4),
                "jxl_e9_decode_seconds": round(jxl_seconds, 4),
                "ratio": round(nnex_seconds / jxl_seconds, 1),
            }
        )
        if args.jxl_threads > 1:
            threaded, back = timed(
                lambda jxl=jxl: imagecodecs.jpegxl_decode(jxl, numthreads=args.jxl_threads),
                args.repeats,
            )
            if not np.array_equal(back, rgb):
                raise AssertionError("JPEG XL threaded round-trip mismatch")
            rows[-1]["jxl_e9_decode_seconds_threaded"] = round(threaded, 4)
            rows[-1]["ratio_threaded"] = round(nnex_seconds / threaded, 1)
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "scope": "Supplementary sequential decode timing, best of repeats, one thread for JXL",
        "machine": platform.machine(),
        "repeats": args.repeats,
        "jxl_threads": args.jxl_threads,
        "median_ratio": statistics.median(r["ratio"] for r in rows),
        "median_ratio_threaded": statistics.median(r["ratio_threaded"] for r in rows)
        if args.jxl_threads > 1
        else None,
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
