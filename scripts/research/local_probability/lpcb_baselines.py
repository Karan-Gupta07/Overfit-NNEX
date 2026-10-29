"""Measure PNG, WebP lossless and JPEG XL (effort 9 and 10) on the LPCB images.

Same settings as fresh_validation.py. Each codec gets the whole image, including
the five that the learned coder splits into strips. Every file is decoded and
compared with its source before its size counts. Resumes from --out.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from fresh_validation import baselines
from PIL import Image

LPCB = Path(__file__).resolve().parents[3] / "data/corpora/lpcb"


def measure(name: str) -> dict:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(LPCB / "ppm" / name) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    start = time.perf_counter()
    sizes = baselines(pixels)
    return {"ppm": name, **sizes, "seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads((LPCB / "manifest.json").read_text())
    rows = json.loads(args.out.read_text()) if args.out.exists() else []
    done = {row["ppm"] for row in rows}
    todo = sorted(
        (item for item in manifest if item["ppm"] not in done),
        key=lambda item: -item["width"] * item["height"],
    )
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(measure, item["ppm"]) for item in todo]
        for future in as_completed(futures):
            rows.append(future.result())
            args.out.write_text(json.dumps(rows, indent=1) + "\n")
            print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
