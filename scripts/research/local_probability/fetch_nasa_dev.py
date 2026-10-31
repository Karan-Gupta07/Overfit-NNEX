"""Fetch NASA Photojournal development cohorts that exclude every LPCB image.

LPCB's 45 NASA images are Photojournal PIA numbers 12811 to 13943 (its "STA"
files are PIA numbers too). Development work on them needs images of the same
kind that LPCB does not contain.

The "space" cohort was chosen from header scans of PIA13700 to PIA13959,
without looking at pixels or results: drop LPCB numbers, keep RGB and palette
TIFFs of 0.3 to 16.8 million pixels, drop an image whose size repeats the
previous kept one (a series), and keep every third; then the first and middle
grayscale TIFFs by the same rule.

The "earth" cohort was added after the LPCB results showed that most of the
NASA gap to paq8px sits in Terra ASTER scenes and one SRTM shaded-relief map.
It holds every Earth-observation image (Terra ASTER and MISR, EO-1) found in
the Photojournal descriptions of PIA13700 to PIA14399 outside LPCB, and every
other SRTM "colored height and shaded relief" map (plan view, no perspective
or anaglyph) found in PIA02700-02779, PIA03300-03399 and PIA04950-04979 whose
TIFF is at most 128 MiB and 0.3 million pixels or more.

Grayscale and palette images become RGB, as LPCB's PPMs are. Images above
1600 x 1600 are center-cropped to it so a development pass stays short.
Images alternate between a tuning half and a check half by list position.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PIA = (
    13700, 13704, 13719, 13730, 13735, 13762, 13765, 13768, 13776, 13791,
    13797, 13802, 13807, 13825, 13836, 13852, 13866, 13885, 13892, 13911,
    13923, 13929, 13933, 13936, 13950, 13954, 13959, 13702, 13835,
)  # fmt: skip
EARTH = (
    13722, 13732, 13775, 13777, 13824, 13888, 13945, 13952, 13979, 14136,
    14137, 14143, 14147, 14174, 14271, 14277, 14278, 13937, 13957,
    2712, 2719, 2732, 2742, 3310, 3318, 3322, 3327, 3347, 3388,
    3394, 3396, 4951, 4958, 4969,
)  # fmt: skip
COHORTS = {"space": PIA, "earth": EARTH}
LPCB = (
    12811, 12813, 13452, 13453, 13454, 13455, 13456, 13457, 13458, 13459,
    13757, 13771, 13779, 13781, 13782, 13785, 13789, 13799, 13803, 13810,
    13812, 13815, 13833, 13843, 13844, 13845, 13855, 13856, 13857, 13859,
    13862, 13872, 13882, 13891, 13894, 13900, 13901, 13904, 13908, 13912,
    13913, 13914, 13915, 13942, 13943,
)  # fmt: skip
URL = "https://assets.science.nasa.gov/content/dam/science/psd/photojournal/pia/pia{g:02d}/pia{n:05d}/PIA{n:05d}.tif"
CROP = 1600
MAX_SOURCE = 128 * 1024 * 1024
CORPORA = Path(__file__).resolve().parents[3] / "data/corpora"
DIRECTORY = {"space": "nasa-dev", "earth": "nasa-dev-earth"}


def ppm(pixels: np.ndarray) -> bytes:
    h, w, _ = pixels.shape
    return f"P6\n{w} {h}\n255\n".encode() + pixels.tobytes()


def rgb(source: bytes) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(io.BytesIO(source)) as image:
        if image.format != "TIFF" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("expected a single-frame TIFF")
        if image.mode == "RGB" and image.tag_v2.get(258) == (8, 8, 8):
            pixels = np.asarray(image)
        elif image.mode == "L" and image.tag_v2.get(258) in (8, (8,)):
            pixels = np.repeat(np.asarray(image)[..., None], 3, axis=2)
        elif image.mode == "P":
            pixels = np.asarray(image.convert("RGB"))
        else:
            raise ValueError(f"unexpected TIFF samples: {image.mode}")
    h, w, _ = pixels.shape
    y, x = max(0, (h - CROP) // 2), max(0, (w - CROP) // 2)
    return np.ascontiguousarray(pixels[y : y + CROP, x : x + CROP])


def fetch(number: int) -> bytes:
    url = URL.format(g=number // 1000, n=number)
    request = urllib.request.Request(url, headers={"User-Agent": "research-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(MAX_SOURCE + 1)
    if len(data) > MAX_SOURCE:
        raise ValueError("oversized source")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=sorted(COHORTS), default="space")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cohort = COHORTS[args.cohort]
    if args.out is None:
        args.out = CORPORA / DIRECTORY[args.cohort]
    if set(cohort) & set(LPCB) or set(PIA) & set(EARTH):
        raise AssertionError("the cohorts overlap LPCB or each other")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {r["image"]: r for r in json.loads(manifest_path.read_text())["images"]}
    rows = []
    for index, number in enumerate(cohort):
        name = f"PIA{number:05d}.ppm"
        target = args.out / name
        if target.exists() and name in previous:
            data = target.read_bytes()
            if hashlib.sha256(data).hexdigest() != previous[name]["ppm_sha256"]:
                raise ValueError(f"changed image: {name}")
            rows.append(previous[name])
            continue
        source = fetch(number)
        pixels = rgb(source)
        data = ppm(pixels)
        row = {
            "image": name,
            "half": "tune" if index % 2 == 0 else "check",
            "source": URL.format(g=number // 1000, n=number),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "width": int(pixels.shape[1]),
            "height": int(pixels.shape[0]),
            "ppm_sha256": hashlib.sha256(data).hexdigest(),
        }
        if name in previous and previous[name]["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"upstream image changed: {name}")
        target.write_bytes(data)
        rows.append(row)
        print(name, row["half"], row["width"], row["height"], flush=True)
    manifest = {
        "selection": __doc__.strip(),
        "excluded_lpcb_numbers": list(LPCB),
        "images": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(rows), "images in", args.out)


if __name__ == "__main__":
    main()
