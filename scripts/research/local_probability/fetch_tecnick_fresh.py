"""Fix a fresh Tecnick cohort from the 16 images the first selection left unused.

Every unused image becomes one centered 512x512 crop; positions 0, 4, 8, 12 of
the sorted unused names also stay at native 2400x2400. Selection uses only the
names recorded in tecnick-manifest.json and is fixed before any compression run
on these images. No setting may be tuned on this cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from fetch_sipi_holdout import immutable_write
from fetch_tecnick import (
    ARCHIVE_SHA256,
    LICENSE,
    PREFIX,
    SIZE,
    URL,
    _png,
    center_crop,
    read_sources,
)
from PIL import Image

NATIVE = slice(0, 16, 4)


def prepare(archive: bytes, root: Path) -> dict:
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("upstream archive changed; the frozen cohort no longer applies")
    first = json.loads((root / "tecnick-manifest.json").read_text())
    unused = sorted(first["unused"])
    if len(unused) != 16:
        raise ValueError("expected 16 unused images")
    sources = read_sources(archive)
    cohorts = {"tecnick-fresh-2400": unused[NATIVE], "tecnick-fresh-crop512": unused}
    manifest = {
        "source": URL,
        "license": LICENSE,
        "archive_sha256": ARCHIVE_SHA256,
        "selection": (
            "The 16 names tecnick-manifest.json lists as unused, sorted: every one "
            "center-cropped to 512x512, positions 0,4,8,12 also native. Fixed before "
            "any compression measurement on these images."
        ),
        "role": "fresh held-out validation; not used to select model parameters",
        "cohorts": {},
    }
    for cohort, names in cohorts.items():
        out = root / cohort
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for name in names:
            raw = sources[name]
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (SIZE, SIZE):
                    raise ValueError(f"unexpected source pixels: {name}")
                pixels = np.asarray(image).copy()
            if cohort.endswith("2400"):
                data, transform = raw, "publisher PNG bytes unchanged"
            else:
                pixels = center_crop(pixels)
                data, transform = _png(pixels), "centered 512x512 crop, exact RGB samples"
            short = name[len(PREFIX) :]
            immutable_write(out / short, data)
            records.append(
                {
                    "image": short,
                    "source_entry": name,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "png_sha256": hashlib.sha256(data).hexdigest(),
                    "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "shape": list(pixels.shape),
                    "transform": transform,
                }
            )
        manifest["cohorts"][cohort] = records
    text = json.dumps(manifest, indent=2) + "\n"
    immutable_write(root / "tecnick-fresh-manifest.json", text.encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3] / "data/corpora"
    )
    parser.add_argument(
        "--archive", type=Path, required=True, help="local copy of the pinned archive"
    )
    args = parser.parse_args()
    manifest = prepare(args.archive.read_bytes(), args.root)
    print({name: [row["image"] for row in rows] for name, rows in manifest["cohorts"].items()})


if __name__ == "__main__":
    main()
