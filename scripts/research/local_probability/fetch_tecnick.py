"""Fix the Tecnick TESTIMAGES validation cohorts before any compression run.

Selection uses only the publisher's sorted file names: positions 0, 3, ..., 33
at native 2400x2400 and positions 1, 4, ..., 34 as one centered 512x512 crop
each. The other 16 images stay unused. A rerun after inspecting results is a
regression, not fresh validation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from fetch_sipi_holdout import immutable_write
from PIL import Image

URL = (
    "https://sourceforge.net/projects/testimages/files/SAMPLING/8BIT/RGB/"
    "SAMPLING_8BIT_RGB_2400x2400.tar.bz2/download"
)
SOURCE_PAGE = "https://sourceforge.net/projects/testimages/files/SAMPLING/8BIT/RGB/"
ARCHIVE_SHA256 = "987e6a5208206b51bb556e808ea052aa588f373124bd0f571888a4754c6afa89"
PREFIX = "img_2400x2400_3x8bit_C00C00_RGB_"
MAX_ARCHIVE = 256 * 1024 * 1024
MAX_MEMBER = 32 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
SIZE = 2400
CROP = 512
LICENSE = "CC BY-NC-SA 4.0, Nicola Asuni (TESTIMAGES); research measurement only"


def select(names: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(names)
    if len(ordered) != 40:
        raise ValueError("expected the 40 published 2400x2400 images")
    return ordered[0:36:3], ordered[1:36:3]


def center_crop(pixels: np.ndarray) -> np.ndarray:
    top = (pixels.shape[0] - CROP) // 2
    left = (pixels.shape[1] - CROP) // 2
    return np.ascontiguousarray(pixels[top : top + CROP, left : left + CROP])


def _png(pixels: np.ndarray) -> bytes:
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    data = encoded.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        if not np.array_equal(np.asarray(image), pixels):
            raise AssertionError("PNG write changed RGB samples")
    return data


def read_sources(archive: bytes) -> dict[str, bytes]:
    if len(archive) > MAX_ARCHIVE:
        raise ValueError("oversized archive")
    sources = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:bz2") as tar:
        for index, member in enumerate(tar):
            total += member.size
            if index >= 256 or not 0 <= member.size <= MAX_MEMBER or total > MAX_TOTAL:
                raise ValueError("archive expansion budget exceeded")
            name = Path(member.name).name
            if not member.isfile() or not name.endswith(".png"):
                continue
            if not name.startswith(PREFIX) or name in sources:
                raise ValueError(f"unexpected archive member: {member.name}")
            file = tar.extractfile(member)
            if file is None:
                raise ValueError("missing archive member data")
            raw = file.read(MAX_MEMBER + 1)
            if len(raw) != member.size:
                raise ValueError("truncated source image")
            sources[name] = raw
    return sources


def prepare(archive: bytes, root: Path) -> dict:
    digest = hashlib.sha256(archive).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("upstream archive changed; the frozen cohort no longer applies")
    sources = read_sources(archive)
    native, cropped = select(list(sources))
    cohorts = {"tecnick-2400": native, "tecnick-crop512": cropped}
    manifest = {
        "source": URL,
        "source_page": SOURCE_PAGE,
        "license": LICENSE,
        "archive_sha256": digest,
        "selection": (
            "Sorted publisher file names; positions 0,3,...,33 native and 1,4,...,34 "
            "center-cropped to 512x512. Fixed before any compression measurement."
        ),
        "unused": sorted(set(sources) - set(native) - set(cropped)),
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
            short = name[len(PREFIX) :]
            if cohort == "tecnick-2400":
                data = raw
                transform = "publisher PNG bytes unchanged"
            else:
                pixels = center_crop(pixels)
                data = _png(pixels)
                transform = f"centered {CROP}x{CROP} crop, exact RGB samples"
            target = out / short
            immutable_write(target, data)
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
    immutable_write(root / "tecnick-manifest.json", text.encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3] / "data/corpora"
    )
    parser.add_argument("--archive", type=Path, help="local copy of the pinned archive")
    args = parser.parse_args()
    if args.archive:
        archive = args.archive.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=300) as response:
            archive = response.read(MAX_ARCHIVE + 1)
    manifest = prepare(archive, args.root)
    print({name: len(rows) for name, rows in manifest["cohorts"].items()})


if __name__ == "__main__":
    main()
