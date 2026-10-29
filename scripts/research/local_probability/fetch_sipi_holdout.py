"""Recreate the fixed SIPI cohort without changing or extracting source files.

The manifest records the original holdout reservation. A rerun after inspecting
results is a regression, not a fresh validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = [f"4.1.{i:02}" for i in range(1, 9)] + [
    "4.2.01",
    "4.2.03",
    "4.2.05",
    "4.2.06",
    "4.2.07",
    "house",
]
URL = "https://sipi.usc.edu/database/misc.tar.gz"
MAX_ARCHIVE = 32 * 1024 * 1024
MAX_IMAGE = 4 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024


@dataclass(frozen=True)
class Cohort:
    names: tuple[str, ...]
    url: str
    source_page: str
    selection: str
    max_archive: int
    sizes: tuple[tuple[int, int], ...] = ((256, 256), (512, 512))


DEFAULT_COHORT = Cohort(
    tuple(NAMES),
    URL,
    "https://sipi.usc.edu/database/database.php?volume=misc",
    "Every color image currently listed by the publisher, excluding withdrawn images; "
    "fixed before running model tests.",
    MAX_ARCHIVE,
)


def immutable_write(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite changed file: {path}")
    else:
        path.write_bytes(data)


def convert_archive(archive: bytes, out: Path, *, cohort: Cohort = DEFAULT_COHORT) -> dict:
    if len(archive) > cohort.max_archive:
        raise ValueError("oversized archive")
    digest = hashlib.sha256(archive).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("archive_sha256") != digest:
            raise ValueError("upstream archive changed; preserve the existing cohort")
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for index, member in enumerate(tar):
            total += member.size
            if index >= 4096 or not 0 <= member.size <= MAX_MEMBER or total > MAX_TOTAL:
                raise ValueError("archive expansion budget exceeded")
            name = Path(member.name).name
            for suffix in (".tiff", ".tif"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            if name not in cohort.names or not member.isfile():
                continue
            if member.size > MAX_IMAGE or name in records:
                raise ValueError("oversized or duplicate source image")
            file = tar.extractfile(member)
            if file is None:
                raise ValueError("missing archive member data")
            raw = file.read(MAX_IMAGE + 1)
            if len(raw) != member.size:
                raise ValueError("truncated source image")
            with Image.open(io.BytesIO(raw)) as image:
                if (
                    image.format != "TIFF"
                    or image.mode != "RGB"
                    or image.size not in cohort.sizes
                    or image.tag_v2.get(258) != (8, 8, 8)
                ):
                    raise ValueError(f"unexpected source pixels: {name}")
                source = np.asarray(image).copy()
                encoded = io.BytesIO()
                image.save(encoded, format="PNG")
            data = encoded.getvalue()
            with Image.open(io.BytesIO(data)) as image:
                if not np.array_equal(np.asarray(image), source):
                    raise AssertionError("TIFF-to-PNG changed RGB samples")
            target = out / (name + ".png")
            immutable_write(target, data)
            records[name] = {
                "image": target.name,
                "source_entry": member.name,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "png_sha256": hashlib.sha256(data).hexdigest(),
                "pixel_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            }
    if set(records) != set(cohort.names):
        raise ValueError(f"missing names: {set(cohort.names) - set(records)}")
    manifest = {
        "source": cohort.url,
        "source_page": cohort.source_page,
        "selection": cohort.selection,
        "transform": "TIFF to PNG, exact RGB samples verified, no resizing",
        "role": "fresh held-out validation; not used to select model parameters",
        "archive_sha256": digest,
        "images": [records[name] for name in cohort.names],
    }
    immutable_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data/corpora/sipi-color-holdout",
    )
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read(MAX_ARCHIVE + 1)
    manifest = convert_archive(archive, args.out)
    print("Prepared fixed", len(manifest["images"]), "image SIPI color cohort.")


if __name__ == "__main__":
    main()
