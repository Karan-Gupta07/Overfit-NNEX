"""Fetch all twelve published native 512x512 color aerials, never previews."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from fetch_sipi_holdout import Cohort, convert_archive

COHORT = Cohort(
    tuple(f"2.1.{i:02}" for i in range(1, 13)),
    "https://sipi.usc.edu/database/aerials.tar.gz",
    "https://sipi.usc.edu/database/database.php?volume=aerials",
    "Every native 512x512 color image in the published aerials volume, selected "
    "before compression measurements. Larger native images are outside this "
    "validation's compute budget. No previews, crops, resizing, or result-based selection.",
    128 * 1024 * 1024,
    sizes=((512, 512),),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data/corpora/sipi-aerial-512",
    )
    args = parser.parse_args()
    with urllib.request.urlopen(COHORT.url, timeout=90) as response:
        archive = response.read(COHORT.max_archive + 1)
    manifest = convert_archive(archive, args.out, cohort=COHORT)
    print("Prepared fixed", len(manifest["images"]), "native color aerial images.")


if __name__ == "__main__":
    main()
