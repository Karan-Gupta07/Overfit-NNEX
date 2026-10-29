"""Download the fixed 24-image Kodak PNG corpus, without resampling.

The first three images are development inputs; images 04..24 are held out.
Images remain subject to their original owners' rights.

    python scripts/fetch_kodak.py --out data/corpora/kodak
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

SOURCE = "https://r0k.us/graphics/kodak/"
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def fetch(out: Path, count: int = 24) -> None:
    if not 1 <= count <= 24:
        raise ValueError("count must be in 1..24")
    out.mkdir(parents=True, exist_ok=True)

    def download(index: int) -> dict[str, object]:
        name = f"kodim{index:02}.png"
        url = f"{SOURCE}kodak/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"invalid or oversized source image: {name}")
        with Image.open(io.BytesIO(data)) as image:
            if image.size not in ((768, 512), (512, 768)) or image.mode != "RGB":
                raise ValueError(f"unexpected source geometry or mode: {name}")
            image.verify()
        target = out / name
        if target.exists():
            if target.read_bytes() != data:
                raise FileExistsError(f"refusing to replace different image: {target}")
        else:
            target.write_bytes(data)
        return {
            "image": name,
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "split": "development" if index <= 3 else "heldout",
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, range(1, count + 1)))
    manifest = {
        "source": SOURCE,
        "selection": "original images in numerical order; 01..03 development, 04..24 held out",
        "transform": "none",
        "images": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Fetched {len(records)} unchanged Kodak PNGs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.out)


if __name__ == "__main__":
    main()
