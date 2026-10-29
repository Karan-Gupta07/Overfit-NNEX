"""Fetch the Lossless Photo Compression Benchmark images and their published results.

LPCB (http://qlic.altervista.org/LPCB.html) ranks codecs on 107 24-bit RGB PPM
photographs: 62 from camera RAW samples and 45 NASA Photojournal images. The
author's original PPM archive survives on the Wayback Machine; every image must
match the MD5 sum the benchmark publishes. Each PPM also becomes a PNG of the
same samples; images over the codec's pixel limit become PNG strips, which
older receipts coded as separate files. lpcb_validation.py now codes each PPM
as one file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import neural_bitstream as coder
import numpy as np
from PIL import Image

ARCHIVE_URL = (
    "https://web.archive.org/web/20140701152353id_/"
    "http://www.compressionratings.com/files/lpcb.ppm.zip"
)
ARCHIVE_BYTES = 2_361_154_127
PUBLISHED_URL = "http://qlic.altervista.org/"
PUBLISHED = (
    "md5sums.txt",
    "list-ppm.txt",
    "paq8px_v216-9L_lpcb_log.txt",
    "cmix-v16-tsv.txt",
    "list-gra111.txt",
)
ROOT = Path(__file__).resolve().parents[3] / "data/corpora/lpcb"


def download(url: str, target: Path) -> None:
    if not target.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, target.open("wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)


def convert(task: tuple[Path, str]) -> dict:
    archive, name = task
    data = zipfile.ZipFile(archive).read(name)
    ppm = ROOT / "ppm" / name
    ppm.write_bytes(data)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(ppm) as image:
        if image.format != "PPM" or image.mode != "RGB":
            raise ValueError(f"unexpected source: {name}")
        rgb = np.asarray(image).copy()
    h, w, _ = rgb.shape
    strips = math.ceil(w * h / coder.MAX_PIXELS)
    rows = math.ceil(h / strips)
    stem = Path(name).stem
    outputs = []
    for k in range(strips):
        part = rgb[k * rows : (k + 1) * rows]
        target = ROOT / "png" / (f"{stem}.png" if strips == 1 else f"{stem}.s{k + 1}.png")
        Image.fromarray(part).save(target, format="PNG", compress_level=6)
        with Image.open(target) as back:
            if not np.array_equal(np.asarray(back), part):
                raise AssertionError(f"PNG round trip: {target.name}")
        outputs.append(
            {
                "png": target.name,
                "width": w,
                "height": int(part.shape[0]),
                "png_bytes": target.stat().st_size,
            }
        )
    return {
        "ppm": name,
        "md5": hashlib.md5(data).hexdigest(),
        "ppm_bytes": len(data),
        "width": w,
        "height": h,
        "strips": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for sub in ("ppm", "png", "published"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED:
        download(PUBLISHED_URL + name, ROOT / "published" / name)
    archive = ROOT / "lpcb.ppm.zip"
    download(ARCHIVE_URL, archive)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("incomplete LPCB archive")
    published = {}
    for line in (ROOT / "published/md5sums.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].endswith(".ppm"):
            published[parts[0]] = parts[1].lower()
    names = [n for n in zipfile.ZipFile(archive).namelist() if n.endswith(".ppm")]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(convert, [(archive, n) for n in names]))
    if {r["ppm"]: r["md5"] for r in rows} != published:
        raise ValueError("images differ from the published LPCB MD5 sums")
    (ROOT / "manifest.json").write_text(json.dumps(rows, indent=1) + "\n")
    split = [r["ppm"] for r in rows if len(r["strips"]) > 1]
    print(len(rows), "images match the published MD5 sums; split into strips:", split)


if __name__ == "__main__":
    main()
