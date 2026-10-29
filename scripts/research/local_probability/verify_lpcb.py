"""Check coded LPCB files against the benchmark's published MD5 sums.

Decodes every <name>.coded file in a directory, rebuilds the PPM bytes LPCB
publishes (binary P6, "P6\\n<width> <height>\\n255\\n" then the RGB samples, the
header all 107 images use) and compares their MD5 with md5sums.txt from
http://qlic.altervista.org/. Exits non-zero unless every listed image is present
and matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import compress as experiment
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MD5SUMS = ROOT / "data/corpora/lpcb/published/md5sums.txt"


def published(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].endswith(".ppm"):
            sums[parts[0]] = parts[1].lower()
    return sums


def ppm_bytes(rgba: np.ndarray) -> bytes:
    if rgba.ndim != 3 or rgba.shape[2] != 4 or not np.all(rgba[..., 3] == 255):
        raise ValueError("expected opaque RGB")
    height, width = rgba.shape[:2]
    return f"P6\n{width} {height}\n255\n".encode() + np.ascontiguousarray(rgba[..., :3]).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, help="directory of <name>.coded files")
    parser.add_argument("--md5sums", type=Path, default=MD5SUMS)
    parser.add_argument("--out", type=Path, help="JSON report")
    args = parser.parse_args()
    expected = published(args.md5sums)
    rows = []
    for name in sorted(expected):
        path = args.files / f"{Path(name).stem}.coded"
        if not path.exists():
            rows.append({"image": name, "match": False, "error": "missing"})
            print(f"{name:24} missing", flush=True)
            continue
        blob = path.read_bytes()
        start = time.perf_counter()
        try:
            md5 = hashlib.md5(ppm_bytes(experiment.decompress(blob))).hexdigest()
        except (ValueError, AssertionError, OSError) as error:
            rows.append({"image": name, "bytes": len(blob), "match": False, "error": str(error)})
            print(f"{name:24} error: {error}", flush=True)
            continue
        seconds = time.perf_counter() - start
        match = md5 == expected[name]
        rows.append(
            {
                "image": name,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "md5": md5,
                "match": match,
                "decode_seconds": round(seconds, 2),
            }
        )
        print(
            f"{name:24} {len(blob):>12,} {'ok' if match else 'MISMATCH'} {seconds:6.1f}s",
            flush=True,
        )
    matched = sum(row["match"] for row in rows)
    total = sum(row.get("bytes", 0) for row in rows)
    print(f"{matched}/{len(expected)} images match the published MD5 sums; {total:,} bytes")
    if args.out:
        args.out.write_text(
            json.dumps(
                {"matched": matched, "images": len(expected), "bytes": total, "rows": rows},
                indent=1,
            )
            + "\n"
        )
    sys.exit(0 if matched == len(expected) else 1)


if __name__ == "__main__":
    main()
