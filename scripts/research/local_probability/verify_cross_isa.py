"""Verify integer NNEX v2/v3/v4/v5 with a second executable; no source image or trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import neural_bitstream as coder
import numpy as np

from semanticcodec.format.container import MAGIC as SCDX_MAGIC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    native = coder.INTEGER_BINARY
    result = {
        "complete": False,
        "native_architecture": platform.machine(),
        "native_backend_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "alternate_backend_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "warning": "Two executable builds on one host, not independent physical machines.",
        "images": [],
        "scdx_skipped": [],
    }
    for path in args.inputs:
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("file budget")
        blob = path.read_bytes()
        if blob.startswith(SCDX_MAGIC):
            result["scdx_skipped"].append(path.name)
            continue
        if not blob.startswith(coder.INTEGER_MAGICS):
            raise ValueError("expected integer NNEX v2/v3/v4/v5")
        pixels = coder.decode(blob)
        try:
            coder.INTEGER_BINARY = args.binary
            if not np.array_equal(coder.decode(blob), pixels):
                raise AssertionError("alternate decoder mismatch")
            if coder.reencode(blob, pixels) != blob:
                raise AssertionError("alternate encoder bytes differ")
        finally:
            coder.INTEGER_BINARY = native
        result["images"].append(
            {
                "file": path.name,
                "file_sha256": hashlib.sha256(blob).hexdigest(),
                "envelope_version": blob[len(coder.MAGIC) - 1],
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "exact_rgb": True,
                "identical_encoded_bytes": True,
            }
        )
        print("CROSS ISA VERIFIED", path.name, flush=True)
    if not result["images"]:
        raise ValueError("no integer neural files selected")
    result["complete"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
