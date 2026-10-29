"""Post-hoc NNEX v4 transcode of existing v3 files: exact bytes and timing.

Uses only each file's in-band model and decoded pixels; nothing is trained or
tuned. The v4 file has the same model and probabilities as its v3 source, so
the size difference is just the row-stream table and per-stream flushes.
Timings are best-of-N wall seconds for the native subprocess path, including
model parsing, temporary files, and the pixel checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import time
from pathlib import Path

import neural_bitstream as coder
import numpy as np


def best(repeats: int, function, *arguments):
    fastest, value = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function(*arguments)
        fastest = min(fastest, time.perf_counter() - start)
    return fastest, value


def legacy_decode(binary: Path, blob: bytes) -> np.ndarray:
    """Decode with a pre-v4 backend that takes no thread argument."""
    model_size, _ = struct.unpack_from("<2I", blob, len(coder.MAGIC))
    payload = blob[coder.HEADER_SIZE : coder.HEADER_SIZE + model_size]
    config, (w, h) = coder.native_configuration(*coder.parse_v3(payload))
    with coder.temporary_directory() as td:
        directory = Path(td)
        (directory / "config").write_bytes(config)
        (directory / "stream").write_bytes(blob[coder.HEADER_SIZE + model_size :])
        command = [
            str(binary),
            "decode",
            *(str(directory / n) for n in ("config", "stream", "out")),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        return np.frombuffer((directory / "out").read_bytes(), np.uint8).reshape(h, w, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--legacy-binary", type=Path, help="pre-v4 backend for decode timing")
    parser.add_argument("--save-v4", type=Path, help="directory for the transcoded files")
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        blob = path.read_bytes()
        if not blob.startswith(coder.MAGIC_V3):
            print("SKIP (not v3)", path.name, flush=True)
            continue
        payload = coder.model_payload(blob)
        pixels = coder.decode(blob)
        coder.THREADS = 1
        v3_encode_1, again = best(1, coder.encode_v3, pixels, payload)
        coder.THREADS = 0
        v3_encode, again_threaded = best(args.repeats, coder.encode_v3, pixels, payload)
        if again != blob or again_threaded != blob:
            raise AssertionError("v3 re-encoding changed bytes")
        v4_encode, v4 = best(args.repeats, coder.encode_v4, pixels, payload)
        v3_decode, back3 = best(args.repeats, coder.decode, blob)
        v4_decode, back4 = best(args.repeats, coder.decode, v4)
        if not (np.array_equal(back3, pixels) and np.array_equal(back4, pixels)):
            raise AssertionError("decode mismatch")
        row = {
            "file": path.name,
            "pixels": int(pixels.shape[0] * pixels.shape[1]),
            "v3_bytes": len(blob),
            "v4_bytes": len(v4),
            "v4_overhead_bytes": len(v4) - len(blob),
            "v3_sha256": hashlib.sha256(blob).hexdigest(),
            "v4_sha256": hashlib.sha256(v4).hexdigest(),
            "v3_reencoded_identically": True,
            "v3_encode_seconds_one_thread": round(v3_encode_1, 3),
            "v3_encode_seconds": round(v3_encode, 3),
            "v4_encode_seconds": round(v4_encode, 3),
            "v3_decode_seconds": round(v3_decode, 3),
            "v4_decode_seconds": round(v4_decode, 3),
        }
        if args.legacy_binary:
            legacy, back = best(args.repeats, legacy_decode, args.legacy_binary, blob)
            if not np.array_equal(back, pixels):
                raise AssertionError("legacy decode mismatch")
            row["legacy_v3_decode_seconds"] = round(legacy, 3)
        if args.save_v4:
            args.save_v4.mkdir(parents=True, exist_ok=True)
            (args.save_v4 / (path.stem + ".v4.coded")).write_bytes(v4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        raise ValueError("no v3 files")
    result = {
        "scope": "Post-hoc v3 to v4 transcode of existing files; no training or tuning",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "backend_sha256": hashlib.sha256(coder.INTEGER_BINARY.read_bytes()).hexdigest(),
        "legacy_backend_sha256": hashlib.sha256(args.legacy_binary.read_bytes()).hexdigest()
        if args.legacy_binary
        else None,
        "repeats": args.repeats,
        "default_streams": coder.DEFAULT_STREAMS,
        "total_v3_bytes": sum(r["v3_bytes"] for r in rows),
        "total_v4_bytes": sum(r["v4_bytes"] for r in rows),
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
