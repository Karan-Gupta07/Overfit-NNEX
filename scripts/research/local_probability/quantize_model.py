"""Post-training weight precision search. Source bytes are always counted."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import neural_bitstream as coder
import numpy as np
from input_image import read_rgb

from semanticcodec.format.container import compress, decode_cbor, decompress, encode_cbor

V3_PRECISIONS = (16, 12, 10, 8, 7, 6)


def _round(raw: bytes, bits: int) -> bytes:
    values = np.frombuffer(raw, dtype="<f2").astype(np.float64)
    maximum = float(np.abs(values).max())
    if maximum == 0:
        return raw
    shift = math.floor(math.log2((2 ** (bits - 1) - 1) / maximum))
    scale = math.ldexp(1.0, shift)
    return (np.rint(values * scale) / scale).astype("<f2").tobytes()


def _quantized(name: str) -> bool:
    # Keep the exact guide predictor and linear mean. Their small
    # coefficients affect every feature and save few bytes.
    return name.endswith(".weight") and name != "affine.weight"


def quantize(metadata: bytes, bits: int) -> bytes:
    """Version 2 CBOR model on a coarser hidden/head weight grid."""
    if bits not in (8, 10, 12, 16):
        raise ValueError("supported precisions: 8, 10, 12, 16")
    coder.configuration(metadata)
    if bits == 16:
        return metadata
    body = decode_cbor(decompress(metadata, max_output=coder.MAX_MODEL_BYTES))
    for tensors in body[5]:
        for tensor in tensors:
            if _quantized(tensor[0]):
                tensor[2] = _round(tensor[2], bits)
    result = compress(encode_cbor(body))
    coder.configuration(result)
    return result


def quantize_v3(payload: bytes, bits: int) -> bytes:
    """Version 3 compact model on a coarser hidden/head weight grid."""
    if bits not in V3_PRECISIONS:
        raise ValueError("supported precisions: " + ", ".join(map(str, V3_PRECISIONS)))
    header, planes = coder.parse_v3(payload)
    if bits == 16:
        return payload
    rounded = [
        {name: _round(raw, bits) if _quantized(name) else raw for name, raw in state.items()}
        for state in planes
    ]
    return coder.encode_model(header, rounded)


def load_v3(model_path: Path) -> bytes:
    """A v3 payload, transcoding an exact v2 CBOR model if needed."""
    data = model_path.read_bytes()
    return data if model_path.suffix == ".nnm" else coder.transcode_v3(data)


def run(
    path: Path, model_path: Path, prefix: Path, precisions=V3_PRECISIONS, *, envelope: int = 4
) -> dict:
    """Encode precisions in order, stopping after the first one that grows the file.

    Sizes are unimodal in practice; the choice is still the smallest actual file
    among those encoded and independently decoded. Envelopes 3 and 4 share the
    model payload and probabilities; 4 adds row streams for parallel decoding.
    """
    if envelope not in (3, 4):
        raise ValueError("supported envelopes: 3, 4")
    encode = coder.encode_v4 if envelope == 4 else coder.encode_v3
    pixels = read_rgb(path, max_pixels=coder.MAX_PIXELS)
    payload = load_v3(model_path)
    candidates = []
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for bits in precisions:
        model = quantize_v3(payload, bits)
        mark = time.perf_counter()
        blob = encode(pixels, model)
        encode_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        if not np.array_equal(coder.decode(blob), pixels):
            raise AssertionError("quantized-model independent decode mismatch")
        decode_seconds = time.perf_counter() - mark
        coded_path = prefix.with_name(prefix.stem + f"-q{bits}.nnexp")
        coded_path.write_bytes(blob)
        model_path_out = prefix.with_name(prefix.stem + f"-q{bits}-model.nnm")
        model_path_out.write_bytes(model)
        candidates.append(
            {
                "bits": bits,
                "envelope_version": envelope,
                "model_bytes": len(model),
                "complete_file_bytes": len(blob),
                "model_sha256": hashlib.sha256(model).hexdigest(),
                "file_sha256": hashlib.sha256(blob).hexdigest(),
                "exact_rgb": True,
                "encode_seconds": round(encode_seconds, 2),
                "decode_seconds": round(decode_seconds, 2),
            }
        )
        if len(candidates) > 1 and len(blob) > candidates[-2]["complete_file_bytes"]:
            break
    selected = min(candidates, key=lambda row: row["complete_file_bytes"])
    result = {
        "image": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "original_model_sha256": hashlib.sha256(payload).hexdigest(),
        "selected": selected,
        "candidates": candidates,
        "warning": (
            f"Post-training precision experiment. All candidates are actual integer "
            f"v{envelope} files, including weights and checksum, independently decoded."
        ),
    }
    prefix.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.image, args.model, args.out)), flush=True)


if __name__ == "__main__":
    main()
