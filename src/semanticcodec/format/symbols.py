"""Exact appearance dictionary payload, independent of OCR and system fonts.

Symbols contain RGB XOR differences against a conventional background image.
Applying the differences restores the original samples, not an approximation
of a character. Each file carries its entire dictionary.
"""

from __future__ import annotations

import struct

import numpy as np

from semanticcodec.core.scene.document import AssetRef
from semanticcodec.format.container import compress, decode_cbor, decompress, encode_cbor

SYMBOL_CODEC = "symbols-v1"
SYMBOL_CODEC_V2 = "symbols-v2"
MAGIC = b"SYMA\x01"
MAGIC_V2 = b"SYMA\x02"
ATLAS_WIDTH = 512
MAX_EDGE = 64
MAX_PIXELS = 16_777_216
MAX_SYMBOLS = 100_000
MAX_INSTANCES = 1_000_000
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_PAINT_PIXELS = 64 * 1024 * 1024


def atlas_layout(sizes: list[tuple[int, int]], width: int = ATLAS_WIDTH
                 ) -> tuple[list[tuple[int, int]], int]:
    """Shelf packing is part of the format; only symbol dimensions are sent."""
    if not 1 <= width <= ATLAS_WIDTH:
        raise ValueError("invalid symbol atlas width")
    x = y = row_height = 0
    positions = []
    for w, h in sizes:
        if not (1 <= w <= min(MAX_EDGE, width) and 1 <= h <= MAX_EDGE):
            raise ValueError("invalid symbol dimensions")
        if x + w > width:
            x = 0
            y += row_height
            row_height = 0
        positions.append((x, y))
        x += w
        row_height = max(row_height, h)
    height = y + row_height
    if not sizes or width * height > MAX_PIXELS:
        raise ValueError("empty or oversized symbol atlas")
    return positions, height


def _put_uint(out: bytearray, value: int) -> None:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("symbol integer outside uint32")
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def _get_uint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 35, 7):
        if offset >= len(data):
            raise ValueError("truncated symbol integer")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value > 0xFFFFFFFF or (shift and byte == 0):
                raise ValueError("invalid symbol integer")
            return value, offset
    raise ValueError("oversized symbol integer")


def pack_symbols(base: AssetRef, atlas: AssetRef, sizes: list[tuple[int, int]],
                 instances: list[tuple[int, int]]) -> bytes:
    """Instances are (row-major pixel offset, dictionary ID), in scan order."""
    dims = bytes(v for size in sizes for v in size)
    commands = bytearray()
    previous = 0
    for position, symbol in instances:
        _put_uint(commands, position - previous)
        _put_uint(commands, symbol)
        previous = position
    body = encode_cbor([
        base.codec, base.payload, atlas.codec, atlas.width, atlas.height, atlas.payload,
        dims, bytes(commands),
    ])
    if len(body) > MAX_PAYLOAD_BYTES:
        raise ValueError("symbol payload exceeds size limit")
    return (MAGIC_V2 if base.codec == "jxl-lossless" else MAGIC) + compress(body)


def decode_symbols(asset: AssetRef) -> np.ndarray:
    """Decode with explicit allocation and painting budgets for untrusted files."""
    from semanticcodec.fallback.codecs import decode_asset

    if not (asset.width >= 1 and asset.height >= 1
            and asset.width * asset.height <= MAX_PIXELS):
        raise ValueError("symbol canvas exceeds pixel limit")
    version2 = asset.codec == SYMBOL_CODEC_V2
    magic = MAGIC_V2 if version2 else MAGIC
    if asset.codec not in (SYMBOL_CODEC, SYMBOL_CODEC_V2) or not asset.payload.startswith(magic):
        raise ValueError("invalid symbol payload magic or version")
    body = decode_cbor(decompress(asset.payload[len(magic):], MAX_PAYLOAD_BYTES))
    if not isinstance(body, list) or len(body) != 8:
        raise ValueError("invalid symbol payload structure")
    base_codec, base_data, atlas_codec, aw, ah, atlas_data, dims, commands = body
    base_codecs = ("png", "webp-lossless", "jxl-lossless") if version2 else ("png", "webp-lossless")
    if base_codec not in base_codecs or atlas_codec not in ("png", "webp-lossless"):
        raise ValueError("symbol layers must use a non-recursive lossless codec")
    if not all(isinstance(v, bytes) for v in (base_data, atlas_data, dims, commands)):
        raise ValueError("symbol layers and commands must be byte strings")
    if type(aw) is not int or type(ah) is not int:
        raise ValueError("symbol atlas dimensions must be integers")
    if len(dims) % 2 or not 1 <= len(dims) // 2 <= MAX_SYMBOLS:
        raise ValueError("invalid symbol dictionary length")
    sizes = list(struct.iter_unpack("BB", dims))
    origins, expected_height = atlas_layout(sizes, aw)
    if ah != expected_height:
        raise ValueError("symbol atlas height disagrees with dictionary")
    # Every instance takes at least two bytes.
    if not commands or len(commands) > MAX_INSTANCES * 10:
        raise ValueError("invalid symbol command length")

    # Validate all commands before decoding either image or starting to paint.
    instances: list[tuple[int, int, int]] = []
    offset = position = paint_pixels = 0
    while offset < len(commands):
        delta, offset = _get_uint(commands, offset)
        symbol, offset = _get_uint(commands, offset)
        position += delta
        if symbol >= len(sizes) or position >= asset.width * asset.height:
            raise ValueError("symbol reference or position out of bounds")
        y, x = divmod(position, asset.width)
        w, h = sizes[symbol]
        if x + w > asset.width or y + h > asset.height:
            raise ValueError("symbol extends outside canvas")
        paint_pixels += w * h
        if len(instances) >= MAX_INSTANCES or paint_pixels > MAX_PAINT_PIXELS:
            raise ValueError("symbol painting budget exceeded")
        instances.append((x, y, symbol))

    base = decode_asset(AssetRef(base_codec, asset.width, asset.height, base_data)).copy()
    atlas = decode_asset(AssetRef(atlas_codec, aw, ah, atlas_data))
    if np.any(base[..., 3] != 255) or np.any(atlas[..., 3] != 255):
        raise ValueError("symbol layers must be opaque")
    for x, y, symbol in instances:
        w, h = sizes[symbol]
        sx, sy = origins[symbol]
        base[y:y + h, x:x + w, :3] ^= atlas[sy:sy + h, sx:sx + w, :3]
    return base
