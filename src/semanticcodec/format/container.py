"""`.scdx` container.

Chunked TLV, PNG-like, so unknown chunks can be skipped for forward
compatibility. Untrusted input on the read path: every length is bounds-checked
against the remaining buffer and every chunk is CRC-verified before use.

Rate note: the graph payload is currently CBOR + zstd, a deliberate
pre-entropy-coder proxy. It is within a modest factor
of a tuned coder and lets the semantic work proceed without blocking on coder
engineering. Writing a real rANS coder against a representation that is still
changing would be premature.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum

import cbor2
import zstandard

MAGIC = b"SCDX\x1a\x0a"
# v1 introduces symbols-v1 and replacement patches. Older decoders must fail
# closed, rather than silently ignore replacement and flatten straight alpha.
VERSION_MAJOR = 1
VERSION_MINOR = 0

# Guard against a hostile or corrupt length field claiming gigabytes.
MAX_CHUNK_BYTES = 256 * 1024 * 1024


class Flags(IntEnum):
    LOSSLESS = 1 << 0
    PROGRESSIVE = 1 << 1
    HAS_ALPHA = 1 << 2
    TIER0_REQUIRED = 1 << 3


class ChunkType(IntEnum):
    """Four-byte ASCII tags as big-endian u32, PNG convention."""

    HEAD = 0x48454144  # canvas, dims, colour space
    SYMT = 0x53594D54  # symbol tables: palette, fonts
    GRPH = 0x47525048  # scene graph
    ASET = 0x41534554  # embedded raster assets
    RESD = 0x52455344  # residual layers
    META = 0x4D455441  # provenance
    TEXT = 0x54455854  # plain-text index for searchable archives
    DBUG = 0x44425547  # per-node bit accounting, stripped in release


# Chunks a decoder must understand. Anything else is skippable, which is what
# makes minor-version additions non-breaking.
REQUIRED = frozenset({ChunkType.HEAD, ChunkType.SYMT, ChunkType.GRPH})


class ContainerError(ValueError):
    pass


@dataclass(slots=True)
class Chunk:
    type: int
    payload: bytes

    @property
    def tag(self) -> str:
        return struct.pack(">I", self.type).decode("ascii", "replace")


def compress(data: bytes, level: int = 19) -> bytes:
    return zstandard.ZstdCompressor(level=level).compress(data)


def decompress(data: bytes, max_output: int = MAX_CHUNK_BYTES) -> bytes:
    # max_output_size alone is NOT a cap when a frame advertises its size:
    # zstd uses the advertised size instead. Check it before allocation.
    try:
        size = zstandard.frame_content_size(data)
        if size != zstandard.CONTENTSIZE_UNKNOWN and size > max_output:
            raise ContainerError("decompressed payload exceeds size cap")
        result = zstandard.ZstdDecompressor().decompress(
            data, max_output_size=max_output, allow_extra_data=False
        )
    except zstandard.ZstdError as exc:
        raise ContainerError(f"invalid compressed payload: {exc}") from exc
    if len(result) > max_output:
        raise ContainerError("decompressed payload exceeds size cap")
    return result


def encode_cbor(obj: object) -> bytes:
    """Canonical CBOR: deterministic key ordering, so identical graphs produce
    identical bytes."""
    return cbor2.dumps(obj, canonical=True)


def decode_cbor(data: bytes) -> object:
    return cbor2.loads(data)


def write_container(chunks: list[Chunk], flags: int = 0) -> bytes:
    out = bytearray(MAGIC)
    out.append(VERSION_MAJOR)
    out.append(VERSION_MINOR)
    out += struct.pack(">H", flags)
    for c in chunks:
        if len(c.payload) > MAX_CHUNK_BYTES:
            raise ContainerError(f"chunk {c.tag} exceeds {MAX_CHUNK_BYTES} bytes")
        out += struct.pack(">II", c.type, len(c.payload))
        out += c.payload
        out += struct.pack(">I", zlib.crc32(c.payload) & 0xFFFFFFFF)
    return bytes(out)


def read_container(data: bytes) -> tuple[int, list[Chunk]]:
    """Parse a container. Returns (flags, chunks). Raises on any malformation.

    Deliberately strict about structure and lenient about unknown chunk types:
    that combination is what gives forward compatibility without accepting
    corruption.
    """
    if len(data) < len(MAGIC) + 4:
        raise ContainerError("truncated header")
    if data[: len(MAGIC)] != MAGIC:
        raise ContainerError("bad magic; not a .scdx file")
    pos = len(MAGIC)
    major, minor = data[pos], data[pos + 1]
    pos += 2
    if major not in (0, VERSION_MAJOR):
        raise ContainerError(
            f"incompatible major version {major} (this build reads {VERSION_MAJOR})"
        )
    del minor  # minor bumps are additive and readable
    (flags,) = struct.unpack_from(">H", data, pos)
    pos += 2

    chunks: list[Chunk] = []
    while pos < len(data):
        if pos + 8 > len(data):
            raise ContainerError("truncated chunk header")
        ctype, length = struct.unpack_from(">II", data, pos)
        pos += 8
        if length > MAX_CHUNK_BYTES:
            raise ContainerError(f"chunk length {length} exceeds cap")
        if pos + length + 4 > len(data):
            raise ContainerError(
                f"chunk claims {length} bytes but only {len(data) - pos - 4} remain"
            )
        payload = data[pos : pos + length]
        pos += length
        (crc,) = struct.unpack_from(">I", data, pos)
        pos += 4
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != actual:
            raise ContainerError(
                f"CRC mismatch in chunk {Chunk(ctype, b'').tag}: "
                f"stored {crc:08x}, computed {actual:08x}"
            )
        chunks.append(Chunk(ctype, payload))

    present: set[int] = {c.type for c in chunks}
    missing = {int(r) for r in REQUIRED} - present
    if missing:
        raise ContainerError(
            "missing required chunks: "
            + ", ".join(sorted(Chunk(m, b"").tag for m in missing))
        )
    return flags, chunks


def find_chunk(chunks: list[Chunk], ctype: ChunkType) -> bytes | None:
    for c in chunks:
        if c.type == ctype:
            return c.payload
    return None


if __name__ == "__main__":
    payload = encode_cbor({"hello": "world", "n": [1, 2, 3]})
    blob = write_container(
        [
            Chunk(ChunkType.HEAD, b"head"),
            Chunk(ChunkType.SYMT, b"symt"),
            Chunk(ChunkType.GRPH, compress(payload)),
            Chunk(0x5A5A5A5A, b"unknown-but-skippable"),
        ],
        flags=Flags.LOSSLESS,
    )
    flags, chunks = read_container(blob)
    assert flags & Flags.LOSSLESS
    assert len(chunks) == 4, "unknown chunk must be preserved, not dropped"
    got = decode_cbor(decompress(find_chunk(chunks, ChunkType.GRPH) or b""))
    assert got == {"hello": "world", "n": [1, 2, 3]}, got

    # canonical CBOR: key order in the input must not change the bytes
    assert encode_cbor({"a": 1, "b": 2}) == encode_cbor({"b": 2, "a": 1})

    def expect_error(bad: bytes, why: str) -> None:
        try:
            read_container(bad)
        except ContainerError:
            return
        raise AssertionError(f"should have rejected: {why}")

    expect_error(b"nope", "bad magic")
    expect_error(blob[:4], "truncated header")
    expect_error(blob[:-1], "truncated final chunk")
    corrupt = bytearray(blob)
    corrupt[-8] ^= 0xFF  # flip a payload byte, leaving the CRC stale
    expect_error(bytes(corrupt), "CRC mismatch")
    bumped = bytearray(blob)
    bumped[len(MAGIC)] = 99
    expect_error(bytes(bumped), "future major version")
    expect_error(
        write_container([Chunk(ChunkType.HEAD, b"h")]), "missing required chunks"
    )
    print("container self-check passed")
