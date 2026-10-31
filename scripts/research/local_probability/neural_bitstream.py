"""A bounded experimental reference stream, not a production asset format.

Decoding needs only the stream and native backend, not PyTorch or the source
image. Version 2 uses integer inference/CDFs and checks a decoded-pixel hash.
Version 3 keeps that envelope with a compact exact model payload and optional
wider causal features. Version 4 keeps the v3 model and probabilities but splits
the arithmetic stream by row so the decoder can run a multithreaded wavefront.
Version 5 keeps the v3 model, bounds each mixture component's mean, and either
codes like v4 or mixes the network with an adaptive context model; it can also
code ranks among the sample values that every plane uses.
Model payload version 2 stores an all-zero slice of a sliced tensor as one
reserved grid shift; version 1 payloads still decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from semanticcodec.core.scene.document import AssetRef
from semanticcodec.fallback.codecs import decode_asset, encode_png, encode_webp_lossless
from semanticcodec.format.container import decode_cbor, decompress, encode_cbor

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "data/research-local-probability/nnproto"
INTEGER_BINARY = ROOT / "data/research-local-probability/nnproto-integer"
FLOAT_MAGIC = b"NNEX\x00\x01"
MAGIC = b"NNEX\x00\x02"
MAGIC_V3 = b"NNEX\x00\x03"
MAGIC_V4 = b"NNEX\x00\x04"
MAGIC_V5 = b"NNEX\x00\x05"
INTEGER_MAGICS = (MAGIC, MAGIC_V3, MAGIC_V4, MAGIC_V5)
FLOAT_HEADER_SIZE = len(MAGIC) + 8
HEADER_SIZE = FLOAT_HEADER_SIZE + 32
MAX_MODEL_BYTES = 1024 * 1024
MAX_STREAM_BYTES = 128 * 1024 * 1024
# Version 4 row streams. Decoding uses at most this many threads.
DEFAULT_STREAMS = 16
# Native backend threads; 0 lets the backend choose.
THREADS = 0
# Version 5 layouts, as nnproto_integer.cpp: "sN" codes the network alone in N
# row streams (v4 with bounded means), "bN" mixes it with a context model in N
# column bands per plane, and "gN" also lets that model defer to the network
# where mixing has not been paying. Mixing bodies whose planes leave samples
# unused also set V5_VALUES and skip the bits those samples decide.
V5_KINDS = {"s": (0, 64), "b": (1, 16), "g": (3, 16)}
V5_VALUES = 4
# Native calls; context mixing codes several times slower than the network alone.
TIMEOUT = 300
V5_TIMEOUT = 1800


# Compact canonical v3 model payload. Each tensor is raw binary16 or Rice-coded
# integers on a power-of-two grid; decoded values are bit-identical binary16.
MODEL_VERSION = 2
MODEL_VERSIONS = (1, 2)
NAMES = (
    "guide_coeff",
    "affine.weight",
    "hidden.0.weight",
    "hidden.0.bias",
    "hidden.2.weight",
    "hidden.2.bias",
    "head.weight",
    "head.bias",
)
BASE = {1: 14, 0: 21, 2: 30}
CONTEXT = {1: 13, 0: 24, 2: 34}
FAR_TAPS = 12
# Feature set 2 adds the plane's level and the sample's row, column and diagonal parity.
LEVEL_PHASE = 4
# Feature set 3 adds the causal-window least-squares fit: the fit, its spread, its
# errors at four causal neighbours and each decoded plane's fit error at the sample.
# Feature set 4 has the same features for two windows: the set 3 window with fewer
# inputs from decoded planes, and a small one.
LOCAL_FIT = {1: 6, 0: 7, 2: 8}
FITS = {3: 1, 4: 2}
FEATURE_SETS = (0, 1, 2, 3, 4)
MAX_PIXELS = 16_777_216
MAX_SHIFT = 24
MAX_RICE = 40
# Tensor kinds. COLUMNS and ROWS give each slice of a 2-D tensor its own grid,
# so a few large weights no longer force a fine grid on the whole tensor.
RAW, RICE, COLUMNS, ROWS = 0, 1, 2, 3
SLICE_SHIFT_BITS = 5
# From model version 2, this slice shift marks a slice of zeros with no values.
ZERO_SLICE = (1 << SLICE_SHIFT_BITS) - 1


def feature_dimensions(channel: int, context: bool, feature_set: int) -> tuple[int, int]:
    if feature_set not in FEATURE_SETS or (feature_set and not context):
        raise ValueError("feature set")
    base = BASE[channel]
    full = base + (CONTEXT[channel] if context else 0) + (FAR_TAPS if feature_set else 0)
    full += LEVEL_PHASE if feature_set >= 2 else 0
    full += LOCAL_FIT[channel] * FITS.get(feature_set, 0)
    return full, base


def tensor_shapes(channel, hidden, mixtures, context, feature_set) -> dict[str, tuple[int, ...]]:
    full, base = feature_dimensions(channel, context, feature_set)
    return {
        "guide_coeff": (base,),
        "affine.weight": (1, full),
        "hidden.0.weight": (hidden, full),
        "hidden.0.bias": (hidden,),
        "hidden.2.weight": (hidden, hidden),
        "hidden.2.bias": (hidden,),
        "head.weight": (3 * mixtures, hidden),
        "head.bias": (3 * mixtures,),
    }


def check_header(w, h, hidden, mixtures, context, feature_set) -> None:
    if (
        any(type(n) is not int for n in (w, h, hidden, mixtures, feature_set))
        or type(context) is not bool
        or w <= 0
        or h <= 0
        or w * h > MAX_PIXELS
        or not 1 <= hidden <= 64
        or not 1 <= mixtures <= 4
    ):
        raise ValueError("model bounds")
    feature_dimensions(1, context, feature_set)


def check_half(raw: bytes) -> None:
    words = np.frombuffer(raw, dtype="<u2")
    # Positive binary16 patterns sort by magnitude; 0x70e2 is exactly 10000.
    if np.any((words & 0x7FFF) >= 0x70E2):
        raise ValueError("parameter bounds")


class _Bits:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.used = 0

    def put(self, value: int, count: int) -> None:
        for i in range(count - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.used += 1
            if self.used == 8:
                self.out.append(self.acc)
                self.acc = self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.out.append(self.acc << (8 - self.used))
            self.acc = self.used = 0
        return bytes(self.out)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def bit(self) -> int:
        if self.position >= 8 * len(self.data):
            raise ValueError("truncated model bits")
        value = (self.data[self.position >> 3] >> (7 - (self.position & 7))) & 1
        self.position += 1
        return value

    def bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if position >= len(data) or shift > 35:
            raise ValueError("invalid model varint")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


def _grid(values: np.ndarray) -> tuple[int, np.ndarray]:
    for shift in range(MAX_SHIFT + 1):
        scaled = values * 2.0**shift
        if np.all(scaled == np.round(scaled)):
            return shift, np.round(scaled).astype(np.int64)
    raise ValueError("non-binary16 value")


def _zigzag(q: np.ndarray) -> np.ndarray:
    return np.where(q >= 0, 2 * q, -2 * q - 1)


def _rice_cost(z: np.ndarray, k: int) -> int:
    return int((z >> k).sum()) + len(z) * (k + 1)


def _rice_k(z: np.ndarray) -> int:
    return min(range(MAX_RICE + 1), key=lambda candidate: (_rice_cost(z, candidate), candidate))


def _put_rice(bits: _Bits, z: np.ndarray, k: int) -> None:
    for value in z.tolist():
        bits.put(0, value >> k)
        bits.put(1, 1)
        bits.put(value & ((1 << k) - 1), k)


def sliceable(shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and min(shape) > 1


def _value_shifts(values: np.ndarray) -> np.ndarray:
    """Smallest s with value * 2**s integral, for each binary16 value."""
    words = values.astype("<f2").view("<u2").astype(np.int64)
    exponent = (words >> 10) & 31
    units = (words & 1023) | np.where(exponent == 0, 0, 1024)
    present = units != 0
    trailing = np.zeros_like(units)
    trailing[present] = np.log2(units[present] & -units[present]).astype(np.int64)
    shifts = np.where(exponent == 0, 24, 25 - exponent) - trailing
    return np.where(present, np.maximum(shifts, 0), 0)


def slice_grids(values: np.ndarray, shape: tuple[int, ...], kind: int):
    """(grid shift per slice, integers with one slice per row) of a 2-D tensor."""
    matrix = values.reshape(shape)
    if kind == COLUMNS:
        matrix = matrix.T
    shifts = _value_shifts(matrix.ravel()).reshape(matrix.shape).max(axis=1)
    return shifts, np.rint(matrix * np.exp2(shifts)[:, None]).astype(np.int64)


def zero_slices(slices: np.ndarray, escape: bool) -> np.ndarray:
    """Slices coded as ZERO_SLICE: the all-zero ones when the version has the escape."""
    return ~slices.any(axis=1) if escape else np.zeros(len(slices), bool)


def _encode_tensor(raw: bytes, shape: tuple[int, ...] = (), escape: bool = True) -> bytes:
    values = np.frombuffer(raw, "<f2").astype(np.float64)
    shift, q = _grid(values)
    z = _zigzag(q)
    k = _rice_k(z)
    bits = _Bits()
    _put_rice(bits, z, k)
    body = bits.finish()
    coded = bytes([RICE, shift, k]) + _varint(len(body)) + body
    if sliceable(shape):
        for kind in (COLUMNS, ROWS):
            shifts, slices = slice_grids(values, shape, kind)
            zs = _zigzag(slices)
            zero = zero_slices(slices, escape)
            k = _rice_k(zs[~zero].ravel())
            size = (SLICE_SHIFT_BITS * len(shifts) + _rice_cost(zs[~zero].ravel(), k) + 7) // 8
            if 2 + len(_varint(size)) + size >= len(coded):
                continue
            bits = _Bits()
            for slice_shift, z, empty in zip(shifts.tolist(), zs, zero.tolist(), strict=True):
                bits.put(ZERO_SLICE if empty else slice_shift, SLICE_SHIFT_BITS)
                if not empty:
                    _put_rice(bits, z, k)
            body = bits.finish()
            coded = bytes([kind, k]) + _varint(len(body)) + body
    return coded if len(coded) < 1 + len(raw) else bytes([RAW]) + raw


def _read_rice(reader: _Reader, count: int, k: int, shift: int) -> np.ndarray:
    # |value| < 10000 with shift <= 24 bounds the zigzag value, hence the unary run.
    limit = 2 * 10_000 * 2**shift
    z = []
    for _ in range(count):
        quotient = 0
        while not reader.bit():
            quotient += 1
            if (quotient << k) > limit:
                raise ValueError("parameter bounds")
        z.append((quotient << k) | reader.bits(k))
    zz = np.array(z, dtype=np.int64)
    return np.where(zz & 1, -((zz + 1) >> 1), zz >> 1)


def _decode_tensor(
    data: bytes, position: int, count: int, shape: tuple[int, ...] = (), escape: bool = True
) -> tuple[bytes, int]:
    if position >= len(data):
        raise ValueError("truncated model")
    kind = data[position]
    position += 1
    if kind == RAW:
        raw = data[position : position + 2 * count]
        if len(raw) != 2 * count:
            raise ValueError("truncated model")
        return raw, position + 2 * count
    if kind == RICE and position + 2 <= len(data):
        shift, k = data[position], data[position + 1]
        position += 2
    elif kind in (COLUMNS, ROWS) and sliceable(shape) and position < len(data):
        shift, k = 0, data[position]
        position += 1
    else:
        raise ValueError("invalid tensor coding")
    if shift > MAX_SHIFT or k > MAX_RICE:
        raise ValueError("invalid tensor grid")
    length, position = _read_varint(data, position)
    body = data[position : position + length]
    if len(body) != length:
        raise ValueError("truncated model")
    reader = _Reader(body)
    if kind == RICE:
        values = _read_rice(reader, count, k, shift).astype(np.float64) / 2.0**shift
    else:
        slices = shape[1] if kind == COLUMNS else shape[0]
        matrix = np.empty((slices, count // slices))
        for index in range(slices):
            slice_shift = reader.bits(SLICE_SHIFT_BITS)
            if escape and slice_shift == ZERO_SLICE:
                matrix[index] = 0.0
                continue
            if slice_shift > MAX_SHIFT:
                raise ValueError("invalid tensor grid")
            q = _read_rice(reader, count // slices, k, slice_shift)
            matrix[index] = q.astype(np.float64) / 2.0**slice_shift
        values = (matrix.T if kind == COLUMNS else matrix).ravel()
    # Padding bits are checked by the canonical re-encoding of the whole payload.
    if (reader.position + 7) // 8 != length:
        raise ValueError("noncanonical tensor padding")
    half = values.astype("<f2")
    if not np.array_equal(half.astype(np.float64), values):
        raise ValueError("value not exactly binary16")
    raw = half.tobytes()
    check_half(raw)
    return raw, position + length


def encode_model(
    header: tuple, planes: list[dict[str, bytes]], version: int = MODEL_VERSION
) -> bytes:
    w, h, hidden, mixtures, context, feature_set = header
    check_header(w, h, hidden, mixtures, context, feature_set)
    if version not in MODEL_VERSIONS:
        raise ValueError("model version")
    out = bytearray([version]) + _varint(w) + _varint(h)
    out += bytes([hidden, mixtures, int(context) | (feature_set << 1)])
    if len(planes) != 3:
        raise ValueError("plane count")
    for channel, tensors in zip((1, 0, 2), planes, strict=True):
        expected = tensor_shapes(channel, hidden, mixtures, context, feature_set)
        if set(tensors) != set(NAMES):
            raise ValueError("parameter names")
        for name in NAMES:
            raw = tensors[name]
            if len(raw) != 2 * int(np.prod(expected[name])):
                raise ValueError("parameter shape or data")
            check_half(raw)
            out += _encode_tensor(raw, tuple(expected[name]), version >= 2)
    return bytes(out)


def decode_model(payload: bytes, max_bytes: int) -> tuple[tuple, list[dict[str, bytes]]]:
    if len(payload) > max_bytes or not payload or payload[0] not in MODEL_VERSIONS:
        raise ValueError("invalid v3 model")
    version = payload[0]
    w, position = _read_varint(payload, 1)
    h, position = _read_varint(payload, position)
    if position + 3 > len(payload):
        raise ValueError("truncated model")
    hidden, mixtures, flags = payload[position : position + 3]
    position += 3
    if flags >> 1 not in FEATURE_SETS:
        raise ValueError("feature set")
    header = (w, h, hidden, mixtures, bool(flags & 1), flags >> 1)
    check_header(*header)
    planes = []
    for channel in (1, 0, 2):
        expected = tensor_shapes(channel, *header[2:])
        tensors = {}
        for name in NAMES:
            tensors[name], position = _decode_tensor(
                payload, position, int(np.prod(expected[name])), tuple(expected[name]), version >= 2
            )
        planes.append(tensors)
    if position != len(payload) or encode_model(header, planes, version) != payload:
        raise ValueError("noncanonical or trailing model data")
    return header, planes


def half_to_fixed(raw: bytes, *, wide: bool) -> bytes:
    """Binary16 to signed Q24/Q16, nearest with ties to even, using integers."""
    words = np.frombuffer(raw, dtype="<u2").astype(np.int64)
    exponent = (words >> 10) & 31
    if np.any(exponent == 31):
        raise ValueError("non-finite parameter")
    units = (words & 1023) + np.where(exponent == 0, 0, 1024)
    shifts = np.where(exponent == 0, 0 if wide else -8, exponent - (1 if wide else 9))
    left = units << np.maximum(shifts, 0)
    denominator = np.int64(1) << np.maximum(-shifts, 0)
    quotient, remainder = units // denominator, units % denominator
    round_up = (2 * remainder > denominator) | (
        (2 * remainder == denominator) & ((quotient & 1) != 0)
    )
    magnitude = np.where(shifts >= 0, left, quotient + round_up)
    if np.any(magnitude >= 10_000 * (2**24 if wide else 2**16)):
        raise ValueError("parameter bounds")
    return (
        np.where((words >> 15) != 0, -magnitude, magnitude)
        .astype("<i8" if wide else "<i4")
        .tobytes()
    )


def parse_v2(metadata: bytes) -> tuple[tuple, list[dict[str, bytes]]]:
    """Validate a v1/v2 compressed CBOR model; return header and binary16 tensors."""
    if len(metadata) > MAX_MODEL_BYTES:
        raise ValueError("oversized model")
    raw_model = decompress(metadata, max_output=MAX_MODEL_BYTES)
    body = decode_cbor(raw_model)
    if encode_cbor(body) != raw_model:
        raise ValueError("noncanonical or trailing model data")
    if not isinstance(body, list) or len(body) != 6:
        raise ValueError("invalid model")
    w, h, hidden, mixtures, context, planes = body
    if (
        any(type(n) is not int for n in (w, h, hidden, mixtures))
        or w <= 0
        or h <= 0
        or w * h > 16_777_216
        or not 1 <= hidden <= 64
        or not 1 <= mixtures <= 4
        or type(context) is not bool
        or not isinstance(planes, list)
        or len(planes) != 3
    ):
        raise ValueError("model bounds")
    header = (w, h, hidden, mixtures, context, 0)
    states = []
    for channel, tensors in zip((1, 0, 2), planes, strict=True):
        shapes = tensor_shapes(channel, hidden, mixtures, context, 0)
        if not isinstance(tensors, list) or len(tensors) != len(shapes):
            raise ValueError("parameter count")
        state = {}
        for tensor in tensors:
            if not isinstance(tensor, list) or len(tensor) != 3:
                raise ValueError("invalid parameter")
            name, shape, raw = tensor
            if (
                not isinstance(name, str)
                or name not in shapes
                or name in state
                or not isinstance(shape, list)
                or any(type(n) is not int for n in shape)
                or tuple(shape) != shapes[name]
                or not isinstance(raw, bytes)
                or len(raw) != 2 * int(np.prod(shapes[name]))
            ):
                raise ValueError("parameter shape or data")
            # Rejects magnitudes >= 10000, all infinities, and NaNs.
            check_half(raw)
            state[name] = raw
        states.append(state)
    return header, states


def parse_v3(payload: bytes) -> tuple[tuple, list[dict[str, bytes]]]:
    return decode_model(payload, MAX_MODEL_BYTES)


def native_configuration(
    header: tuple, planes: list[dict[str, bytes]], *, integer: bool = True
) -> tuple[bytes, tuple[int, int]]:
    w, h, hidden, mixtures, context, feature_set = header
    if feature_set and not integer:
        raise ValueError("wider features need the integer backend")
    magic = (0x33494E4E if feature_set else 0x32494E4E) if integer else 0x31504E4E
    data = bytearray(struct.pack("<6I", magic, w, h, hidden, mixtures, context))
    if feature_set:
        data += struct.pack("<I", feature_set)
    for channel, state in zip((1, 0, 2), planes, strict=True):
        dimensions, base = feature_dimensions(channel, context, feature_set)
        data += struct.pack("<2I", dimensions, base)
        for name in NAMES:
            raw = state[name]
            if integer:
                data += half_to_fixed(raw, wide=name in ("guide_coeff", "affine.weight"))
            else:
                data += np.frombuffer(raw, dtype="<f2").astype("<f8").tobytes()
    return bytes(data), (w, h)


def configuration(metadata: bytes, *, integer: bool = True) -> tuple[bytes, tuple[int, int]]:
    return native_configuration(*parse_v2(metadata), integer=integer)


def transcode_v3(metadata: bytes) -> bytes:
    """Exact compact payload for an existing v2 model (identical parameters)."""
    return encode_model(*parse_v2(metadata))


def temporary_directory():
    directory = ROOT / "data/research-local-probability/tmp"
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=directory)


def local_fits(pixels: np.ndarray, feature_set: int = 3) -> np.ndarray:
    """Least-squares fits of known pixels, exactly as the integer backend codes them.

    Returns int16 (2 * FITS[feature_set], 3, h, w): for each window of the feature
    set, the fit in eighths of a sample, then 16 times the window fit RMS, for
    planes G, R, B.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("expected opaque uint8 RGB")
    if feature_set not in FITS:
        raise ValueError("feature set without least-squares fits")
    count = FITS[feature_set]
    h, w = pixels.shape[:2]
    with temporary_directory() as td:
        source, target = Path(td) / "pixels", Path(td) / "fits"
        source.write_bytes(np.ascontiguousarray(pixels).tobytes())
        command = [str(INTEGER_BINARY), "local-fit", str(w), str(h), str(source), str(target)]
        command += [str(THREADS), str(feature_set)]
        proc = subprocess.run(command, capture_output=True, check=False, timeout=600)
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode("utf8", "replace"))
        fits = np.fromfile(target, "<i2")
    if fits.size != 6 * count * h * w:
        raise ValueError("local fit size")
    return fits.reshape(2 * count, 3, h, w)


def call(
    mode: str,
    config: Path,
    source: Path,
    target: Path,
    *,
    integer: bool = False,
    streams=None,
    fits=None,
    timeout: int = TIMEOUT,
) -> None:
    binary = INTEGER_BINARY if integer else BINARY
    command = [str(binary), mode, str(config), str(source), str(target)]
    if integer:
        command.append(str(THREADS))
        if streams is not None:
            command.append(str(streams))
            if fits is not None:
                command.append(str(fits))
    proc = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf8", "replace"))


def _stream(
    config: bytes,
    w: int,
    h: int,
    pixels: np.ndarray,
    *,
    integer: bool,
    streams=None,
    fits=None,
    mode=None,
) -> bytes:
    if pixels.dtype != np.uint8 or pixels.shape != (h, w, 3):
        raise ValueError("expected matching opaque uint8 RGB")
    with temporary_directory() as td:
        directory = Path(td)
        cf, sf, out = (directory / name for name in ("config", "pixels", "coded"))
        cf.write_bytes(config)
        sf.write_bytes(np.ascontiguousarray(pixels).tobytes())
        if mode is None:
            mode = "encode" if streams is None else "encode-v4"
        timeout = V5_TIMEOUT if mode == "encode-v5" else TIMEOUT
        call(mode, cf, sf, out, integer=integer, streams=streams, fits=fits, timeout=timeout)
        stream = out.read_bytes()
    if len(stream) > MAX_STREAM_BYTES:
        raise ValueError("oversized arithmetic stream")
    return stream


def encode(pixels: np.ndarray, metadata: bytes, *, integer: bool = True) -> bytes:
    config, (w, h) = configuration(metadata, integer=integer)
    stream = _stream(config, w, h, pixels, integer=integer)
    digest = hashlib.sha256(pixels.tobytes()).digest() if integer else b""
    magic = MAGIC if integer else FLOAT_MAGIC
    return magic + struct.pack("<2I", len(metadata), len(stream)) + digest + metadata + stream


def encode_v3(pixels: np.ndarray, payload: bytes) -> bytes:
    config, (w, h) = native_configuration(*parse_v3(payload))
    stream = _stream(config, w, h, pixels, integer=True)
    digest = hashlib.sha256(pixels.tobytes()).digest()
    return MAGIC_V3 + struct.pack("<2I", len(payload), len(stream)) + digest + payload + stream


def encode_v4(
    pixels: np.ndarray, payload: bytes, *, streams: int = DEFAULT_STREAMS, fits=None
) -> bytes:
    """The v3 model and probabilities; row y goes to stream y % min(streams, height).

    fits optionally names local_fits output for these pixels and the payload's
    feature set, which the backend then reuses. Wrong fits code a file that does
    not decode, so decode the result.
    """
    if type(streams) is not int or not 1 <= streams <= 64:
        raise ValueError("stream count")
    config, (w, h) = native_configuration(*parse_v3(payload))
    stream = _stream(config, w, h, pixels, integer=True, streams=streams, fits=fits)
    digest = hashlib.sha256(pixels.tobytes()).digest()
    return MAGIC_V4 + struct.pack("<2I", len(payload), len(stream)) + digest + payload + stream


def shared_values(pixels: np.ndarray):
    """Sorted sample values if every plane uses the same set and it has interior gaps."""
    used = [np.bincount(pixels[..., c].ravel(), minlength=256) > 0 for c in range(3)]
    if not (np.array_equal(used[0], used[1]) and np.array_equal(used[0], used[2])):
        return None
    values = np.flatnonzero(used[0]).astype(np.uint8)
    return values if int(values[-1]) - int(values[0]) + 1 > len(values) else None


def pack_values(pixels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each sample's rank among values, which must hold every sample."""
    used = np.zeros(256, bool)
    used[values] = True
    if not used[pixels].all():
        raise ValueError("sample outside the value map")
    ranks = np.zeros(256, np.uint8)
    ranks[values] = np.arange(len(values))
    return ranks[pixels]


def _v5_head(values) -> bytes:
    if values is None:
        return b"\0"
    values = np.asarray(values)
    if (
        values.dtype != np.uint8
        or values.ndim != 1
        or not len(values)
        or np.any(np.diff(values.astype(int)) <= 0)
    ):
        raise ValueError("value map")
    used = np.zeros(256, bool)
    used[values] = True
    return b"\1" + np.packbits(used, bitorder="little").tobytes()


def _v5_split(stream: bytes):
    """(sorted values or None, native body) of a v5 stream."""
    if stream[:1] == b"\0":
        return None, stream[1:]
    if stream[:1] != b"\1" or len(stream) < 33:
        raise ValueError("value map")
    values = np.flatnonzero(np.unpackbits(np.frombuffer(stream[1:33], np.uint8), bitorder="little"))
    if not len(values):
        raise ValueError("value map")
    return values.astype(np.uint8), stream[33:]


def check_layout(layout) -> None:
    if (
        type(layout) is not str
        or len(layout) < 2
        or layout[0] not in V5_KINDS
        or not layout[1:].isdigit()
        or layout[1] == "0"
        or not 1 <= int(layout[1:]) <= V5_KINDS[layout[0]][1]
    ):
        raise ValueError("v5 layout")


def encode_v5(
    pixels: np.ndarray, payload: bytes, *, layout: str = "g1", values=None, fits=None
) -> bytes:
    """The v3 model with bounded component means, coded in a V5_KINDS layout.

    With values (shared_values of these pixels) the backend codes each sample's
    rank among them (pack_values), so the payload's network and any fits must
    be for the packed pixels. fits is as for encode_v4.
    """
    check_layout(layout)
    head = _v5_head(values)
    coded = pixels if values is None else pack_values(pixels, values)
    config, (w, h) = native_configuration(*parse_v3(payload))
    native = _stream(config, w, h, coded, integer=True, streams=layout, fits=fits, mode="encode-v5")
    stream = head + native
    if len(stream) > MAX_STREAM_BYTES:
        raise ValueError("oversized arithmetic stream")
    digest = hashlib.sha256(pixels.tobytes()).digest()
    return MAGIC_V5 + struct.pack("<2I", len(payload), len(stream)) + digest + payload + stream


def v5_layout(blob: bytes) -> str:
    """The layout of a v5 file."""
    payload = model_payload(blob)
    _, native = _v5_split(blob[HEADER_SIZE + len(payload) :])
    kinds = {flags: kind for kind, (flags, _) in V5_KINDS.items()}
    if len(native) < 2:
        raise ValueError("v5 layout")
    flags = native[0] & ~V5_VALUES if native[0] & 1 else native[0]
    if flags not in kinds:
        raise ValueError("v5 layout")
    return f"{kinds[flags]}{native[1]}"


def model_payload(blob: bytes) -> bytes:
    if not blob.startswith(INTEGER_MAGICS) or len(blob) < HEADER_SIZE:
        raise ValueError("magic")
    (model_size,) = struct.unpack_from("<I", blob, len(MAGIC))
    return blob[HEADER_SIZE : HEADER_SIZE + model_size]


def reencode(blob: bytes, pixels: np.ndarray) -> bytes:
    """Encode pixels again with the in-band model (and stream count) of an integer file."""
    payload = model_payload(blob)
    if blob.startswith(MAGIC_V5):
        values, _ = _v5_split(blob[HEADER_SIZE + len(payload) :])
        return encode_v5(pixels, payload, layout=v5_layout(blob), values=values)
    if blob.startswith(MAGIC_V4):
        position = HEADER_SIZE + len(payload)
        if position >= len(blob):
            raise ValueError("length")
        return encode_v4(pixels, payload, streams=blob[position])
    return encode_v3(pixels, payload) if blob.startswith(MAGIC_V3) else encode(pixels, payload)


def decode(blob: bytes) -> np.ndarray:
    version5 = blob.startswith(MAGIC_V5)
    version4 = blob.startswith(MAGIC_V4)
    version3 = version5 or version4 or blob.startswith(MAGIC_V3)
    integer = version3 or blob.startswith(MAGIC)
    header_size = HEADER_SIZE if integer else FLOAT_HEADER_SIZE
    if (not integer and not blob.startswith(FLOAT_MAGIC)) or len(blob) < header_size:
        raise ValueError("magic")
    model_size, stream_size = struct.unpack_from("<2I", blob, len(MAGIC))
    if (
        model_size > MAX_MODEL_BYTES
        or stream_size > MAX_STREAM_BYTES
        or stream_size < 4
        or len(blob) != header_size + model_size + stream_size
    ):
        raise ValueError("length")
    metadata = blob[header_size : header_size + model_size]
    stream = blob[header_size + model_size :]
    values = None
    if version5:
        values, stream = _v5_split(stream)
    if version3:
        config, (w, h) = native_configuration(*parse_v3(metadata))
    else:
        config, (w, h) = configuration(metadata, integer=integer)
    with temporary_directory() as td:
        directory = Path(td)
        cf, sf, out = (directory / name for name in ("config", "stream", "out"))
        cf.write_bytes(config)
        sf.write_bytes(stream)
        mode = "decode-v5" if version5 else "decode-v4" if version4 else "decode"
        call(mode, cf, sf, out, integer=integer, timeout=V5_TIMEOUT if version5 else TIMEOUT)
        pixels = np.frombuffer(out.read_bytes(), np.uint8)
        if pixels.size != w * h * 3:
            raise ValueError("decoded dimensions")
        if values is not None:
            if pixels.max(initial=0) >= len(values):
                raise ValueError("decoded rank outside the value map")
            pixels = values[pixels]
        if (
            integer
            and hashlib.sha256(pixels.tobytes()).digest() != blob[FLOAT_HEADER_SIZE:HEADER_SIZE]
        ):
            raise ValueError("decoded pixel checksum mismatch")
        return pixels.reshape(h, w, 3).copy()


def measure_baselines(pixels: np.ndarray) -> dict[str, int]:
    import imagecodecs

    baseline = imagecodecs.jpegxl_encode(pixels, lossless=True, effort=9, numthreads=1)
    if not np.array_equal(imagecodecs.jpegxl_decode(baseline, index=0, numthreads=1), pixels):
        raise AssertionError("JPEG XL baseline mismatch")
    baselines = {"jxl-lossless-e9": len(baseline)}
    h, w = pixels.shape[:2]
    for name, function in (("png", encode_png), ("webp-lossless", encode_webp_lossless)):
        payload = function(pixels)
        reconstructed = decode_asset(AssetRef(name, w, h, payload))[..., :3]
        if not np.array_equal(reconstructed, pixels):
            raise AssertionError(f"{name} baseline mismatch")
        baselines[name] = len(payload)
    return baselines


def run(path: Path, model_path: Path, output: Path, *, integer: bool = True) -> dict:
    from input_image import read_rgb

    pixels = read_rgb(path)
    metadata = model_path.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    blob = encode(pixels, metadata, integer=integer)
    encode_seconds = time.perf_counter() - start
    output.write_bytes(blob)
    start = time.perf_counter()
    reconstructed = decode(blob)
    decode_seconds = time.perf_counter() - start
    if not np.array_equal(reconstructed, pixels):
        raise AssertionError("independent decoder mismatch")
    baselines = measure_baselines(pixels)
    _, stream_size = struct.unpack_from("<2I", blob, len(MAGIC))
    return {
        "image": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256(metadata).hexdigest(),
        "native_backend_sha256": hashlib.sha256(
            (INTEGER_BINARY if integer else BINARY).read_bytes()
        ).hexdigest(),
        "envelope_version": 2 if integer else 1,
        "numerical_backend": "integer-q16" if integer else "floating-reference",
        "model_bytes": len(metadata),
        "stream_bytes": stream_size,
        "complete_file_bytes": len(blob),
        "baseline_jxl_e9": baselines["jxl-lossless-e9"],
        "baselines": baselines,
        "ratio": len(blob) / baselines["jxl-lossless-e9"],
        "vs_best": len(blob) / min(baselines.values()),
        "encode_s": encode_seconds,
        "decode_s": decode_seconds,
        "exact_rgb": True,
        "shared_image_assets_bytes": 0,
        "warning": (
            "Experimental integer-only stream. Requires C++17 and 128-bit integers; "
            "ARM64/x86_64 Rosetta tested on one Mac; broader hardware validation "
            "and production integration remain outstanding."
            if integer
            else "Experimental floating reference stream, not a portable format."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--float-reference", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.decode:
        if args.model or args.receipt or args.float_reference:
            parser.error("--decode needs only a bitstream and --out")
        if args.input.stat().st_size > MAX_MODEL_BYTES + MAX_STREAM_BYTES + HEADER_SIZE:
            parser.error("experimental bitstream file byte budget exceeded")
        Image.fromarray(decode(args.input.read_bytes())).save(args.out, format="PNG")
        return
    if not args.model or not args.receipt:
        parser.error("encoding requires --model and --receipt")
    row = run(args.input, args.model, args.out, integer=not args.float_reference)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
