"""The experimental decoder needs neither PyTorch nor an original image."""

from __future__ import annotations

import copy
import importlib.util
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semanticcodec.format.container import compress, encode_cbor

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "scripts/research/local_probability"
spec = importlib.util.spec_from_file_location(
    "_local_probability_reference", SOURCE / "neural_bitstream.py"
)
assert spec is not None and spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def model_body(w=7, h=5, context=True):
    rng = np.random.default_rng(239)
    hidden, mixtures = 4, 1
    planes = []
    for channel in (1, 0, 2):
        base = {1: 14, 0: 21, 2: 30}[channel]
        dimensions = base + ({1: 13, 0: 24, 2: 34}[channel] if context else 0)
        shapes = {
            "guide_coeff": (base,),
            "affine.weight": (1, dimensions),
            "hidden.0.weight": (hidden, dimensions),
            "hidden.0.bias": (hidden,),
            "hidden.2.weight": (hidden, hidden),
            "hidden.2.bias": (hidden,),
            "head.weight": (3 * mixtures, hidden),
            "head.bias": (3 * mixtures,),
        }
        state = []
        for name, shape in shapes.items():
            values = rng.normal(0, 0.003, shape).astype("<f2")
            state.append([name, list(shape), values.tobytes()])
        planes.append(state)
    return [w, h, hidden, mixtures, context, planes]


@pytest.fixture(scope="session")
def backend(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("optional research test needs a C++17 compiler")
    directory = tmp_path_factory.mktemp("probability-native")
    binaries = []
    for source in ("nnproto.cpp", "nnproto_integer.cpp"):
        binary = directory / source.removesuffix(".cpp")
        subprocess.run(
            [
                compiler,
                "-O2",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffp-contract=off",
                "-fno-fast-math",
                str(SOURCE / source),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        binaries.append(binary)
    return binaries


@pytest.fixture
def coder(backend, tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "BINARY", backend[0])
    monkeypatch.setattr(reference, "INTEGER_BINARY", backend[1])
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    return reference


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (5, 7)])
@pytest.mark.parametrize("context", [False, True])
def test_independent_reference_roundtrip(coder, shape, context):
    h, w = shape
    pixels = np.random.default_rng(71).integers(0, 256, (h, w, 3), np.uint8)
    metadata = compress(encode_cbor(model_body(w, h, context)))
    stream = coder.encode(pixels, metadata)
    assert np.array_equal(coder.decode(stream), pixels)
    assert coder.encode(pixels, metadata) == stream
    assert (
        len(stream)
        == coder.HEADER_SIZE
        + len(metadata)
        + struct.unpack_from("<2I", stream, len(coder.MAGIC))[1]
    )


def test_research_decode_cli_has_no_source_image(coder, tmp_path):
    pixels = np.random.default_rng(2).integers(0, 256, (5, 7, 3), np.uint8)
    blob = coder.encode(pixels, compress(encode_cbor(model_body())))
    coded, output = tmp_path / "only-stream.nnexp", tmp_path / "reconstructed.png"
    coded.write_bytes(blob)
    # A fresh interpreter imports the decoder, not the trainer. There is no
    # original image on disk, no model filename, and no Torch import.
    program = (
        "import importlib.util,pathlib,sys;"
        "spec=importlib.util.spec_from_file_location('ref',sys.argv[1]);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "m.INTEGER_BINARY=pathlib.Path(sys.argv[2]);m.ROOT=pathlib.Path(sys.argv[3]);"
        "sys.argv=['decoder',sys.argv[4],'--decode','--out',sys.argv[5]];"
        "m.main();assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(SOURCE / "neural_bitstream.py"),
            str(coder.INTEGER_BINARY),
            str(tmp_path),
            str(coded),
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    with Image.open(output) as image:
        assert np.array_equal(np.asarray(image), pixels)


@pytest.mark.parametrize(
    ("field", "value"),
    [(0, True), (0, -7), (1, 0), (1, 16_777_217), (2, 65), (2, True), (3, 5), (4, 1), (5, [])],
)
def test_invalid_architecture(field, value):
    body = model_body()
    body[field] = value
    with pytest.raises(ValueError, match="bounds"):
        reference.configuration(compress(encode_cbor(body)))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "shape", "data", "nan", "large"])
def test_invalid_parameters(kind):
    body = model_body()
    tensors = body[5][0]
    if kind == "missing":
        tensors.pop()
    elif kind == "duplicate":
        tensors[-1] = copy.deepcopy(tensors[0])
    elif kind == "shape":
        tensors[0][1] = [1_000_000_000]
    elif kind == "data":
        tensors[0][2] = b"x"
    else:
        values = np.zeros(14, dtype="<f2")
        values[0] = np.nan if kind == "nan" else 12_000
        tensors[0][2] = values.tobytes()
    with pytest.raises(ValueError):
        reference.configuration(compress(encode_cbor(body)))


def test_model_expansion_and_trailing_data_rejected():
    with pytest.raises(ValueError, match="size cap"):
        reference.configuration(compress(b"x" * (reference.MAX_MODEL_BYTES + 1)))
    with pytest.raises(ValueError, match="trailing"):
        reference.configuration(compress(encode_cbor(model_body()) + b"x"))


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"not a stream",
        b"NNEX\x00\x02" + b"\0" * 20,
        b"NNEX\x00\x01" + struct.pack("<2I", 2**30, 0),
    ],
)
def test_invalid_envelope(blob):
    with pytest.raises(ValueError):
        reference.decode(blob)


def test_truncated_stream_rejected(coder):
    metadata = compress(encode_cbor(model_body()))
    blob = coder.MAGIC + struct.pack("<2I", len(metadata), 4) + b"\0" * 32 + metadata + b"\0" * 4
    with pytest.raises(RuntimeError, match="truncated arithmetic"):
        coder.decode(blob)


def test_rgb_type_and_dimensions_checked(coder):
    metadata = compress(encode_cbor(model_body()))
    for pixels in (
        np.zeros((5, 7, 4), np.uint8),
        np.zeros((5, 7, 3), np.uint16),
        np.zeros((6, 7, 3), np.uint8),
    ):
        with pytest.raises(ValueError, match="uint8 RGB"):
            coder.encode(pixels, metadata)


@pytest.mark.parametrize("wide", [False, True])
def test_every_supported_binary16_value_converts_exactly(wide):
    words = np.arange(65536, dtype="<u2")
    words = words[((words >> 10) & 31) != 31]
    values = words.view("<f2").astype(np.float64)
    supported = np.isfinite(values) & (np.abs(values) < 10_000)
    raw = words[supported].tobytes()
    expected = np.rint(values[supported] * (2**24 if wide else 2**16)).astype(
        "<i8" if wide else "<i4"
    )
    assert reference.half_to_fixed(raw, wide=wide) == expected.tobytes()


def test_legacy_floating_reference_is_still_readable(coder):
    pixels = np.full((5, 7, 3), 117, np.uint8)
    blob = coder.encode(pixels, compress(encode_cbor(model_body())), integer=False)
    assert blob.startswith(coder.FLOAT_MAGIC)
    assert np.array_equal(coder.decode(blob), pixels)


def test_integer_pixel_checksum_rejected(coder):
    pixels = np.full((5, 7, 3), 117, np.uint8)
    blob = bytearray(coder.encode(pixels, compress(encode_cbor(model_body()))))
    blob[coder.FLOAT_HEADER_SIZE] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        coder.decode(bytes(blob))


def test_integer_backend_matches_without_optimizations(coder, tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None
    alternate = tmp_path / "integer-unoptimized"
    subprocess.run(
        [compiler, "-O0", "-std=c++17", str(SOURCE / "nnproto_integer.cpp"), "-o", str(alternate)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    pixels = np.random.default_rng(82).integers(0, 256, (5, 7, 3), np.uint8)
    metadata = compress(encode_cbor(model_body()))
    # Feature set 4 adds two binary64 solves; the image is larger than either window.
    textured = local_fit_pixels(37, 41)
    local = v3_model(41, 37, 4, grid=True, scale=0.3)
    optimized = coder.encode(pixels, metadata)
    optimized_local = coder.encode_v4(textured, local, streams=3)
    optimized_mixed = coder.encode_v5(textured, local, layout="g3")
    optimized_fits = coder.local_fits(textured, 4)
    coder.INTEGER_BINARY = alternate
    unoptimized = coder.encode(pixels, metadata)
    assert optimized == unoptimized
    assert np.array_equal(coder.decode(optimized), pixels)
    assert coder.encode_v4(textured, local, streams=3) == optimized_local
    assert coder.encode_v5(textured, local, layout="g3") == optimized_mixed
    assert np.array_equal(coder.local_fits(textured, 4), optimized_fits)
    assert np.array_equal(coder.decode(optimized_local), textured)
    assert np.array_equal(coder.decode(optimized_mixed), textured)


@pytest.fixture
def quantizer(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import quantize_model

    return quantize_model


@pytest.mark.parametrize("bits", [8, 10, 12, 16])
def test_quantized_model_preserves_guides_and_decodes_exactly(coder, quantizer, bits):
    from semanticcodec.format.container import decode_cbor, decompress

    original = model_body()
    metadata = compress(encode_cbor(original))
    quantized = quantizer.quantize(metadata, bits)
    body = decode_cbor(decompress(quantized))
    for before, after in zip(original[5], body[5], strict=True):
        for index in (0, 1):
            assert before[index] == after[index]
    if bits == 16:
        assert quantized == metadata
    pixels = np.random.default_rng(51).integers(0, 256, (5, 7, 3), np.uint8)
    blob = coder.encode(pixels, quantized)
    assert np.array_equal(coder.decode(blob), pixels)


@pytest.mark.parametrize("bits", [7, 9, 24])
def test_unsupported_precision_rejected(quantizer, bits):
    with pytest.raises(ValueError, match="precisions"):
        quantizer.quantize(compress(encode_cbor(model_body())), bits)


def v3_model(w=7, h=5, feature_set=0, *, grid=False, scale=0.003, version=None):
    rng = np.random.default_rng(613 + feature_set)
    hidden, mixtures = 4, 2
    planes = []
    for channel in (1, 0, 2):
        state = {}
        shapes = reference.tensor_shapes(channel, hidden, mixtures, True, feature_set)
        for name, shape in shapes.items():
            values = rng.normal(0, scale, shape)
            if grid and name.endswith(".weight"):
                values = np.rint(values * 2**12) / 2**12
            state[name] = values.astype("<f2").tobytes()
        planes.append(state)
    header = (w, h, hidden, mixtures, True, feature_set)
    return reference.encode_model(header, planes, version or reference.MODEL_VERSION)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (5, 7)])
@pytest.mark.parametrize("feature_set", [0, 1, 2, 3, 4])
def test_v3_roundtrip_and_reencoding(coder, shape, feature_set):
    h, w = shape
    pixels = np.random.default_rng(19).integers(0, 256, (h, w, 3), np.uint8)
    payload = v3_model(w, h, feature_set, grid=True)
    blob = coder.encode_v3(pixels, payload)
    assert blob.startswith(coder.MAGIC_V3)
    assert np.array_equal(coder.decode(blob), pixels)
    assert coder.reencode(blob, pixels) == blob
    assert coder.model_payload(blob) == payload


def test_v3_transcoding_keeps_the_exact_v2_stream(coder):
    pixels = np.random.default_rng(23).integers(0, 256, (5, 7, 3), np.uint8)
    metadata = compress(encode_cbor(model_body()))
    v2 = coder.encode(pixels, metadata)
    payload = coder.transcode_v3(metadata)
    v3 = coder.encode_v3(pixels, payload)
    assert v2[coder.HEADER_SIZE + len(metadata) :] == v3[coder.HEADER_SIZE + len(payload) :]
    assert coder.parse_v3(payload) == coder.parse_v2(metadata)


def test_v3_grid_tensors_use_rice_coding_exactly():
    payload = v3_model(grid=True)
    header, planes = reference.parse_v3(payload)
    raw = reference.encode_model(header, planes)
    assert raw == payload
    plain = v3_model(grid=False)
    assert len(payload) < len(plain)


def _mixed_scale_weights(shape=(6, 9), seed=29):
    # Column j sits on the grid 2**-(j + 2); one tensor-wide grid would be fine everywhere.
    rng = np.random.default_rng(seed)
    return rng.integers(-60, 61, shape) * np.exp2(-np.arange(shape[1]) - 2.0)


@pytest.mark.parametrize("transpose", [False, True])
def test_v3_sliced_grids_code_mixed_scale_weights_exactly(transpose):
    matrix = _mixed_scale_weights()
    matrix = matrix.T.copy() if transpose else matrix
    raw = matrix.astype("<f2").tobytes()
    coded = reference._encode_tensor(raw, matrix.shape)
    assert coded[0] == (reference.ROWS if transpose else reference.COLUMNS)
    assert len(coded) < len(reference._encode_tensor(raw))
    decoded, position = reference._decode_tensor(coded, 0, matrix.size, matrix.shape)
    assert position == len(coded)
    assert np.array_equal(np.frombuffer(decoded, "<f2"), matrix.ravel())


def test_v3_sliced_grids_need_a_matrix_and_valid_shifts():
    matrix = _mixed_scale_weights()
    coded = bytearray(reference._encode_tensor(matrix.astype("<f2").tobytes(), matrix.shape))
    with pytest.raises(ValueError, match="invalid tensor coding"):
        reference._decode_tensor(bytes(coded), 0, matrix.size)
    _length, body = reference._read_varint(coded, 2)
    # The first slice's shift becomes 25: above MAX_SHIFT, below ZERO_SLICE.
    coded[body] = (coded[body] & 0x07) | (25 << 3)
    with pytest.raises(ValueError, match="invalid tensor grid"):
        reference._decode_tensor(bytes(coded), 0, matrix.size, matrix.shape)


def _pruned_weights(shape=(6, 9), columns=(0, 3, 4)):
    matrix = _mixed_scale_weights(shape)
    matrix[:, list(columns)] = 0.0
    return matrix


def test_v3_zero_slices_cost_one_reserved_shift():
    matrix = _pruned_weights()
    raw = matrix.astype("<f2").tobytes()
    coded = reference._encode_tensor(raw, matrix.shape)
    old = reference._encode_tensor(raw, matrix.shape, escape=False)
    assert coded[0] == old[0] == reference.COLUMNS
    assert len(coded) < len(old)
    decoded, position = reference._decode_tensor(coded, 0, matrix.size, matrix.shape)
    assert position == len(coded)
    assert np.array_equal(np.frombuffer(decoded, "<f2"), matrix.ravel())
    # Version 1 reserves no shift: its zero slices are ordinary slices.
    with pytest.raises(ValueError, match="invalid tensor grid"):
        reference._decode_tensor(coded, 0, matrix.size, matrix.shape, escape=False)
    decoded, _ = reference._decode_tensor(old, 0, matrix.size, matrix.shape, escape=False)
    assert np.array_equal(np.frombuffer(decoded, "<f2"), matrix.ravel())


def test_v3_model_versions_differ_only_in_zero_slices():
    header, planes = reference.parse_v3(v3_model(grid=True))
    shape = reference.tensor_shapes(1, 4, 2, True, 0)["hidden.0.weight"]
    planes[0]["hidden.0.weight"] = _pruned_weights(shape).astype("<f2").tobytes()
    current = reference.encode_model(header, planes)
    first = reference.encode_model(header, planes, 1)
    assert current[0] == 2 and first[0] == 1 and len(current) < len(first)
    assert reference.parse_v3(current) == reference.parse_v3(first) == (header, planes)
    # A version 2 payload must use the escape for every zero slice.
    with pytest.raises(ValueError, match="noncanonical"):
        reference.parse_v3(bytes([2]) + first[1:])
    plain = v3_model(grid=True)
    assert reference.encode_model(*reference.parse_v3(plain), 1)[1:] == plain[1:]


def test_v3_models_with_sliced_grids_roundtrip(coder):
    header, planes = reference.parse_v3(v3_model(grid=True))
    shape = reference.tensor_shapes(1, 4, 2, True, 0)["hidden.0.weight"]
    planes[0]["hidden.0.weight"] = _mixed_scale_weights(shape).astype("<f2").tobytes()
    payload = reference.encode_model(header, planes)
    assert reference.parse_v3(payload) == (header, planes)
    pixels = np.random.default_rng(31).integers(0, 256, (5, 7, 3), np.uint8)
    blob = coder.encode_v3(pixels, payload)
    assert np.array_equal(coder.decode(blob), pixels)
    assert coder.reencode(blob, pixels) == blob


@pytest.mark.parametrize("kind", ["version", "trailing", "truncated", "feature"])
def test_v3_payload_must_be_canonical(kind):
    payload = bytearray(v3_model(grid=True))
    if kind == "version":
        payload[0] = 3
    elif kind == "trailing":
        payload += b"\0"
    elif kind == "truncated":
        del payload[-3:]
    else:
        assert payload[1:4] == bytes([7, 5, 4])
        payload[5] = 1 | (5 << 1)
    with pytest.raises(ValueError):
        reference.parse_v3(bytes(payload))


def test_v3_rice_padding_bits_must_be_zero():
    planes = [
        {
            name: bytes(2 * int(np.prod(shape)))
            for name, shape in reference.tensor_shapes(channel, 1, 1, True, 0).items()
        }
        for channel in (1, 0, 2)
    ]
    payload = bytearray(reference.encode_model((1, 1, 1, 1, True, 0), planes))
    # Fourteen zero guide coefficients: one-bit Rice codes, two padding bits.
    assert payload[6:12] == bytes([reference.RICE, 0, 0, 2, 0xFF, 0xFC])
    reference.parse_v3(bytes(payload))
    payload[11] |= 1
    with pytest.raises(ValueError, match="noncanonical"):
        reference.parse_v3(bytes(payload))


def test_v3_feature_set_needs_context():
    with pytest.raises(ValueError, match="feature set"):
        reference.check_header(7, 5, 4, 2, False, 1)


@pytest.mark.parametrize("feature_set", [0, 5])
def test_native_backend_rejects_unknown_feature_sets(coder, tmp_path, feature_set):
    header, planes = reference.parse_v3(v3_model(feature_set=1))
    config, _ = reference.native_configuration(header, planes)
    config = bytearray(config)
    struct.pack_into("<I", config, 24, feature_set)
    paths = [tmp_path / name for name in ("config", "pixels", "out")]
    paths[0].write_bytes(bytes(config))
    paths[1].write_bytes(bytes(105))
    with pytest.raises(RuntimeError, match="feature set"):
        reference.call("encode", *paths, integer=True)


@pytest.mark.parametrize("bits", [6, 7, 8, 16])
def test_v3_quantized_model_decodes_exactly(coder, quantizer, bits):
    payload = v3_model(feature_set=1)
    quantized = quantizer.quantize_v3(payload, bits)
    assert (quantized == payload) == (bits == 16)
    pixels = np.random.default_rng(53).integers(0, 256, (5, 7, 3), np.uint8)
    assert np.array_equal(coder.decode(coder.encode_v3(pixels, quantized)), pixels)


@pytest.mark.parametrize("bits", [5, 9, 24])
def test_v3_unsupported_precision_rejected(quantizer, bits):
    with pytest.raises(ValueError, match="precisions"):
        quantizer.quantize_v3(v3_model(), bits)


# Hashes of complete v3 files written by the single-threaded backend before the
# encoder evaluated distributions in parallel. They pin the v3 bitstream.
V3_GOLDEN = {
    (0, "smooth"): "6bfc70b8914f0b597abcb4e93a6e7cf64c7d27b92b17a20e9d6109dfe13f0ff7",
    (0, "noisy"): "7473ca09dfc27b51bd0518eb95049a295448e8d984e9c8c7d521d143b5b94a3a",
    (1, "smooth"): "ca0c4c581d47fa3788d84e73d0f2d799da862c71ddfe9f5da3a6abf119c45d33",
    (1, "noisy"): "c18b8f5c993d246e7be01886a374396183e288d7b48ca1a27380b2a85b1d8c31",
    # Recorded when feature set 2 was added.
    (2, "smooth"): "5190faaa994461101841d03d55cfd1606f9be4806a8d5a2d840b0d872a2838dd",
    (2, "noisy"): "9e64ad42aea379f0d2eb853e1704ffd205e241f0d4f50ae4c119c9cda9b8bb2e",
    # Recorded when feature set 3 was added; -O0 and -O2 builds agreed.
    (3, "smooth"): "7dad6d38bf58e3938fd4d12302dbce14386b4d71ed0aeec8db9391a80ecd6fe3",
    (3, "noisy"): "4d670057bb0eecda339ead3e0a6de648ce7eefbc1ee844fab2a6057f7bd1f13c",
    # Recorded when feature set 4 was added; -O0 and -O2 builds agreed.
    (4, "smooth"): "e4155958d6e2c7eb78e2f46f8db04eb9403ead7748fa7148a66147114c19810f",
    (4, "noisy"): "cee9be152db346d67f61f1a6eeff75231252af5857bff9c8a2949be8fc5683ba",
}


def golden_pixels(kind, h=19, w=23):
    if kind == "smooth":
        ramp = np.add.outer(np.arange(h), np.arange(w))[..., None] * np.array([3, 5, 7])
        return (ramp % 256).astype(np.uint8)
    return np.random.default_rng(97).integers(0, 256, (h, w, 3), np.uint8)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize(("feature_set", "kind"), sorted(V3_GOLDEN))
def test_v3_stream_matches_the_single_threaded_backend(
    coder, monkeypatch, threads, feature_set, kind
):
    import hashlib

    monkeypatch.setattr(coder, "THREADS", threads)
    payload = v3_model(23, 19, feature_set, grid=True, scale=0.3, version=1)
    pixels = golden_pixels(kind)
    blob = coder.encode_v3(pixels, payload)
    assert hashlib.sha256(blob).hexdigest() == V3_GOLDEN[feature_set, kind]
    assert np.array_equal(coder.decode(blob), pixels)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (5, 7), (19, 23)])
@pytest.mark.parametrize("streams", [1, 3, 16])
def test_v4_roundtrip_and_reencoding(coder, monkeypatch, shape, streams):
    h, w = shape
    pixels = np.random.default_rng(29).integers(0, 256, (h, w, 3), np.uint8)
    payload = v3_model(w, h, 1, grid=True, scale=0.3)
    blob = coder.encode_v4(pixels, payload, streams=streams)
    assert blob.startswith(coder.MAGIC_V4)
    assert blob[coder.HEADER_SIZE + len(payload)] == min(streams, h)
    for threads in (1, 2, 8):
        monkeypatch.setattr(coder, "THREADS", threads)
        assert np.array_equal(coder.decode(blob), pixels)
        assert coder.reencode(blob, pixels) == blob
    assert coder.model_payload(blob) == payload


def test_v4_keeps_v3_probabilities(coder):
    pixels = golden_pixels("smooth")
    payload = v3_model(23, 19, 1, grid=True, scale=0.3)
    v3 = coder.encode_v3(pixels, payload)[coder.HEADER_SIZE + len(payload) :]
    v4 = coder.encode_v4(pixels, payload, streams=1)[coder.HEADER_SIZE + len(payload) :]
    # One row stream is the v3 stream without its four zero padding bytes.
    assert v3.endswith(bytes(4))
    assert v4 == bytes([1]) + reference._varint(len(v3) - 4) + v3[:-4]


def _v4_stream(coder, blob, payload):
    return blob[coder.HEADER_SIZE + len(payload) :]


def _with_stream(coder, blob, payload, stream):
    head = coder.MAGIC_V4 + struct.pack("<2I", len(payload), len(stream))
    return head + blob[coder.FLOAT_HEADER_SIZE : coder.HEADER_SIZE] + payload + stream


@pytest.mark.parametrize("kind", ["zero", "too-many", "short", "long"])
def test_v4_stream_table_is_checked(coder, kind):
    pixels = golden_pixels("noisy")
    payload = v3_model(23, 19, 1, grid=True, scale=0.3)
    blob = coder.encode_v4(pixels, payload, streams=3)
    stream = bytearray(_v4_stream(coder, blob, payload))
    if kind == "zero":
        stream[0] = 0
    elif kind == "too-many":
        stream[0] = 20
    elif kind == "short":
        stream += b"\0"
    else:
        del stream[-1]
    with pytest.raises(RuntimeError, match="stream"):
        coder.decode(_with_stream(coder, blob, payload, bytes(stream)))


def test_v4_threaded_decode_fails_cleanly_on_a_truncated_row_stream(coder, monkeypatch):
    pixels = golden_pixels("noisy")
    payload = v3_model(23, 19, 1, grid=True, scale=0.3)
    blob = coder.encode_v4(pixels, payload, streams=3)
    stream = _v4_stream(coder, blob, payload)
    position, sizes = 1, []
    for _ in range(3):
        size, position = reference._read_varint(stream, position)
        sizes.append(size)
    # Drop row stream 0 entirely. Its rows run out of bits while other threads
    # wait on them; the error must surface instead of a hang.
    table = bytes([3, 0]) + reference._varint(sizes[1]) + reference._varint(sizes[2])
    broken = _with_stream(coder, blob, payload, table + stream[position + sizes[0] :])
    monkeypatch.setattr(coder, "THREADS", 3)
    with pytest.raises(RuntimeError, match="truncated arithmetic"):
        coder.decode(broken)


@pytest.mark.parametrize("extra", [["x"], ["-1"], ["65"], ["0", "4"]])
def test_native_backend_rejects_bad_counts(coder, tmp_path, extra):
    header, planes = reference.parse_v3(v3_model(feature_set=1))
    config, _ = reference.native_configuration(header, planes)
    paths = [tmp_path / name for name in ("config", "pixels", "out")]
    paths[0].write_bytes(config)
    paths[1].write_bytes(bytes(105))
    result = subprocess.run(
        [str(coder.INTEGER_BINARY), "encode", *map(str, paths), *extra],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert b"count" in result.stderr


def local_fit_pixels(h, w):
    """A smooth colour ramp with noise, so window fits are well conditioned."""
    rng = np.random.default_rng(71)
    ramp = np.add.outer(np.arange(h) * 3, np.arange(w) * 2)[..., None] + np.array([0, 40, 90])
    return np.clip(ramp + rng.integers(-9, 10, (h, w, 3)), 0, 255).astype(np.uint8)


CAUSAL = [(0, -1), (-1, 0), (-1, -1), (-1, 1), (0, -2), (-2, 0)]
CAUSAL += [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (-2, -2), (-2, 2)]
NEIGHBOURS = [(0, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (0, 1), (1, 0), (1, -1), (1, 1)]
CROSS = [(0, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]
# Windows of feature sets 3 and 4 as (rows above, columns either side, decoded-plane
# taps, ridge scale), as in nnproto_integer.cpp.
FIT_WINDOWS = {
    3: [(12, 12, NEIGHBOURS, 2**-20)],
    4: [(12, 12, CROSS, 2**-20), (4, 4, CROSS, 2**-13)],
}


def reference_local_fits(pixels, window):
    """Least-squares fits by a direct float64 ridge solve of every causal window."""
    above, side, taps, scale = window
    h, w = pixels.shape[:2]
    out = np.zeros((2, 3, h, w))
    for g, channel in enumerate((1, 0, 2)):
        rows = []
        for y in range(h):
            for x in range(w):
                own = [
                    pixels[y + dy, x + dx, channel] if y + dy >= 0 and 0 <= x + dx < w else 0
                    for dy, dx in CAUSAL
                ]
                prior = [
                    pixels[min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1), plane]
                    for plane in {1: (), 0: (1,), 2: (1, 0)}[channel]
                    for dy, dx in taps
                ]
                rows.append(own + prior)
        inputs = np.array(rows, float).reshape(h, w, -1)
        target = pixels[..., channel].astype(float)
        n = inputs.shape[2]
        for y in range(h):
            for x in range(w):
                cells = [
                    (r, c)
                    for r in range(max(0, y - above), y)
                    for c in range(max(0, x - side), min(w, x + side + 1))
                ]
                cells += [(y, c) for c in range(max(0, x - side), x)]
                index = tuple(np.array(cells, int).reshape(-1, 2).T)
                f, t = inputs[index], target[index]
                gram = f.T @ f
                ridge = max(1.0, np.trace(gram) / n) * scale
                beta = np.linalg.solve(gram + ridge * np.eye(n), f.T @ t)
                residual = max(0.0, (t @ t - beta @ (f.T @ t)) / max(1, len(t)))
                out[0, g, y, x] = np.clip(np.floor(8 * beta @ inputs[y, x] + 0.5), 0, 2040)
                out[1, g, y, x] = min(4095, np.floor(16 * np.sqrt(residual) + 0.5))
    return out


@pytest.mark.parametrize("feature_set", [3, 4])
def test_local_fits_match_a_direct_least_squares_solve(coder, monkeypatch, feature_set):
    pixels = local_fit_pixels(21, 30)
    windows = FIT_WINDOWS[feature_set]
    want = np.concatenate([reference_local_fits(pixels, window) for window in windows])
    for threads in (1, 3):
        monkeypatch.setattr(coder, "THREADS", threads)
        fits = coder.local_fits(pixels, feature_set)
        assert fits.shape == (2 * len(windows), 3, 21, 30)
        # Rounding near half an eighth may differ from an unrelated float order.
        assert np.abs(fits - want).max() <= 1
        assert np.mean(fits == want) > 0.99


@pytest.mark.parametrize("feature_set", [3, 4])
@pytest.mark.parametrize("streams", [1, 3, 16])
def test_least_squares_wavefront_decoding(coder, monkeypatch, streams, feature_set):
    pixels = local_fit_pixels(37, 41)
    payload = v3_model(41, 37, feature_set, grid=True, scale=0.3)
    blob = coder.encode_v4(pixels, payload, streams=streams)
    for threads in (1, 2, 8):
        monkeypatch.setattr(coder, "THREADS", threads)
        assert np.array_equal(coder.decode(blob), pixels)
        assert coder.reencode(blob, pixels) == blob


@pytest.mark.parametrize("feature_set", [3, 4])
def test_v4_encode_reuses_precomputed_fits(coder, tmp_path, feature_set):
    pixels = local_fit_pixels(37, 41)
    payload = v3_model(41, 37, feature_set, grid=True, scale=0.3)
    fits = tmp_path / "fits"
    coder.local_fits(pixels, feature_set).tofile(fits)
    assert coder.encode_v4(pixels, payload, fits=fits) == coder.encode_v4(pixels, payload)
    fits.write_bytes(fits.read_bytes()[:-2])
    with pytest.raises(RuntimeError, match="local fit size"):
        coder.encode_v4(pixels, payload, fits=fits)


def test_v4_fits_need_a_least_squares_feature_set(coder, tmp_path):
    pixels = local_fit_pixels(37, 41)
    fits = tmp_path / "fits"
    coder.local_fits(pixels, 3).tofile(fits)
    with pytest.raises(RuntimeError, match="no least-squares fits"):
        coder.encode_v4(pixels, v3_model(41, 37, 2, grid=True, scale=0.3), fits=fits)


def test_v4_stream_count_is_checked_before_encoding(coder):
    pixels = np.zeros((5, 7, 3), np.uint8)
    for streams in (0, 65, True):
        with pytest.raises(ValueError, match="stream count"):
            coder.encode_v4(pixels, v3_model(), streams=streams)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (5, 7), (19, 23), (9, 40)])
@pytest.mark.parametrize("layout", ["s1", "s3", "b1", "b3", "g2"])
def test_v5_roundtrip_and_reencoding(coder, monkeypatch, shape, layout):
    h, w = shape
    pixels = np.random.default_rng(31).integers(0, 256, (h, w, 3), np.uint8)
    payload = v3_model(w, h, 1, grid=True, scale=0.3)
    blob = coder.encode_v5(pixels, payload, layout=layout)
    assert blob.startswith(coder.MAGIC_V5)
    # Row streams are capped by the height, bands by their 13-column minimum width.
    limit = h if layout[0] == "s" else max(1, w // 13)
    assert coder.v5_layout(blob) == f"{layout[0]}{min(int(layout[1:]), limit)}"
    for threads in (1, 2, 8):
        monkeypatch.setattr(coder, "THREADS", threads)
        assert np.array_equal(coder.decode(blob), pixels)
        assert coder.reencode(blob, pixels) == blob
    assert coder.model_payload(blob) == payload


@pytest.mark.parametrize("feature_set", [3, 4])
@pytest.mark.parametrize("layout", ["s3", "b2", "g3"])
def test_v5_least_squares_wavefront_and_precomputed_fits(
    coder, monkeypatch, tmp_path, feature_set, layout
):
    pixels = local_fit_pixels(37, 41)
    payload = v3_model(41, 37, feature_set, grid=True, scale=0.3)
    blob = coder.encode_v5(pixels, payload, layout=layout)
    fits = tmp_path / "fits"
    coder.local_fits(pixels, feature_set).tofile(fits)
    assert coder.encode_v5(pixels, payload, layout=layout, fits=fits) == blob
    assert coder.encode_v5(pixels, payload, layout=layout, fits=fits, checked=True) == blob
    for threads in (1, 2, 8):
        monkeypatch.setattr(coder, "THREADS", threads)
        assert np.array_equal(coder.decode(blob), pixels)


def test_v5_mixing_codes_structure_the_network_misses(coder):
    # Every fifth value only, and B a function of G: a weak network leaves both.
    rng = np.random.default_rng(43)
    g = 5 * rng.integers(0, 40, (64, 64))
    pixels = np.stack([5 * rng.integers(0, 40, (64, 64)), g, 195 - g], axis=2).astype(np.uint8)
    payload = v3_model(64, 64, 1, grid=True, scale=0.3)
    alone = coder.encode_v5(pixels, payload, layout="s1")
    for layout in ("b1", "g1"):
        mixed = coder.encode_v5(pixels, payload, layout=layout)
        assert len(mixed) < 0.9 * len(alone)
        assert np.array_equal(coder.decode(mixed), pixels)


def test_constant_planes_keep_only_the_head_bias(coder):
    rng = np.random.default_rng(59)
    g = rng.integers(0, 256, (24, 24))
    pixels = np.stack([g, g, (g + 7) % 256], axis=2).astype(np.uint8)
    payload = v3_model(24, 24, 1, grid=True, scale=0.3)
    cut = coder.constant_planes(payload, (1, 2))
    assert len(cut) < len(payload)
    header, planes = coder.parse_v3(payload)
    cut_header, cut_planes = coder.parse_v3(cut)
    assert cut_header == header and cut_planes[0] == planes[0]
    for plane in (1, 2):
        for name, raw in cut_planes[plane].items():
            assert raw == (bytes(len(raw)) if name in coder.CONSTANT_ZEROS else planes[plane][name])
    for layout in ("s1", "g1"):
        blob = coder.encode_v5(pixels, cut, layout=layout)
        assert np.array_equal(coder.decode(blob), pixels)
    # fold moves the network's output for zero inputs into the head bias.
    folded = coder.parse_v3(coder.constant_planes(payload, (2,), fold=True))[1]
    assert folded[:2] == planes[:2]
    shapes = coder.tensor_shapes(2, *header[2:])

    def value(name):
        return np.frombuffer(planes[2][name], "<f2").astype(np.float32).reshape(shapes[name])

    first = np.maximum(value("hidden.0.bias"), 0)
    second = np.maximum(value("hidden.2.weight") @ first + value("hidden.2.bias"), 0)
    output = value("head.weight") @ second + value("head.bias")
    bias = np.frombuffer(folded[2]["head.bias"], "<f2")
    assert np.allclose(bias, output, rtol=1e-2, atol=1e-2)


def test_v5_value_map(coder, monkeypatch):
    rng = np.random.default_rng(47)
    pixels = (5 * rng.integers(0, 52, (19, 23, 3))).astype(np.uint8)
    pixels[0, 0] = (0, 255, 5)
    pixels[0, 1] = (255, 0, 0)
    pixels[0, 2] = (5, 5, 255)
    values = coder.shared_values(pixels)
    assert np.array_equal(values, np.arange(0, 256, 5))
    packed = coder.pack_values(pixels, values)
    assert packed.max() == 51 and np.array_equal(values[packed], pixels)
    payload = v3_model(23, 19, 1, grid=True, scale=0.3)
    for layout in ("s3", "g2"):
        blob = coder.encode_v5(pixels, payload, layout=layout, values=values)
        stream = blob[coder.HEADER_SIZE + len(payload) :]
        assert stream[0] == 1
        for threads in (1, 4):
            monkeypatch.setattr(coder, "THREADS", threads)
            assert np.array_equal(coder.decode(blob), pixels)
        assert coder.reencode(blob, pixels) == blob
        # A map with fewer values than the ranks coded is rejected.
        short = bytearray(blob)
        start = coder.HEADER_SIZE + len(payload) + 1
        short[start : start + 32] = bytes([1]) + bytes(31)
        with pytest.raises(ValueError, match="value map"):
            coder.decode(bytes(short))


@pytest.mark.parametrize("layout", ["s3", "b2", "g3"])
def test_v5_checked_encodes_are_the_same_file_and_verified(coder, layout):
    pixels = (5 * np.random.default_rng(71).integers(0, 52, (19, 40, 3))).astype(np.uint8)
    payload = v3_model(40, 19, 1, grid=True, scale=0.3)
    for values in (None, coder.shared_values(pixels)):
        blob = coder.encode_v5(pixels, payload, layout=layout, values=values)
        assert not coder.verified(blob, pixels)
        assert coder.encode_v5(pixels, payload, layout=layout, values=values, checked=True) == blob
        assert coder.verified(blob, pixels)
    wrong = pixels.copy()
    wrong[3, 4, 1] ^= 1
    assert not coder.verified(blob, wrong)
    assert not coder.verified(blob, np.ascontiguousarray(pixels[:, :-1]))
    assert not coder.verified(blob[:-1] + bytes([blob[-1] ^ 1]), pixels)


def test_v5_bits_of_unused_samples_are_not_coded(coder, monkeypatch):
    # Each plane uses its own comb, so no shared map packs them.
    rng = np.random.default_rng(53)
    pixels = np.stack([k * rng.integers(0, 256 // k, (29, 31)) for k in (3, 2, 5)], axis=2).astype(
        np.uint8
    )
    assert coder.shared_values(pixels) is None
    payload = v3_model(31, 29, 1, grid=True, scale=0.3)
    flags = coder.HEADER_SIZE + len(payload) + 1
    for layout in ("b1", "g2"):
        blob = coder.encode_v5(pixels, payload, layout=layout)
        assert blob[flags] & coder.V5_VALUES and blob[flags + 2] == 7
        assert coder.v5_layout(blob) == layout
        for threads in (1, 4):
            monkeypatch.setattr(coder, "THREADS", threads)
            assert np.array_equal(coder.decode(blob), pixels)
        for mask in (0, 8):
            broken = bytearray(blob)
            broken[flags + 2] = mask
            with pytest.raises(RuntimeError, match="value map"):
                coder.decode(bytes(broken))
    # A plane that uses nearly every sample carries no map.
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    full = np.stack([ramp, ramp, 2 * (ramp // 2)], axis=2)
    payload = v3_model(16, 16, 1, grid=True, scale=0.3)
    blob = coder.encode_v5(full, payload, layout="g1")
    flags = coder.HEADER_SIZE + len(payload) + 1
    assert blob[flags] & coder.V5_VALUES and blob[flags + 2] == 4
    assert np.array_equal(coder.decode(blob), full)


def test_v5_table_bytes_follow_the_band_models(coder):
    pixels = np.random.default_rng(67).integers(0, 256, (7, 32, 3), np.uint8)
    payload = v3_model(32, 7, 1, grid=True, scale=0.3)
    assert coder.v5_table_bytes(coder.encode_v5(pixels, payload, layout="s2")) == 0
    blob = coder.encode_v5(pixels, payload, layout="g2")
    assert coder.v5_table_bytes(blob) == 2 * 3 * 128 << coder.V5_TABLE_BITS[0]
    assert coder.layout_table_bytes(32, 7, "g2") == coder.v5_table_bytes(blob)
    assert coder.layout_table_bytes(32, 7, "s2") == 0
    assert coder.layout_table_bytes(4096, 4096, "b1") == 3 * 128 << coder.V5_TABLE_BITS[1]
    with pytest.raises(ValueError):
        coder.layout_table_bytes(32, 7, "x1")


def test_shared_values_need_planes_to_share_a_gapped_set(coder):
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert coder.shared_values(np.stack([ramp] * 3, axis=2)) is None
    sparse = (ramp // 2 * 2).astype(np.uint8)
    assert coder.shared_values(np.stack([sparse, sparse, ramp], axis=2)) is None
    assert len(coder.shared_values(np.stack([sparse] * 3, axis=2))) == 128
    # The union is shared while every plane uses SHARED_SHARE of it.
    missing = np.where(sparse == 8, 10, sparse).astype(np.uint8)
    assert len(coder.shared_values(np.stack([sparse, missing, sparse], axis=2))) == 128
    quarter = (ramp // 4 * 4).astype(np.uint8)
    assert coder.shared_values(np.stack([sparse, quarter, sparse], axis=2)) is None
    with pytest.raises(ValueError, match="value map"):
        coder.pack_values(np.stack([ramp] * 3, axis=2), np.arange(0, 256, 2, dtype=np.uint8))


@pytest.mark.parametrize("layout", ["", "x1", "s0", "s65", "b17", "g01", "g", 3, None])
def test_v5_layout_is_checked_before_encoding(coder, layout):
    with pytest.raises(ValueError, match="v5 layout"):
        coder.encode_v5(np.zeros((5, 7, 3), np.uint8), v3_model(), layout=layout)


@pytest.mark.parametrize("flags", [2, 4, 255])
def test_v5_flags_are_checked(coder, flags):
    pixels = golden_pixels("noisy")
    payload = v3_model(23, 19, 1, grid=True, scale=0.3)
    blob = bytearray(coder.encode_v5(pixels, payload, layout="g2"))
    blob[coder.HEADER_SIZE + len(payload) + 1] = flags
    with pytest.raises(RuntimeError, match="v5 flags"):
        coder.decode(bytes(blob))


def test_v5_band_table_is_checked(coder):
    pixels = golden_pixels("noisy")
    payload = v3_model(23, 19, 1, grid=True, scale=0.3)
    blob = coder.encode_v5(pixels, payload, layout="b2")
    position = coder.HEADER_SIZE + len(payload) + 2
    # 23 columns hold one band of the minimum width.
    assert blob[position] == 1
    for bands in (0, 2, 17):
        broken = bytearray(blob)
        broken[position] = bands
        with pytest.raises(RuntimeError, match="band count"):
            coder.decode(bytes(broken))
    stream = blob[coder.HEADER_SIZE + len(payload) :] + b"\0"
    head = coder.MAGIC_V5 + struct.pack("<2I", len(payload), len(stream))
    longer = head + blob[coder.FLOAT_HEADER_SIZE : coder.HEADER_SIZE] + payload + stream
    with pytest.raises(RuntimeError, match="stream lengths"):
        coder.decode(longer)
