from __future__ import annotations

import copy
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
import zstandard
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from semanticcodec.core.color import Color, Palette
from semanticcodec.core.geometry import Rect
from semanticcodec.core.scene.document import AssetRef, Canvas, Document
from semanticcodec.core.scene.nodes import Group, RasterPatch
from semanticcodec.decode.pipeline import decode_bytes, decode_file
from semanticcodec.encode.pipeline import EncodeOptions, build_floor_document, encode_image
from semanticcodec.encode.symbols import available, build_symbol_asset
from semanticcodec.fallback.codecs import (
    decode_asset,
    encode_png,
    encode_webp_lossless,
)
from semanticcodec.format.container import (
    VERSION_MAJOR,
    ContainerError,
    compress,
    decode_cbor,
    decompress,
    encode_cbor,
)
from semanticcodec.format.serialize import deserialize, serialize
from semanticcodec.format.symbols import (
    MAGIC,
    MAX_PIXELS,
    SYMBOL_CODEC,
    _get_uint,
    _put_uint,
    atlas_layout,
    decode_symbols,
    pack_symbols,
)
from semanticcodec.render.raster import render


@pytest.fixture
def marks() -> np.ndarray:
    """Several colors and backgrounds, disconnected marks and overlapping boxes."""
    image = np.full((144, 256, 4), 255, np.uint8)
    image[:, 128:, :3] = (24, 28, 35)
    rng = np.random.default_rng(17)
    for y in range(4, 132, 16):
        for x in range(4, 244, 12):
            fg = (35, 40, 45) if x < 128 else (180, 200, 230)
            image[y : y + 8, x : x + 2, :3] = fg
            image[y + 6 : y + 8, x : x + 7, :3] = fg
            if x % 36 == 4:
                image[y + 1, x + 2 : x + 5, :3] = rng.integers(50, 180, (3, 3))
    # A ring encloses a separate mark. Copying full bounding boxes naively
    # would count that inner mark twice.
    image[130:142, 4:16, :3] = 0
    image[132:140, 6:14, :3] = 255
    image[134:137, 8:11, :3] = (10, 20, 30)
    return image


@pytest.mark.skipif(not available(), reason="symbol analysis needs the cv extra")
@pytest.mark.parametrize("min_uses", [1, 2, 3])
def test_exact_symbols(marks: np.ndarray, min_uses: int) -> None:
    candidate = build_symbol_asset(marks, colors=8, min_uses=min_uses)
    assert candidate is not None
    asset, stats = candidate
    assert stats.symbols > 0 and stats.instances > 0
    assert np.array_equal(decode_asset(asset), marks)


def test_complete_file_floor_and_determinism(marks: np.ndarray) -> None:
    blob, doc, report = encode_image(marks)
    floor = serialize(build_floor_document(marks, EncodeOptions()))
    assert report.output_bytes == len(blob) <= len(floor)
    assert len(blob) == min(report.candidate_bytes.values())
    assert np.array_equal(decode_bytes(blob)[0], marks)
    assert encode_image(marks)[0] == blob
    assert serialize(deserialize(blob)) == blob
    assert doc.assets[0].codec == report.selected_codec


@pytest.mark.parametrize("profile", ["fast", "balanced", "exhaustive"])
def test_profiles(profile: str, marks: np.ndarray) -> None:
    blob, _, report = encode_image(marks, EncodeOptions(profile=profile))
    assert report.output_bytes <= report.floor_bytes
    assert np.array_equal(decode_bytes(blob)[0], marks)


def test_no_cv_is_a_supported_install(marks: np.ndarray) -> None:
    with patch("semanticcodec.encode.symbols.available", return_value=False):
        blob, _, report = encode_image(marks)
    assert report.floor_used
    assert list(report.candidate_bytes) == ["floor"]
    assert np.array_equal(decode_bytes(blob)[0], marks)


def test_no_cwebp_uses_native_webp(marks: np.ndarray) -> None:
    with patch("shutil.which", return_value=None):
        data = encode_webp_lossless(marks)
    assert np.array_equal(decode_asset(AssetRef("webp-lossless", 256, 144, data)), marks)


def test_no_webp_still_has_png_floor(marks: np.ndarray) -> None:
    from semanticcodec.fallback.codecs import best_lossless

    save = Image.Image.save

    def without_webp(self, fp, format=None, **kwargs):
        if format == "WEBP":
            raise OSError("WebP was not compiled")
        return save(self, fp, format=format, **kwargs)

    with patch("shutil.which", return_value=None), patch.object(Image.Image, "save", without_webp):
        codec, data = best_lossless(marks)
    assert codec == "png"
    assert np.array_equal(decode_asset(AssetRef(codec, 256, 144, data)), marks)


def test_rgba_file_roundtrip(tmp_path) -> None:
    image = np.random.default_rng(31).integers(0, 256, (19, 23, 4), np.uint8)
    image[0, 0] = (123, 45, 67, 0)
    for native in (False, True):
        with patch("shutil.which", return_value=None) if native else nullcontext():
            blob, _, report = encode_image(image)
        assert report.floor_used
        assert np.array_equal(decode_bytes(blob)[0], image)
        encoded, decoded = tmp_path / "a.scdx", tmp_path / "a.png"
        encoded.write_bytes(blob)
        decode_file(encoded, decoded)
        with Image.open(decoded) as im:
            assert np.array_equal(np.asarray(im.convert("RGBA")), image)


def test_replace_does_not_change_source_over() -> None:
    pixels = np.full((2, 3, 4), (200, 0, 0, 128), np.uint8)
    doc = Document(Canvas(3, 2), palette=Palette([Color(255, 255, 255)]))
    doc.assets = [AssetRef("png", 3, 2, encode_png(pixels))]
    patch_node = RasterPatch(rect=Rect.from_px(0, 0, 3, 2))
    doc.root = Group(rect=Rect.from_px(0, 0, 3, 2), children=[patch_node])
    ordinary = render(doc)[0]
    assert np.all(ordinary[..., 3] == 255)
    patch_node.replace = True
    assert np.array_equal(render(deserialize(serialize(doc)))[0], pixels)


def make_tiny_asset() -> AssetRef:
    base = AssetRef("png", 8, 8, encode_png(np.full((8, 8, 3), 255, np.uint8)))
    atlas = AssetRef("png", 512, 2, encode_png(np.full((2, 512, 3), 127, np.uint8)))
    return AssetRef(SYMBOL_CODEC, 8, 8, pack_symbols(base, atlas, [(2, 2)], [(0, 0)]))


@pytest.fixture
def tiny_asset() -> AssetRef:
    return make_tiny_asset()


def _mutate(asset: AssetRef, index: int, value: object) -> AssetRef:
    out = copy.copy(asset)
    body = decode_cbor(decompress(out.payload[len(MAGIC) :]))
    assert isinstance(body, list)
    body[index] = value
    out.payload = MAGIC + compress(encode_cbor(body))
    return out


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, SYMBOL_CODEC),
        (0, "webp-lossy"),
        (0, []),
        (1, "not bytes"),
        (2, SYMBOL_CODEC),
        (3, True),
        (3, 0),
        (3, 513),
        (4, 1),
        (4, MAX_PIXELS),
        (5, b"not an image"),
        (6, b""),
        (6, b"\x02"),
        (6, b"\x00\x02"),
        (6, b"\x41\x02"),
        (7, b""),
        (7, b"\x00"),
        (7, b"\x00\x01"),
        (7, b"\x3f\x00"),
        (7, b"\x80\x00\x00"),
        (7, b"\xff\xff\xff\xff\x7f\x00"),
    ],
)
def test_malformed_symbols_rejected(tiny_asset: AssetRef, index: int, value: object) -> None:
    with pytest.raises((ValueError, OSError)):
        decode_symbols(_mutate(tiny_asset, index, value))


def test_payload_guards(tiny_asset: AssetRef) -> None:
    assert decode_symbols(tiny_asset).shape == (8, 8, 4)
    bad = copy.copy(tiny_asset)
    bad.width = MAX_PIXELS + 1
    with pytest.raises(ValueError, match="pixel limit"):
        decode_symbols(bad)
    for payload in (
        b"",
        b"SYMA\x02",
        MAGIC + compress(encode_cbor({})),
        MAGIC + compress(encode_cbor([1] * 9)),
    ):
        bad = copy.copy(tiny_asset)
        bad.payload = payload
        with pytest.raises(ValueError):
            decode_symbols(bad)
    with pytest.raises(ValueError, match="opaque"):
        transparent = encode_png(np.full((8, 8, 4), 127, np.uint8))
        decode_symbols(_mutate(tiny_asset, 1, transparent))


def test_compressed_size_limit() -> None:
    for known_size in (False, True):
        data = zstandard.ZstdCompressor(write_content_size=known_size).compress(b"a" * 4096)
        with pytest.raises(ContainerError):
            decompress(data, max_output=64)
    with pytest.raises(ContainerError):
        decompress(compress(b"a") + compress(b"b"))


def test_container_version_and_legacy_floor() -> None:
    from semanticcodec.format.container import MAGIC as CONTAINER_MAGIC

    image = np.full((8, 8, 4), 255, np.uint8)
    blob = serialize(build_floor_document(image, EncodeOptions()))
    assert blob[len(CONTAINER_MAGIC)] == VERSION_MAJOR == 1
    # A legacy L0 container has the same chunks, with header version 0.1.
    legacy = bytearray(blob)
    legacy[len(CONTAINER_MAGIC) : len(CONTAINER_MAGIC) + 2] = b"\x00\x01"
    assert np.array_equal(decode_bytes(bytes(legacy))[0], image)


@given(st.integers(min_value=0, max_value=0xFFFFFFFF))
def test_unsigned_integer_roundtrip(value: int) -> None:
    buf = bytearray()
    _put_uint(buf, value)
    assert _get_uint(bytes(buf), 0) == (value, len(buf))


@given(data=st.binary(min_size=0, max_size=120))
@settings(max_examples=100)
def test_random_command_stream_is_rejected_or_bounded(data: bytes) -> None:
    try:
        image = decode_symbols(_mutate(make_tiny_asset(), 7, data))
    except ValueError:
        return
    assert image.shape == (8, 8, 4)


def test_atlas_budget() -> None:
    with pytest.raises(ValueError):
        atlas_layout([])
    with pytest.raises(ValueError):
        atlas_layout([(65, 2)])


def test_decoder_paint_budgets(tiny_asset: AssetRef) -> None:
    with (
        patch("semanticcodec.format.symbols.MAX_PAINT_PIXELS", 3),
        pytest.raises(ValueError, match="budget"),
    ):
        decode_symbols(tiny_asset)
    with (
        patch("semanticcodec.format.symbols.MAX_INSTANCES", 1),
        pytest.raises(ValueError, match="budget"),
    ):
        decode_symbols(_mutate(tiny_asset, 7, b"\x00\x00" * 2))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 4, 4), np.uint8),
        np.zeros((4, 4, 3), np.uint8),
        np.zeros((4, 4, 4), np.float32),
    ],
)
def test_input_validation(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encode_image(image)


def test_option_validation(marks: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encode_image(marks, EncodeOptions(profile="not-a-profile"))
    with pytest.raises(ValueError):
        encode_image(marks, EncodeOptions(quality=101))
