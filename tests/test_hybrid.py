from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pytest

from semanticcodec.core.scene.document import AssetRef
from semanticcodec.decode.pipeline import decode_bytes
from semanticcodec.encode.partition import MIN_SIDE, propose_cuts
from semanticcodec.encode.pipeline import EncodeOptions, encode_image
from semanticcodec.encode.symbols import available, build_symbol_asset
from semanticcodec.fallback.codecs import (
    CodecUnavailable,
    decode_asset,
    encode_jxl_lossless,
    encode_png,
    jxl_available,
)
from semanticcodec.format.container import compress, decode_cbor, decompress, encode_cbor
from semanticcodec.format.serialize import deserialize, serialize
from semanticcodec.format.symbols import (
    MAGIC,
    MAGIC_V2,
    SYMBOL_CODEC,
    SYMBOL_CODEC_V2,
    decode_symbols,
    pack_symbols,
)

native = pytest.mark.skipif(not jxl_available(), reason="needs the hybrid extra")


@native
@pytest.mark.parametrize("kind", ["rgb", "gray", "opaque-rgba"])
def test_jxl_exact(kind: str) -> None:
    image = np.random.default_rng(72).integers(0, 256, (17, 29, 3), np.uint8)
    if kind == "gray":
        image[..., 1:] = image[..., :1]
    source = np.concatenate([image, np.full((17, 29, 1), 255, np.uint8)], axis=2)
    data = encode_jxl_lossless(source if kind == "opaque-rgba" else image)
    assert np.array_equal(decode_asset(AssetRef("jxl-lossless", 29, 17, data)), source)


@native
@pytest.mark.parametrize("shape", [(17, 28), (16, 29), (1, 1)])
def test_jxl_checks_native_dimensions(shape: tuple[int, int]) -> None:
    data = encode_jxl_lossless(np.zeros((17, 29, 3), np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        decode_asset(AssetRef("jxl-lossless", shape[1], shape[0], data))


@native
@pytest.mark.parametrize("mode", ["gray", "uint16", "rgba"])
def test_jxl_checks_native_sample_type(mode: str) -> None:
    import imagecodecs

    shape = (8, 8) if mode == "gray" else (8, 8, 4 if mode == "rgba" else 3)
    pixels = np.zeros(shape, np.uint16 if mode == "uint16" else np.uint8)
    data = imagecodecs.jpegxl_encode(pixels, lossless=True, numthreads=1)
    with pytest.raises(ValueError):
        decode_asset(AssetRef("jxl-lossless", 8, 8, data))


@pytest.mark.parametrize(("w", "h"), [(0, 1), (True, 1), (1, -1), (16777217, 1)])
def test_jxl_canvas_guard_precedes_decoder(w: int, h: int) -> None:
    with (
        patch("semanticcodec.fallback.codecs.jxl_available") as available_mock,
        pytest.raises(ValueError, match="pixel limit"),
    ):
        decode_asset(AssetRef("jxl-lossless", w, h, b"not a JXL"))
    available_mock.assert_not_called()


@native
def test_jxl_malformed_payload() -> None:
    for data in (b"garbage", b"\xff\x0a", b"\xff\x0a\x00\x00"):
        with pytest.raises(ValueError):
            decode_asset(AssetRef("jxl-lossless", 8, 8, data))


def test_missing_native_dependency() -> None:
    pixels = np.zeros((8, 8, 4), np.uint8)
    with patch("semanticcodec.fallback.codecs.jxl_available", return_value=False):
        with pytest.raises(CodecUnavailable, match="hybrid extra"):
            encode_jxl_lossless(pixels)
        with pytest.raises(CodecUnavailable, match="hybrid extra"):
            decode_asset(AssetRef("jxl-lossless", 8, 8, b"x"))
    with (
        patch("semanticcodec.encode.pipeline.jxl_available", return_value=False),
        pytest.raises(ValueError, match="hybrid"),
    ):
        encode_image(pixels, EncodeOptions(hybrid=True))


@native
def test_jxl_alpha_stays_on_exact_conventional_path() -> None:
    pixels = np.random.default_rng(9).integers(0, 256, (8, 8, 4), np.uint8)
    pixels[0, 0] = (89, 12, 200, 0)
    with pytest.raises(CodecUnavailable, match="opaque"):
        encode_jxl_lossless(pixels)
    blob, doc, report = encode_image(pixels, EncodeOptions(hybrid=True, partition_depth=2))
    assert doc.assets[0].codec in ("png", "webp-lossless")
    assert report.regions == 1
    assert np.array_equal(decode_bytes(blob)[0], pixels)


@native
def test_v2_symbols_and_v1_compatibility() -> None:
    base_rgb = np.full((8, 8, 3), 187, np.uint8)
    base = AssetRef("jxl-lossless", 8, 8, encode_jxl_lossless(base_rgb))
    atlas = AssetRef("png", 512, 2, encode_png(np.full((2, 512, 3), 7, np.uint8)))
    payload = pack_symbols(base, atlas, [(2, 2)], [(0, 0), (32, 0)])
    assert payload.startswith(MAGIC_V2)
    asset = AssetRef(SYMBOL_CODEC_V2, 8, 8, payload)
    expected = np.full((8, 8, 4), (187, 187, 187, 255), np.uint8)
    expected[:2, :2, :3] ^= 7
    expected[4:6, :2, :3] ^= 7
    assert np.array_equal(decode_symbols(asset), expected)
    asset.codec = SYMBOL_CODEC
    with pytest.raises(ValueError, match="version"):
        decode_symbols(asset)
    asset.payload = MAGIC + payload[len(MAGIC_V2) :]
    with pytest.raises(ValueError, match="non-recursive"):
        decode_symbols(asset)


@native
@pytest.mark.parametrize(
    ("field", "value"), [(0, SYMBOL_CODEC_V2), (2, "jxl-lossless"), (2, SYMBOL_CODEC), (3, False)]
)
def test_v2_keeps_nested_layer_guards(field: int, value: object) -> None:
    base = AssetRef("jxl-lossless", 8, 8, encode_jxl_lossless(np.zeros((8, 8, 3), np.uint8)))
    atlas = AssetRef("png", 512, 2, encode_png(np.zeros((2, 512, 3), np.uint8)))
    payload = pack_symbols(base, atlas, [(2, 2)], [(0, 0)])
    body = decode_cbor(decompress(payload[len(MAGIC_V2) :]))
    body[field] = value
    with pytest.raises(ValueError):
        decode_symbols(
            AssetRef(
                SYMBOL_CODEC_V2,
                8,
                8,
                MAGIC_V2 + compress(encode_cbor(body)),
            )
        )


@pytest.mark.skipif(not available(), reason="needs cv extra")
def test_hybrid_symbols_keep_unprofitable_jxl_out() -> None:
    rgba = np.full((64, 128, 4), 255, np.uint8)
    for y in range(4, 56, 12):
        for x in range(4, 120, 12):
            rgba[y : y + 6, x : x + 3, :3] = 25
    baseline = build_symbol_asset(rgba)
    assert baseline is not None
    unprofitable = np.random.default_rng(94).integers(0, 256, 100000, np.uint8).tobytes()
    with patch("semanticcodec.encode.symbols.encode_jxl_lossless", return_value=unprofitable):
        candidate = build_symbol_asset(rgba, include_jxl=True)
    assert candidate is not None and candidate[0].payload == baseline[0].payload
    assert np.array_equal(decode_asset(candidate[0]), rgba)


def test_cut_budgets_and_small_regions() -> None:
    assert propose_cuts(np.zeros((127, 255, 3), np.uint8)) == []
    rgb = np.random.default_rng(3).integers(0, 256, (320, 512, 3), np.uint8)
    for axis, coordinate in propose_cuts(rgb):
        assert MIN_SIDE <= coordinate <= rgb.shape[axis] - MIN_SIDE
    assert propose_cuts(rgb) == propose_cuts(rgb.copy())


@pytest.mark.parametrize("depth", [1, 2])
def test_partition_wins_and_renders_every_region(depth: int) -> None:
    rgba = np.random.default_rng(14).integers(0, 256, (256, 384, 4), np.uint8)
    rgba[..., 3] = 255

    def artificial_cost(pixels, **kwargs):
        # Penalize the root only, to exercise the accepted-partition branch
        # independently of any native encoder's heuristics.
        data = encode_png(pixels)
        return "png", data + (b"\0" * 20000 if pixels.shape[:2] == (256, 384) else b"")

    with (
        patch("semanticcodec.encode.symbols.available", return_value=False),
        patch("semanticcodec.fallback.codecs.best_lossless", side_effect=artificial_cost),
    ):
        blob, doc, report = encode_image(rgba, EncodeOptions(partition_depth=depth))
    assert report.selected_codec == "partitioned"
    assert 2 <= report.regions <= 2**depth
    assert report.regions == len(doc.root.children) == len(doc.assets)
    assert len(blob) == min(report.candidate_bytes.values()) < report.floor_bytes
    assert serialize(deserialize(blob)) == blob
    assert np.array_equal(decode_bytes(blob)[0], rgba)
    assert report.node_counts["RasterPatch"] == report.regions


def test_partition_skips_nonopaque_input() -> None:
    rgba = np.full((256, 256, 4), 127, np.uint8)
    with patch("semanticcodec.encode.partition.partition_document") as search:
        blob, _, _ = encode_image(rgba, EncodeOptions(partition_depth=2))
    search.assert_not_called()
    assert np.array_equal(decode_bytes(blob)[0], rgba)


@native
def test_native_hybrid_pipeline_whole_file_floor() -> None:
    rgba = np.random.default_rng(18).integers(0, 256, (256, 256, 4), np.uint8)
    rgba[:128, :, :3] = 255
    rgba[..., 3] = 255
    options = EncodeOptions(hybrid=True, profile="fast", partition_depth=1)
    blob, _, report = encode_image(rgba, options)
    assert len(blob) == min(report.candidate_bytes.values()) <= report.floor_bytes
    assert np.array_equal(decode_bytes(blob)[0], rgba)
    assert encode_image(rgba, copy.copy(options))[0] == blob


@pytest.mark.parametrize("depth", [-1, 3, True, 1.5])
def test_invalid_partition_depth(depth) -> None:
    with pytest.raises(ValueError, match="partition_depth"):
        encode_image(np.full((8, 8, 4), 255, np.uint8), EncodeOptions(partition_depth=depth))
