from __future__ import annotations

import importlib.util
import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"
spec = importlib.util.spec_from_file_location("_probability_input", SOURCE / "input_image.py")
assert spec is not None and spec.loader is not None
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def test_rgb_loader_preserves_samples(tmp_path):
    pixels = np.random.default_rng(7).integers(0, 256, (5, 7, 3), np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    assert np.array_equal(loader.read_rgb(path), pixels)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_rgb_loader_refuses_channel_conversions(tmp_path, mode):
    path = tmp_path / "source.png"
    Image.new(mode, (5, 7)).save(path)
    with pytest.raises(ValueError, match="RGB"):
        loader.read_rgb(path)


def test_pixel_guard_precedes_loading_samples(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (5, 7)).save(path)
    with (
        patch.object(Image.Image, "tobytes", side_effect=AssertionError("loaded pixels")),
        pytest.raises(ValueError, match="pixel budget"),
    ):
        loader.read_rgb(path, max_pixels=1)


def test_byte_guard_precedes_opening_image(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    Image.new("RGB", (5, 7)).save(path)
    monkeypatch.setattr(loader, "MAX_SOURCE_BYTES", 1)
    with (
        patch.object(Image, "open", side_effect=AssertionError("opened oversized file")),
        pytest.raises(ValueError, match="byte budget"),
    ):
        loader.read_rgb(path)


def test_png16_rgb_cannot_be_silently_reduced(tmp_path):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" + struct.pack(">3H", 65535, 257, 5)))
        + chunk(b"IEND", b"")
    )
    path = tmp_path / "rgb16.png"
    path.write_bytes(blob)
    with Image.open(path) as image:
        assert image.mode == "RGB"  # Mode alone would not detect precision loss.
    with pytest.raises(ValueError, match="8-bit"):
        loader.read_rgb(path)


def test_binary_ppm_preserves_samples(tmp_path):
    pixels = np.random.default_rng(11).integers(0, 256, (5, 7, 3), np.uint8)
    path = tmp_path / "source.ppm"
    path.write_bytes(b"P6\n# camera\n7 5\n255\n" + pixels.tobytes())
    assert np.array_equal(loader.read_rgb(path), pixels)


@pytest.mark.parametrize(
    "blob",
    [
        b"P6\n1 1\n100\n" + bytes([100, 50, 0]),
        b"P6\n1 1\n65535\n" + struct.pack(">3H", 65535, 257, 5),
        b"P3\n1 1\n255\n1 2 3\n",
    ],
)
def test_ppm_samples_cannot_be_silently_rescaled(tmp_path, blob):
    path = tmp_path / "source.ppm"
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="8-bit"):
        loader.read_rgb(path)


def test_byte_guard_can_be_raised_per_call(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    Image.new("RGB", (5, 7), (3, 1, 4)).save(path)
    monkeypatch.setattr(loader, "MAX_SOURCE_BYTES", 1)
    assert loader.read_raster(path, max_bytes=path.stat().st_size).shape == (7, 5, 3)
