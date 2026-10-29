"""Bound image-local training memory and reject silent sample conversions."""

import re
from pathlib import Path

import numpy as np
from PIL import Image

MAX_TRAIN_PIXELS = 1_048_576
MAX_SOURCE_BYTES = 64 * 1024 * 1024


FORMATS = ("PNG", "JPEG", "TIFF", "WEBP", "BMP", "PPM")
# Pillow rescales PPM samples to 8 bits whatever the header's maxval, so only
# binary RGB PPM with maxval 255 is read as is.
_SEPARATOR = rb"(?:\s+|#[^\n]*\n)+"
PPM_HEADER = re.compile(rb"P6" + (_SEPARATOR + rb"(\d+)") * 3 + rb"\s")


def read_raster(
    path: Path,
    max_pixels: int = MAX_TRAIN_PIXELS,
    *,
    allow_rgba: bool = False,
    max_bytes: int | None = None,
) -> np.ndarray:
    if path.stat().st_size > (MAX_SOURCE_BYTES if max_bytes is None else max_bytes):
        raise ValueError("research source byte budget exceeded")
    with path.open("rb") as file:
        prefix = file.read(1024)
    if prefix.startswith(b"\x89PNG\r\n\x1a\n") and (len(prefix) < 26 or prefix[24] != 8):
        raise ValueError("only 8-bit RGB samples are supported")
    with Image.open(path, formats=FORMATS) as image:
        if image.width <= 0 or image.height <= 0 or image.width * image.height > max_pixels:
            raise ValueError("research training pixel budget exceeded")
        if image.mode not in (("RGB", "RGBA") if allow_rgba else ("RGB",)):
            raise ValueError("only opaque 8-bit RGB is supported")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("only still images are supported")
        if image.format not in FORMATS:
            raise ValueError("unsupported input format for the 8-bit RGB experiment")
        if image.format == "PPM" and (
            not (header := PPM_HEADER.match(prefix)) or int(header[3]) != 255
        ):
            raise ValueError("only 8-bit binary PPM (P6, maxval 255) is supported")
        if image.format == "TIFF" and image.tag_v2.get(258) != (8,) * len(image.getbands()):
            raise ValueError("only 8-bit TIFF samples are supported")
        if image.format == "JPEG" and getattr(image, "bits", 8) != 8:
            raise ValueError("only 8-bit JPEG samples are supported")
        return np.asarray(image).copy()


def read_rgb(path: Path, max_pixels: int = MAX_TRAIN_PIXELS) -> np.ndarray:
    return read_raster(path, max_pixels=max_pixels)
