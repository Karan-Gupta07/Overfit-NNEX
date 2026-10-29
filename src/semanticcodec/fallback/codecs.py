"""Raster patch codecs.

The fallback path is what makes the codec safe: anything the semantic pass
cannot explain is coded conventionally here, so the worst case is a bigger file
rather than a wrong image.

The normal lossless floor compares PNG and WebP. Hybrid encoding optionally
includes JPEG XL, with an explicitly sized native output buffer on decode.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from semanticcodec.core.scene.document import AssetRef


class CodecUnavailable(RuntimeError):
    pass


# libjxl and cwebp -mt write the same bytes for any thread count; threads only
# change encode time.
ENCODER_THREADS = max(1, min(4, os.cpu_count() or 1))


def _pil_from_rgba(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(rgba, mode="RGBA" if rgba.shape[2] == 4 else "RGB")


def encode_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    # optimize=True is deterministic in Pillow and costs little at these sizes.
    _pil_from_rgba(rgba).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_webp_lossless(rgba: np.ndarray) -> bytes:
    """cwebp -lossless -z 9, matching the baseline configuration in eval."""
    import shutil

    if shutil.which("cwebp") is None:
        # Pillow bundles libwebp. Keep the strongest available native floor on
        # machines without command-line tools, including exact hidden RGB.
        buf = io.BytesIO()
        try:
            _pil_from_rgba(rgba).save(
                buf, format="WEBP", lossless=True, quality=100, method=6, exact=True
            )
        except (OSError, KeyError) as exc:
            raise CodecUnavailable("Pillow WebP encoder is unavailable") from exc
        return buf.getvalue()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "s.png"
        dst = Path(td) / "d.webp"
        _pil_from_rgba(rgba).save(src, format="PNG")
        proc = subprocess.run(
            [
                "cwebp", "-quiet", "-lossless", "-z", "9", "-exact", "-mt",
                str(src), "-o", str(dst),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not dst.exists():
            raise CodecUnavailable(
                f"cwebp failed: {proc.stderr.decode('utf8', 'replace').strip()[:200]}"
            )
        return dst.read_bytes()


def encode_webp_lossy(rgba: np.ndarray, quality: int = 90) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "s.png"
        dst = Path(td) / "d.webp"
        _pil_from_rgba(rgba).save(src, format="PNG")
        proc = subprocess.run(
            ["cwebp", "-quiet", "-q", str(int(quality)), "-m", "6", str(src), "-o", str(dst)],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not dst.exists():
            raise CodecUnavailable("cwebp lossy failed")
        return dst.read_bytes()


def jxl_available() -> bool:
    try:
        import imagecodecs
    except ImportError:
        return False
    return bool(imagecodecs.JPEGXL.available)


def encode_jxl_lossless(rgba: np.ndarray) -> bytes:
    """Optional opaque, 8-bit RGB path. Alpha stays with PNG or exact WebP."""
    if not jxl_available():
        raise CodecUnavailable("JPEG XL requires the hybrid extra")
    if rgba.shape[2] == 4 and np.any(rgba[..., 3] != 255):
        raise CodecUnavailable("the JPEG XL path only supports opaque images")
    import imagecodecs

    rgb = np.ascontiguousarray(rgba[..., :3])
    return bytes(imagecodecs.jpegxl_encode(
        rgb, lossless=True, effort=9, numthreads=ENCODER_THREADS, usecontainer=False,
    ))


def _decode_jxl(asset: AssetRef) -> np.ndarray:
    if not (type(asset.width) is int and type(asset.height) is int
            and asset.width >= 1 and asset.height >= 1
            and asset.width * asset.height <= 16_777_216):
        raise ValueError("JPEG XL asset exceeds pixel limit")
    if not jxl_available():
        raise CodecUnavailable("decoding JPEG XL assets requires the hybrid extra")
    import imagecodecs

    # Unlike the Pillow plugin, this binding checks the stream's shape and
    # dtype against our output array before allocating the decoded image.
    # Frame 0 avoids allocating an entire animation. Non-RGB/8-bit streams
    # are deliberately unsupported by this asset codec.
    rgb = np.empty((asset.height, asset.width, 3), dtype=np.uint8)
    try:
        result = imagecodecs.jpegxl_decode(
            asset.payload, index=0, numthreads=1, out=rgb,
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError("invalid JPEG XL payload, dimensions, or sample type") from exc
    if result is not rgb:
        raise ValueError("JPEG XL decoder did not honor its output buffer")
    rgba = np.full((asset.height, asset.width, 4), 255, dtype=np.uint8)
    rgba[..., :3] = rgb
    return rgba


ENCODERS = {
    "png": encode_png,
    "webp-lossless": encode_webp_lossless,
}


def best_lossless(rgba: np.ndarray, *, include_jxl: bool = False) -> tuple[str, bytes]:
    """Smallest lossless encoding available, with PNG as the guaranteed floor.

    PNG always works; WebP uses cwebp or Pillow. JPEG XL is opt-in.
    """
    use_jxl = include_jxl and rgba.shape[0] * rgba.shape[1] <= 16_777_216
    # The encoders release the GIL; selection order matches the serial loop.
    with ThreadPoolExecutor(3) as pool:
        png = pool.submit(encode_png, rgba)
        webp = pool.submit(encode_webp_lossless, rgba)
        jxl = pool.submit(encode_jxl_lossless, rgba) if use_jxl else None
        best_name, best_payload = "png", png.result()
        try:
            wl = webp.result()
            if len(wl) < len(best_payload):
                best_name, best_payload = "webp-lossless", wl
        except (CodecUnavailable, FileNotFoundError):
            pass
        if jxl is not None:
            try:
                data = jxl.result()
                if len(data) < len(best_payload):
                    best_name, best_payload = "jxl-lossless", data
            except CodecUnavailable:
                pass
    return best_name, best_payload


def decode_asset(asset: AssetRef) -> np.ndarray:
    """Decode an asset payload to HxWx4 uint8 RGBA.

    Pillow reads PNG and WebP natively, so no subprocess is needed on the
    decode path — which matters because decode must stay fast.
    """
    if not asset.payload:
        raise ValueError("asset has no payload")
    from semanticcodec.format.symbols import SYMBOL_CODEC, SYMBOL_CODEC_V2, decode_symbols

    if asset.codec in (SYMBOL_CODEC, SYMBOL_CODEC_V2):
        return decode_symbols(asset)
    if asset.codec == "jxl-lossless":
        return _decode_jxl(asset)
    if asset.codec not in ("png", "webp-lossless", "webp-lossy"):
        raise ValueError(f"unsupported asset codec {asset.codec!r}")
    with Image.open(io.BytesIO(asset.payload)) as im:
        if im.size != (asset.width, asset.height):
            raise ValueError(
                f"asset dimensions disagree: header says {asset.width}x{asset.height}, "
                f"payload decodes to {im.width}x{im.height}"
            )
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    if (rgba.shape[1], rgba.shape[0]) != (asset.width, asset.height):
        raise ValueError(
            f"asset dimensions disagree: header says {asset.width}x{asset.height}, "
            f"payload decodes to {rgba.shape[1]}x{rgba.shape[0]}"
        )
    return rgba


def make_asset(rgba: np.ndarray, *, lossless: bool = True, quality: int = 90,
               include_jxl: bool = False) -> AssetRef:
    h, w = rgba.shape[:2]
    if lossless:
        codec, payload = best_lossless(rgba, include_jxl=include_jxl)
    else:
        try:
            codec, payload = "webp-lossy", encode_webp_lossy(rgba, quality)
        except (CodecUnavailable, FileNotFoundError):
            codec, payload = best_lossless(rgba)
    return AssetRef(codec=codec, width=w, height=h, payload=payload)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.full((32, 48, 4), 255, dtype=np.uint8)
    img[8:24, 8:40, :3] = 40  # a hard edge: the UI-relevant case
    img[0:4, 0:4, 3] = 0  # transparent corner must survive

    a = make_asset(img)
    back = decode_asset(a)
    assert back.shape == img.shape, (back.shape, img.shape)
    assert np.array_equal(back, img), "lossless round-trip must be exact"
    assert a.codec in ENCODERS, a.codec
    print(f"  chosen codec: {a.codec}  {len(a.payload)} B for {img.shape[1]}x{img.shape[0]}")

    # PNG path alone must also round-trip (the no-cwebp floor)
    p = AssetRef("png", 48, 32, encode_png(img))
    assert np.array_equal(decode_asset(p), img)

    # A header that disagrees with the payload must be rejected, not silently
    # rendered at the wrong size.
    bad = AssetRef("png", 999, 999, encode_png(img))
    try:
        decode_asset(bad)
        raise AssertionError("should have rejected dimension mismatch")
    except ValueError:
        pass
    print("codecs self-check passed")
