"""Pixel-derived, lossless symbol proposals.

The analysis is optional (OpenCV, encoder only). No font, OCR answer key, DOM,
external dictionary, model, or approximate character matching reaches decode.
All antialiasing samples survive as exact appearance data.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from semanticcodec.core.scene.document import AssetRef
from semanticcodec.fallback.codecs import CodecUnavailable, best_lossless, encode_jxl_lossless
from semanticcodec.format.symbols import (
    ATLAS_WIDTH,
    MAX_EDGE,
    MAX_INSTANCES,
    MAX_PAINT_PIXELS,
    MAX_PIXELS,
    MAX_SYMBOLS,
    SYMBOL_CODEC,
    SYMBOL_CODEC_V2,
    atlas_layout,
    pack_symbols,
)


@dataclass
class SymbolReport:
    symbols: int = 0
    instances: int = 0
    background_bytes: int = 0
    atlas_bytes: int = 0
    extracted_pixels: int = 0


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _background_colors(rgb: np.ndarray, count: int) -> list[np.ndarray]:
    flat = rgb.reshape(-1, 3)
    # Sampling limits analysis cost, never reconstruction fidelity.
    sample = flat[::max(1, (len(flat) + 199_999) // 200_000)]
    colors, counts = np.unique(sample, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [colors[i] for i in order[:count] if counts[i] >= max(8, len(sample) // 1000)]


def build_symbol_asset(rgba: np.ndarray, *, colors: int = 6,
                       min_uses: int = 1, include_jxl: bool = False,
                       ) -> tuple[AssetRef, SymbolReport] | None:
    """Remove small connected marks, intern exact XOR patches, code the rest.

    One-off symbols are useful too: their tightly packed appearances compress
    together without the whitespace and layout interrupting pixel references.
    Repeated symbols additionally pay for their appearance only once.
    """
    import cv2

    h, w = rgba.shape[:2]
    if w * h > MAX_PIXELS or np.any(rgba[..., 3] != 255):
        return None
    residual = rgba[..., :3].copy()
    appearances: list[np.ndarray] = []
    dictionary: dict[tuple[int, int, bytes], int] = {}
    # (linear position, appearance ID), in extraction order until final sorting.
    instances: list[tuple[int, int]] = []
    paint_pixels = 0
    for color in _background_colors(residual, colors):
        mask = np.any(residual != color, axis=2).astype(np.uint8)
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, n):
            x, y, sw, sh, area = (int(v) for v in stats[label])
            if not (area >= 2 and sw <= MAX_EDGE and sh <= MAX_EDGE):
                continue
            if (len(appearances) >= MAX_SYMBOLS or len(instances) >= MAX_INSTANCES
                    or paint_pixels + sw * sh > MAX_PAINT_PIXELS):
                break
            active = labels[y:y + sh, x:x + sw] == label
            region = residual[y:y + sh, x:x + sw]
            delta = np.zeros((sh, sw, 3), dtype=np.uint8)
            delta[active] = region[active] ^ color
            key = (sw, sh, delta.tobytes())
            symbol = dictionary.get(key)
            if symbol is None:
                symbol = len(appearances)
                dictionary[key] = symbol
                appearances.append(delta)
            instances.append((y * w + x, symbol))
            region[active] = color
            paint_pixels += sw * sh
    if not instances:
        return None

    if min_uses > 1:
        counts = np.bincount([s for _, s in instances], minlength=len(appearances))
        kept = []
        # XOR makes restoration exact even if successive extraction passes overlap.
        for position, symbol in instances:
            if counts[symbol] >= min_uses:
                kept.append((position, symbol))
            else:
                y, x = divmod(position, w)
                patch = appearances[symbol]
                sh, sw = patch.shape[:2]
                residual[y:y + sh, x:x + sw] ^= patch
        instances = kept
        if not instances:
            return None

    used = {symbol for _, symbol in instances}
    counts = np.bincount([symbol for _, symbol in instances], minlength=len(appearances))
    removed = sum(
        int(np.count_nonzero(np.any(appearances[symbol] != 0, axis=2))) * int(counts[symbol])
        for symbol in used
    )
    order = sorted(used, key=lambda s: (
        appearances[s].shape[0], appearances[s].shape[1], appearances[s].tobytes()
    ))
    remap = {old: new for new, old in enumerate(order)}
    sizes = [(appearances[s].shape[1], appearances[s].shape[0]) for s in order]
    try:
        origins, atlas_height = atlas_layout(sizes)
    except ValueError:
        return None
    atlas = np.zeros((atlas_height, ATLAS_WIDTH, 3), dtype=np.uint8)
    for symbol, (sx, sy) in zip(order, origins, strict=True):
        patch = appearances[symbol]
        sh, sw = patch.shape[:2]
        atlas[sy:sy + sh, sx:sx + sw] = patch
    instances = sorted((position, remap[symbol]) for position, symbol in instances)
    with ThreadPoolExecutor(1) as pool:
        jxl = pool.submit(encode_jxl_lossless, residual) if include_jxl else None
        base_codec, base_data = best_lossless(residual)
        atlas_codec, atlas_data = best_lossless(atlas)
    base = AssetRef(base_codec, w, h, base_data)
    atlas_asset = AssetRef(atlas_codec, ATLAS_WIDTH, atlas_height, atlas_data)
    data = pack_symbols(base, atlas_asset, sizes, instances)
    codec = SYMBOL_CODEC
    if jxl is not None:
        try:
            jxl_data = jxl.result()
            jxl_base = AssetRef("jxl-lossless", w, h, jxl_data)
            alternative = pack_symbols(jxl_base, atlas_asset, sizes, instances)
            # Compare complete compressed payloads, not just inner image sizes.
            if len(alternative) < len(data):
                codec, data, base_data = SYMBOL_CODEC_V2, alternative, jxl_data
        except CodecUnavailable:
            pass
    return AssetRef(codec, w, h, data), SymbolReport(
        symbols=len(sizes), instances=len(instances), background_bytes=len(base_data),
        atlas_bytes=len(atlas_data), extracted_pixels=removed,
    )
