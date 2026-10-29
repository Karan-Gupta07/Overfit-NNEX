"""Measured encoder search with a conventional floor and exact symbol path."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from semanticcodec.core.color import Color, Palette
from semanticcodec.core.geometry import Rect
from semanticcodec.core.scene.document import Canvas, Document
from semanticcodec.core.scene.nodes import Group, RasterPatch
from semanticcodec.fallback.codecs import jxl_available, make_asset
from semanticcodec.format.serialize import serialize


@dataclass
class EncodeOptions:
    """One knob the whole pipeline respects."""

    profile: str = "balanced"  # fast | balanced | exhaustive
    lossless: bool = True
    quality: int = 90  # for lossy patch coding
    # Emit the conventional encoding when the semantic path does not win.
    floor_guarantee: bool = True
    hybrid: bool = False  # optional JPEG XL whole-image and residual candidates
    partition_depth: int = 0  # bounded spatial search, 0..2


@dataclass
class EncodeReport:
    """Per-encode accounting. Bits-by-node-type is the single
    most useful plot, and reconstructing it after the fact is painful."""

    width: int = 0
    height: int = 0
    source_bytes: int = 0
    output_bytes: int = 0
    semantic_bytes: int = 0
    floor_bytes: int = 0
    floor_used: bool = False
    node_counts: dict[str, int] = field(default_factory=dict)
    background: str = ""
    selected_codec: str = ""
    candidate_bytes: dict[str, int] = field(default_factory=dict)
    symbols: int = 0
    symbol_instances: int = 0
    symbol_background_bytes: int = 0
    symbol_atlas_bytes: int = 0
    analysis: str = ""
    regions: int = 1

    @property
    def bpp(self) -> float:
        px = self.width * self.height
        return (self.output_bytes * 8.0 / px) if px else 0.0


def load_image(path: Path) -> np.ndarray:
    """Load to HxWx4 uint8 RGBA. Screenshots are usually opaque; alpha is kept
    rather than flattened so window shadows and rounded corners survive."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def estimate_background(rgba: np.ndarray, sample_cap: int = 200_000) -> Color:
    """Modal colour, which for UI is overwhelmingly the page background.

    Subsamples large images: the mode of a strided sample is the same colour in
    practice and this runs on every encode.
    """
    flat = rgba.reshape(-1, 4)
    if flat.shape[0] > sample_cap:
        flat = flat[:: flat.shape[0] // sample_cap + 1]
    # Pack RGBA into one integer so np.unique counts whole colours, not channels.
    packed = (
        flat[:, 0].astype(np.uint32) << 24
        | flat[:, 1].astype(np.uint32) << 16
        | flat[:, 2].astype(np.uint32) << 8
        | flat[:, 3].astype(np.uint32)
    )
    vals, counts = np.unique(packed, return_counts=True)
    v = int(vals[int(np.argmax(counts))])
    return Color((v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)


def count_colors(rgba: np.ndarray, cap: int = 100_000) -> int:
    """Distinct colour count, capped. A cheap screenshot-vs-photo signal:
    UI images have tens to hundreds, photographs have many
    thousands."""
    flat = rgba.reshape(-1, 4)
    if flat.shape[0] > cap:
        flat = flat[:: flat.shape[0] // cap + 1]
    packed = (
        flat[:, 0].astype(np.uint32) << 24
        | flat[:, 1].astype(np.uint32) << 16
        | flat[:, 2].astype(np.uint32) << 8
        | flat[:, 3].astype(np.uint32)
    )
    return int(np.unique(packed).size)


def build_floor_document(rgba: np.ndarray, opts: EncodeOptions) -> Document:
    """The trivial baseline: the whole image as one RasterPatch.

    This is both the L0 encoder output and the floor that every later layer must
    beat. Keeping it as one function means the two can never drift apart.
    """
    h, w = rgba.shape[:2]
    bg = estimate_background(rgba)
    doc = Document(canvas=Canvas(width=w, height=h))
    doc.palette = Palette([bg])
    doc.canvas.background = 0
    asset = doc.add_asset(make_asset(
        rgba, lossless=opts.lossless, quality=opts.quality, include_jxl=opts.hybrid,
    ))
    doc.root = Group(
        rect=Rect.from_px(0, 0, w, h),
        children=[RasterPatch(
            rect=Rect.from_px(0, 0, w, h), asset=asset,
            replace=bool(np.any(rgba[..., 3] != 255)), tag="floor",
        )],
    )
    return doc


def encode_image(rgba: np.ndarray, opts: EncodeOptions | None = None
                 ) -> tuple[bytes, Document, EncodeReport]:
    """Encode an RGBA array. Returns (bitstream, document, report)."""
    opts = opts or EncodeOptions()
    if opts.profile not in ("fast", "balanced", "exhaustive"):
        raise ValueError("profile must be fast, balanced, or exhaustive")
    if not 0 <= opts.quality <= 100:
        raise ValueError("quality must be in 0..100")
    if type(opts.partition_depth) is not int or not 0 <= opts.partition_depth <= 2:
        raise ValueError("partition_depth must be an integer in 0..2")
    if opts.hybrid and not jxl_available():
        raise ValueError("hybrid encoding requires pip install 'semanticcodec[hybrid]'")
    if (rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4
            or not rgba.shape[0] or not rgba.shape[1]):
        raise ValueError("input must be a nonempty HxWx4 uint8 RGBA array")
    h, w = rgba.shape[:2]
    report = EncodeReport(width=w, height=h)

    # No lossy symbol substitution: even when lossy coding is allowed, this
    # candidate is exact and competes with the conventional lossy floor.
    from semanticcodec.encode.symbols import available, build_symbol_asset

    trials = []
    if available():
        trials = [(3 if opts.profile == "fast" else 6, 1)]
        if opts.profile == "exhaustive":
            trials.extend([(12, 1), (6, 2)])
    # Symbol trials run beside the floor; results are consumed in trial order.
    with ThreadPoolExecutor(1) as pool:
        pending = [
            pool.submit(
                build_symbol_asset, rgba, colors=colors, min_uses=min_uses,
                include_jxl=opts.hybrid,
            )
            for colors, min_uses in trials
        ]
        floor_doc = build_floor_document(rgba, opts)
        candidates = [job.result() for job in pending]
    floor_bytes = serialize(floor_doc, lossless=opts.lossless)

    report.floor_bytes = len(floor_bytes)
    report.semantic_bytes = len(floor_bytes)
    report.output_bytes = len(floor_bytes)
    report.floor_used = True
    report.background = estimate_background(rgba).to_hex()
    report.node_counts = {"RasterPatch": 1, "Group": 1}
    report.selected_codec = floor_doc.assets[0].codec
    report.candidate_bytes["floor"] = len(floor_bytes)
    best_bytes, best_doc = floor_bytes, floor_doc

    if trials:
        report.analysis = "pixel-derived symbols"
        semantic_sizes = []
        for (colors, min_uses), candidate in zip(trials, candidates, strict=True):
            if candidate is None:
                continue
            asset, stats = candidate
            doc = copy.deepcopy(floor_doc)
            doc.assets[0] = asset
            candidate_bytes = serialize(doc, lossless=True)
            size = len(candidate_bytes)
            semantic_sizes.append(size)
            report.candidate_bytes[f"symbols-c{colors}-n{min_uses}"] = size
            if size < len(best_bytes) or (not opts.floor_guarantee and best_doc is floor_doc):
                best_bytes, best_doc = candidate_bytes, doc
                report.floor_used = False
                report.selected_codec = asset.codec
                report.symbols = stats.symbols
                report.symbol_instances = stats.instances
                report.symbol_background_bytes = stats.background_bytes
                report.symbol_atlas_bytes = stats.atlas_bytes
        if semantic_sizes:
            report.semantic_bytes = min(semantic_sizes)
    else:
        report.analysis = "symbols unavailable (install the cv extra)"
    if opts.hybrid:
        report.analysis += "; JPEG XL whole-image/residual search"
    if opts.partition_depth and w * h <= 16_777_216 and np.all(rgba[..., 3] == 255):
        from semanticcodec.encode.partition import partition_document

        partition_blob, partition_doc, leaf_reports = partition_document(
            rgba, opts, best_doc, best_bytes, report,
        )
        report.candidate_bytes[f"partition-d{opts.partition_depth}"] = len(partition_blob)
        report.analysis += f"; spatial search depth {opts.partition_depth}"
        if len(partition_blob) < len(best_bytes):
            best_bytes, best_doc = partition_blob, partition_doc
            report.floor_used = False
            report.selected_codec = "partitioned"
            report.regions = len(leaf_reports)
            report.node_counts = {"RasterPatch": report.regions, "Group": 1}
            report.symbols = sum(r.symbols for r in leaf_reports)
            report.symbol_instances = sum(r.symbol_instances for r in leaf_reports)
            report.symbol_background_bytes = sum(r.symbol_background_bytes for r in leaf_reports)
            report.symbol_atlas_bytes = sum(r.symbol_atlas_bytes for r in leaf_reports)
            report.semantic_bytes = min(report.semantic_bytes, len(partition_blob))
    report.output_bytes = len(best_bytes)
    return best_bytes, best_doc, report


def encode_file(src: Path, dst: Path, opts: EncodeOptions | None = None) -> EncodeReport:
    rgba = load_image(src)
    blob, _doc, report = encode_image(rgba, opts)
    report.source_bytes = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)
    return report


if __name__ == "__main__":
    import tempfile

    # A UI-ish test image: flat background, a panel, a hard edge.
    # The panel is deliberately a minority of the area so the modal colour is
    # the background — an earlier version of this check had a 52% panel and
    # "failed" because blue really was the mode.
    img = np.full((60, 90, 4), 255, dtype=np.uint8)
    img[10:30, 10:40, :3] = (9, 105, 218)
    img[..., 3] = 255

    bg = estimate_background(img)
    assert bg == Color(255, 255, 255), bg
    assert count_colors(img) == 2, count_colors(img)

    blob, doc, rep = encode_image(img)
    assert doc.root is not None
    assert rep.output_bytes == len(blob)
    assert rep.bpp > 0
    print(f"  encoded 90x60 -> {len(blob)} B ({rep.bpp:.3f} bpp), bg {rep.background}")

    # The document the encoder produced must survive its own serialiser.
    from semanticcodec.format.serialize import deserialize

    back = deserialize(blob)
    assert back.canvas.width == 90 and back.canvas.height == 60
    assert len(back.assets) == 1

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "in.png"
        o = Path(td) / "out.scdx"
        Image.fromarray(img).save(p)
        r = encode_file(p, o)
        assert o.exists() and r.source_bytes > 0
    print("encode self-check passed")
