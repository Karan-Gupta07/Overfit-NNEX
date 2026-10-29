"""Deterministic reference renderer.

This is the most correctness-critical component in the system, because the
encoder's residual is computed against *its* output and applied to the
decoder's. Invariant: `render(doc)` is a pure function of the document and
library version, bit-identical on any machine.

What that forbids, and why this file looks the way it does:
  - no system font rasterisers (glyph shapes must be pinned)
  - no GPU paths
  - no order-dependent float accumulation in the composite loop
  - integer sRGB compositing, matching browsers rather than physics

Coverage is computed by supersampling on an integer grid. Slower than an
analytic scanline rasteriser and trivially reproducible, which is the correct
trade for a reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semanticcodec.core.color import Color
from semanticcodec.core.geometry import Radii, Rect, q2_to_px
from semanticcodec.core.scene.document import Document
from semanticcodec.core.scene.nodes import (
    Border,
    Fill,
    Group,
    Node,
    RasterPatch,
    Style,
    TextRun,
)

# Antialiasing supersample factor per axis. 4 gives 16 coverage levels per
# pixel, enough for UI edges; it is a format-level constant because changing it
# changes every rendered pixel and therefore every residual.
AA = 4

# Straight (non-premultiplied) RGBA canvas. Straight alpha keeps the
# composite arithmetic identical to the CSS model we are reproducing.
Canvas = np.ndarray


def new_canvas(width: int, height: int, background: Color) -> Canvas:
    buf = np.empty((height, width, 4), dtype=np.uint8)
    buf[..., 0] = background.r
    buf[..., 1] = background.g
    buf[..., 2] = background.b
    buf[..., 3] = background.a
    return buf


def _rounded_rect_coverage(
    rect: Rect, radii: Radii, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    """Coverage in 0..1 for a rounded rectangle, over the given pixel bounds.

    Supersamples AAxAA per pixel. Sample points sit at subpixel centres so a
    rect exactly on a pixel boundary yields coverage 1.0, not 0.5 — matching
    how browsers fill an integer-aligned box.
    """
    x0, y0, x1, y1 = bounds
    w_px, h_px = x1 - x0, y1 - y0
    if w_px <= 0 or h_px <= 0:
        return np.zeros((max(h_px, 0), max(w_px, 0)), dtype=np.float64)

    # Sample coordinates in pixel space.
    step = 1.0 / AA
    offs = (np.arange(AA) + 0.5) * step
    xs = (np.arange(x0, x1)[:, None] + offs[None, :]).ravel()
    ys = (np.arange(y0, y1)[:, None] + offs[None, :]).ravel()

    rx0, ry0 = q2_to_px(rect.x), q2_to_px(rect.y)
    rx1, ry1 = q2_to_px(rect.x1), q2_to_px(rect.y1)

    inside_x = (xs >= rx0) & (xs < rx1)
    inside_y = (ys >= ry0) & (ys < ry1)
    cov = (inside_y[:, None] & inside_x[None, :]).astype(np.float64)

    if not radii.is_zero():
        r = radii.clamped_to(rect)
        corners = (
            (q2_to_px(r.tl), rx0, ry0, +1, +1),
            (q2_to_px(r.tr), rx1, ry0, -1, +1),
            (q2_to_px(r.br), rx1, ry1, -1, -1),
            (q2_to_px(r.bl), rx0, ry1, +1, -1),
        )
        for rad, cx_edge, cy_edge, sx, sy in corners:
            if rad <= 0:
                continue
            cx = cx_edge + sx * rad
            cy = cy_edge + sy * rad
            # Only the quadrant outside the corner arc is carved away.
            in_quad_x = (xs - cx) * sx < 0
            in_quad_y = (ys - cy) * sy < 0
            dx = xs - cx
            dy = ys - cy
            outside = (dx[None, :] ** 2 + dy[:, None] ** 2) > rad * rad
            carve = outside & in_quad_y[:, None] & in_quad_x[None, :]
            cov[carve] = 0.0

    # Average each AAxAA block back down to one pixel.
    return cov.reshape(h_px, AA, w_px, AA).mean(axis=(1, 3))


def _composite(dst: Canvas, bounds: tuple[int, int, int, int], color: Color,
               coverage: np.ndarray, opacity: int) -> None:
    """Source-over composite a solid colour through a coverage mask.

    Integer sRGB, straight alpha. Rounding is round-half-up via +0.5 floor,
    fixed so two implementations cannot disagree on a .5 boundary.
    """
    x0, y0, x1, y1 = bounds
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return
    alpha = coverage * (color.a / 255.0) * (opacity / 255.0)
    if not alpha.any():
        return
    region = dst[y0:y1, x0:x1, :]
    da = region[..., 3:4].astype(np.float64) / 255.0
    sa = alpha[..., None]
    out_a = sa + da * (1.0 - sa)
    src_rgb = np.array(color.rgb(), dtype=np.float64)[None, None, :]
    dst_rgb = region[..., :3].astype(np.float64)
    num = src_rgb * sa + dst_rgb * da * (1.0 - sa)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(out_a > 0, num / np.maximum(out_a, 1e-12), 0.0)
    region[..., :3] = np.floor(np.clip(rgb, 0, 255) + 0.5).astype(np.uint8)
    region[..., 3] = np.floor(np.clip(out_a[..., 0] * 255.0, 0, 255) + 0.5).astype(np.uint8)


def _clip_bounds(rect: Rect, canvas_w: int, canvas_h: int,
                 clip: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect.to_pixel_bounds()
    if clip is not None:
        x0, y0 = max(x0, clip[0]), max(y0, clip[1])
        x1, y1 = min(x1, clip[2]), min(y1, clip[3])
    return (max(0, x0), max(0, y0), min(canvas_w, x1), min(canvas_h, y1))


@dataclass
class RenderStats:
    """Diagnostics. Per-node accounting pays for itself fast."""

    nodes_drawn: int = 0
    nodes_skipped: int = 0
    text_runs: int = 0
    glyphs: int = 0
    patches: int = 0


def render(doc: Document, *, glyph_source: object | None = None) -> tuple[Canvas, RenderStats]:
    """Render a document to a straight-alpha RGBA canvas.

    `glyph_source` supplies glyph bitmaps for TextRun nodes; when None, text is
    skipped and counted in stats. That keeps L0/L1 runnable before the text
    stack exists, and makes "render without text" a free ablation.
    """
    cw, ch = doc.canvas.width, doc.canvas.height
    bg = doc.palette[doc.canvas.background] if len(doc.palette) else Color(255, 255, 255)
    canvas = new_canvas(cw, ch, bg)
    stats = RenderStats()
    if doc.root is not None:
        _render_node(doc, doc.root, canvas, 0, 0, None, Style(), stats, glyph_source)
    return canvas, stats


def _render_node(doc: Document, node: Node, canvas: Canvas, ox: int, oy: int,
                 clip: tuple[int, int, int, int] | None, inherited: Style,
                 stats: RenderStats, glyph_source: object | None) -> None:
    cw, ch = doc.canvas.width, doc.canvas.height
    absolute = Rect(ox + node.rect.x, oy + node.rect.y, node.rect.w, node.rect.h)

    if node.opacity == 0 or absolute.is_empty():
        stats.nodes_skipped += 1
        if not isinstance(node, Group):
            return

    bounds = _clip_bounds(absolute, cw, ch, clip)

    if isinstance(node, Fill):
        cov = _rounded_rect_coverage(absolute, node.radii, bounds)
        _composite(canvas, bounds, doc.palette[node.color], cov, node.opacity)
        stats.nodes_drawn += 1

    elif isinstance(node, Border):
        _render_border(doc, node, canvas, absolute, bounds, stats)

    elif isinstance(node, TextRun):
        stats.text_runs += 1
        if glyph_source is None:
            stats.nodes_skipped += 1
            return
        from semanticcodec.render.text import draw_text_run  # local: keeps core light

        drawn = draw_text_run(doc, node, canvas, absolute, bounds, glyph_source)
        stats.glyphs += drawn
        stats.nodes_drawn += 1

    elif isinstance(node, RasterPatch):
        _render_patch(doc, node, canvas, absolute, bounds, stats)

    elif isinstance(node, Group):
        child_clip = bounds if node.clip else clip
        style = node.style.merged_over(inherited)
        order = sorted(
            range(len(node.children)),
            key=lambda i: (node.children[i].z is not None, node.children[i].z or 0, i),
        )
        for i in order:
            _render_node(doc, node.children[i], canvas, absolute.x, absolute.y,
                         child_clip, style, stats, glyph_source)


def _render_border(doc: Document, node: Border, canvas: Canvas, absolute: Rect,
                   bounds: tuple[int, int, int, int], stats: RenderStats) -> None:
    """Draw a border as outer-minus-inner coverage, per edge.

    Doing it as a difference of two rounded rects (rather than four strips)
    keeps mitred corners correct for free, which is what browsers produce.
    """
    edges = node.edges()
    if not any(e.is_drawn() for e in edges):
        stats.nodes_skipped += 1
        return
    outer = _rounded_rect_coverage(absolute, node.radii, bounds)
    top, right, bottom, left = edges
    inner_rect = Rect(
        absolute.x + left.width,
        absolute.y + top.width,
        max(0, absolute.w - left.width - right.width),
        max(0, absolute.h - top.width - bottom.width),
    )
    inner_radii = Radii(
        max(0, node.radii.tl - max(left.width, top.width)),
        max(0, node.radii.tr - max(right.width, top.width)),
        max(0, node.radii.br - max(right.width, bottom.width)),
        max(0, node.radii.bl - max(left.width, bottom.width)),
    )
    inner = _rounded_rect_coverage(inner_rect, inner_radii, bounds)
    ring = np.clip(outer - inner, 0.0, 1.0)

    if node.is_uniform():
        _composite(canvas, bounds, doc.palette[top.color], ring, node.opacity)
        stats.nodes_drawn += 1
        return

    # Per-edge: mask the ring by which side of the box each pixel is on.
    x0, y0, x1, y1 = bounds
    ys = np.arange(y0, y1)[:, None] + 0.5
    xs = np.arange(x0, x1)[None, :] + 0.5
    ax0, ay0 = q2_to_px(absolute.x), q2_to_px(absolute.y)
    ax1, ay1 = q2_to_px(absolute.x1), q2_to_px(absolute.y1)
    masks = (
        (top, ys < ay0 + q2_to_px(top.width)),
        (bottom, ys >= ay1 - q2_to_px(bottom.width)),
        (left, xs < ax0 + q2_to_px(left.width)),
        (right, xs >= ax1 - q2_to_px(right.width)),
    )
    for edge, m in masks:
        if not edge.is_drawn():
            continue
        _composite(canvas, bounds, doc.palette[edge.color],
                   ring * np.broadcast_to(m, ring.shape), node.opacity)
    stats.nodes_drawn += 1


def _render_patch(doc: Document, node: RasterPatch, canvas: Canvas, absolute: Rect,
                  bounds: tuple[int, int, int, int], stats: RenderStats) -> None:
    """Blit a decoded raster asset. Nearest sampling when sizes differ.

    Resampling is one of the places implementations disagree, so the
    filter is pinned and simple rather than good. An encoder should emit patches
    at native size; a scaled patch is a bug worth seeing.
    """
    from semanticcodec.fallback.codecs import decode_asset

    asset = doc.assets[node.asset]
    rgba = decode_asset(asset)
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        stats.nodes_skipped += 1
        return
    ax0, ay0 = absolute.to_pixel_bounds()[0], absolute.to_pixel_bounds()[1]
    tgt_w = max(1, absolute.to_pixel_bounds()[2] - ax0)
    tgt_h = max(1, absolute.to_pixel_bounds()[3] - ay0)
    src_h, src_w = rgba.shape[:2]
    yy = np.minimum(((np.arange(y0, y1) - ay0) * src_h) // tgt_h, src_h - 1)
    xx = np.minimum(((np.arange(x0, x1) - ax0) * src_w) // tgt_w, src_w - 1)
    sample = rgba[np.clip(yy, 0, src_h - 1)[:, None], np.clip(xx, 0, src_w - 1)[None, :]]

    # Replacement is explicit, so ordinary scene patches retain source-over
    # semantics, including when one happens to cover the entire canvas.
    if node.replace:
        canvas[y0:y1, x0:x1] = sample
        stats.patches += 1
        stats.nodes_drawn += 1
        return

    cov = sample[..., 3].astype(np.float64) / 255.0
    region = canvas[y0:y1, x0:x1, :]
    sa = (cov * (node.opacity / 255.0))[..., None]
    da = region[..., 3:4].astype(np.float64) / 255.0
    out_a = sa + da * (1.0 - sa)
    num = (sample[..., :3].astype(np.float64) * sa
           + region[..., :3].astype(np.float64) * da * (1.0 - sa))
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(out_a > 0, num / np.maximum(out_a, 1e-12), 0.0)
    region[..., :3] = np.floor(np.clip(rgb, 0, 255) + 0.5).astype(np.uint8)
    region[..., 3] = np.floor(
        np.clip(out_a[..., 0] * 255.0, 0, 255) + 0.5
    ).astype(np.uint8)
    stats.patches += 1
    stats.nodes_drawn += 1


OPAQUE_WHITE = Color(255, 255, 255)


def to_rgb(canvas: Canvas, background: Color | None = None) -> np.ndarray:
    """Flatten RGBA onto an opaque background. Screenshots are opaque, so this
    is what comparison and PNG output use."""
    background = background or OPAQUE_WHITE
    a = canvas[..., 3:4].astype(np.float64) / 255.0
    bg = np.array(background.rgb(), dtype=np.float64)[None, None, :]
    rgb = canvas[..., :3].astype(np.float64) * a + bg * (1.0 - a)
    return np.floor(np.clip(rgb, 0, 255) + 0.5).astype(np.uint8)
