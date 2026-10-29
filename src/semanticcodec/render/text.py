"""Glyph run rendering.

Deliberately a stub until L2. `render.raster` calls `draw_text_run` only when a
glyph source is supplied, so L0/L1 run without it and "render without text" is
a free ablation.

When implemented, earlier text-rendering measurements already fixed the
configuration:
  - freetype with FT_LOAD_NO_HINTING (Skia on macOS does not hint)
  - LINEAR coverage -> alpha, no gamma curve (a fitted curve came within 0.04
    of linear and made held-out residuals 7% worse)
  - quarter-pixel (q2) pen quantisation, which beat both whole-pixel and 1/64
  - explicit per-glyph advances; accumulated freetype advances matched the
    browser exactly (457.09 vs 457.09 px, zero drift)
  - NO residual: glyph runs cost 3.3 KiB where raster costs 110 KiB, but
    correcting the remaining edge error costs 141 KiB, more than the pixels.
"""

from __future__ import annotations

from typing import Any


def draw_text_run(doc: Any, node: Any, canvas: Any, absolute: Any,
                  bounds: Any, glyph_source: Any) -> int:
    """Draw a TextRun; returns glyphs drawn. Not implemented until L2."""
    raise NotImplementedError("text rendering arrives with L2")
