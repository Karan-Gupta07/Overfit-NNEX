"""Decoding pipeline.

Deliberately boring. All intelligence is in the encoder; the decoder is a
deterministic interpreter. That asymmetry is standard in codec design and it is
what makes the format durable — improving the encoder never breaks old files.

Stages: parse and verify, resolve library deps, deserialise and validate, render,
composite residual, finalise. Residual composition is absent because nothing
emits residuals yet; measurements showed that exact residual correction for
text costs more than the pixels it corrects, so when it arrives it will be
region-gated rather than always-on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from semanticcodec.core.scene.document import Document
from semanticcodec.format.serialize import deserialize
from semanticcodec.render.raster import RenderStats, render


@dataclass
class DecodeReport:
    width: int = 0
    height: int = 0
    input_bytes: int = 0
    stats: RenderStats | None = None
    text_skipped: int = 0


def decode_document(doc: Document, *, glyph_source: object | None = None
                    ) -> tuple[np.ndarray, RenderStats]:
    """Render a document to HxWx4 RGBA."""
    return render(doc, glyph_source=glyph_source)


def decode_bytes(data: bytes, *, glyph_source: object | None = None,
                 check: bool = True) -> tuple[np.ndarray, DecodeReport]:
    """Decode a `.scdx` bitstream to RGBA.

    `check=True` validates the parsed graph. This is a trust boundary: a
    bitstream is untrusted input, and a decoder that renders a graph with
    dangling references produces silently wrong pixels.
    """
    doc = deserialize(data, check=check)
    canvas, stats = decode_document(doc, glyph_source=glyph_source)
    report = DecodeReport(
        width=doc.canvas.width,
        height=doc.canvas.height,
        input_bytes=len(data),
        stats=stats,
        text_skipped=stats.text_runs if glyph_source is None else 0,
    )
    return canvas, report


def decode_file(src: Path, dst: Path, *, glyph_source: object | None = None
                ) -> DecodeReport:
    """Decode to PNG, preserving alpha and hidden RGB when present."""
    data = src.read_bytes()
    canvas, report = decode_bytes(data, glyph_source=glyph_source)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pixels = canvas[..., :3] if np.all(canvas[..., 3] == 255) else canvas
    Image.fromarray(pixels).save(dst, format="PNG", optimize=True)
    return report


if __name__ == "__main__":
    from semanticcodec.encode.pipeline import encode_image
    from semanticcodec.render.raster import to_rgb

    # Round-trip an image through encode -> .scdx -> decode and require the
    # pixels back exactly. This is the check that matters for L0: the whole
    # point of the RasterPatch path is that it is lossless.
    img = np.full((40, 70, 4), 255, dtype=np.uint8)
    img[6:34, 6:64, :3] = (31, 35, 40)
    img[15:25, 20:50, :3] = (255, 200, 0)

    blob, _doc, _rep = encode_image(img)
    canvas, report = decode_bytes(blob)
    assert (report.width, report.height) == (70, 40), (report.width, report.height)

    out_rgb = to_rgb(canvas)
    assert np.array_equal(out_rgb, img[..., :3]), (
        "L0 round-trip must be pixel-exact; "
        f"max abs diff {np.abs(out_rgb.astype(int) - img[..., :3].astype(int)).max()}"
    )
    assert report.stats is not None and report.stats.patches == 1

    # A corrupt bitstream must raise rather than render garbage.
    from semanticcodec.format.container import ContainerError

    bad = bytearray(blob)
    bad[-8] ^= 0xFF
    try:
        decode_bytes(bytes(bad))
        raise AssertionError("should have rejected CRC-corrupt bitstream")
    except ContainerError:
        pass

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        s = Path(td) / "a.scdx"
        d = Path(td) / "a.png"
        s.write_bytes(blob)
        decode_file(s, d)
        again = np.asarray(Image.open(d).convert("RGB"), dtype=np.uint8)
        assert np.array_equal(again, img[..., :3]), "PNG output must match input"
    print("decode self-check passed")
