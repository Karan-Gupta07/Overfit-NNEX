"""Document <-> `.scdx` bitstream.

`json -> binary -> json` must round-trip exactly. That holds here
because the scene graph contains no floats: every value is already a quantised
integer, so CBOR carries it losslessly and there is nothing to drift.

The graph payload reuses the JSON-shaped dict from `core.scene.json_io` rather
than defining a second encoding. One representation, two containers — a second
hand-written mapping would be a place for the two to silently disagree.
"""

from __future__ import annotations

from semanticcodec.core.scene.document import Document, validate
from semanticcodec.core.scene.json_io import document_from_json, document_to_json
from semanticcodec.core.scene.nodes import Group, TextRun
from semanticcodec.format.container import (
    Chunk,
    ChunkType,
    ContainerError,
    Flags,
    compress,
    decode_cbor,
    decompress,
    encode_cbor,
    find_chunk,
    read_container,
    write_container,
)


def serialize(doc: Document, *, lossless: bool = True, include_text: bool = True,
              validate_first: bool = True) -> bytes:
    """Encode a document to `.scdx` bytes.

    `validate_first` defaults on: serialising an invalid graph produces a file
    that fails at decode time, far from the bug that caused it.
    """
    if validate_first:
        validate(doc)

    j = document_to_json(doc, include_payloads=False)

    head = {
        "format_version": j["format_version"],
        "canvas": j["canvas"],
    }
    symt = {
        "palette": j["palette"],
        "fonts": j["fonts"],
    }
    # Assets go in their own chunk with raw payloads, not base64: the JSON form
    # base64s them, which would cost 33% for nothing in the binary container.
    aset = [
        {"codec": a.codec, "width": a.width, "height": a.height, "payload": a.payload}
        for a in doc.assets
    ]

    chunks = [
        Chunk(ChunkType.HEAD, compress(encode_cbor(head))),
        Chunk(ChunkType.SYMT, compress(encode_cbor(symt))),
        Chunk(ChunkType.GRPH, compress(encode_cbor(j["root"]))),
    ]
    if aset:
        chunks.append(Chunk(ChunkType.ASET, encode_cbor(aset)))

    if include_text:
        # Searchable archive capability. Costs little and no
        # pixel codec can offer it.
        strings = [
            n.text for n in doc.nodes() if isinstance(n, TextRun) and n.text.strip()
        ]
        if strings:
            chunks.append(Chunk(ChunkType.TEXT, compress(encode_cbor(strings))))

    flags = Flags.LOSSLESS if lossless else 0
    return write_container(chunks, flags=flags)


def deserialize(data: bytes, *, check: bool = True) -> Document:
    """Decode `.scdx` bytes to a document. Validates by default: this is a
    trust boundary."""
    _flags, chunks = read_container(data)

    head_raw = find_chunk(chunks, ChunkType.HEAD)
    symt_raw = find_chunk(chunks, ChunkType.SYMT)
    grph_raw = find_chunk(chunks, ChunkType.GRPH)
    if head_raw is None or symt_raw is None or grph_raw is None:
        raise ContainerError("required chunk missing after parse")

    head = decode_cbor(decompress(head_raw))
    symt = decode_cbor(decompress(symt_raw))
    root = decode_cbor(decompress(grph_raw))
    if not isinstance(head, dict) or not isinstance(symt, dict):
        raise ContainerError("HEAD and SYMT must be maps")

    aset_raw = find_chunk(chunks, ChunkType.ASET)
    assets: list[dict[str, object]] = []
    if aset_raw is not None:
        decoded = decode_cbor(aset_raw)
        if not isinstance(decoded, list):
            raise ContainerError("ASET must be a list")
        for a in decoded:
            if not isinstance(a, dict):
                raise ContainerError("ASET entry must be a map")
            payload = a.get("payload", b"")
            if not isinstance(payload, bytes):
                raise ContainerError("asset payload must be a byte string")
            assets.append(
                {
                    "codec": a.get("codec", "png"),
                    "width": a.get("width", 0),
                    "height": a.get("height", 0),
                    "payload_b64": None,  # replaced below; keeps one code path
                    "_raw": payload,
                }
            )

    doc = document_from_json(
        {
            "format_version": head.get("format_version", "0.1"),
            "canvas": head.get("canvas", {}),
            "palette": symt.get("palette", []),
            "fonts": symt.get("fonts", []),
            "assets": [
                {"codec": a["codec"], "width": a["width"], "height": a["height"]}
                for a in assets
            ],
            "root": root,
        }
    )
    # Attach raw payloads directly rather than round-tripping through base64.
    for ref, a in zip(doc.assets, assets, strict=True):
        raw = a["_raw"]
        assert isinstance(raw, bytes)
        ref.payload = raw

    if check:
        validate(doc)
    return doc


if __name__ == "__main__":
    import json

    import numpy as np

    from semanticcodec.core.color import BLACK, WHITE, Color
    from semanticcodec.core.geometry import Radii, Rect, px_to_q2
    from semanticcodec.core.scene.document import Canvas, FontRef
    from semanticcodec.core.scene.nodes import (
        Border,
        Edge,
        Fill,
        Glyph,
        RasterPatch,
        Style,
    )
    from semanticcodec.fallback.codecs import make_asset

    doc = Document(canvas=Canvas(width=200, height=100))
    w = doc.intern_color(WHITE)
    blue = doc.intern_color(Color(9, 105, 218))
    k = doc.intern_color(BLACK)
    doc.canvas.background = w
    fi = doc.add_font(FontRef("Arial", 400, False, "/tmp/Arial.ttf"))

    patch = np.full((20, 20, 4), 128, dtype=np.uint8)
    ai = doc.add_asset(make_asset(patch))

    e = Edge(px_to_q2(1), blue)
    doc.root = Group(
        rect=Rect.from_px(0, 0, 200, 100),
        style=Style(font=fi, size=px_to_q2(14)),
        children=[
            Fill(rect=Rect.from_px(8, 8, 100, 32), color=blue,
                 radii=Radii(*(px_to_q2(6),) * 4)),
            Border(rect=Rect.from_px(8, 8, 100, 32), top=e, right=e, bottom=e, left=e),
            TextRun(rect=Rect.from_px(16, 14, 60, 20), font=fi, size=px_to_q2(14),
                    color=w, baseline_x=px_to_q2(16), baseline_y=px_to_q2(28),
                    glyphs=[Glyph(54), Glyph(68, px_to_q2(7.25))], text="Save"),
            RasterPatch(rect=Rect.from_px(120, 8, 20, 20), asset=ai),
        ],
    )

    blob = serialize(doc)
    back = deserialize(blob)

    a = json.dumps(document_to_json(doc, include_payloads=False), sort_keys=True)
    b = json.dumps(document_to_json(back, include_payloads=False), sort_keys=True)
    assert a == b, "binary round-trip changed the graph"
    assert back.assets[0].payload == doc.assets[0].payload, "asset payload lost"
    assert back.palette.colors() == doc.palette.colors(), "palette order changed"
    assert isinstance(back.root, Group)
    bt = back.root.children[2]
    assert isinstance(bt, TextRun)
    assert bt.glyphs[1].advance == px_to_q2(7.25), "explicit advance lost"
    assert bt.glyphs[0].advance is None, "default advance became explicit"

    print(f"  .scdx size: {len(blob)} B  (graph+symbols+asset for 4 nodes)")

    # Truncation must be detected, never silently half-decoded.
    try:
        deserialize(blob[: len(blob) // 2])
        raise AssertionError("should have rejected truncated file")
    except ContainerError:
        pass
    print("serialize self-check passed")
