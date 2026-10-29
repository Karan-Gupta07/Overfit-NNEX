"""`.scg.json` read/write.

The JSON form is the research and debugging representation; `.scdx` is the
binary one. `json -> binary -> json` must round-trip exactly,
which is why every value written here is already an integer: there is no float
anywhere in the graph, so there is nothing to drift.

Raster payloads are base64'd. That bloats the JSON, which is fine — the JSON
form is never the thing whose size we report.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from semanticcodec.core.color import Color, Palette
from semanticcodec.core.geometry import Radii, Rect
from semanticcodec.core.scene.document import (
    AssetRef,
    Canvas,
    Document,
    FontRef,
    ValidationError,
    validate,
)
from semanticcodec.core.scene.nodes import (
    AAMode,
    Border,
    BorderStyle,
    Edge,
    Fill,
    Glyph,
    Group,
    Node,
    NodeType,
    RasterPatch,
    Style,
    TextRun,
)


def _rect_to_json(r: Rect) -> list[int]:
    return [r.x, r.y, r.w, r.h]


def _rect_from_json(v: Any) -> Rect:
    if not (isinstance(v, list) and len(v) == 4 and all(isinstance(i, int) for i in v)):
        raise ValidationError(f"rect must be 4 integers, got {v!r}")
    return Rect(*v)


def _base_to_json(n: Node) -> dict[str, Any]:
    d: dict[str, Any] = {"type": n.type.value, "rect": _rect_to_json(n.rect)}
    # Omit defaults. Keeps files readable and diffs small.
    if n.z is not None:
        d["z"] = n.z
    if n.opacity != 255:
        d["opacity"] = n.opacity
    if n.clip:
        d["clip"] = True
    if n.tag:
        d["tag"] = n.tag
    return d


def _radii_to_json(r: Radii) -> list[int] | None:
    return None if r.is_zero() else list(r)


def _edge_to_json(e: Edge) -> dict[str, Any]:
    d: dict[str, Any] = {"width": e.width, "color": e.color}
    if e.style is not BorderStyle.SOLID:
        d["style"] = e.style.value
    return d


def _edge_from_json(v: Any) -> Edge:
    if not isinstance(v, dict):
        raise ValidationError(f"edge must be an object, got {v!r}")
    return Edge(
        width=int(v.get("width", 0)),
        color=int(v.get("color", 0)),
        style=BorderStyle(v.get("style", "solid")),
    )


def node_to_json(n: Node) -> dict[str, Any]:
    d = _base_to_json(n)
    if isinstance(n, Fill):
        d["color"] = n.color
        if (r := _radii_to_json(n.radii)) is not None:
            d["radii"] = r
    elif isinstance(n, Border):
        if n.is_uniform():
            d["uniform"] = _edge_to_json(n.top)
        else:
            d["edges"] = [_edge_to_json(e) for e in n.edges()]
        if (r := _radii_to_json(n.radii)) is not None:
            d["radii"] = r
    elif isinstance(n, TextRun):
        d.update(
            font=n.font,
            size=n.size,
            color=n.color,
            baseline=[n.baseline_x, n.baseline_y],
            text=n.text,
        )
        # A glyph with a default advance serialises as a bare int, which is the
        # common case; only overridden advances need the pair form.
        d["glyphs"] = [g.gid if g.advance is None else [g.gid, g.advance] for g in n.glyphs]
        if n.letter_spacing:
            d["letter_spacing"] = n.letter_spacing
        if n.aa_mode is not AAMode.GRAYSCALE:
            d["aa_mode"] = n.aa_mode.value
    elif isinstance(n, RasterPatch):
        d["asset"] = n.asset
        if n.replace:
            d["replace"] = True
    elif isinstance(n, Group):
        if not n.style.is_empty():
            d["style"] = {
                k: v
                for k, v in (
                    ("font", n.style.font),
                    ("size", n.style.size),
                    ("color", n.style.color),
                    ("letter_spacing", n.style.letter_spacing),
                )
                if v is not None
            }
        d["children"] = [node_to_json(c) for c in n.children]
    else:  # pragma: no cover - Node union is closed
        raise ValidationError(f"unknown node type {type(n).__name__}")
    return d


def node_from_json(v: Any) -> Node:
    if not isinstance(v, dict):
        raise ValidationError(f"node must be an object, got {type(v).__name__}")
    try:
        t = NodeType(v["type"])
    except KeyError:
        raise ValidationError("node missing 'type'") from None
    except ValueError:
        raise ValidationError(f"unknown node type {v['type']!r}") from None

    # Explicit locals rather than **base: mypy cannot verify a **dict against a
    # dataclass signature, and silently-wrong field types in the format parser
    # are exactly the bug class validation exists to prevent.
    rect: Rect = _rect_from_json(v.get("rect"))
    z: int | None = int(v["z"]) if v.get("z") is not None else None
    opacity: int = int(v.get("opacity", 255))
    clip: bool = bool(v.get("clip", False))
    tag: str | None = str(v["tag"]) if v.get("tag") is not None else None
    radii = Radii(*v["radii"]) if "radii" in v else Radii()

    if t is NodeType.FILL:
        return Fill(
            rect=rect, z=z, opacity=opacity, clip=clip, tag=tag,
            color=int(v.get("color", 0)), radii=radii,
        )  # fmt: skip
    if t is NodeType.BORDER:
        if "uniform" in v:
            e = _edge_from_json(v["uniform"])
            return Border(
                rect=rect, z=z, opacity=opacity, clip=clip, tag=tag,
                top=e, right=e, bottom=e, left=e, radii=radii,
            )  # fmt: skip
        edges = [_edge_from_json(e) for e in v.get("edges", [])]
        if len(edges) != 4:
            raise ValidationError(f"Border needs 4 edges or 'uniform', got {len(edges)}")
        return Border(
            rect=rect, z=z, opacity=opacity, clip=clip, tag=tag,
            top=edges[0], right=edges[1], bottom=edges[2], left=edges[3], radii=radii,
        )  # fmt: skip
    if t is NodeType.TEXT_RUN:
        bl = v.get("baseline", [0, 0])
        if not (isinstance(bl, list) and len(bl) == 2):
            raise ValidationError(f"baseline must be [x, y], got {bl!r}")
        glyphs = []
        for g in v.get("glyphs", []):
            if isinstance(g, bool):
                raise ValidationError(f"glyph must be int or [gid, advance], got {g!r}")
            if isinstance(g, int):
                glyphs.append(Glyph(g))
            elif isinstance(g, list) and len(g) == 2:
                glyphs.append(Glyph(int(g[0]), int(g[1])))
            else:
                raise ValidationError(f"glyph must be int or [gid, advance], got {g!r}")
        return TextRun(
            rect=rect, z=z, opacity=opacity, clip=clip, tag=tag,
            font=int(v.get("font", 0)),
            size=int(v.get("size", 0)),
            color=int(v.get("color", 0)),
            baseline_x=int(bl[0]),
            baseline_y=int(bl[1]),
            glyphs=glyphs,
            text=str(v.get("text", "")),
            letter_spacing=int(v.get("letter_spacing", 0)),
            aa_mode=AAMode(v.get("aa_mode", "grayscale")),
        )  # fmt: skip
    if t is NodeType.RASTER_PATCH:
        replace = v.get("replace", False)
        if type(replace) is not bool:
            raise ValidationError("RasterPatch replace must be boolean")
        return RasterPatch(
            rect=rect, z=z, opacity=opacity, clip=clip, tag=tag,
            asset=int(v.get("asset", 0)), replace=replace,
        )  # fmt: skip
    if t is NodeType.GROUP:
        st = v.get("style") or {}
        return Group(
            rect=rect, z=z, opacity=opacity, clip=clip, tag=tag,
            style=Style(
                font=st.get("font"),
                size=st.get("size"),
                color=st.get("color"),
                letter_spacing=st.get("letter_spacing"),
            ),
            children=[node_from_json(c) for c in v.get("children", [])],
        )  # fmt: skip
    raise ValidationError(f"unhandled node type {t}")  # pragma: no cover


def document_to_json(doc: Document, *, include_payloads: bool = True) -> dict[str, Any]:
    return {
        "format_version": doc.format_version,
        "canvas": {
            "width": doc.canvas.width,
            "height": doc.canvas.height,
            "device_pixel_ratio": doc.canvas.device_pixel_ratio,
            "background": doc.canvas.background,
            "color_space": doc.canvas.color_space,
            "composite_space": doc.canvas.composite_space,
        },
        "palette": [c.to_hex() for c in doc.palette.colors()],
        "fonts": [
            {
                k: val
                for k, val in (
                    ("family", f.family),
                    ("weight", f.weight),
                    ("italic", f.italic),
                    ("file", f.file),
                    ("hash", f.hash),
                )
                if val not in (None, False)
            }
            for f in doc.fonts
        ],
        "assets": [
            {
                "codec": a.codec,
                "width": a.width,
                "height": a.height,
                **(
                    {"payload_b64": base64.b64encode(a.payload).decode("ascii")}
                    if include_payloads and a.payload
                    else {"payload_bytes": len(a.payload)}
                ),
            }
            for a in doc.assets
        ],
        "root": node_to_json(doc.root) if doc.root is not None else None,
    }


def document_from_json(v: Any) -> Document:
    if not isinstance(v, dict):
        raise ValidationError("document must be an object")
    c = v.get("canvas") or {}
    doc = Document(
        canvas=Canvas(
            width=int(c.get("width", 0)),
            height=int(c.get("height", 0)),
            device_pixel_ratio=int(c.get("device_pixel_ratio", 1)),
            background=int(c.get("background", 0)),
            color_space=str(c.get("color_space", "srgb")),
            composite_space=str(c.get("composite_space", "srgb")),
        ),
        format_version=str(v.get("format_version", "0.1")),
    )
    # Rebuild the palette positionally: indices in the graph refer to position,
    # so intern() order must be preserved exactly.
    pal = Palette()
    for hexs in v.get("palette", []):
        pal.intern(Color.from_css(hexs))
    doc.palette = pal
    doc.fonts = [
        FontRef(
            family=str(f.get("family", "")),
            weight=int(f.get("weight", 400)),
            italic=bool(f.get("italic", False)),
            file=f.get("file"),
            hash=f.get("hash"),
        )
        for f in v.get("fonts", [])
    ]
    doc.assets = [
        AssetRef(
            codec=str(a.get("codec", "png")),
            width=int(a.get("width", 0)),
            height=int(a.get("height", 0)),
            payload=base64.b64decode(a["payload_b64"]) if "payload_b64" in a else b"",
        )
        for a in v.get("assets", [])
    ]
    root = v.get("root")
    doc.root = node_from_json(root) if root is not None else None
    return doc


def write_scg(doc: Document, path: Path, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document_to_json(doc), indent=indent, sort_keys=False) + "\n"
    )


def read_scg(path: Path, *, check: bool = True) -> Document:
    doc = document_from_json(json.loads(path.read_text()))
    if check:
        validate(doc)
    return doc


if __name__ == "__main__":
    from semanticcodec.core.color import BLACK, WHITE
    from semanticcodec.core.geometry import px_to_q2

    doc = Document(canvas=Canvas(width=200, height=100))
    w = doc.intern_color(WHITE)
    b = doc.intern_color(Color(9, 105, 218))
    k = doc.intern_color(BLACK)
    doc.canvas.background = w
    fi = doc.add_font(FontRef("Arial", 400, False, "/tmp/Arial.ttf"))
    ai = doc.add_asset(AssetRef("png", 10, 10, b"\x89PNG_fake"))

    e = Edge(px_to_q2(1), b)
    doc.root = Group(
        rect=Rect.from_px(0, 0, 200, 100),
        style=Style(font=fi, size=px_to_q2(14), color=k),
        children=[
            Fill(rect=Rect.from_px(8, 8, 100, 32), color=b, radii=Radii(*(px_to_q2(6),) * 4)),
            Border(rect=Rect.from_px(8, 8, 100, 32), top=e, right=e, bottom=e, left=e),
            Border(
                rect=Rect.from_px(8, 48, 100, 2),
                top=Edge(px_to_q2(2), k),
                right=Edge(),
                bottom=Edge(),
                left=Edge(),
            ),
            TextRun(
                rect=Rect.from_px(16, 14, 60, 20),
                font=fi,
                size=px_to_q2(14),
                color=w,
                baseline_x=px_to_q2(16),
                baseline_y=px_to_q2(28),
                glyphs=[Glyph(54), Glyph(68, px_to_q2(7.25)), Glyph(12)],
                text="Save",
                opacity=200,
                tag="button-label",
            ),
            RasterPatch(rect=Rect.from_px(120, 8, 40, 40), asset=ai, clip=True),
        ],
    )
    validate(doc)

    # Round-trip must be exact through JSON text, twice (idempotent).
    j1 = json.dumps(document_to_json(doc), sort_keys=True)
    back = document_from_json(json.loads(j1))
    validate(back)
    j2 = json.dumps(document_to_json(back), sort_keys=True)
    assert j1 == j2, "JSON round-trip is not exact"

    assert back.palette.colors() == doc.palette.colors(), "palette order must survive"
    assert back.assets[0].payload == b"\x89PNG_fake", "asset payload must survive"
    assert isinstance(back.root, Group)
    kids = back.root.children
    bt = kids[3]
    assert isinstance(bt, TextRun)
    assert bt.glyphs[1].advance == px_to_q2(7.25), "explicit advance must survive"
    assert bt.glyphs[0].advance is None, "default advance must stay default"
    assert bt.opacity == 200 and bt.tag == "button-label"
    assert isinstance(kids[1], Border) and kids[1].is_uniform(), "uniform border must survive"
    assert isinstance(kids[2], Border) and not kids[2].is_uniform(), "per-edge border must survive"
    assert kids[4].clip is True

    # File round-trip
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.scg.json"
        write_scg(doc, p)
        again = read_scg(p)
        assert json.dumps(document_to_json(again), sort_keys=True) == j1

    # Malformed input must raise, not silently produce a wrong graph
    for bad in ({}, {"type": "Nope"}, {"type": "Fill"}, {"type": "Fill", "rect": [1, 2]}):
        try:
            node_from_json(bad)
            raise AssertionError(f"should have rejected {bad!r}")
        except ValidationError:
            pass
    print("json_io self-check passed")
