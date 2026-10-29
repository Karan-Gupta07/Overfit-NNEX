"""Document container and validation.

Validation runs on BOTH sides: in the encoder before serialising, and in the
decoder after parsing. The encoder-side check catches optimiser bugs where they
are cheap to diagnose; the decoder-side check is a trust boundary, since a
bitstream is untrusted input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from semanticcodec.core.color import Color, Palette
from semanticcodec.core.geometry import Q2_ONE, Rect, q2_to_px
from semanticcodec.core.scene.nodes import (
    Group,
    Node,
    NodeType,
    RasterPatch,
    Style,
    TextRun,
    children_of,
)

FORMAT_VERSION = "0.1"

# Nodes may legitimately overflow the canvas: shadows, focus rings, and text
# that the browser clipped. Allow a margin rather than rejecting valid graphs.
OVERFLOW_MARGIN_PX = 64

# Expansion-bomb guards. Not yet reachable via Use/defs, but the
# caps belong with validation from the start rather than bolted on later.
MAX_DEPTH = 64
MAX_NODES = 500_000


@dataclass(slots=True)
class FontRef:
    """A font the document references.

    `file` is a local path during research. It becomes a content-addressed
    library reference once a component library exists; the
    hash field is already here so bitstreams written now record what they used.
    """

    family: str
    weight: int = 400
    italic: bool = False
    file: str | None = None
    hash: str | None = None


@dataclass(slots=True)
class AssetRef:
    """An embedded raster payload (stored in the ASET chunk)."""

    codec: str  # "png" | "webp-lossless" | "webp-lossy" | "avif-lossy"
    width: int  # whole pixels
    height: int
    payload: bytes = b""


@dataclass(slots=True)
class Canvas:
    width: int  # whole pixels, logical (pre-DPR)
    height: int
    device_pixel_ratio: int = 1
    background: int = 0  # palette index
    color_space: str = "srgb"
    # Recorded explicitly rather than left implicit: we composite in sRGB to
    # match browsers, and a future reader must not have to guess.
    composite_space: str = "srgb"

    def rect(self) -> Rect:
        return Rect.from_px(0, 0, self.width, self.height)


@dataclass(slots=True)
class Document:
    canvas: Canvas
    palette: Palette = field(default_factory=Palette)
    fonts: list[FontRef] = field(default_factory=list)
    assets: list[AssetRef] = field(default_factory=list)
    root: Node | None = None
    format_version: str = FORMAT_VERSION

    def intern_color(self, c: Color) -> int:
        return self.palette.intern(c)

    def add_font(self, f: FontRef) -> int:
        for i, existing in enumerate(self.fonts):
            if (existing.family, existing.weight, existing.italic) == (
                f.family,
                f.weight,
                f.italic,
            ):
                return i
            del existing
        self.fonts.append(f)
        return len(self.fonts) - 1

    def add_asset(self, a: AssetRef) -> int:
        self.assets.append(a)
        return len(self.assets) - 1

    def nodes(self) -> list[Node]:
        if self.root is None:
            return []
        out: list[Node] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(reversed(children_of(n)))
        return out


class ValidationError(ValueError):
    """Raised with a node path so failures are locatable in a large graph."""


def _fail(path: str, msg: str) -> None:
    raise ValidationError(f"{path}: {msg}")


def validate(doc: Document, *, strict_bounds: bool = True) -> None:
    """Check every scene-graph rule. Raises ValidationError on the first
    violation.

    `strict_bounds=False` skips the canvas-containment check, which is useful
    mid-encode when a candidate node is being fitted before being clipped.
    """
    if doc.format_version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
        _fail("document", f"incompatible format_version {doc.format_version!r}")
    if doc.canvas.width <= 0 or doc.canvas.height <= 0:
        _fail("canvas", f"non-positive size {doc.canvas.width}x{doc.canvas.height}")
    if doc.canvas.device_pixel_ratio < 1:
        _fail("canvas", f"device_pixel_ratio {doc.canvas.device_pixel_ratio} < 1")
    if doc.canvas.composite_space != "srgb":
        _fail("canvas", f"unsupported composite_space {doc.canvas.composite_space!r}")
    if len(doc.palette) == 0:
        _fail("palette", "empty; canvas.background cannot resolve")
    if not 0 <= doc.canvas.background < len(doc.palette):
        _fail("canvas", f"background index {doc.canvas.background} out of range")

    if doc.root is None:
        return

    canvas = doc.canvas.rect()
    margin = int(OVERFLOW_MARGIN_PX * Q2_ONE)
    allowed = Rect(
        canvas.x - margin,
        canvas.y - margin,
        canvas.w + 2 * margin,
        canvas.h + 2 * margin,
    )

    count = 0
    # (node, absolute origin x, absolute origin y, depth, path label)
    stack: list[tuple[Node, int, int, int, str]] = [(doc.root, 0, 0, 0, "root")]
    while stack:
        node, ox, oy, depth, path = stack.pop()
        count += 1
        if count > MAX_NODES:
            _fail(path, f"node count exceeds {MAX_NODES}")
        if depth > MAX_DEPTH:
            _fail(path, f"depth exceeds {MAX_DEPTH}")

        r = node.rect
        if r.w < 0 or r.h < 0:
            _fail(path, f"negative size {r.w}x{r.h}")
        if not 0 <= node.opacity <= 255:
            _fail(path, f"opacity {node.opacity} outside 0..255")

        absolute = Rect(ox + r.x, oy + r.y, r.w, r.h)
        if strict_bounds and not r.is_empty() and allowed.intersect(absolute).is_empty():
            _fail(
                path,
                f"rect {tuple(q2_to_px(v) for v in absolute)} px lies entirely "
                f"outside the canvas + {OVERFLOW_MARGIN_PX}px margin",
            )

        _validate_refs(doc, node, path)

        for i, child in enumerate(children_of(node)):
            stack.append(
                (child, absolute.x, absolute.y, depth + 1, f"{path}.{node.type.value}[{i}]")
            )


def _validate_refs(doc: Document, node: Node, path: str) -> None:
    """Every palette/font/asset index must resolve. A dangling reference means
    the decoder would render a different colour or glyph than the encoder saw,
    which is silent corruption — the worst failure available to this format."""
    np_ = len(doc.palette)

    def check_color(i: int, what: str) -> None:
        if not 0 <= i < np_:
            _fail(path, f"{what} palette index {i} out of range (palette has {np_})")

    t = node.type
    if t == NodeType.FILL:
        check_color(node.color, "fill")  # type: ignore[union-attr]
    elif t == NodeType.BORDER:
        for name, edge in zip(
            ("top", "right", "bottom", "left"),
            node.edges(),  # type: ignore[union-attr]
            strict=True,
        ):
            if edge.width < 0:
                _fail(path, f"border {name} width {edge.width} negative")
            if edge.is_drawn():
                check_color(edge.color, f"border {name}")
    elif t == NodeType.TEXT_RUN:
        assert isinstance(node, TextRun)
        check_color(node.color, "text")
        if not 0 <= node.font < len(doc.fonts):
            _fail(path, f"font index {node.font} out of range ({len(doc.fonts)} fonts)")
        if node.size <= 0:
            _fail(path, f"text size {node.size} must be positive")
        if not node.glyphs:
            _fail(path, "TextRun has no glyphs (a node that renders nothing)")
        for g in node.glyphs:
            if g.gid < 0:
                _fail(path, f"negative glyph id {g.gid}")
    elif t == NodeType.RASTER_PATCH:
        assert isinstance(node, RasterPatch)
        if type(node.replace) is not bool or (node.replace and node.opacity != 255):
            _fail(path, "replacement patches require boolean replace and full opacity")
        if not 0 <= node.asset < len(doc.assets):
            _fail(path, f"asset index {node.asset} out of range")
    elif t == NodeType.GROUP:
        assert isinstance(node, Group)
        s: Style = node.style
        if s.color is not None:
            check_color(s.color, "group style")
        if s.font is not None and not 0 <= s.font < len(doc.fonts):
            _fail(path, f"group style font index {s.font} out of range")


if __name__ == "__main__":
    from semanticcodec.core.color import BLACK, WHITE
    from semanticcodec.core.geometry import px_to_q2
    from semanticcodec.core.scene.nodes import Fill, Glyph

    doc = Document(canvas=Canvas(width=200, height=100))
    white = doc.intern_color(WHITE)
    black = doc.intern_color(BLACK)
    doc.canvas.background = white
    fi = doc.add_font(FontRef("Arial", 400, False, "/dev/null"))
    assert doc.add_font(FontRef("Arial", 400, False, "/dev/null")) == fi, "fonts must dedupe"

    doc.root = Group(
        rect=Rect.from_px(0, 0, 200, 100),
        children=[
            Fill(rect=Rect.from_px(10, 10, 80, 30), color=white),
            TextRun(
                rect=Rect.from_px(10, 10, 80, 20),
                font=fi,
                size=px_to_q2(14),
                color=black,
                baseline_x=px_to_q2(10),
                baseline_y=px_to_q2(24),
                glyphs=[Glyph(1)],
                text="x",
            ),
        ],
    )
    validate(doc)
    assert len(doc.nodes()) == 3

    def kid(d: Document, i: int) -> Node:
        assert isinstance(d.root, Group)
        return d.root.children[i]

    def expect_fail(mutate: Callable[[Document], None], why: str) -> None:
        import copy

        d = copy.deepcopy(doc)
        mutate(d)
        try:
            validate(d)
        except ValidationError:
            return
        raise AssertionError(f"validation should have rejected: {why}")

    expect_fail(lambda d: setattr(kid(d, 0), "color", 99), "bad palette index")
    expect_fail(lambda d: setattr(kid(d, 1), "font", 5), "bad font index")
    expect_fail(lambda d: setattr(kid(d, 1), "glyphs", []), "empty glyph run")
    expect_fail(lambda d: setattr(kid(d, 1), "size", 0), "zero text size")
    expect_fail(lambda d: setattr(kid(d, 0), "opacity", 300), "opacity > 255")
    expect_fail(
        lambda d: setattr(kid(d, 0), "rect", Rect(0, 0, -4, 4)), "negative size"
    )
    expect_fail(
        lambda d: setattr(kid(d, 0), "rect", Rect.from_px(9000, 9000, 10, 10)),
        "far outside canvas",
    )
    expect_fail(lambda d: setattr(d.canvas, "background", 42), "bad background index")

    # a RasterPatch pointing at a nonexistent asset must be rejected
    d2 = Document(canvas=Canvas(width=10, height=10))
    d2.intern_color(WHITE)
    d2.root = RasterPatch(rect=Rect.from_px(0, 0, 10, 10), asset=0)
    try:
        validate(d2)
        raise AssertionError("should have rejected dangling asset ref")
    except ValidationError:
        pass

    print("document self-check passed")
