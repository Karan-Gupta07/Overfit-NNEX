"""Scene graph node types.

Scope: only the nodes L0-L2 actually needs — Group, Fill, Border, TextRun,
RasterPatch. Shadow, Path, IconRef, ImageRef, Use, Layout and Repeat were
planned but are deliberately absent until an encoder stage emits them.
Every node type is a permanent format and renderer commitment; adding one
before anything produces it is a commitment bought with no evidence.

Design principles that the code enforces:
  - renderability over semantics: a node exists because it renders cheaply
  - every node is optional: removing one leaves a valid graph
  - relative coordinates: children are positioned inside their parent
  - quantised parameters: integers only, so JSON and binary agree exactly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Union

from semanticcodec.core.geometry import Radii, Rect


class NodeType(StrEnum):
    GROUP = "Group"
    FILL = "Fill"
    BORDER = "Border"
    TEXT_RUN = "TextRun"
    RASTER_PATCH = "RasterPatch"


class BorderStyle(StrEnum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"


class AAMode(StrEnum):
    """Text antialiasing mode.

    Text-rendering measurements showed that grayscale coverage with a *linear* alpha ramp
    matches headless Chromium on macOS; a fitted transfer curve was within
    0.04 of linear and made held-out residuals worse. SUBPIXEL exists because
    Windows/ClearType produces per-channel coloured glyph edges, and a
    mismatch there shows as visible fringing. Nothing renders
    SUBPIXEL yet; it is recorded so captures are not silently mislabelled.
    """

    GRAYSCALE = "grayscale"
    SUBPIXEL = "subpixel"


@dataclass(slots=True)
class Glyph:
    """One positioned glyph within a run.

    `advance` is stored explicitly and in q2 because measurements showed that
    cumulative position drift across a line is far more visible than mild shape
    mismatch. Paying a few bits per glyph to pin position is the right trade.
    `advance=None` means "use the font's default advance for this glyph id",
    which is the common case and costs one flag bit.
    """

    gid: int
    advance: int | None = None


@dataclass(slots=True)
class NodeBase:
    """Fields shared by every node. `rect` is relative to the parent origin."""

    rect: Rect
    z: int | None = None
    opacity: int = 255
    clip: bool = False
    # Debug provenance. Stripped at serialisation; never affects rendering.
    tag: str | None = None


@dataclass(slots=True)
class Fill(NodeBase):
    """Solid colour rectangle with optional corner radii."""

    color: int = 0  # palette index
    radii: Radii = field(default_factory=Radii)
    type: Literal[NodeType.FILL] = NodeType.FILL


@dataclass(slots=True)
class Edge:
    """One border edge. `width` is q2."""

    width: int = 0
    color: int = 0
    style: BorderStyle = BorderStyle.SOLID

    def is_drawn(self) -> bool:
        return self.width > 0


@dataclass(slots=True)
class Border(NodeBase):
    """Per-edge border. Order is CSS order: top, right, bottom, left."""

    top: Edge = field(default_factory=Edge)
    right: Edge = field(default_factory=Edge)
    bottom: Edge = field(default_factory=Edge)
    left: Edge = field(default_factory=Edge)
    radii: Radii = field(default_factory=Radii)
    type: Literal[NodeType.BORDER] = NodeType.BORDER

    def edges(self) -> tuple[Edge, Edge, Edge, Edge]:
        return (self.top, self.right, self.bottom, self.left)

    def is_uniform(self) -> bool:
        """All four edges identical — the overwhelmingly common case, and worth
        a single flag in the bitstream instead of four edge records."""
        e = self.edges()
        return all(x == e[0] for x in e[1:])


@dataclass(slots=True)
class TextRun(NodeBase):
    """A single rendered line of text in one style.

    Highest-value node type in the format: measurements put glyph runs at
    3.3 KiB against 110 KiB of raster for the same text, a 33x reduction.

    `baseline_x/baseline_y` are q2, relative to the parent origin — not derived
    from `rect`. The line box top is NOT the baseline (it includes leading
    above the ink), and conflating them was a real bug that cost ~2px at 10px
    text.

    `text` is retained for the searchable-archive capability (the TEXT chunk)
    and so OCR output can be checked against it. It is not used for rendering;
    `glyphs` is authoritative. Glyph mode is the default because shaping
    determinism across HarfBuzz versions cannot be assumed, and rendering must
    be bit-identical everywhere.
    """

    font: int = 0  # index into document font table
    size: int = 0  # q2
    color: int = 0  # palette index
    baseline_x: int = 0  # q2, relative to parent
    baseline_y: int = 0  # q2, relative to parent
    glyphs: list[Glyph] = field(default_factory=list)
    text: str = ""
    letter_spacing: int = 0  # q2
    aa_mode: AAMode = AAMode.GRAYSCALE
    type: Literal[NodeType.TEXT_RUN] = NodeType.TEXT_RUN


@dataclass(slots=True)
class RasterPatch(NodeBase):
    """Pixels the encoder declined to model.

    Guarantees that anything unexplainable degrades to conventional
    coding rather than to garbage. `asset` indexes the document asset table.
    """

    asset: int = 0
    # Exact straight-alpha samples, rather than source-over painting.
    replace: bool = False
    type: Literal[NodeType.RASTER_PATCH] = NodeType.RASTER_PATCH


@dataclass(slots=True)
class Style:
    """Inheritable style carried by a Group.

    Only text properties inherit; geometry never does. Inheritance is resolved
    at parse time into concrete per-node values so the renderer never walks
    ancestors — that keeps rendering cacheable, which matters because the
    encoder's RD search calls it thousands of times per image.
    """

    font: int | None = None
    size: int | None = None
    color: int | None = None
    letter_spacing: int | None = None

    def is_empty(self) -> bool:
        return (
            self.font is None
            and self.size is None
            and self.color is None
            and self.letter_spacing is None
        )

    def merged_over(self, parent: Style) -> Style:
        """Resolve self against an inherited parent style; self wins."""
        return Style(
            font=self.font if self.font is not None else parent.font,
            size=self.size if self.size is not None else parent.size,
            color=self.color if self.color is not None else parent.color,
            letter_spacing=(
                self.letter_spacing
                if self.letter_spacing is not None
                else parent.letter_spacing
            ),
        )


@dataclass(slots=True)
class Group(NodeBase):
    """Container. Children are positioned relative to this node's origin."""

    children: list[Node] = field(default_factory=list)
    style: Style = field(default_factory=Style)
    type: Literal[NodeType.GROUP] = NodeType.GROUP


Node = Union[Group, Fill, Border, TextRun, RasterPatch]  # noqa: UP007

# Paint order within a parent when z is unset: document order. Nodes with an
# explicit z sort after, ascending. Kept as a module constant so the renderer
# and the canonicaliser cannot disagree about it.
LEAF_TYPES = (NodeType.FILL, NodeType.BORDER, NodeType.TEXT_RUN, NodeType.RASTER_PATCH)


def children_of(node: Node) -> list[Node]:
    """Uniform child access. Only Group has children; the rest are leaves."""
    return node.children if isinstance(node, Group) else []


def walk(node: Node) -> list[Node]:
    """Depth-first preorder over the subtree rooted at `node`."""
    out = [node]
    for c in children_of(node):
        out.extend(walk(c))
    return out


def absolute_rect(path: list[Node]) -> Rect:
    """Absolute rect of the last node given its ancestor chain.

    Children store parent-relative coordinates, so absolute
    position requires the chain. Returned in q2.
    """
    x = y = 0
    for n in path[:-1]:
        x += n.rect.x
        y += n.rect.y
    last = path[-1].rect
    return Rect(x + last.x, y + last.y, last.w, last.h)


if __name__ == "__main__":
    from semanticcodec.core.geometry import px_to_q2

    r = Rect.from_px(0, 0, 100, 40)
    fill = Fill(rect=r, color=1, radii=Radii(*(px_to_q2(6),) * 4))
    run = TextRun(
        rect=Rect.from_px(8, 8, 60, 20),
        font=0,
        size=px_to_q2(14),
        color=2,
        baseline_x=px_to_q2(8),
        baseline_y=px_to_q2(22),
        glyphs=[Glyph(54), Glyph(68, px_to_q2(7.5))],
        text="Save",
    )
    g = Group(rect=r, children=[fill, run])

    assert [n.type for n in walk(g)] == [
        NodeType.GROUP,
        NodeType.FILL,
        NodeType.TEXT_RUN,
    ]
    assert children_of(fill) == []
    # relative -> absolute must accumulate ancestor origins
    outer = Group(rect=Rect.from_px(40, 120, 200, 80), children=[g])
    abs_run = absolute_rect([outer, g, run])
    assert abs_run.x == px_to_q2(48), abs_run
    assert abs_run.y == px_to_q2(128), abs_run
    assert abs_run.w == px_to_q2(60)

    # style inheritance: child overrides win, unset fields fall through
    parent = Style(font=0, size=px_to_q2(14), color=1)
    child = Style(color=7)
    m = child.merged_over(parent)
    assert (m.font, m.size, m.color) == (0, px_to_q2(14), 7), m
    assert Style().is_empty() and not child.is_empty()

    b = Border(rect=r, top=Edge(4, 3), right=Edge(4, 3), bottom=Edge(4, 3), left=Edge(4, 3))
    assert b.is_uniform()
    assert not Border(rect=r, top=Edge(4, 3)).is_uniform()
    assert Edge().is_drawn() is False
    print("nodes self-check passed")
