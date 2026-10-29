"""Colour, palette, and compositing.

Compositing is in **integer sRGB, not linear light**. That is "less correct"
colorimetrically and it is the right choice here: browsers composite in sRGB
space and we are reproducing browser output. Matching the source rasteriser
beats matching physics — a glyph edge blended in the wrong space is wrong in
exactly the places that cost us the most bits (measured on rendered text).
"""

from __future__ import annotations

import re
from typing import NamedTuple

_CSS_NUM = re.compile(r"[-+]?[\d.]+")


class Color(NamedTuple):
    """8-bit straight (non-premultiplied) RGBA."""

    r: int
    g: int
    b: int
    a: int = 255

    @classmethod
    def from_css(cls, s: str) -> Color:
        """Parse `rgb(r, g, b)` / `rgba(r, g, b, a)` / `#rgb` / `#rrggbb`.

        Raises on anything else: silently defaulting an unparseable colour to
        black produces reconstructions that look plausibly wrong, which is the
        worst failure mode for a codec.
        """
        s = s.strip()
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                return cls(*(int(c * 2, 16) for c in h))
            if len(h) == 6:
                return cls(*(int(h[i : i + 2], 16) for i in (0, 2, 4)))
            if len(h) == 8:
                return cls(*(int(h[i : i + 2], 16) for i in (0, 2, 4, 6)))
            raise ValueError(f"bad hex colour: {s!r}")
        nums = _CSS_NUM.findall(s)
        if len(nums) < 3:
            raise ValueError(f"unparseable colour: {s!r}")
        r, g, b = (max(0, min(255, int(round(float(n))))) for n in nums[:3])
        a = 255
        if len(nums) >= 4:
            a = max(0, min(255, int(round(float(nums[3]) * 255))))
        return cls(r, g, b, a)

    def is_opaque(self) -> bool:
        return self.a == 255

    def is_transparent(self) -> bool:
        return self.a == 0

    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_hex(self) -> str:
        if self.a == 255:
            return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}{self.a:02x}"


WHITE = Color(255, 255, 255)
BLACK = Color(0, 0, 0)
TRANSPARENT = Color(0, 0, 0, 0)


def blend_over(src: Color, dst: Color) -> Color:
    """Source-over composite of two straight-alpha colours, integer sRGB.

    Used for scalar colour maths (e.g. solving for a text foreground given a
    known background). Bulk pixel compositing lives in the renderer.
    """
    if src.a == 255 or dst.a == 0:
        return src
    if src.a == 0:
        return dst
    sa = src.a / 255.0
    da = dst.a / 255.0
    out_a = sa + da * (1.0 - sa)
    if out_a == 0.0:
        return TRANSPARENT

    def ch(sc: int, dc: int) -> int:
        v = (sc * sa + dc * da * (1.0 - sa)) / out_a
        return max(0, min(255, int(round(v))))

    return Color(
        ch(src.r, dst.r), ch(src.g, dst.g), ch(src.b, dst.b), int(round(out_a * 255))
    )


class Palette:
    """Ordered colour table. Nodes reference colours by index.

    An index into a short table costs 2-6 bits where an inline
    RGB costs 24+. UI images typically have 10-100 distinct colours, so this
    is one of the cheapest wins available.
    """

    __slots__ = ("_colors", "_index", "_counts")

    def __init__(self, colors: list[Color] | None = None) -> None:
        self._colors: list[Color] = []
        self._index: dict[Color, int] = {}
        self._counts: list[int] = []
        for c in colors or []:
            self.intern(c)

    def intern(self, color: Color) -> int:
        """Return the index for `color`, adding it if absent."""
        i = self._index.get(color)
        if i is None:
            i = len(self._colors)
            self._colors.append(color)
            self._index[color] = i
            self._counts.append(0)
        self._counts[i] += 1
        return i

    def __getitem__(self, i: int) -> Color:
        return self._colors[i]

    def __len__(self) -> int:
        return len(self._colors)

    def __contains__(self, color: Color) -> bool:
        return color in self._index

    def colors(self) -> list[Color]:
        return list(self._colors)

    def usage(self, i: int) -> int:
        return self._counts[i]

    def canonical(self) -> tuple[Palette, list[int]]:
        """Reorder by descending usage; returns (new_palette, old->new map).

        Canonical form requires frequency ordering so the entropy
        coder assigns short codes to common colours, and so two encoders that
        produce identical pixels produce identical bytes.
        """
        order = sorted(
            range(len(self._colors)), key=lambda i: (-self._counts[i], self._colors[i])
        )
        new = Palette()
        remap = [0] * len(self._colors)
        for new_i, old_i in enumerate(order):
            new._colors.append(self._colors[old_i])
            new._index[self._colors[old_i]] = new_i
            new._counts.append(self._counts[old_i])
            remap[old_i] = new_i
        return new, remap


if __name__ == "__main__":
    assert Color.from_css("rgb(1, 2, 3)") == Color(1, 2, 3, 255)
    assert Color.from_css("rgba(1, 2, 3, 0.5)").a == 128
    assert Color.from_css("rgba(0, 0, 0, 0)").is_transparent()
    assert Color.from_css("#fff") == WHITE
    assert Color.from_css("#1f2328") == Color(31, 35, 40)
    assert Color.from_css("  #1F2328  ").to_hex() == "#1f2328"
    try:
        Color.from_css("chartreuse")
        raise AssertionError("should have rejected a named colour")
    except ValueError:
        pass

    # opaque source replaces destination; transparent source is a no-op
    assert blend_over(Color(10, 20, 30), WHITE) == Color(10, 20, 30)
    assert blend_over(TRANSPARENT, WHITE) == WHITE
    # half-alpha black over white lands mid-grey
    mid = blend_over(Color(0, 0, 0, 128), WHITE)
    assert 125 <= mid.r <= 129, mid

    p = Palette()
    a = p.intern(WHITE)
    p.intern(BLACK)
    p.intern(BLACK)
    p.intern(BLACK)
    assert len(p) == 2 and p[a] == WHITE
    canon, remap = p.canonical()
    assert canon[0] == BLACK, "most-used colour must sort first"
    assert remap[a] == 1
    assert canon.usage(0) == 3
    print("color self-check passed")
