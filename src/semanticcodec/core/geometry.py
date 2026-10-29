"""Geometry in q2 fixed point.

All coordinates are integers in quarter-pixel units. One q2 unit = 0.25 px.

Sizes are q2 too, not whole pixels: mixing units invites conversion bugs for
no real saving. Text-rendering measurements independently confirmed quarter-pixel is
the right precision — it beat both whole-pixel and 1/64 for text positioning.

No floats are stored anywhere in the scene graph. Floats enter only at the
rasteriser boundary, which is what keeps the format exactly round-trippable
and the renderer deterministic.
"""

from __future__ import annotations

from typing import NamedTuple

# Quarter-pixel: 4 q2 units per pixel.
Q2_SHIFT = 2
Q2_ONE = 1 << Q2_SHIFT


def px_to_q2(v: float) -> int:
    """Quantise pixels to q2. Round-half-away-from-zero for symmetry."""
    return int(v * Q2_ONE + (0.5 if v >= 0 else -0.5))


def q2_to_px(v: int) -> float:
    return v / Q2_ONE


def q2_floor_px(v: int) -> int:
    """Largest whole pixel <= v. Floor division handles negatives correctly."""
    return v >> Q2_SHIFT


def q2_ceil_px(v: int) -> int:
    return -((-v) >> Q2_SHIFT)


class Rect(NamedTuple):
    """Axis-aligned rectangle, q2 units. `x, y` may be negative (shadows,
    overflow); `w, h` may not."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_px(cls, x: float, y: float, w: float, h: float) -> Rect:
        return cls(px_to_q2(x), px_to_q2(y), px_to_q2(w), px_to_q2(h))

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def translate(self, dx: int, dy: int) -> Rect:
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def intersect(self, other: Rect) -> Rect:
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return Rect(x0, y0, 0, 0)
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def union(self, other: Rect) -> Rect:
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        x0 = min(self.x, other.x)
        y0 = min(self.y, other.y)
        return Rect(x0, y0, max(self.x1, other.x1) - x0, max(self.y1, other.y1) - y0)

    def contains(self, other: Rect) -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def to_pixel_bounds(self) -> tuple[int, int, int, int]:
        """Whole-pixel bounds covering this rect, as (x0, y0, x1, y1).

        Conservative outward rounding: a rect that partially covers a pixel
        must include that pixel or antialiased edges get clipped.
        """
        return (
            q2_floor_px(self.x),
            q2_floor_px(self.y),
            q2_ceil_px(self.x1),
            q2_ceil_px(self.y1),
        )

    def area_px(self) -> float:
        return q2_to_px(self.w) * q2_to_px(self.h)


class Radii(NamedTuple):
    """Corner radii in q2, clockwise from top-left."""

    tl: int = 0
    tr: int = 0
    br: int = 0
    bl: int = 0

    def is_zero(self) -> bool:
        return self.tl == 0 and self.tr == 0 and self.br == 0 and self.bl == 0

    def clamped_to(self, rect: Rect) -> Radii:
        """CSS-style clamp: adjacent radii cannot exceed the edge they share.

        Browsers scale all radii by a single factor f <= 1 so no edge is
        over-subscribed. Reproducing that exactly matters because we are
        matching browser output.
        """
        if self.is_zero():
            return self
        pairs = (
            (self.tl + self.tr, rect.w),
            (self.br + self.bl, rect.w),
            (self.tl + self.bl, rect.h),
            (self.tr + self.br, rect.h),
        )
        f = 1.0
        for total, extent in pairs:
            if total > extent > 0:
                f = min(f, extent / total)
        if f >= 1.0:
            return self
        return Radii(*(int(r * f) for r in self))


if __name__ == "__main__":
    # Self-check: q2 round-trips, rect algebra, CSS radius clamping.
    assert px_to_q2(1.0) == 4
    assert px_to_q2(0.25) == 1
    assert px_to_q2(-0.25) == -1
    assert q2_to_px(px_to_q2(13.25)) == 13.25
    assert q2_floor_px(5) == 1 and q2_ceil_px(5) == 2
    assert q2_floor_px(-5) == -2 and q2_ceil_px(-5) == -1

    a = Rect.from_px(0, 0, 10, 10)
    b = Rect.from_px(5, 5, 10, 10)
    assert a.intersect(b) == Rect.from_px(5, 5, 5, 5)
    assert a.union(b) == Rect.from_px(0, 0, 15, 15)
    assert a.intersect(Rect.from_px(20, 20, 1, 1)).is_empty()
    assert a.contains(Rect.from_px(1, 1, 2, 2))
    assert not a.contains(b)
    assert Rect.from_px(0.5, 0.5, 1.0, 1.0).to_pixel_bounds() == (0, 0, 2, 2)

    # 40+40 > 50 wide, so radii scale by 50/80
    clamped = Radii(*(px_to_q2(v) for v in (40, 40, 0, 0))).clamped_to(
        Rect.from_px(0, 0, 50, 100)
    )
    assert clamped.tl + clamped.tr <= px_to_q2(50), clamped
    assert Radii().clamped_to(a).is_zero()
    print("geometry self-check passed")
