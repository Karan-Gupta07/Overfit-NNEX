"""Bounded, exact spatial mixtures, selected on complete serialized file size.

The decoder needs no partition algorithm: the result is ordinary RasterPatch
nodes. Cuts are proposals only; they never determine reconstruction fidelity.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from semanticcodec.core.geometry import Rect
from semanticcodec.core.scene.document import AssetRef, Document
from semanticcodec.core.scene.nodes import Group, Node, RasterPatch
from semanticcodec.encode.pipeline import EncodeOptions, EncodeReport, encode_image
from semanticcodec.format.serialize import serialize

MIN_SIDE = 128


def propose_cuts(rgb: np.ndarray) -> list[tuple[int, int]]:
    """One variance-reducing boundary per axis, with a minimum child size."""
    h, w = rgb.shape[:2]
    values = rgb[..., :3].astype(np.int16)
    cuts = []
    for axis, length in ((0, h), (1, w)):
        if length < 2 * MIN_SIDE:
            continue
        differences = np.abs(values - np.roll(values, 1, axis=1 - axis)).mean(axis=2)
        profile = differences.mean(axis=1 - axis)
        prefix = np.cumsum(profile)
        positions = np.arange(MIN_SIDE, length - MIN_SIDE + 1)
        delta = (prefix[positions - 1] / positions
                 - (prefix[-1] - prefix[positions - 1]) / (length - positions))
        gain = delta**2 * positions * (length - positions) / length
        position = int(positions[int(np.argmax(gain))])
        position = max(MIN_SIDE, min(length - MIN_SIDE, 16 * round(position / 16)))
        cuts.append((axis, position))
    return cuts


@dataclass
class _Region:
    x: int
    y: int
    asset: AssetRef
    report: EncodeReport


def partition_document(rgba: np.ndarray, opts: EncodeOptions, seed: Document,
                       seed_blob: bytes, seed_report: EncodeReport,
                       ) -> tuple[bytes, Document, list[EncodeReport]]:
    """At most four output regions and 21 analyzed rectangles at depth two.

    Subproblem choices include graph and asset overhead. The root candidate
    must still beat the original complete file, including its conventional
    floor; this heuristic search is not a claim of a globally optimal split.
    """
    h, w = rgba.shape[:2]
    child_options = replace(opts, partition_depth=0, floor_guarantee=True)

    def document(regions: list[_Region]) -> Document:
        doc = Document(canvas=copy.deepcopy(seed.canvas), palette=copy.deepcopy(seed.palette))
        children: list[Node] = []
        for region in regions:
            asset = region.asset
            index = doc.add_asset(asset)
            children.append(RasterPatch(
                rect=Rect.from_px(region.x, region.y, asset.width, asset.height),
                asset=index,
            ))
        doc.root = Group(rect=Rect.from_px(0, 0, w, h), children=children)
        return doc

    memo: dict[tuple[int, int, int, int], tuple[AssetRef, EncodeReport]] = {}

    def search(x: int, y: int, sw: int, sh: int, depth: int,
               ) -> tuple[list[_Region], Document, bytes]:
        key = (x, y, sw, sh)
        block = rgba[y:y + sh, x:x + sw]
        if key == (0, 0, w, h):
            regions = [_Region(x, y, seed.assets[0], seed_report)]
            best_doc, best_blob = seed, seed_blob
        else:
            if key not in memo:
                _, child_doc, child_report = encode_image(block, child_options)
                memo[key] = child_doc.assets[0], child_report
            asset, report = memo[key]
            regions = [_Region(x, y, asset, report)]
            best_doc = document(regions)
            best_blob = serialize(best_doc, lossless=opts.lossless)
        if depth:
            for axis, position in propose_cuts(block):
                if axis == 0:
                    rectangles = ((x, y, sw, position),
                                  (x, y + position, sw, sh - position))
                else:
                    rectangles = ((x, y, position, sh),
                                  (x + position, y, sw - position, sh))
                proposal = []
                for rectangle in rectangles:
                    proposal.extend(search(*rectangle, depth - 1)[0])
                proposed_doc = document(proposal)
                proposed_blob = serialize(proposed_doc, lossless=opts.lossless)
                if len(proposed_blob) < len(best_blob):
                    regions, best_doc, best_blob = proposal, proposed_doc, proposed_blob
        return regions, best_doc, best_blob

    regions, doc, blob = search(0, 0, w, h, opts.partition_depth)
    return blob, doc, [r.report for r in regions]
