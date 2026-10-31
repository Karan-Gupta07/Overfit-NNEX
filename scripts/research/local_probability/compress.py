"""Experimental best-file encoder: SCDX floor or a self-contained integer NNEX.

This is a research multiplexer, not a new .scdx asset. Its decoder recognizes
SCDX and integer NNEX v2/v3/v4/v5. Learned candidates compete on their actual
complete bytes; the encoder writes v5 (the network, optionally mixed with a
context model) by default and v4 or v3 on request.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import itertools
import json
import math
import multiprocessing
import os
import queue
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import neural_bitstream as coder
import numpy as np
from input_image import MAX_TRAIN_PIXELS, read_raster, read_rgb
from PIL import Image

from semanticcodec.core.color import Palette
from semanticcodec.core.geometry import Rect
from semanticcodec.core.scene.document import AssetRef, Canvas, Document
from semanticcodec.core.scene.nodes import Group, RasterPatch
from semanticcodec.decode.pipeline import decode_bytes
from semanticcodec.encode.pipeline import (
    EncodeOptions,
    count_colors,
    encode_image,
    estimate_background,
)
from semanticcodec.fallback.codecs import ENCODER_THREADS, jxl_available
from semanticcodec.format.container import MAGIC as SCDX_MAGIC
from semanticcodec.format.serialize import serialize

MAX_FILE_BYTES = 256 * 1024 * 1024
# Images over coder.MAX_PIXELS, up to MAX_STRIPS times that, become equal
# horizontal strips in one STRIPS_MAGIC file. Each strip is a complete SCDX or
# NNEX file with its own model, or a SHARED_MAGIC part, and its first rows do
# not see the strip above.
STRIPS_MAGIC = b"NNST\x00\x01"
MAX_STRIPS = 4
_STRIPS_HEAD = struct.Struct("<2IB")
_STRIP = struct.Struct("<2I")
# A SHARED_MAGIC part is a strip coded with a donor strip's model: the donor's
# index, then the strip's v5 file less its model payload, which is the donor's
# (coder.with_height). A strip shares only if its file comes out smaller; it is
# tried where coding a screen band (screen_mixing) with the donor's model
# promises to save more than the strip's own model costs. Strip models cost
# 12 KB to 17 KB each on LPCB. The models of the two PIA13757 strips cost 2% and
# 4% more stream each than they save across strips; on STA13843, 2% to 4%.
SHARED_MAGIC = b"NNSH\x00\x01"
# Strips decode side by side while their context tables (coder.v5_table_bytes)
# fit in STRIP_MEMORY of the machine's memory: a mixed strip of a 46 MP image
# allocates 3 GB, and three at once beside other work thrashed a 24 GB machine.
STRIP_MEMORY = 0.5
# Raw RGBA samples of the largest image the strips hold, plus a header: an
# uncompressed PPM of a 46-megapixel LPCB image is 139 MB.
MAX_SOURCE_BYTES = MAX_STRIPS * coder.MAX_PIXELS * 4 + 4096
# (maximum pixels, hidden widths): weights cost less per pixel on larger images,
# but not on very compressible ones once mixed: width 48 weights were a quarter
# of the mixed PIA13779 (5.4 MP), and width 24 made it 12% smaller. Larger images
# whose residual_bytes are under COMPRESSIBLE_BYTES race COMPRESSIBLE_WIDTHS;
# the others skip a second worker's memory and time. Of the 83 LPCB images (or
# strips) over 4 MP, four had residual_bytes of 0.9 to 3.0 MB, the rest 6.2 MB up.
WIDTH_TIERS = ((1_000_000, (12, 32)), (4_000_000, (24, 48)), (coder.MAX_PIXELS, (48,)))
COMPRESSIBLE_WIDTHS = (24, 48)
COMPRESSIBLE_BYTES = 4_000_000
# Quantization-aware snapshots, in chain order, and their learning rate. The
# weight search never picked 8- or 7-bit snapshots on 20 Tecnick images (crops
# and natives), so they were dropped. With GPTQ rounding the 10-bit snapshot
# still won 8 of 72 development planes on estimates, but without it the crops
# coded 0.047% smaller and the natives 0.003% smaller, about 0.5 s and 3 s
# faster; an 11-bit snapshot instead cost 0.016% on the crops.
QAT_BITS = (12,)
QAT_LR = 0.002
# Fused training sees every pixel of images up to TRAIN_PIXELS (binary16
# features, about 0.5 KB of device memory per pixel); PyTorch training sees
# MAX_TRAIN_PIXELS. Large images train for fewer epochs so that no model
# sees more than MAX_SAMPLES training samples.
TRAIN_PIXELS = 8_388_608
MAX_SAMPLES = 51_200 * 8192
BATCH = 8192
# Fused training of images up to SMALL_IMAGE pixels: at equal samples, half
# the batch (twice the steps) coded development crops 0.8% smaller.
SMALL_IMAGE = 1_000_000
SMALL_BATCH = 4096
# Small images also clip each plane's gradient more loosely (the half batch's
# noisier gradients hit the default clip on a third or more of early steps)
# and prune first-layer inputs, which cost more than they predict when weights
# are a large share of the file. Neither helped 2400 by 2400 natives.
MAX_NORM = 10.0
SMALL_MAX_NORM = 30.0
# Width racing: parallel widths meet after 1/RACE_FRACTION of their epochs, and
# only widths whose race score (neural_probability_probe.race_score) is within
# RACE_MARGIN of the best keep training. The wider net gains most late in
# training: from a 32nd of training to the end, width 32 gained 1.7% to 5.4% on
# width 12 on the development crops. It trailed by 1.6% to 2.4% on cards_a
# and screws, which it won, 2.9% to 3.1% on cushions (a tie), and 3.8% to
# 9.9% on the other nine, which it lost, in each of three runs.
RACE_FRACTION = 32
RACE_MARGIN = 0.03
# Mixed, weights were a quarter of PIA13779 at width 48, and width 24 won by
# 12%; at a 32nd of training, the race had dropped it.
RACE_KEEP_SHARE = 0.1
RACE_TIMEOUT = 3600.0
# The floor waits until every worker has its features on the GPU: building them
# is CPU-bound, while the floor has all of training to finish beside it.
DEFER_FLOOR = True
# The full floor (PNG, WebP -z 9 and JPEG XL effort 9 beside pixel-derived
# symbols) takes about 16 s on all cores at 2400 by 2400, and slows the GPU
# training beside it. With a parallel fused encode, the floor is first only
# JPEG XL at SCREEN_EFFORT (0.9 s there), and the full floor runs unless the
# learned models' first estimates (race scores after 1/RACE_FRACTION of
# training) are at least FLOOR_SCREEN smaller than that. Those estimates came
# out 10% to 19% above the final files on the development crops, and effort 9
# saves 3% to 4% over effort 7 on photographs.
SCREEN_FLOOR = True
SCREEN_EFFORT = 7
FLOOR_SCREEN = 0.1
# Images with fewer distinct sampled colours (count_colors) always get the full
# floor: on graphics, WebP and the symbol path can beat JPEG XL by far. Every
# development photograph has over 14,000.
SCREEN_COLORS = 4096
# Envelope 5 codes learned models as v5 files of the network alone (bounded
# means, V5_STREAMS row streams). The best learned model then mixes with the
# context model (mixing_layouts) if that saves at least MIX_GAIN bytes per
# pixel, since mixing decodes several times slower. Images over SCREEN_MIX
# pixels first code a centred band of 1/MIX_CROP of their rows both ways, and
# mix only if mixing saves MIX_SCREEN bytes per pixel of the band.
V5_STREAMS = coder.DEFAULT_STREAMS
MIX_GAIN = 0.002
# The model that mixes comes from a weight search that counts only MIX_ABSORB
# of each network stream bit: the context model absorbs most of what coarser
# weights cost the network. On six PIA images 0.3 saved 0.10% of the mixed
# files (13785: 0.5 and 0.15 saved less).
MIX_ABSORB = 0.3
SCREEN_MIX = 2_000_000
MIX_CROP = 32
MIX_SCREEN = 0.001
# The band's bytes are those of coding it after SCREEN_LEAD rows less those of
# the lead alone: a band's first rows lack the rows above, which costs the
# network far more than mixing. Without the lead, mixing saved 2.4% of a band
# of olympus_xz1_01 but 0.07% of the whole photo; with it, 0.08%. Mixing each
# file's own network on five LPCB images and two strips, where mixing changed
# the whole stream by -0.14% to +0.07%, the screen said -0.10% to +0.16%; it
# said 2.2% and 4.2% where mixing saved 1.6% and 5.8%. The network alone is the
# best file's: on the lower strip of PIA13757 the mixing model alone was already
# 2.3% smaller than that, and mixing it saved only 0.1% more.
SCREEN_LEAD = 32
# Images up to BOTH_GATES pixels code ungated and gated mixing and keep the
# smaller; above it the gate costs little and saves much decoding time. Mixing
# codes a thread per plane and band, so more bands decode faster, but each
# band's model learns alone. Images up to SCREEN_MIX pixels (over BOTH_GATES)
# mix in one band and in two and keep the smaller: of the seven such LPCB images, two bands
# saved 7.0 KB on five and lost on two. Larger ones screen in one band (two
# over MIX_BANDS_FROM pixels) and mix in one more unless mixing saved BAND_GAIN
# of the network's bytes there. On LPCB the extra band decoded 14% to 45%
# faster; it cost under 0.2% (and saved 0.21% on PIA13803) where screens saved
# under 10%, but 0.8% on PIA13810 (21%) and 0.6% on PIA13915 (33%).
BOTH_GATES = 1_000_000
MIX_BANDS_FROM = 8_000_000
BAND_GAIN = 0.1
# R and B planes that nearly repeat G (order-0 entropy of their difference from
# G under GRAY_BITS per sample, on every GRAY_STEP-th row and column) also code
# with their networks cut to a constant (coder.constant_planes), and the
# smaller file wins. On grey PIA images such networks cost more than they
# save: 13943 7.9 KB, 13856 4.6 KB, 12813 4.0 KB and 13857 2.5 KB smaller.
GRAY_BITS = 1.0
GRAY_STEP = 2


class Race:
    """Checkpoint rendezvous for models training in parallel workers."""

    def __init__(self, manager, count: int, margin: float = RACE_MARGIN):
        self.scores = manager.dict()
        self.barrier = manager.Barrier(count)
        self.margin = margin

    @staticmethod
    def checkpoint(epochs: int) -> int:
        return max(1, epochs // RACE_FRACTION)

    def report(
        self, hidden: int, feature_set: int, score: float, absorbed: float | None = None
    ) -> bool:
        """Whether to keep training: score within margin of the best, or for mixing, see below.

        absorbed scores the model for mixing, counting MIX_ABSORB of its bits.
        Every width keeps training while the best absorbed score's weights are
        at least RACE_KEEP_SHARE of it: early scores do not tell which width
        wins once mixed there.
        """
        self.scores[(hidden, feature_set)] = (score, absorbed)
        try:
            self.barrier.wait(RACE_TIMEOUT)
        except threading.BrokenBarrierError:
            # A worker failed or stalled: keep training rather than guess.
            return True
        rows = list(self.scores.values())
        if score <= min(row[0] for row in rows) * (1 + self.margin):
            return True
        mixed = [row for row in rows if row[1] is not None]
        if absorbed is None or not mixed:
            return False
        plain, best = min(mixed, key=lambda row: row[1])
        weights = plain - (plain - best) / (1 - MIX_ABSORB)
        return absorbed <= best * (1 + self.margin) or weights >= RACE_KEEP_SHARE * best

    def abort(self) -> None:
        self.barrier.abort()


def fast_floor(rgba: np.ndarray) -> bytes:
    """The floor's SCDX document with only a JPEG XL asset at SCREEN_EFFORT (opaque only)."""
    import imagecodecs

    h, w = rgba.shape[:2]
    payload = imagecodecs.jpegxl_encode(
        np.ascontiguousarray(rgba[..., :3]),
        lossless=True,
        effort=SCREEN_EFFORT,
        numthreads=ENCODER_THREADS,
        usecontainer=False,
    )
    doc = Document(canvas=Canvas(width=w, height=h))
    doc.palette = Palette([estimate_background(rgba)])
    doc.canvas.background = 0
    asset = doc.add_asset(AssetRef(codec="jxl-lossless", width=w, height=h, payload=bytes(payload)))
    rect = Rect.from_px(0, 0, w, h)
    doc.root = Group(rect=rect, children=[RasterPatch(rect=rect, asset=asset, tag="floor")])
    return serialize(doc, lossless=True)


def residual_bytes(rgb: np.ndarray, step: int = 4) -> float:
    """Order-0 entropy of MED (LOCO-I) residuals on every step-th row, in bytes of the image."""
    h, w = rgb.shape[:2]
    rows = np.arange(1, h, step)
    here = rgb[rows].astype(np.int16)
    above = rgb[rows - 1].astype(np.int16)
    a, b, c, x = here[:, :-1], above[:, 1:], above[:, :-1], here[:, 1:]
    high, low = np.maximum(a, b), np.minimum(a, b)
    prediction = np.where(c >= high, low, np.where(c <= low, high, a + b - c))
    residuals = ((x - prediction) & 255).astype(np.uint8)
    bits = 0.0
    for plane in range(residuals.shape[2]):
        counts = np.bincount(residuals[..., plane].ravel(), minlength=256)
        p = counts[counts > 0] / max(1, residuals[..., plane].size)
        bits -= float((p * np.log2(p)).sum())
    return bits * h * w / 8


def floor_screened(estimates: list[float], fast_bytes: int, colors: int) -> bool:
    """Whether the fast floor stands in for the full floor (see FLOOR_SCREEN)."""
    return (
        bool(estimates)
        and colors >= SCREEN_COLORS
        and min(estimates) <= fast_bytes * (1 - FLOOR_SCREEN)
    )


def _await_estimates(estimates, count: int, pending) -> list[float]:
    """Up to count scores from the estimates queue; fewer once any worker has ended."""
    scores: list[float] = []
    while len(scores) < count:
        try:
            scores.append(estimates.get(timeout=0.05))
        except queue.Empty:
            if any(item.ready() for item in pending):
                break
    while len(scores) < count:
        try:
            scores.append(estimates.get_nowait())
        except queue.Empty:
            break
    return scores


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def strip_rows(width: int, height: int) -> int:
    """Rows per strip of an image over coder.MAX_PIXELS; the last strip may be shorter."""
    count = math.ceil(width * height / coder.MAX_PIXELS)
    while count <= MAX_STRIPS and math.ceil(height / count) * width > coder.MAX_PIXELS:
        count += 1
    rows = math.ceil(height / count)
    if count > MAX_STRIPS or math.ceil(height / rows) > MAX_STRIPS:
        raise ValueError("research training pixel budget exceeded")
    return rows


def pack_strips(width: int, parts: list[tuple[int, bytes]]) -> bytes:
    """One file of (rows, complete file) strips, top to bottom."""
    height = sum(rows for rows, _ in parts)
    return (
        STRIPS_MAGIC
        + _STRIPS_HEAD.pack(width, height, len(parts))
        + b"".join(_STRIP.pack(rows, len(part)) for rows, part in parts)
        + b"".join(part for _, part in parts)
    )


def _strip_files(blob: bytes) -> tuple[int, list[tuple[int, bytes]]]:
    """(width, [(rows, complete file)]) of a STRIPS_MAGIC file, top to bottom."""
    offset = len(STRIPS_MAGIC) + _STRIPS_HEAD.size
    if len(blob) < offset:
        raise ValueError("truncated strip file")
    width, height, count = _STRIPS_HEAD.unpack_from(blob, len(STRIPS_MAGIC))
    if (
        not 2 <= count <= MAX_STRIPS
        or width == 0
        or width * height > MAX_STRIPS * coder.MAX_PIXELS
        or len(blob) < offset + count * _STRIP.size
    ):
        raise ValueError("strip file bounds")
    table = [_STRIP.unpack_from(blob, offset + k * _STRIP.size) for k in range(count)]
    offset += count * _STRIP.size
    if (
        any(rows == 0 for rows, _ in table)
        or sum(rows for rows, _ in table) != height
        or offset + sum(length for _, length in table) != len(blob)
    ):
        raise ValueError("strip file bounds")
    parts = []
    for _, length in table:
        part = blob[offset : offset + length]
        offset += length
        if part.startswith(STRIPS_MAGIC):
            raise ValueError("nested strip file")
        parts.append(part)
    files = [
        (rows, _unshare(part, parts, rows) if part.startswith(SHARED_MAGIC) else part)
        for (rows, _), part in zip(table, parts, strict=True)
    ]
    return width, files


def _decode_strips(width: int, files: list[tuple[int, bytes]]) -> list[np.ndarray]:
    if not files:
        return []
    # Strips are independent files, and a mixed strip decodes on only a few
    # threads (one per plane and band), so strips decode side by side.
    with ThreadPoolExecutor(strip_decoders([file for _, file in files])) as pool:
        strips = list(pool.map(decompress, [file for _, file in files]))
    if any(rgba.shape != (rows, width, 4) for (rows, _), rgba in zip(files, strips, strict=True)):
        raise ValueError("strip shape")
    return strips


def _decompress_strips(blob: bytes) -> np.ndarray:
    return np.concatenate(_decode_strips(*_strip_files(blob)), axis=0)


def check_strips(blob: bytes, rgba: np.ndarray) -> int:
    """Raises unless decompress(blob) is rgba; returns how many strips it decoded.

    Strips are read as decompress reads them; one whose file decoded as it was
    coded (coder.verified) against its rows of an opaque rgba is not decoded.
    """
    if len(blob) > MAX_FILE_BYTES or not blob.startswith(STRIPS_MAGIC):
        raise ValueError("expected a strip file within the byte budget")
    width, files = _strip_files(blob)
    if width != rgba.shape[1]:
        raise AssertionError("strip file width")
    pending, top = [], 0
    for rows, file in files:
        tile = rgba[top : top + rows]
        top += rows
        opaque = tile.shape[0] == rows and bool(np.all(tile[..., 3] == 255))
        if not (opaque and coder.verified(file, np.ascontiguousarray(tile[..., :3]))):
            pending.append((rows, file, tile))
    if top != rgba.shape[0]:
        raise AssertionError("strip file height")
    decoded = _decode_strips(width, [(rows, file) for rows, file, _ in pending])
    for (_, _, tile), strip in zip(pending, decoded, strict=True):
        if not np.array_equal(strip, tile):
            raise AssertionError("selected complete file failed independent RGBA decoding")
    return len(pending)


def _memory() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def strip_decoders(files: list[bytes]) -> int:
    """How many strip files decode at once (STRIP_MEMORY)."""
    tables = max(coder.v5_table_bytes(f) if f.startswith(coder.MAGIC_V5) else 0 for f in files)
    if not tables:
        return len(files)
    return max(1, min(len(files), int(STRIP_MEMORY * _memory() // tables)))


def _fitting_coders(shape: tuple[int, ...], layouts, checked: bool) -> int:
    h, w = shape[:2]
    cores = os.cpu_count() or 1
    copies = 2 if checked else 1
    threads = max(cores if kind[0] == "s" else copies * 3 * int(kind[1:]) for kind in layouts)
    tables = copies * max(coder.layout_table_bytes(w, h, kind) for kind in layouts)
    count = max(1, cores // threads)
    if tables:
        count = min(count, int(STRIP_MEMORY * _memory() // tables))
    return count


def parallel_coders(shape: tuple[int, ...], layouts, jobs: int, checked: bool = False) -> int:
    """How many of jobs v5 encodes of an image of shape in layouts run at once.

    A mixed encode runs one thread per plane and band, so several share the
    cores while their context tables fit in STRIP_MEMORY of the machine's
    memory; checked (coder.encode_v5), its decoder takes as many threads and
    tables again. A network-only encode uses every core.
    """
    return max(1, min(jobs, _fitting_coders(shape, layouts, checked)))


def checked_mixing(shape: tuple[int, ...], layouts, jobs: int) -> bool:
    """Whether jobs mixed encodes in layouts decode as they code (coder.encode_v5 checked).

    A file that decoded beside its encoder needs no final decode, which for a
    mixed file takes longer than the encode. They check only if all of them,
    with their decoders, run at once and their tables fit in memory; one may
    take more threads than there are cores: on 12 cores a three-band strip of
    STA13843 coded in 21 s and decoded in 24 s, and took 38 s checked.
    """
    return (
        jobs > 0
        and all(kind[0] != "s" for kind in layouts)
        and _fitting_coders(shape, layouts, checked=True) >= jobs
    )


def _unshare(part: bytes, parts: list[bytes], rows: int) -> bytes:
    """The complete v5 file of a SHARED_MAGIC part of a strip of rows."""
    position = len(SHARED_MAGIC) + 1
    donor = part[position - 1] if len(part) >= position else len(parts)
    rest = part[position:]
    if (
        donor >= len(parts)
        or not parts[donor].startswith(coder.MAGIC_V5)
        or not rest.startswith(coder.MAGIC_V5)
        or len(rest) < coder.HEADER_SIZE
    ):
        raise ValueError("shared strip model")
    payload = coder.with_height(coder.model_payload(parts[donor]), rows)
    return rest[: coder.HEADER_SIZE] + payload + rest[coder.HEADER_SIZE :]


def _shared_part(rgb: np.ndarray, own: bytes, index: int, donor: bytes) -> bytes:
    """rgb, whose own file is own, coded with the model of donor strip index."""
    payload = coder.with_height(coder.model_payload(donor), rgb.shape[0])
    values = coder.v5_values(donor)
    layout = coder.v5_layout(own)
    checked = checked_mixing(rgb.shape, [layout], 1)
    blob = coder.encode_v5(rgb, payload, layout=layout, values=values, checked=checked)
    body = blob[: coder.HEADER_SIZE] + blob[coder.HEADER_SIZE + len(payload) :]
    return SHARED_MAGIC + bytes([index]) + body


def _covers(values, rgb: np.ndarray) -> bool:
    if values is None:
        return True
    used = np.zeros(256, bool)
    used[values] = True
    return bool(used[rgb].all())


def share_models(
    strips: list[np.ndarray], parts: list[tuple[int, bytes]]
) -> tuple[list[tuple[int, bytes]], dict]:
    """parts with some strips coded with one donor strip's model (SHARED_MAGIC).

    Every v5 strip is a possible donor; the one whose screens promise the most
    in total donates to each strip it promises to save bytes on, if the shared
    part comes out smaller.
    """
    files = [blob for _, blob in parts]
    eligible = [k for k, blob in enumerate(files) if blob.startswith(coder.MAGIC_V5)]
    report: dict = {"screens": [], "donor": None, "shared": []}
    if len(eligible) < 2:
        return parts, report
    rgbs = {k: np.ascontiguousarray(strips[k][..., :3]) for k in eligible}
    layouts = {k: coder.v5_layout(files[k]) for k in eligible}
    values = {k: coder.v5_values(files[k]) for k in eligible}
    models = {k: coder.model_payload(files[k]) for k in eligible}
    pairs = [
        (donor, k)
        for donor in eligible
        for k in eligible
        if donor != k and _covers(values[donor], rgbs[k])
    ]
    jobs = [(k, k) for k in eligible] + pairs
    # Bands are small, so up to four code at once while their threads fit the cores.
    cores = os.cpu_count() or 1
    threads = max(cores if kind[0] == "s" else 3 * int(kind[1:]) for kind in layouts.values())
    with ThreadPoolExecutor(max(1, min(4, cores // threads))) as pool:
        bands = dict(
            zip(
                jobs,
                pool.map(
                    lambda job: band_stream_bytes(
                        rgbs[job[1]], models[job[0]], layouts[job[1]], values[job[0]]
                    ),
                    jobs,
                ),
                strict=True,
            )
        )
    gains = {}
    for donor, k in pairs:
        rows = screen_band(rgbs[k].shape[0])[2]
        scale = rgbs[k].shape[0] / rows
        gains[donor, k] = (bands[k, k] - bands[donor, k]) * scale + len(models[k])
        report["screens"].append(
            {"donor": donor, "strip": k, "estimated_saving": round(gains[donor, k])}
        )
    totals = {d: sum(max(0.0, g) for (e, _), g in gains.items() if e == d) for d in eligible}
    donor = max(totals, key=lambda d: totals[d])
    if totals[donor] <= 0:
        return parts, report
    report["donor"] = donor
    shared = list(parts)
    for (d, k), gain in gains.items():
        if d != donor or gain <= 0:
            continue
        part = _shared_part(rgbs[k], files[k], donor, files[donor])
        report["shared"].append({"strip": k, "own": len(files[k]), "shared": len(part)})
        if len(part) < len(files[k]):
            shared[k] = (parts[k][0], part)
    return shared, report


def _compress_strips(path: Path, pixels: np.ndarray, options: dict) -> tuple[bytes, dict]:
    """compress() of each strip_rows strip, packed into one STRIPS_MAGIC file."""
    start = time.perf_counter()
    height, width = pixels.shape[:2]
    rows = strip_rows(width, height)
    parts, reports, tiles = [], [], []
    with coder.temporary_directory() as directory:
        for index, top in enumerate(range(0, height, rows), 1):
            strip = np.ascontiguousarray(pixels[top : top + rows])
            # BMP is quick to write and read back; PNG keeps alpha.
            suffix = "bmp" if strip.shape[2] == 3 else "png"
            target = Path(directory) / f"strip-{index}.{suffix}"
            Image.fromarray(strip).save(target)
            blob, report = compress(target, **{**options, "check": False})
            parts.append((strip.shape[0], blob))
            reports.append(report)
            tiles.append(strip)
    mark = time.perf_counter()
    parts, sharing = share_models(tiles, parts)
    share_seconds = time.perf_counter() - mark
    best = pack_strips(width, parts)
    if pixels.shape[2] == 3:
        rgba = np.concatenate([pixels, np.full((height, width, 1), 255, np.uint8)], axis=2)
    else:
        rgba = pixels
    mark = time.perf_counter()
    decoded = check_strips(best, rgba)
    check_seconds = time.perf_counter() - mark
    strips = [
        {
            "rows": strip_height,
            "complete_file_bytes": report["complete_file_bytes"],
            "selected": report["selected"],
            "seconds": report["seconds"],
            "phase_seconds": report["phase_seconds"],
            "shared_values": report["shared_values"],
            "mixing": report["mixing"],
        }
        for (strip_height, _), report in zip(parts, reports, strict=True)
    ]
    return best, {
        "image": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
        "complete_file_bytes": len(best),
        "selected": {"format": "strips", "count": len(parts)},
        "strips": strips,
        "shared_models": sharing,
        "exact_rgba": True,
        "checked_encode": len(parts) - decoded,
        "seconds": time.perf_counter() - start,
        "phase_seconds": {
            **{f"strip-{k}": round(row["seconds"], 2) for k, row in enumerate(strips, 1)},
            "share-models": round(share_seconds, 2),
            "final-check": round(check_seconds, 2),
        },
        "warning": reports[0]["warning"],
    }


def decompress(blob: bytes) -> np.ndarray:
    if len(blob) > MAX_FILE_BYTES:
        raise ValueError("experimental file byte budget exceeded")
    if blob.startswith(STRIPS_MAGIC):
        return _decompress_strips(blob)
    if blob.startswith(SCDX_MAGIC):
        return decode_bytes(blob)[0]
    if not blob.startswith(coder.INTEGER_MAGICS):
        raise ValueError("expected SCDX, strips or integer NNEX v2/v3/v4/v5")
    rgb = coder.decode(blob)
    return np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)


def mixing_layouts(pixels: int, gain: float | None = None) -> tuple[str, ...]:
    """v5 context-mixing layouts tried for an image of this many pixels.

    gain is the share of the network's bytes that mixing saved on the screen
    band (screen_mixing in mixing_layouts(pixels)[-1]), if one was coded.
    """
    if pixels <= BOTH_GATES:
        return ("b1", "g1")
    if pixels <= SCREEN_MIX:
        return ("g1", "g2")
    bands = 2 if pixels > MIX_BANDS_FROM else 1
    return (f"g{bands + 1 if gain is not None and gain < BAND_GAIN else bands}",)


def _stream_bytes(blob: bytes) -> int:
    return len(blob) - coder.HEADER_SIZE - len(coder.model_payload(blob))


def screen_band(h: int) -> tuple[int, int, int]:
    """(top, lead, rows) of the band screens code: rows after lead rows of lead-in from top.

    The band is centred, and its first row is even, keeping the network's
    parity features.
    """
    rows = max(1, h // MIX_CROP)
    lead = min(SCREEN_LEAD, (h - rows) // 2 * 2)
    return (h - rows - lead) // 2 // 2 * 2, lead, rows


def _top_stream_bytes(rgb: np.ndarray, top: int, count: int, layout: str, model: bytes, values):
    """Stream bytes of count rows of rgb from top, coded with model in layout."""
    if count == 0:
        return 0
    part = np.ascontiguousarray(rgb[top : top + count])
    payload = coder.with_height(model, count)
    return _stream_bytes(coder.encode_v5(part, payload, layout=layout, values=values))


def band_stream_bytes(rgb: np.ndarray, model: bytes, layout: str, values) -> int:
    """Stream bytes of screen_band's band of rgb, less those of its lead-in alone."""
    top, lead, rows = screen_band(rgb.shape[0])
    whole = _top_stream_bytes(rgb, top, lead + rows, layout, model, values)
    return whole - _top_stream_bytes(rgb, top, lead, layout, model, values)


def screen_mixing(
    rgb: np.ndarray, network: bytes, payload: bytes, values, layout: str = "g1"
) -> dict:
    """Stream bytes of a centred band of rows: network alone, and payload mixed (layout).

    The band follows up to SCREEN_LEAD rows of lead-in, whose own bytes are
    subtracted. Its first row is even, keeping the network's parity features.
    """
    top, lead, rows = screen_band(rgb.shape[0])
    counts = (lead + rows, lead)
    # The network codes its rows on one thread; mixed bands at once only while
    # their threads fit the cores (parallel_coders).
    band = (lead + rows, *rgb.shape[1:])
    with (
        ThreadPoolExecutor(2) as network_pool,
        ThreadPoolExecutor(parallel_coders(band, [layout], 2)) as mixed_pool,
    ):
        plain = network_pool.map(
            lambda count: _top_stream_bytes(rgb, top, count, "s1", network, values), counts
        )
        coded = mixed_pool.map(
            lambda count: _top_stream_bytes(rgb, top, count, layout, payload, values), counts
        )
        (whole, alone), (mixed, mixed_alone) = plain, coded
    return {
        "rows": rows,
        "lead": lead,
        "pixels": rows * rgb.shape[1],
        "network": whole - alone,
        "mixed": mixed - mixed_alone,
    }


def _mix(
    rgb: np.ndarray, network: bytes, models: list[tuple[bytes, object]], values, best: int
) -> list[tuple[bytes | None, dict]]:
    """Per (payload, fits) of models, its smallest mixed file if that saves enough over best.

    network is the payload of the best learned file, which codes the network
    alone. The models and layouts code side by side (parallel_coders).
    """
    pixels = rgb.shape[0] * rgb.shape[1]
    reports: list[dict] = [{"layouts": []} for _ in models]
    if pixels > SCREEN_MIX:
        # Screened with the image's own (banded) layout: two bands cost about 0.8%.
        with ThreadPoolExecutor(len(models)) as pool:
            screens = pool.map(
                lambda model: screen_mixing(
                    rgb, network, model[0], values, mixing_layouts(pixels)[-1]
                ),
                models,
            )
            for report, screen in zip(reports, screens, strict=True):
                report["screen"] = screen

    def saving(k: int, share: float) -> bool:
        screen = reports[k].get("screen")
        return screen is None or screen["network"] - screen["mixed"] >= share * screen["pixels"]

    def layouts(k: int) -> tuple[str, ...]:
        screen = reports[k].get("screen")
        if screen is None:
            return mixing_layouts(pixels)
        gain = (screen["network"] - screen["mixed"]) / max(1, screen["network"])
        return mixing_layouts(pixels, gain)

    jobs = [(k, kind) for k in range(len(models)) if saving(k, MIX_SCREEN) for kind in layouts(k)]
    kinds = sorted({layout for _, layout in jobs}) or list(mixing_layouts(pixels))
    # Candidates check as they code (checked_mixing) unless their screen says
    # they will not save MIX_GAIN, so the file that wins needs no final decode.
    check = checked_mixing(rgb.shape, kinds, len(jobs))
    checks = [check and saving(k, MIX_GAIN) for k, _ in jobs]
    mixed: list[bytes | None] = [None] * len(models)
    coders = parallel_coders(rgb.shape, kinds, len(jobs), checked=any(checks))
    with ThreadPoolExecutor(coders) as pool:
        blobs = pool.map(
            lambda job, checked: coder.encode_v5(
                rgb,
                models[job[0]][0],
                layout=job[1],
                values=values,
                fits=models[job[0]][1],
                checked=checked,
            ),
            jobs,
            checks,
        )
        for (k, layout), checked, candidate in zip(jobs, checks, blobs, strict=True):
            row = {"layout": layout, "complete_file_bytes": len(candidate), "checked": checked}
            reports[k]["layouts"].append(row)
            if mixed[k] is None or len(candidate) < len(mixed[k]):
                mixed[k] = candidate
    return [
        (blob if blob is not None and best - len(blob) >= MIX_GAIN * pixels else None, report)
        for blob, report in zip(mixed, reports, strict=True)
    ]


def gray_planes(rgb: np.ndarray) -> tuple[int, ...]:
    """Model planes (1 R, 2 B) whose samples nearly repeat G's (see GRAY_BITS)."""
    sampled = rgb[::GRAY_STEP, ::GRAY_STEP].astype(np.int16)
    planes = []
    for plane, channel in ((1, 0), (2, 2)):
        difference = (sampled[..., channel] - sampled[..., 1]) & 255
        counts = np.bincount(difference.ravel(), minlength=256)
        p = counts[counts > 0] / max(1, difference.size)
        if -float((p * np.log2(p)).sum()) < GRAY_BITS:
            planes.append(plane)
    return tuple(planes)


def _constant_gray(
    rgb: np.ndarray, source: bytes, planes, values, fits, best: int
) -> tuple[bytes | None, dict]:
    """source coded with the networks of planes cut to a constant, if smaller than best.

    Both constants (coder.constant_planes with and without fold) code in
    source's layout. A network-only source also mixes the smaller one, and
    keeps a mixed file only if that saves MIX_GAIN bytes per pixel over the
    smaller network-only file.
    """
    pixels = rgb.shape[0] * rgb.shape[1]
    layout = coder.v5_layout(source)
    model = coder.model_payload(source)
    report: dict = {"planes": list(planes), "layouts": []}

    def code(jobs: list[tuple[bool, str]]) -> list[bytes]:
        kinds = [kind for _, kind in jobs]
        checked = checked_mixing(rgb.shape, kinds, len(jobs))
        coders = parallel_coders(rgb.shape, kinds, len(jobs), checked=checked)
        with ThreadPoolExecutor(coders) as pool:
            blobs = list(
                pool.map(
                    lambda job: coder.encode_v5(
                        rgb,
                        payloads[job[0]],
                        layout=job[1],
                        values=values,
                        fits=fits,
                        checked=checked,
                    ),
                    jobs,
                )
            )
        for (fold, kind), blob in zip(jobs, blobs, strict=True):
            report["layouts"].append(
                {"layout": kind, "fold": fold, "complete_file_bytes": len(blob), "checked": checked}
            )
        return blobs

    payloads = {fold: coder.constant_planes(model, planes, fold) for fold in (False, True)}
    coded = dict(zip((False, True), code([(False, layout), (True, layout)]), strict=True))
    fold = min(coded, key=lambda key: len(coded[key]))
    smallest = coded[fold] if len(coded[fold]) < best else None
    if layout.startswith("s"):
        limit = min(best, len(coded[fold])) - MIX_GAIN * pixels
        for candidate in code([(fold, kind) for kind in mixing_layouts(pixels)]):
            if len(candidate) < min(limit, len(smallest) if smallest else best):
                smallest = candidate
    return smallest, report


# The uniform-grid finalist is coded only when its estimate is within this
# fraction of the searched model's. In 47 distinct Tecnick encodes that coded
# both, the uniform model never won (smallest estimated gap 0.094%, actual
# 0.091%); below an estimated 0.5% the two gaps differed by at most 0.018%.
UNIFORM_MARGIN = 0.0005


def finalist_encoder(envelope: int, fits=None, values=None):
    """encode(pixels, model) for learned candidates of an envelope.

    fits names precomputed local fits of the training pixels, reused by v4 and
    v5 encodes; v5 takes the shared values the training pixels were packed with.
    """
    if envelope == 5:
        layout = f"s{V5_STREAMS}"
        return functools.partial(coder.encode_v5, layout=layout, values=values, fits=fits)
    return functools.partial(coder.encode_v4, fits=fits) if envelope == 4 else coder.encode_v3


def training_pixels(snapshot: Path, values=None) -> np.ndarray:
    """The image a snapshot of (possibly packed) training pixels came from."""
    pixels = read_rgb(snapshot, max_pixels=coder.MAX_PIXELS)
    return pixels if values is None else np.asarray(values, np.uint8)[pixels]


def _code_finalists(
    snapshot: Path, work: Path, search: dict, envelope: int, fits: str | None = None, values=None
) -> dict:
    """Actual files for the searched model and, if close, the best uniform-grid model.

    The model searched for mixing also codes alone. fits and values are as for
    finalist_encoder. The final whole-file check decodes whichever file wins,
    so finalists are not decoded here.
    """
    pixels = training_pixels(snapshot, values)
    encode = finalist_encoder(envelope, fits, values)
    finalists = [("searched", "selected_file", "estimated_bytes", search["labels"])]
    uniform = search["uniform_label"]
    if search["uniform_estimated_bytes"] <= search["estimated_bytes"] * (1 + UNIFORM_MARGIN):
        finalists.append((uniform, "uniform_file", "uniform_estimated_bytes", [uniform] * 3))
    if search.get("mixing_file"):
        labels = search["mixing_labels"]
        finalists.append(("mixing", "mixing_file", "mixing_estimated_bytes", labels))
    candidates, coded = [], {}
    for label, key, estimate, weights in finalists:
        model = Path(search[key]).read_bytes()
        if model in coded:
            continue
        blob = encode(pixels, model)
        target = work / f"{label}.nnexp"
        target.write_bytes(blob)
        coded[model] = target
        candidates.append(
            {
                "weights": weights,
                "bits": int(label.rsplit("-q", 1)[1]) if key == "uniform_file" else None,
                "envelope_version": envelope,
                "model_bytes": len(model),
                "complete_file_bytes": len(blob),
                "estimated_file_bytes": round(search[estimate]),
                "model_sha256": hashlib.sha256(model).hexdigest(),
                "file_sha256": hashlib.sha256(blob).hexdigest(),
                "file": str(target),
            }
        )
    selected = min(candidates, key=lambda row: row["complete_file_bytes"])
    return {"selected": selected, "candidates": candidates}


def learn(job: dict) -> dict:
    """Train one model and choose its weight grids."""
    import neural_probability_probe as trainer
    import torch
    from quantize_model import run as precision_search

    torch.set_num_threads(4)
    work = Path(job["work"])
    work.mkdir()
    snapshot = Path(job["snapshot"])
    mark = time.perf_counter()
    try:
        trained = trainer.run(
            snapshot,
            argparse.Namespace(**job["training"], race=job.get("race"), out=work / "model.json"),
        )
    except trainer.Dropped:
        return {"dropped": True, "train_seconds": time.perf_counter() - mark}
    except BaseException:
        # Release the other workers instead of leaving them at the barrier.
        if job.get("race") is not None:
            job["race"].abort()
        raise
    train_seconds = time.perf_counter() - mark
    mark = time.perf_counter()
    values = None if job.get("values") is None else np.asarray(job["values"], np.uint8)
    search = trained.get("search")
    if search:
        choices = _code_finalists(
            snapshot,
            work,
            search,
            job["envelope"],
            job["training"].get("local_fits"),
            values,
        )
        if search.get("mixing_file"):
            choices["mixing_model"] = search["mixing_file"]
            choices["mixing_estimate"] = search["mixing_absorbed_bytes"]
    else:
        choices = precision_search(
            snapshot,
            Path(trained["model_v3_file"]),
            work / "precision.json",
            envelope=job["envelope"],
            values=values,
        )
        for row in choices["candidates"]:
            row["file"] = str(work / f"precision-q{row['bits']}.nnexp")
        choices["selected"]["file"] = str(work / f"precision-q{choices['selected']['bits']}.nnexp")
    return {
        "dropped": False,
        "fused": trained.get("fused", False),
        "training": trained["training"],
        "training_pixels": trained["training_pixels"],
        "epochs": trained["epochs"],
        "choices": choices,
        "train_seconds": train_seconds,
        "precision_seconds": time.perf_counter() - mark,
    }


# Set in pool workers once the parent has written the local fits; each worker
# releases _FEATURES_READY once its features are on the device, and puts its
# first estimate on _ESTIMATES for the floor screen.
_FITS_READY = None
_FEATURES_READY = None
_ESTIMATES = None


def _warm_worker(fits_ready, features_ready=None, estimates=None) -> None:
    global _FITS_READY, _FEATURES_READY, _ESTIMATES
    _FITS_READY = fits_ready
    _FEATURES_READY = features_ready
    _ESTIMATES = estimates
    # A missing module must fail in learn(): a failing initializer respawns forever.
    try:
        for name in ("neural_probability_probe", "quantize_model"):
            importlib.import_module(name)
    except ImportError:
        pass


def _learn_in_worker(job: dict) -> dict:
    # A spawned interpreter does not see the parent's module overrides.
    coder.INTEGER_BINARY = Path(job["binary"])
    coder.ROOT = Path(job["root"])
    coder.THREADS = job["threads"]
    if "local_fits" in job["training"]:
        job["training"]["local_ready"] = _FITS_READY.wait
    if _FEATURES_READY is not None:
        job["training"]["features_ready"] = _FEATURES_READY.release
    if _ESTIMATES is not None:
        job["training"]["estimate"] = _ESTIMATES.put
    return learn(job)


def _await_features(ready, count: int, pending) -> None:
    """Returns once count workers released ready, or any worker has ended."""
    for _ in range(count):
        while not ready.acquire(timeout=0.05):
            if any(item.ready() for item in pending):
                return


def training_budget(pixels: int, epochs: int, fused: bool) -> tuple[int, int, int]:
    """(training pixel budget, epochs, batch) for an image of `pixels`."""
    budget = TRAIN_PIXELS if fused else MAX_TRAIN_PIXELS
    batch = SMALL_BATCH if fused and pixels <= SMALL_IMAGE else BATCH
    return budget, max(1, min(epochs, round(MAX_SAMPLES / min(pixels, budget)))), batch


def small_image_training(batch: int) -> dict:
    small = batch == SMALL_BATCH
    return {"max_norm": SMALL_MAX_NORM if small else MAX_NORM, "prune": small}


def default_widths(pixels: int, residual: float | None = None) -> tuple[int, ...]:
    """Hidden widths tried for an image; wider nets pay off once weights are cheap.

    residual is the image's residual_bytes, needed only over WIDTH_TIERS[-2]'s pixels.
    """
    widths = next((tier for limit, tier in WIDTH_TIERS if pixels <= limit), WIDTH_TIERS[-1][1])
    if pixels > WIDTH_TIERS[-2][0] and residual is not None and residual < COMPRESSIBLE_BYTES:
        return COMPRESSIBLE_WIDTHS
    return widths


def compress(
    path: Path,
    *,
    widths=None,
    feature_sets=(4,),
    mixtures=4,
    epochs=800,
    schedule="cosine",
    lr=0.015,
    device="auto",
    conventional=True,
    hybrid=True,
    partition_depth=0,
    sequential=False,
    envelope=5,
    parallel=True,
    compile_training=False,
    fused_training=None,
    search=None,
    race=None,
    check=True,
) -> tuple[bytes, dict]:
    """Smallest complete file among the floor and every learned candidate.

    With parallel=True each learned model trains in its own spawned process
    while this process builds the floor. Every worker seeds its own RNGs exactly
    as the serial loop does, so the files are byte-identical either way.
    compile_training=True is faster on MPS but trains (slightly) different models.
    fused_training=None uses the fused Metal trainer whenever MPS is available;
    search=None then adds quantization-aware snapshots and an estimated
    per-plane weight search, coding only two finalists. widths=None picks
    default_widths for the image. Unless race=False, parallel fused
    models race (see Race): a model that keeps training is byte-identical to
    its unraced run, but a dropped width is never fully trained. check=False
    skips decoding the chosen file (strips decode their whole file instead).
    """
    options = {name: value for name, value in locals().items() if name != "path"}
    if widths is not None and not isinstance(widths, tuple):
        raise ValueError("invalid learned model budget")
    if (
        any(type(h) is not int or not 1 <= h <= 64 for h in widths or ())
        or any(type(s) is not int or s not in coder.FEATURE_SETS for s in feature_sets)
        or type(mixtures) is not int
        or not 1 <= mixtures <= 4
        or type(epochs) is not int
        or epochs < 0
        or schedule not in ("constant", "cosine")
        or type(lr) is not float
        or not math.isfinite(lr)
        or lr <= 0
        or device not in ("auto", "cpu", "mps")
        or envelope not in (3, 4, 5)
        or type(parallel) is not bool
        or type(compile_training) is not bool
        or type(check) is not bool
        or any(v is not None and type(v) is not bool for v in (fused_training, search, race))
    ):
        raise ValueError("invalid learned model budget")
    start = time.perf_counter()
    pixels = read_raster(
        path,
        max_pixels=MAX_STRIPS * coder.MAX_PIXELS,
        allow_rgba=True,
        max_bytes=MAX_SOURCE_BYTES,
    )
    if pixels.shape[0] * pixels.shape[1] > coder.MAX_PIXELS:
        return _compress_strips(path, pixels, options)
    if pixels.shape[2] == 3:
        rgba = np.concatenate([pixels, np.full((*pixels.shape[:2], 1), 255, np.uint8)], axis=2)
    else:
        rgba = pixels
    best = None
    selected = None
    candidates = []
    dropped: list[dict] = []
    phases: dict[str, float] = {}
    learned_best = None
    mixing_models: list[tuple[float, dict, bytes]] = []
    mixing: dict | None = None
    opaque = bool(np.all(rgba[..., 3] == 255))
    rgb = np.ascontiguousarray(rgba[..., :3])
    # Learned models train on, and v5 codes, ranks among shared sample values.
    values = coder.shared_values(rgb) if envelope == 5 and opaque else None
    learn_rgb = rgb if values is None else coder.pack_values(rgb, values)
    if widths is None:
        area = rgba.shape[0] * rgba.shape[1]
        widths = default_widths(area, residual_bytes(rgb) if area > WIDTH_TIERS[-2][0] else None)
    if fused_training is None:
        fused_training = (
            bool(widths) and device != "cpu" and not compile_training and _mps_available()
        )
    if search is None:
        search = fused_training
    learned = list(itertools.product(widths, feature_sets))
    models = learned if opaque else []
    with coder.temporary_directory() as directory:
        directory = Path(directory)
        jobs = []
        pool = manager = None
        race = race is not False and parallel and fused_training and len(models) > 1
        fits_ready = features_ready = estimates = None
        if parallel and models:
            # Workers start and import torch while the snapshot and fits are made.
            context = multiprocessing.get_context("spawn")
            fits_ready = context.Event()
            if DEFER_FLOOR and fused_training and conventional:
                features_ready = context.Semaphore(0)
            if SCREEN_FLOOR and fused_training and conventional and hybrid and jxl_available():
                estimates = context.Queue()
            pool = context.Pool(
                len(models),
                initializer=_warm_worker,
                initargs=(fits_ready, features_ready, estimates),
            )
            if race:
                manager = context.Manager()
                rendezvous = Race(manager, len(models))
        try:
            local = {}
            if models:
                # Uncompressed: workers only read it back, and PNG costs 0.3 s at 2400 by 2400.
                snapshot = directory / "source.bmp"
                Image.fromarray(learn_rgb).save(snapshot, format="BMP")
                budget, passes, batch = training_budget(
                    rgba.shape[0] * rgba.shape[1], epochs, fused_training
                )
                local = {
                    s: directory / f"local-fits-f{s}"
                    for s in sorted({s for s in feature_sets if s in coder.FITS})
                }
                for hidden, feature_set in models:
                    training = {
                        "epochs": passes,
                        "train_pixels": budget,
                        "hidden": hidden,
                        "mixtures": mixtures,
                        "batch": batch,
                        "lr": lr,
                        "schedule": schedule,
                        "device": device,
                        "causal_context": True,
                        "feature_set": feature_set,
                        "sequential": sequential,
                        "baseline": False,
                        "compile": compile_training,
                        "fused": fused_training,
                        "search": search,
                        "qat_bits": QAT_BITS,
                        "qat_epochs": max(1, passes // 10),
                        "qat_lr": QAT_LR,
                        "mixing_absorb": MIX_ABSORB if envelope == 5 else None,
                        **small_image_training(batch),
                    }
                    if feature_set in local:
                        training["local_fits"] = str(local[feature_set])
                    if estimates is not None:
                        training["estimate_at"] = Race.checkpoint(passes)
                    jobs.append(
                        {
                            "training": training,
                            "snapshot": str(snapshot),
                            "work": str(directory / f"h{hidden}-f{feature_set}"),
                            "envelope": envelope,
                            "binary": str(coder.INTEGER_BINARY),
                            "root": str(coder.ROOT),
                            "threads": coder.THREADS,
                        }
                        | ({"race": rendezvous} if race else {})
                        | ({} if values is None else {"values": values.tolist()})
                    )
            pending = [pool.apply_async(_learn_in_worker, (job,)) for job in jobs] if pool else []
            # Least-squares fits are computed once here rather than in every worker,
            # which meanwhile build the features that do not need them.
            mark = time.perf_counter()
            for feature_set, target in local.items():
                coder.local_fits(learn_rgb, feature_set).tofile(target)
            if local:
                phases["local-fit"] = time.perf_counter() - mark
            if fits_ready is not None:
                fits_ready.set()
            if conventional:
                if features_ready is not None:
                    _await_features(features_ready, len(pending), pending)
                mark = time.perf_counter()
                if estimates is not None:
                    fast = fast_floor(rgba)
                    colors = count_colors(rgba)
                    phases["floor"] = time.perf_counter() - mark
                    scores = _await_estimates(estimates, len(pending), pending)
                    if floor_screened(scores, len(fast), colors):
                        best = fast
                        selected = {"format": "scdx", "codec": "jxl-lossless", "screened": True}
                    mark = time.perf_counter()
                if best is None:
                    best, _document, report = encode_image(
                        rgba, EncodeOptions(hybrid=hybrid, partition_depth=partition_depth)
                    )
                    selected = {"format": "scdx", "codec": report.selected_codec}
                candidates.append({**selected, "complete_file_bytes": len(best)})
                phases["floor"] = phases.get("floor", 0.0) + time.perf_counter() - mark
            results = [item.get() for item in pending] if pool else [learn(job) for job in jobs]
            scores = dict(rendezvous.scores) if manager is not None else {}
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
            if manager is not None:
                manager.shutdown()
        for job, result in zip(jobs, results, strict=True):
            hidden, feature_set = job["training"]["hidden"], job["training"]["feature_set"]
            phases[f"train-h{hidden}-f{feature_set}"] = result["train_seconds"]
            score, _absorbed = scores.get((hidden, feature_set), (None, None))
            if result["dropped"]:
                dropped.append(
                    {"hidden": hidden, "feature_set": feature_set}
                    | ({} if score is None else {"race_score": round(score)})
                )
                continue
            phases[f"precision-h{hidden}-f{feature_set}"] = result["precision_seconds"]
            labels = {
                "format": f"nnex-v{envelope}",
                "hidden": hidden,
                "mixtures": mixtures,
                "feature_set": feature_set,
                "training": result["training"],
                "training_pixels": result["training_pixels"],
                "epochs": result["epochs"],
                "batch": job["training"]["batch"],
            }
            if score is not None:
                labels["race_score"] = round(score)
            for candidate in result["choices"]["candidates"]:
                candidates.append({**labels, **{k: v for k, v in candidate.items() if k != "file"}})
            chosen = result["choices"]["selected"]
            blob = Path(chosen["file"]).read_bytes()
            if learned_best is None or len(blob) < len(learned_best[0]):
                learned_best = (blob, labels)
            mixing_model = result["choices"].get("mixing_model")
            if mixing_model is not None:
                estimate = result["choices"]["mixing_estimate"]
                mixing_models.append((estimate, labels, Path(mixing_model).read_bytes()))
            if best is None or len(blob) < len(best):
                best = blob
                selected = {
                    **labels,
                    "bits": chosen["bits"],
                    "model_bytes": chosen["model_bytes"],
                }
                if "weights" in chosen:
                    selected["weights"] = chosen["weights"]
        if envelope == 5 and learned_best is not None:
            mark = time.perf_counter()
            # Every width that finished training mixes: the narrower one can win
            # once mixed (PIA13799: 4.8% smaller at width 12 than at 32), and
            # estimates with MIX_ABSORB of the bits did not tell which.
            blob, labels = learned_best
            network = coder.model_payload(blob)
            models = sorted(mixing_models, key=lambda row: row[0]) or [(0.0, labels, network)]
            mixing = {"layouts": [], "screens": []}
            smallest = None
            jobs = [(payload, local.get(labels["feature_set"])) for _, labels, payload in models]
            results = _mix(rgb, network, jobs, values, len(best))
            for (_estimate, labels, _payload), (mixed, tried) in zip(models, results, strict=True):
                for row in tried["layouts"]:
                    candidates.append({**labels, **row})
                    mixing["layouts"].append({"hidden": labels["hidden"], **row})
                if "screen" in tried:
                    mixing["screens"].append({"hidden": labels["hidden"], **tried["screen"]})
                if mixed is not None and (smallest is None or len(mixed) < len(smallest[0])):
                    smallest = (mixed, labels)
            phases["mixing"] = time.perf_counter() - mark
            if smallest is not None:
                best, labels = smallest
                selected = {
                    **labels,
                    "layout": coder.v5_layout(best),
                    "model_bytes": len(coder.model_payload(best)),
                }
            gray = gray_planes(rgb)
            if gray:
                mark = time.perf_counter()
                source, labels = smallest or learned_best
                fits = local.get(labels["feature_set"])
                cut, tried = _constant_gray(rgb, source, gray, values, fits, len(best))
                mixing["gray"] = tried
                for row in tried["layouts"]:
                    candidates.append({**labels, **row, "constant_planes": list(gray)})
                phases["gray"] = time.perf_counter() - mark
                if cut is not None:
                    best = cut
                    selected = {
                        **labels,
                        "layout": coder.v5_layout(best),
                        "model_bytes": len(coder.model_payload(best)),
                        "constant_planes": list(gray),
                    }
    if best is None:
        raise ValueError("no eligible candidate; enable the conventional floor")
    # A file that decoded as it was coded (coder.verified) is not decoded again.
    checked_encode = opaque and coder.verified(best, rgb)
    if check and not checked_encode:
        mark = time.perf_counter()
        if not np.array_equal(decompress(best), rgba):
            raise AssertionError("selected complete file failed independent RGBA decoding")
        phases["final-check"] = time.perf_counter() - mark
    return best, {
        "image": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
        "complete_file_bytes": len(best),
        "selected": selected,
        "candidates": candidates,
        "exact_rgba": check,
        "checked_encode": checked_encode,
        "learned_skipped": None
        if learned and opaque
        else ("disabled" if not learned else "nonopaque"),
        "seconds": time.perf_counter() - start,
        "learned_in_parallel": pool is not None,
        "raced": bool(race),
        "dropped_at_race": dropped,
        "compiled_training": compile_training,
        "fused_training": fused_training,
        "weight_search": search,
        "shared_values": None if values is None else len(values),
        "mixing": mixing,
        "phase_seconds": {name: round(value, 2) for name, value in phases.items()},
        "warning": (
            "Experimental multi-format tool, not a stable release. "
            "The floor is a complete wrapped SCDX, not an unwrapped oracle. "
            "Torch training seeds global RNGs and is intended for CLI use."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument(
        "--epochs", type=int, default=800, help="large images train for fewer (MAX_SAMPLES)"
    )
    parser.add_argument("--hidden", nargs="+", type=int, help="default: chosen by image size")
    parser.add_argument("--feature-sets", nargs="+", type=int, default=[4])
    parser.add_argument("--mixtures", type=int, default=4)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--without-floor", action="store_true")
    parser.add_argument("--floor-only", action="store_true")
    parser.add_argument("--partition-depth", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--envelope", type=int, choices=[3, 4, 5], default=5)
    parser.add_argument("--serial", action="store_true", help="train models one at a time")
    parser.add_argument("--no-race", action="store_true", help="train every width fully (see Race)")
    parser.add_argument("--threads", type=int, default=0, help="native threads; 0 chooses")
    parser.add_argument(
        "--compile-training", action="store_true", help="faster, not byte-identical training"
    )
    parser.add_argument(
        "--torch-training", action="store_true", help="PyTorch training and full precision search"
    )
    args = parser.parse_args()
    if not 0 <= args.threads <= 64:
        parser.error("--threads must be between 0 and 64")
    coder.THREADS = args.threads
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.decode:
        if args.input.stat().st_size > MAX_FILE_BYTES:
            parser.error("experimental file byte budget exceeded")
        Image.fromarray(decompress(args.input.read_bytes())).save(args.out, format="PNG")
        return
    blob, report = compress(
        args.input,
        widths=() if args.floor_only else (tuple(args.hidden) if args.hidden else None),
        feature_sets=tuple(args.feature_sets),
        mixtures=args.mixtures,
        epochs=args.epochs,
        schedule=args.schedule,
        lr=args.lr,
        device=args.device,
        conventional=not args.without_floor,
        hybrid=not args.no_hybrid,
        partition_depth=args.partition_depth,
        envelope=args.envelope,
        parallel=not args.serial,
        compile_training=args.compile_training,
        fused_training=False if args.torch_training else None,
        race=False if args.no_race else None,
    )
    args.out.write_bytes(blob)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
