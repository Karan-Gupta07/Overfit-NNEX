"""Experimental best-file encoder: SCDX floor or a self-contained integer NNEX.

This is a research multiplexer, not a new .scdx asset. Its decoder recognizes
SCDX and integer NNEX v2/v3/v4. Learned candidates compete on their actual
complete bytes; the encoder writes v4 by default and v3 on request.
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
import queue
import struct
import threading
import time
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
# NNEX file with its own model, and its first rows do not see the strip above.
STRIPS_MAGIC = b"NNST\x00\x01"
MAX_STRIPS = 4
_STRIPS_HEAD = struct.Struct("<2IB")
_STRIP = struct.Struct("<2I")
# Raw RGBA samples of the largest image the strips hold, plus a header: an
# uncompressed PPM of a 46-megapixel LPCB image is 139 MB.
MAX_SOURCE_BYTES = MAX_STRIPS * coder.MAX_PIXELS * 4 + 4096
# (maximum pixels, hidden widths): weights cost less per pixel on larger images.
WIDTH_TIERS = ((1_000_000, (12, 32)), (4_000_000, (24, 48)), (coder.MAX_PIXELS, (48,)))
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


class Race:
    """Checkpoint rendezvous for models training in parallel workers."""

    def __init__(self, manager, count: int, margin: float = RACE_MARGIN):
        self.scores = manager.dict()
        self.barrier = manager.Barrier(count)
        self.margin = margin

    @staticmethod
    def checkpoint(epochs: int) -> int:
        return max(1, epochs // RACE_FRACTION)

    def report(self, hidden: int, feature_set: int, score: float) -> bool:
        self.scores[(hidden, feature_set)] = score
        try:
            self.barrier.wait(RACE_TIMEOUT)
        except threading.BrokenBarrierError:
            # A worker failed or stalled: keep training rather than guess.
            return True
        return score <= min(self.scores.values()) * (1 + self.margin)

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


def _decompress_strips(blob: bytes) -> np.ndarray:
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
    strips = []
    for rows, length in table:
        part = blob[offset : offset + length]
        offset += length
        if part.startswith(STRIPS_MAGIC):
            raise ValueError("nested strip file")
        rgba = decompress(part)
        if rgba.shape != (rows, width, 4):
            raise ValueError("strip shape")
        strips.append(rgba)
    return np.concatenate(strips, axis=0)


def _compress_strips(path: Path, pixels: np.ndarray, options: dict) -> tuple[bytes, dict]:
    """compress() of each strip_rows strip, packed into one STRIPS_MAGIC file."""
    start = time.perf_counter()
    height, width = pixels.shape[:2]
    rows = strip_rows(width, height)
    parts, reports = [], []
    with coder.temporary_directory() as directory:
        for index, top in enumerate(range(0, height, rows), 1):
            strip = np.ascontiguousarray(pixels[top : top + rows])
            # BMP is quick to write and read back; PNG keeps alpha.
            suffix = "bmp" if strip.shape[2] == 3 else "png"
            target = Path(directory) / f"strip-{index}.{suffix}"
            Image.fromarray(strip).save(target)
            blob, report = compress(target, **options)
            parts.append((strip.shape[0], blob))
            reports.append(report)
    best = pack_strips(width, parts)
    if pixels.shape[2] == 3:
        rgba = np.concatenate([pixels, np.full((height, width, 1), 255, np.uint8)], axis=2)
    else:
        rgba = pixels
    mark = time.perf_counter()
    if not np.array_equal(decompress(best), rgba):
        raise AssertionError("selected complete file failed independent RGBA decoding")
    check_seconds = time.perf_counter() - mark
    strips = [
        {
            "rows": strip_height,
            "complete_file_bytes": report["complete_file_bytes"],
            "selected": report["selected"],
            "seconds": report["seconds"],
            "phase_seconds": report["phase_seconds"],
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
        "exact_rgba": True,
        "seconds": time.perf_counter() - start,
        "phase_seconds": {
            **{f"strip-{k}": round(row["seconds"], 2) for k, row in enumerate(strips, 1)},
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
        raise ValueError("expected SCDX, strips or integer NNEX v2/v3/v4")
    rgb = coder.decode(blob)
    return np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)


# The uniform-grid finalist is coded only when its estimate is within this
# fraction of the searched model's. In 47 distinct Tecnick encodes that coded
# both, the uniform model never won (smallest estimated gap 0.094%, actual
# 0.091%); below an estimated 0.5% the two gaps differed by at most 0.018%.
UNIFORM_MARGIN = 0.0005


def _code_finalists(
    snapshot: Path, work: Path, search: dict, envelope: int, fits: str | None = None
) -> dict:
    """Actual files for the searched model and, if close, the best uniform-grid model.

    fits names the snapshot's precomputed local fits, reused by v4 encodes. The
    final whole-file check decodes whichever file wins, so finalists are not
    decoded here.
    """
    pixels = read_rgb(snapshot, max_pixels=coder.MAX_PIXELS)
    encode = functools.partial(coder.encode_v4, fits=fits) if envelope == 4 else coder.encode_v3
    finalists = [("searched", "selected_file", "estimated_bytes", search["labels"])]
    uniform = search["uniform_label"]
    if search["uniform_estimated_bytes"] <= search["estimated_bytes"] * (1 + UNIFORM_MARGIN):
        finalists.append((uniform, "uniform_file", "uniform_estimated_bytes", [uniform] * 3))
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
                "bits": int(label.rsplit("-q", 1)[1]) if label != "searched" else None,
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
    if trained.get("search"):
        choices = _code_finalists(
            snapshot, work, trained["search"], job["envelope"], job["training"].get("local_fits")
        )
    else:
        choices = precision_search(
            snapshot,
            Path(trained["model_v3_file"]),
            work / "precision.json",
            envelope=job["envelope"],
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


def default_widths(pixels: int) -> tuple[int, ...]:
    """Hidden widths tried for an image; wider nets pay off once weights are cheap."""
    for limit, widths in WIDTH_TIERS:
        if pixels <= limit:
            return widths
    return WIDTH_TIERS[-1][1]


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
    envelope=4,
    parallel=True,
    compile_training=False,
    fused_training=None,
    search=None,
    race=None,
) -> tuple[bytes, dict]:
    """Smallest complete file among the floor and every learned candidate.

    With parallel=True each learned model trains in its own spawned process
    while this process builds the floor. Every worker seeds its own RNGs exactly
    as the serial loop does, so the files are byte-identical either way.
    compile_training=True is faster on MPS but trains (slightly) different models.
    fused_training=None uses the fused Metal trainer whenever MPS is available;
    search=None then adds quantization-aware snapshots and an estimated
    per-plane weight search, coding only two finalists. widths=None picks
    default_widths for the image size. Unless race=False, parallel fused
    models race (see Race): a model that keeps training is byte-identical to
    its unraced run, but a dropped width is never fully trained.
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
        or envelope not in (3, 4)
        or type(parallel) is not bool
        or type(compile_training) is not bool
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
    opaque = bool(np.all(rgba[..., 3] == 255))
    if widths is None:
        widths = default_widths(rgba.shape[0] * rgba.shape[1])
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
                Image.fromarray(rgba[..., :3]).save(snapshot, format="BMP")
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
                    )
            pending = [pool.apply_async(_learn_in_worker, (job,)) for job in jobs] if pool else []
            # Least-squares fits are computed once here rather than in every worker,
            # which meanwhile build the features that do not need them.
            mark = time.perf_counter()
            for feature_set, target in local.items():
                coder.local_fits(np.ascontiguousarray(rgba[..., :3]), feature_set).tofile(target)
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
            score = scores.get((hidden, feature_set))
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
            if best is None or len(blob) < len(best):
                best = blob
                selected = {
                    **labels,
                    "bits": chosen["bits"],
                    "model_bytes": chosen["model_bytes"],
                }
                if "weights" in chosen:
                    selected["weights"] = chosen["weights"]
    if best is None:
        raise ValueError("no eligible candidate; enable the conventional floor")
    mark = time.perf_counter()
    back = decompress(best)
    if not np.array_equal(back, rgba):
        raise AssertionError("selected complete file failed independent RGBA decoding")
    phases["final-check"] = time.perf_counter() - mark
    return best, {
        "image": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
        "complete_file_bytes": len(best),
        "selected": selected,
        "candidates": candidates,
        "exact_rgba": True,
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
    parser.add_argument("--envelope", type=int, choices=[3, 4], default=4)
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
