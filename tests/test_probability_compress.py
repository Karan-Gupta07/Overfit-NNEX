from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import compress

    return compress


def test_transparency_and_hidden_rgb_use_exact_floor(experiment, tmp_path):
    pixels = np.random.default_rng(37).integers(0, 256, (5, 7, 4), np.uint8)
    pixels[0, 0] = (137, 2, 91, 0)
    path = tmp_path / "alpha.png"
    Image.fromarray(pixels).save(path)
    blob, report = experiment.compress(path, hybrid=False)
    assert report["selected"]["format"] == "scdx"
    assert report["learned_skipped"] == "nonopaque"
    assert report["exact_rgba"]
    assert np.array_equal(experiment.decompress(blob), pixels)


def test_floor_only_needs_no_torch(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (5, 7), (31, 67, 89)).save(path)
    program = (
        "import pathlib,sys;sys.path.insert(0,sys.argv[1]);import compress;"
        "blob,report=compress.compress(pathlib.Path(sys.argv[2]),widths=(),hybrid=False);"
        "assert report['exact_rgba'];assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program, str(SOURCE), str(path)],
        check=True,
        capture_output=True,
        timeout=30,
    )


def test_decoder_cli_refuses_oversized_file_before_reading(
    experiment, tmp_path, monkeypatch, capsys
):
    coder = experiment.coder
    source = tmp_path / "oversized"
    with source.open("wb") as file:
        file.truncate(coder.MAX_MODEL_BYTES + coder.MAX_STREAM_BYTES + coder.HEADER_SIZE + 1)
    monkeypatch.setattr(
        sys,
        "argv",
        ["neural_bitstream", str(source), "--decode", "--out", str(tmp_path / "out.png")],
    )
    with pytest.raises(SystemExit):
        coder.main()
    assert "file byte budget" in capsys.readouterr().err


def test_strip_rows_fit_the_pixel_limit(experiment, monkeypatch):
    monkeypatch.setattr(experiment.coder, "MAX_PIXELS", 10)
    # Three strips would need 3 rows of 4 pixels, over the limit.
    assert experiment.strip_rows(4, 7) == 2
    with pytest.raises(ValueError, match="pixel budget"):
        experiment.strip_rows(4, 9)


@pytest.mark.parametrize("channels", [3, 4])
def test_large_image_is_one_file_of_strips(experiment, tmp_path, monkeypatch, channels):
    monkeypatch.setattr(experiment.coder, "MAX_PIXELS", 20)
    pixels = np.random.default_rng(41).integers(0, 256, (7, 5, channels), np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    blob, report = experiment.compress(path, widths=(), hybrid=False)
    assert blob.startswith(experiment.STRIPS_MAGIC)
    assert report["selected"] == {"format": "strips", "count": 2}
    assert [strip["rows"] for strip in report["strips"]] == [4, 3]
    assert report["complete_file_bytes"] == len(blob)
    assert report["checked_encode"] == 0
    if channels == 3:
        pixels = np.concatenate([pixels, np.full((7, 5, 1), 255, np.uint8)], axis=2)
    assert np.array_equal(experiment.decompress(blob), pixels)


def test_strip_files_are_bounds_checked(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.coder, "MAX_PIXELS", 20)
    path = tmp_path / "source.png"
    Image.fromarray(np.random.default_rng(43).integers(0, 256, (7, 5, 3), np.uint8)).save(path)
    blob, _ = experiment.compress(path, widths=(), hybrid=False)
    head = experiment._STRIPS_HEAD
    start = len(experiment.STRIPS_MAGIC)
    table = start + head.size
    width, height, count = head.unpack_from(blob, start)
    first = experiment._STRIP.unpack_from(blob, table)[1]
    body = table + count * experiment._STRIP.size
    parts = [blob[body : body + first], blob[body + first :]]
    cases = [
        (blob[: start + 4], "truncated"),
        (blob + b"\0", "bounds"),
        (blob[:start] + head.pack(width, height + 1, count) + blob[table:], "bounds"),
        (blob[:start] + head.pack(width, height, 1) + blob[table:], "bounds"),
        (blob[:start] + head.pack(0, height, count) + blob[table:], "bounds"),
        (experiment.pack_strips(width, [(3, parts[0]), (4, parts[1])]), "strip shape"),
        (experiment.pack_strips(width, [(7, blob), (1, parts[1])]), "nested"),
    ]
    for corrupt, message in cases:
        with pytest.raises(ValueError, match=message):
            experiment.decompress(corrupt)


def test_strips_can_share_one_model(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.coder, "MAX_PIXELS", 40)
    pixels = np.random.default_rng(61).integers(0, 256, (12, 5, 3), np.uint8)
    rgba = np.concatenate([pixels, np.full((12, 5, 1), 255, np.uint8)], axis=2)
    parts, strips = [], []
    for top in (0, 6):
        strip = np.ascontiguousarray(pixels[top : top + 6])
        path = tmp_path / f"strip-{top}.png"
        Image.fromarray(strip).save(path)
        blob, _ = experiment.compress(
            path,
            widths=(4,),
            feature_sets=(1,),
            epochs=0,
            device="cpu",
            conventional=False,
            hybrid=False,
            check=False,
        )
        parts.append((6, blob))
        strips.append(strip)
    shared = experiment._shared_part(strips[1], parts[1][1], 0, parts[0][1])
    assert shared.startswith(experiment.SHARED_MAGIC)
    assert len(shared) < len(parts[1][1])
    blob = experiment.pack_strips(5, [parts[0], (6, shared)])
    assert np.array_equal(experiment.decompress(blob), rgba)
    head = len(experiment.SHARED_MAGIC)
    for donor in (1, 2):
        wrong = shared[:head] + bytes([donor]) + shared[head + 1 :]
        with pytest.raises(ValueError, match="shared strip model"):
            experiment.decompress(experiment.pack_strips(5, [parts[0], (6, wrong)]))
    for truncated in (shared[:head], shared[: head + 1 + 8]):
        with pytest.raises(ValueError, match="shared strip model"):
            experiment.decompress(experiment.pack_strips(5, [parts[0], (6, truncated)]))
    files = [blob for _, blob in parts]
    tables = max(experiment.coder.v5_table_bytes(f) for f in files)
    assert experiment.strip_decoders(files) == 2
    cores = os.cpu_count() or 1
    assert experiment.parallel_coders((7, 32, 3), ["b1", "g1"], 4) == max(1, min(4, cores // 3))
    assert experiment.parallel_coders((7, 32, 3), ["s16", "g1"], 2) == 1
    assert experiment.parallel_coders((7, 32, 3), ["g1"], 0) == 1
    checked = max(1, min(4, cores // 6))
    assert experiment.parallel_coders((7, 32, 3), ["b1", "g1"], 4, checked=True) == checked
    assert experiment.checked_mixing((7, 32, 3), ["b1", "g1"], 2) is (checked >= 2)
    assert experiment.checked_mixing((7, 32, 3), ["s16", "g1"], 1) is False
    assert experiment.checked_mixing((7, 32, 3), ["g1"], 0) is False
    with monkeypatch.context() as patch:
        # A checked one-band encode and its decoder take six threads.
        patch.setattr(experiment.os, "cpu_count", lambda: 6)
        assert experiment.checked_mixing((7, 32, 3), ["g1"], 1)
        assert not experiment.checked_mixing((7, 32, 3), ["g1"], 2)
        # One checked encode may take more threads than there are cores.
        assert experiment.checked_mixing((7, 32, 3), ["g2"], 1)
        assert not experiment.checked_mixing((7, 32, 3), ["g2"], 2)
    monkeypatch.setattr(experiment, "STRIP_MEMORY", 0.0)
    assert experiment.strip_decoders(files) == (1 if tables else 2)
    assert experiment.parallel_coders((7, 32, 3), ["b1", "g1"], 4) == 1
    assert not experiment.checked_mixing((7, 32, 3), ["g1"], 1)
    assert np.array_equal(experiment.decompress(blob), rgba)
    chosen, report = experiment.share_models(strips, parts)
    assert len(report["screens"]) == 2
    assert np.array_equal(experiment.decompress(experiment.pack_strips(5, chosen)), rgba)
    # Screens that promise the model bytes code the shared parts.
    monkeypatch.setattr(experiment, "band_stream_bytes", lambda *args: 0)
    chosen, report = experiment.share_models(strips, parts)
    donor = report["donor"]
    assert donor is not None and chosen[donor] == parts[donor]
    assert chosen[1 - donor][1].startswith(experiment.SHARED_MAGIC)
    assert np.array_equal(experiment.decompress(experiment.pack_strips(5, chosen)), rgba)


def test_strip_check_decodes_only_strips_not_checked_as_coded(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.coder, "MAX_PIXELS", 40)
    coder = experiment.coder
    pixels = np.random.default_rng(73).integers(0, 256, (12, 5, 3), np.uint8)
    rgba = np.concatenate([pixels, np.full((12, 5, 1), 255, np.uint8)], axis=2)
    parts = []
    for top in (0, 6):
        path = tmp_path / f"strip-{top}.png"
        Image.fromarray(pixels[top : top + 6]).save(path)
        blob, _ = experiment.compress(
            path,
            widths=(4,),
            feature_sets=(1,),
            epochs=0,
            device="cpu",
            conventional=False,
            hybrid=False,
            check=False,
        )
        parts.append((6, blob))
    blob = experiment.pack_strips(5, parts)
    assert experiment.check_strips(blob, rgba) == 2
    first = parts[0][1]
    checked = coder.encode_v5(
        np.ascontiguousarray(pixels[:6]),
        coder.model_payload(first),
        layout=coder.v5_layout(first),
        values=coder.v5_values(first),
        checked=True,
    )
    assert checked == first and coder.verified(first, pixels[:6])
    assert experiment.check_strips(blob, rgba) == 1
    for row in (2, 9):
        wrong = rgba.copy()
        wrong[row, 1, 0] ^= 1
        with pytest.raises(AssertionError, match="failed independent"):
            experiment.check_strips(blob, wrong)
    with pytest.raises(AssertionError, match="height"):
        experiment.check_strips(blob, rgba[:11])
    with pytest.raises(ValueError, match="strip file"):
        experiment.check_strips(first, rgba)


def test_no_candidate_is_an_error(experiment, tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (5, 7)).save(path)
    with pytest.raises(ValueError, match="no eligible candidate"):
        experiment.compress(path, widths=(), conventional=False)
    with pytest.raises(ValueError):
        experiment.decompress(b"not an experimental file")


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": True},
        {"epochs": -1},
        {"widths": (True,)},
        {"feature_sets": (5,)},
        {"mixtures": 5},
        {"schedule": "linear"},
        {"lr": 0},
        {"lr": float("nan")},
        {"device": "invalid"},
        {"envelope": 2},
        {"parallel": 1},
        {"race": 1},
    ],
)
def test_invalid_training_budgets_fail_before_loading(experiment, tmp_path, options):
    with pytest.raises(ValueError, match="invalid learned"):
        experiment.compress(tmp_path / "absent.png", **options)


def test_only_small_images_prune_and_clip_loosely(experiment):
    for pixels, small in ((512 * 512, True), (2400 * 2400, False)):
        _budget, _epochs, batch = experiment.training_budget(pixels, 800, True)
        settings = experiment.small_image_training(batch)
        assert settings["prune"] is small
        assert settings["max_norm"] == (experiment.SMALL_MAX_NORM if small else 10.0)
    _budget, _epochs, batch = experiment.training_budget(512 * 512, 800, False)
    assert experiment.small_image_training(batch) == {"max_norm": 10.0, "prune": False}


def test_large_images_race_a_narrow_width_only_when_compressible(experiment):
    flat = np.zeros((64, 64, 3), np.uint8)
    noise = np.random.default_rng(5).integers(0, 256, (64, 64, 3), np.uint8)
    assert experiment.residual_bytes(flat) == 0.0
    assert experiment.residual_bytes(noise) > 0.95 * noise.size
    assert experiment.default_widths(500_000) == (12, 32)
    assert experiment.default_widths(3_000_000) == (24, 48)
    assert experiment.default_widths(3_000_000, 10**9) == (24, 48)
    assert experiment.default_widths(12_000_000) == (48,)
    assert experiment.default_widths(12_000_000, experiment.COMPRESSIBLE_BYTES) == (48,)
    assert experiment.default_widths(12_000_000, 10**6) == experiment.COMPRESSIBLE_WIDTHS


def test_mixing_adds_a_band_where_the_screen_gains_little(experiment, monkeypatch):
    layouts = experiment.mixing_layouts
    assert layouts(experiment.BOTH_GATES) == ("b1", "g1")
    assert layouts(experiment.SCREEN_MIX) == ("g1", "g2")
    mid, large = experiment.SCREEN_MIX + 1, experiment.MIX_BANDS_FROM + 1
    assert layouts(mid) == layouts(mid, experiment.BAND_GAIN) == ("g1",)
    assert layouts(mid, experiment.BAND_GAIN / 2) == ("g2",)
    assert layouts(large) == layouts(large, 0.5) == ("g2",)
    assert layouts(large, 0.0) == ("g3",)
    # Each model mixes in the layout its own screen chose.
    monkeypatch.setattr(experiment, "BOTH_GATES", 0)
    monkeypatch.setattr(experiment, "SCREEN_MIX", 0)
    mixed = {b"high": 50, b"low": 95}
    monkeypatch.setattr(
        experiment,
        "screen_mixing",
        lambda rgb, network, payload, values, layout: {
            "pixels": 64,
            "network": 100,
            "mixed": mixed[payload],
            "layout": layout,
        },
    )
    coded = []

    def encode(rgb, payload, *, layout, values, fits, checked):
        coded.append((payload, layout))
        return b"x" * 10

    monkeypatch.setattr(experiment.coder, "encode_v5", encode)
    rgb = np.zeros((8, 8, 3), np.uint8)
    results = experiment._mix(rgb, b"network", [(b"high", None), (b"low", None)], None, 1000)
    assert sorted(coded) == [(b"high", "g1"), (b"low", "g2")]
    assert [report["screen"]["layout"] for _, report in results] == ["g1", "g1"]


def test_gray_planes_code_with_constant_networks_when_smaller(experiment, monkeypatch):
    rng = np.random.default_rng(59)
    g, noise = rng.integers(0, 256, (2, 16, 16))

    def rgb(r, b):
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    assert experiment.gray_planes(rgb(g, noise)) == (1,)
    assert experiment.gray_planes(rgb(noise, (g + 3) % 256)) == (2,)
    assert experiment.gray_planes(rgb(noise, noise)) == ()
    coder = experiment.coder
    sizes, mixed = {}, []
    monkeypatch.setattr(experiment, "MIX_GAIN", 1.0)
    monkeypatch.setattr(coder, "model_payload", lambda blob: b"model")
    monkeypatch.setattr(
        coder, "constant_planes", lambda payload, planes, fold: b"fold" if fold else b"zero"
    )
    monkeypatch.setattr(coder, "v5_layout", lambda blob: blob.decode())

    def encode(pixels, payload, layout, values, fits, checked=False):
        size = sizes[layout]
        if isinstance(size, tuple):
            return bytes(size[payload == b"fold"])
        mixed.append(payload)
        return bytes(size)

    monkeypatch.setattr(coder, "encode_v5", encode)

    def cut(source, best, mixing=None, **coded):
        sizes.clear()
        sizes.update(coded)
        mixed.clear()
        blob, report = experiment._constant_gray(rgb(g, g), source, (1, 2), None, None, best)
        layouts = [row["layout"] for row in report["layouts"]]
        assert layouts == [source.decode(), *coded]
        assert mixed == [mixing] * (len(coded) - 1)
        return None if blob is None else len(blob)

    # (bytes without fold, with fold) in the source layout; the smaller one mixes,
    # and mixing must save MIX_GAIN per pixel (256 bytes here) over the smaller
    # network-only file.
    assert cut(b"s16", 1100, b"zero", s16=(1000, 1010), b1=900, g1=760) == 1000
    assert cut(b"s16", 1100, b"fold", s16=(1200, 1150), b1=900, g1=800) == 800
    assert cut(b"s16", 1100, b"zero", s16=(1200, 1200), b1=900, g1=845) is None
    assert cut(b"g1", 1100, g1=(1050, 1040)) == 1040
    assert cut(b"g1", 1100, g1=(1100, 1100)) is None


def test_race_continues_only_the_best_score(experiment):
    import multiprocessing
    import threading

    manager = multiprocessing.get_context("spawn").Manager()

    def results(margin, rows):
        race = experiment.Race(manager, len(rows), margin)
        got = [None] * len(rows)

        def run(i, row):
            got[i] = race.report(i, 2, *row)

        threads = [threading.Thread(target=run, args=item) for item in enumerate(rows)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        return got

    def mixed(stream, weights):
        return stream + weights, stream * experiment.MIX_ABSORB + weights

    try:
        for margin, want in ((0.0, [True, False, False]), (0.06, [True, True, False])):
            assert results(margin, [(100,), (105,), (120,)]) == want
        # A width also continues when it leads with the bits mixing absorbs,
        # and a narrower one does while the leader's weights are a large share of that.
        rows = [mixed(1000, 30), mixed(1040, 10), mixed(1100, 20)]
        assert results(0.0, rows) == [True, True, False]
        assert results(0.0, [mixed(130, 25), mixed(100, 30)]) == [True, True]
        assert results(0.0, [mixed(100, 30), mixed(130, 25)]) == [True, False]
        race = experiment.Race(manager, 2)
        race.abort()
        assert race.report(12, 2, 1e9)
    finally:
        manager.shutdown()
    assert experiment.Race.checkpoint(800) == 800 // experiment.RACE_FRACTION
    assert experiment.Race.checkpoint(3) == 1


def test_floor_screen_needs_a_clear_estimate_on_a_photograph(experiment):
    fast, colors = 1000, experiment.SCREEN_COLORS
    limit = fast * (1 - experiment.FLOOR_SCREEN)
    assert experiment.floor_screened([limit, 2000.0], fast, colors)
    assert not experiment.floor_screened([limit + 1], fast, colors)
    assert not experiment.floor_screened([limit], fast, colors - 1)
    assert not experiment.floor_screened([], fast, colors)


def test_estimates_stop_waiting_once_a_worker_ends(experiment):
    import queue

    class Item:
        def __init__(self, done):
            self.done = done

        def ready(self):
            return self.done

    estimates = queue.Queue()
    for score in (5.0, 7.0, 3.0):
        estimates.put(score)
    assert experiment._await_estimates(estimates, 2, [Item(False)]) == [5.0, 7.0]
    assert experiment._await_estimates(estimates, 2, [Item(True)]) == [3.0]
    assert experiment._await_estimates(estimates, 1, [Item(True)]) == []


@pytest.mark.skipif(
    importlib.util.find_spec("imagecodecs") is None, reason="JPEG XL needs the hybrid extra"
)
def test_fast_floor_decodes_exactly(experiment):
    rgb = np.random.default_rng(41).integers(0, 256, (9, 11, 3), np.uint8)
    rgba = np.concatenate([rgb, np.full((9, 11, 1), 255, np.uint8)], axis=2)
    blob = experiment.fast_floor(rgba)
    assert blob.startswith(experiment.SCDX_MAGIC)
    assert np.array_equal(experiment.decompress(blob), rgba)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("imagecodecs") is None,
    reason="optional learned encoder requires ml and hybrid extras",
)
def test_whole_file_selection_with_actual_learned_model(experiment, tmp_path, monkeypatch):
    compiler = __import__("shutil").which("c++")
    if compiler is None:
        pytest.skip("needs optional research C++ backend")
    binary = tmp_path / "backend"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", str(SOURCE / "nnproto_integer.cpp"), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    monkeypatch.setattr(experiment.coder, "INTEGER_BINARY", binary)
    monkeypatch.setattr(experiment.coder, "ROOT", tmp_path)
    pixels = np.random.default_rng(51).integers(0, 256, (5, 7, 3), np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    for conventional in (False, True):
        blob, report = experiment.compress(
            path,
            widths=(4,),
            feature_sets=(0, 1),
            epochs=0,
            device="cpu",
            conventional=conventional,
            hybrid=False,
        )
        assert report["learned_in_parallel"]
        assert {c.get("feature_set") for c in report["candidates"]} >= {0, 1}
        assert report["complete_file_bytes"] == min(
            candidate["complete_file_bytes"] for candidate in report["candidates"]
        )
        assert np.array_equal(experiment.decompress(blob)[..., :3], pixels)
        if not conventional:
            assert report["selected"]["format"] == "nnex-v5"
            assert [row["layout"] for row in report["mixing"]["layouts"]] == ["b1", "g1"]
            coder = experiment.coder
            payload = coder.model_payload(blob)
            values, _ = coder._v5_split(blob[coder.HEADER_SIZE + len(payload) :])
            screen = experiment.screen_mixing(pixels, payload, payload, values)
            assert (screen["rows"], screen["lead"], screen["pixels"]) == (1, 4, 7)
            assert screen["network"] > 0 and screen["mixed"] > 0
            serial, serial_report = experiment.compress(
                path,
                widths=(4,),
                feature_sets=(0, 1),
                epochs=2,
                device="cpu",
                conventional=False,
                hybrid=False,
                parallel=False,
            )
            parallel, _ = experiment.compress(
                path,
                widths=(4,),
                feature_sets=(0, 1),
                epochs=2,
                device="cpu",
                conventional=False,
                hybrid=False,
            )
            assert not serial_report["learned_in_parallel"]
            assert serial == parallel
            v3, v3_report = experiment.compress(
                path,
                widths=(4,),
                feature_sets=(1,),
                epochs=0,
                device="cpu",
                conventional=False,
                hybrid=False,
                envelope=3,
            )
            assert v3.startswith(experiment.coder.MAGIC_V3)
            assert v3_report["selected"]["format"] == "nnex-v3"
            assert np.array_equal(experiment.decompress(v3)[..., :3], pixels)
            coded = tmp_path / "learned.coded"
            coded.write_bytes(blob)
            program = (
                "import pathlib,sys;sys.path.insert(0,sys.argv[1]);import compress;"
                "compress.coder.INTEGER_BINARY=pathlib.Path(sys.argv[2]);"
                "compress.coder.ROOT=pathlib.Path(sys.argv[4]);"
                "back=compress.decompress(pathlib.Path(sys.argv[3]).read_bytes());"
                "assert back.shape==(5,7,4);assert 'torch' not in sys.modules"
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(SOURCE),
                    str(binary),
                    str(coded),
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
    # Every plane holds the same gapped set of values, so models train on ranks.
    g = 5 * np.random.default_rng(53).integers(0, 40, (5, 7))
    sparse = np.stack([g, g[::-1], g[:, ::-1]], axis=2).astype(np.uint8)
    Image.fromarray(sparse).save(path)
    blob, report = experiment.compress(
        path,
        widths=(4,),
        feature_sets=(1,),
        epochs=0,
        device="cpu",
        conventional=False,
        hybrid=False,
    )
    assert report["learned_in_parallel"]
    assert report["shared_values"] == len(np.unique(g))
    assert np.array_equal(experiment.decompress(blob)[..., :3], sparse)


def test_uniform_finalist_is_coded_only_when_close(experiment, tmp_path, monkeypatch):
    Image.new("RGB", (5, 7), (31, 67, 89)).save(tmp_path / "source.png")
    (tmp_path / "searched.bin").write_bytes(b"searched")
    (tmp_path / "uniform.bin").write_bytes(b"uniform grid")
    reused = []

    def encode_v4(pixels, model, fits=None):
        reused.append(fits)
        return b"NN" + model

    monkeypatch.setattr(experiment.coder, "encode_v4", encode_v4)
    search = {
        "labels": ["float-t12", "float-t12", "float-t12"],
        "selected_file": str(tmp_path / "searched.bin"),
        "estimated_bytes": 1000.0,
        "uniform_label": "float-q10",
        "uniform_file": str(tmp_path / "uniform.bin"),
    }
    coded = []
    for uniform_estimate in (1000.0 * (1 + experiment.UNIFORM_MARGIN), 1003.0):
        work = tmp_path / str(uniform_estimate)
        work.mkdir()
        candidate = {**search, "uniform_estimated_bytes": uniform_estimate}
        choices = experiment._code_finalists(
            tmp_path / "source.png", work, candidate, 4, "local-fits"
        )
        coded.append([row["bits"] for row in choices["candidates"]])
        assert choices["selected"]["weights"] == search["labels"]
    assert coded == [[None, 10], [None]]
    assert reused == ["local-fits"] * 3
