"""Tiled training inputs must equal whole-image inputs at every interior pixel."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("imagecodecs") is None,
    reason="optional trainer requires ml and hybrid extras",
)


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import neural_probability_probe

    monkeypatch.setattr(neural_probability_probe, "TILE", 16)
    return neural_probability_probe


def test_small_images_train_on_every_pixel(trainer):
    assert trainer.training_tiles(30, 40, budget=1200) == [(slice(0, 30), slice(0, 40), None)]


def test_tiles_are_deterministic_disjoint_and_within_budget(trainer):
    tiles = trainer.training_tiles(70, 90, budget=16 * 16 * 7)
    again = trainer.training_tiles(70, 90, budget=16 * 16 * 7)
    assert [(ys, xs) for ys, xs, _ in tiles] == [(ys, xs) for ys, xs, _ in again]
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(tiles, again, strict=True))
    cover = np.zeros((70, 90), int)
    for ys, xs, mask in tiles:
        cover[ys, xs] += mask
    assert cover.max() == 1
    assert cover.sum() <= 16 * 16 * 7


def fake_local_fits(pixels, feature_set=3):
    """Deterministic stand-in for the native least-squares fits."""
    h, w = pixels.shape[:2]
    rng = np.random.default_rng(int(pixels.sum()) + feature_set)
    count = {3: 1, 4: 2}[feature_set]
    planes = [rng.integers(0, 4096 if k % 2 else 2041, (3, h, w)) for k in range(2 * count)]
    return np.stack(planes).astype(np.int16)


@pytest.mark.parametrize("feature_set", [0, 1, 2, 3, 4])
def test_tiled_features_equal_whole_image_features(trainer, monkeypatch, feature_set):
    monkeypatch.setattr(trainer.coder, "local_fits", fake_local_fits)
    rng = np.random.default_rng(5)
    rgb = np.clip(np.cumsum(rng.integers(-3, 4, (70, 90, 3)), axis=1) + 128, 0, 255)
    rgb = rgb.astype(np.uint8)
    args = argparse.Namespace(causal_context=True, feature_set=feature_set)
    h, w = rgb.shape[:2]
    tiles = trainer.training_tiles(h, w, budget=16 * 16 * 7)
    whole = [(slice(0, h), slice(0, w), None)]
    index = np.arange(h * w).reshape(h, w)
    pick = np.concatenate([index[ys, xs][mask] for ys, xs, mask in tiles])
    prior_whole, prior_tiles = {}, {}
    original = trainer.guide_fit
    for channel in (1, 0, 2):
        monkeypatch.setattr(trainer, "guide_fit", original)
        xf, target, anchor, co, co_base, errors = trainer.tile_inputs(
            rgb, channel, args, prior_whole, whole
        )
        # Compare features under the same transmitted guide coefficients.
        fixed = co[: len(co_base)]
        monkeypatch.setattr(trainer, "guide_fit", lambda *_, fixed=fixed: fixed)
        xt, tt, at, _, _, tile_errors = trainer.tile_inputs(rgb, channel, args, prior_tiles, tiles)
        assert np.array_equal(xt, xf[pick])
        assert np.array_equal(tt, target[pick])
        assert np.array_equal(at, anchor[pick])
        prior_whole[channel] = errors
        prior_tiles[channel] = tile_errors


@pytest.mark.parametrize("feature_set", [0, 2, 4])
def test_banded_features_equal_unbanded_features(trainer, monkeypatch, feature_set):
    monkeypatch.setattr(trainer.coder, "local_fits", fake_local_fits)
    rng = np.random.default_rng(6)
    rgb = np.clip(np.cumsum(rng.integers(-3, 4, (70, 90, 3)), axis=0) + 128, 0, 255)
    rgb = rgb.astype(np.uint8)
    args = argparse.Namespace(causal_context=True, feature_set=feature_set)
    whole = [(slice(0, 70), slice(0, 90), None)]
    monkeypatch.setattr(trainer, "BANDS", 9)
    prior = {False: {}, True: {}}
    for channel in (1, 0, 2):
        outputs = {}
        for banded in (False, True):
            monkeypatch.setattr(trainer, "BAND_MIN_ROWS", 1 if banded else 1000)
            outputs[banded] = trainer.tile_inputs(rgb, channel, args, prior[banded], whole)
            prior[banded][channel] = outputs[banded][5]
        for unbanded, banded in zip(outputs[False][:5], outputs[True][:5], strict=True):
            assert np.array_equal(unbanded, banded)


@pytest.mark.parametrize("feature_set", [0, 4])
def test_training_features_equal_exact_float_features(trainer, monkeypatch, feature_set):
    """Integer base features, predicted in chunks, give plane_inputs' float64 features."""
    monkeypatch.setattr(trainer.coder, "local_fits", fake_local_fits)
    monkeypatch.setattr(trainer, "GUIDE_CHUNK", 1000)
    rng = np.random.default_rng(7)
    rgb = np.clip(np.cumsum(rng.integers(-4, 5, (70, 90, 3)), axis=1) + 128, 0, 255)
    rgb = rgb.astype(np.uint8)
    args = argparse.Namespace(causal_context=True, feature_set=feature_set)
    whole = [(slice(0, 70), slice(0, 90), None)]
    prior, prior_exact = {}, {}
    for channel in (1, 0, 2):
        xf, target, anchor, _, co_base, errors = trainer.tile_inputs(
            rgb, channel, args, prior, whole
        )
        exact, exact_target, exact_anchor, base_error = trainer.plane_inputs(
            rgb, channel, co_base, prior_exact, True, feature_set=feature_set
        )
        assert np.array_equal(xf, (exact / 32).astype(np.float32))
        assert np.array_equal(target, exact_target)
        assert np.array_equal(anchor, exact_anchor.ravel())
        assert np.array_equal(errors[0], base_error)
        prior[channel] = errors
        prior_exact[channel] = base_error
