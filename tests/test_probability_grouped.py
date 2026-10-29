"""Grouped plane training must be equivalent to independent per-plane networks."""

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

    return neural_probability_probe


def _nets(trainer, dimensions):
    import torch

    torch.manual_seed(3)
    nets = [trainer.ProbabilityNet(d, 8, 3, np.linspace(-1, 1, d)) for d in dimensions]
    with torch.no_grad():
        for net in nets:
            net.head.weight.normal_(0, 0.1)
    return nets


def test_grouped_forward_matches_each_plane(trainer):
    import torch

    dimensions = [5, 9, 7]
    nets = _nets(trainer, dimensions)
    grouped = trainer.GroupedNet(nets, dimensions)
    x = torch.zeros(3, 40, 9)
    for g, d in enumerate(dimensions):
        x[g, :, :d] = torch.randn(40, d)
    together = grouped(x)
    for g, (net, d) in enumerate(zip(nets, dimensions, strict=True)):
        alone = net(x[g, :, :d])
        for joint, single in zip(together, alone, strict=True):
            assert torch.allclose(joint[g], single, atol=1e-5)


def test_padding_stays_zero_and_export_round_trips(trainer):
    import torch

    dimensions = [4, 6, 6]
    nets = _nets(trainer, dimensions)
    grouped = trainer.GroupedNet(nets, dimensions)
    optimizer = torch.optim.Adam(grouped.parameters(), lr=0.01)
    x = torch.zeros(3, 64, 6)
    for g, d in enumerate(dimensions):
        x[g, :, :d] = torch.randn(64, d)
    target = torch.randint(-5, 6, (3, 64)).float()
    anchor = torch.full((3, 64), 128.0)
    for _ in range(5):
        optimizer.zero_grad()
        trainer.mixture_bits(*grouped(x), target, anchor).mean(1).sum().backward()
        trainer.clip_per_plane(grouped.parameters(), 10)
        optimizer.step()
    assert torch.count_nonzero(grouped.w1[0, 4:]) == 0
    assert torch.count_nonzero(grouped.affine[0, 4:]) == 0
    grouped.export(nets, dimensions)
    after = grouped(x)
    for g, (net, d) in enumerate(zip(nets, dimensions, strict=True)):
        assert torch.allclose(net(x[g, :, :d])[1], after[1][g], atol=1e-5)


def test_per_plane_clipping_matches_clip_grad_norm(trainer):
    import torch

    dimensions = [3, 3]
    nets = _nets(trainer, dimensions)
    grouped = trainer.GroupedNet(nets, dimensions)
    for p in grouped.parameters():
        p.grad = torch.randn_like(p) * 50
    expected = []
    for g in range(2):
        grads = [p.grad[g].clone() for p in grouped.parameters()]
        params = [torch.nn.Parameter(torch.zeros_like(v)) for v in grads]
        for param, grad in zip(params, grads, strict=True):
            param.grad = grad
        torch.nn.utils.clip_grad_norm_(params, 10)
        expected.append([param.grad for param in params])
    trainer.clip_per_plane(grouped.parameters(), 10)
    for g in range(2):
        for p, want in zip(grouped.parameters(), expected[g], strict=True):
            assert torch.allclose(p.grad[g], want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("feature_set", [0, 1, 2, 3, 4])
def test_grouped_planes_have_sequential_tensor_layout(trainer, monkeypatch, feature_set):
    def fake_local_fits(pixels, feature_set=3):
        rng = np.random.default_rng(int(pixels.sum()))
        shape = (3, *pixels.shape[:2])
        count = {3: 1, 4: 2}[feature_set]
        fits = [rng.integers(0, 4096 if k % 2 else 2041, shape) for k in range(2 * count)]
        return np.stack(fits).astype(np.int16)

    monkeypatch.setattr(trainer.coder, "local_fits", fake_local_fits)
    rng = np.random.default_rng(9)
    rgb = np.clip(np.cumsum(rng.integers(-3, 4, (24, 20, 3)), axis=1) + 128, 0, 255)
    rgb = rgb.astype(np.uint8)
    args = argparse.Namespace(
        causal_context=True,
        feature_set=feature_set,
        hidden=8,
        mixtures=2,
        lr=0.003,
        epochs=2,
        batch=128,
        schedule="cosine",
    )
    tiles = trainer.training_tiles(24, 20)
    grouped, prior = trainer.fit_planes(rgb, args, "cpu", tiles)
    sequential, errors = [], {}
    for channel in (1, 0, 2):
        tensors, report, error = trainer.fit(rgb, channel, args, "cpu", errors, tiles=tiles)
        errors[channel] = error
        sequential.append((tensors, report))
    for (g_tensors, g_report), (s_tensors, s_report) in zip(grouped, sequential, strict=True):
        assert [(n, s) for n, s, _ in g_tensors] == [(n, s) for n, s, _ in s_tensors]
        assert g_report["channel"] == s_report["channel"]
        assert g_report["parameters"] == s_report["parameters"]
        assert np.isfinite(g_report["conditional_bits"])
    for channel in (1, 0, 2):
        assert all(
            np.array_equal(a, b) for a, b in zip(prior[channel], errors[channel], strict=True)
        )
