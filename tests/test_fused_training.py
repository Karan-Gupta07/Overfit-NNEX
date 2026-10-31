"""Fused Metal training and the estimated weight search used by the encoder."""

from __future__ import annotations

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
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import model_search
    import neural_bitstream
    import neural_probability_probe

    return neural_probability_probe, model_search, neural_bitstream


def _mps() -> bool:
    import torch

    return bool(torch.backends.mps.is_available())


def _nets(trainer, dimensions, hidden, mixtures=4, seed=3):
    import torch

    torch.manual_seed(seed)
    nets = []
    for d in dimensions:
        net = trainer.ProbabilityNet(d, hidden, mixtures, np.linspace(-1, 1, d))
        with torch.no_grad():
            net.head.weight.normal_(0, 0.3)
        net.register_buffer("guide_coeff", torch.zeros(3))
        nets.append(net)
    return nets


def _data(dimensions, count, seed=5):
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(dimensions), count, max(dimensions))
    for g, d in enumerate(dimensions):
        x[g, :, :d] = torch.randn(count, d, generator=generator) * 0.5
    a = torch.randint(0, 256, (len(dimensions), count), generator=generator).float()
    y = torch.randint(-30, 31, (len(dimensions), count), generator=generator).float()
    return x, torch.clamp(y, -a, 255 - a), a


def test_tensor_bytes_is_the_exact_coded_length(modules):
    _trainer, search, coder = modules
    rng = np.random.default_rng(1)
    for scale in (0.0, 1e-3, 0.3, 5.0, 900.0):
        for size in (1, 7, 300):
            raw = (rng.standard_normal(size) * scale).astype("<f2").tobytes()
            assert search.tensor_bytes(raw) == len(coder._encode_tensor(raw))
            for bits in (12, 8, 5):
                rounded = search.round_state({"head.weight": raw}, bits)["head.weight"]
                assert search.tensor_bytes(rounded) == len(coder._encode_tensor(rounded))


def test_tensor_bytes_matches_sliced_grid_coding(modules):
    _trainer, search, coder = modules
    rng = np.random.default_rng(4)
    for h, d in ((2, 3), (12, 43), (12, 12), (32, 80)):
        matrix = rng.standard_normal((h, d)) * np.exp2(rng.integers(-8, 3, d))
        # Pruned inputs: zero columns, which the coder stores as zero slices.
        matrix[:, rng.permutation(d)[: d // 3]] = 0.0
        state = {
            "affine.weight": bytes(2 * d),
            "hidden.0.bias": bytes(2 * h),
            "head.bias": bytes(6),
            "hidden.0.weight": matrix.astype("<f2").tobytes(),
        }
        for code in (16, 8, search.COLUMN + 8, search.ROW + 6):
            rounded = search.round_state(state, {"hidden.0.weight": code})
            assert search.round_state(rounded, {"hidden.0.weight": code}) == rounded
            raw = rounded["hidden.0.weight"]
            assert search.tensor_bytes(raw, (h, d)) == len(coder._encode_tensor(raw, (h, d)))
            if code > search.COLUMN:
                values = np.frombuffer(raw, "<f2").astype(np.float64).reshape(h, d)
                kind = coder.COLUMNS if code < search.ROW else coder.ROWS
                _shifts, slices = coder.slice_grids(values.ravel(), (h, d), kind)
                assert np.abs(slices).max(axis=1).max() <= 2 ** (code % 100 - 1) - 1


def test_round_state_rounds_only_hidden_and_head_weights(modules):
    _trainer, search, _coder = modules
    rng = np.random.default_rng(2)
    state = {name: rng.standard_normal(12).astype("<f2").tobytes() for name in search.WEIGHTS}
    state.update(
        {
            name: rng.standard_normal(4).astype("<f2").tobytes()
            for name in ("affine.weight", "head.bias")
        }
    )
    once = search.round_state(state, 7)
    assert once["affine.weight"] == state["affine.weight"]
    assert once["head.bias"] == state["head.bias"]
    assert all(once[name] != state[name] for name in search.WEIGHTS)
    assert search.round_state(once, 7) == once
    mixed = search.round_state(state, {"head.weight": 6})
    assert mixed["hidden.0.weight"] == state["hidden.0.weight"]
    assert mixed["head.weight"] == search.round_state(state, 6)["head.weight"]


def test_coded_bits_follow_the_integer_coder(modules):
    import math

    import torch

    trainer, _search, _coder = modules
    sharp = math.exp(-4.0)
    # Rows: a mean 0.9 below the lowest value (the coder's logistic saturates
    # there; the training objective's clamped float mass comes out negative),
    # the same beside a component in range, and two ordinary samples.
    mean = torch.tensor([[-0.9, -0.9], [-0.9, 0.0], [100.2, 90.0], [254.6, 250.0]])
    scale = torch.tensor([[sharp, sharp], [sharp, 1.0], [1.0, 3.0], [0.7, 2.0]])
    target = torch.tensor([0.0, 0.0, 100.0, 255.0])
    anchor = torch.zeros(4)
    logits = torch.zeros(4, 2)
    bits = trainer.coded_bits(logits, mean, scale, target, anchor)
    clamped = trainer.mixture_bits(logits, mean, scale, target, anchor)
    assert clamped[0] < 0 and bits[0] == 8.0
    # Only the component in range carries mass.
    share = (torch.sigmoid(torch.tensor(0.5)) - 0.5) * 2 / (1 - torch.sigmoid(torch.tensor(-0.5)))
    assert abs(float(bits[1]) + math.log2(float(share))) < 1e-3
    np.testing.assert_allclose(bits[2:].numpy(), clamped[2:].numpy(), atol=1e-3)


def test_search_is_never_worse_than_the_best_uniform_grid(modules):
    import torch

    trainer, search, coder = modules
    channels = (1, 0, 2)
    dimensions = [coder.feature_dimensions(c, True, 1)[0] for c in channels]
    hidden = 6
    nets = _nets(trainer, dimensions, hidden)
    x, y, a = _data(dimensions, 2000)
    sources = []
    for net, channel in zip(nets, channels, strict=True):
        state = {k: v.detach().numpy().astype("<f2").tobytes() for k, v in net.state_dict().items()}
        base = coder.feature_dimensions(channel, True, 1)[1]
        state["guide_coeff"] = np.zeros(base, "<f2").tobytes()
        sources.append({"float": state})
    estimator = search.Estimator(x, y, a, dimensions, 4, trainer.coded_bits)
    header = (40, 50, hidden, 4, True, 1)
    with torch.no_grad():
        result = search.search(estimator, header, sources, 1.0)
    assert result["estimated_bytes"] <= result["uniform_estimated_bytes"] + 1e-6
    assert len(result["labels"]) == 3
    for key in ("payload", "uniform_payload"):
        parsed_header, planes = coder.parse_v3(result[key])
        assert parsed_header == header and len(planes) == 3


def test_gptq_rounding_stays_on_its_grids_and_beats_nearest(modules):
    _trainer, search, coder = modules
    rng = np.random.default_rng(6)
    h, d, n = 8, 20, 4000
    # Nonnegative inputs, like ReLU outputs, leave a mean error for the bias.
    x = np.abs(rng.standard_normal((n, d)) @ rng.standard_normal((d, d)) * 0.3)
    xa = np.hstack([x, np.ones((n, 1))])
    w = rng.standard_normal((h, d)) * 0.4
    w[:, 5] = 0.0
    raw = w.astype("<f2").tobytes()
    bias = (rng.standard_normal(h) * 0.1).astype("<f2").tobytes()
    exact = np.hstack(
        [
            np.frombuffer(raw, "<f2").astype(np.float64).reshape(h, d),
            np.frombuffer(bias, "<f2").astype(np.float64)[:, None],
        ]
    )
    gram = xa.T @ xa

    def error(matrix, b=bias):
        rounded = np.hstack(
            [
                np.frombuffer(matrix, "<f2").astype(np.float64).reshape(h, d),
                np.frombuffer(b, "<f2").astype(np.float64)[:, None],
            ]
        )
        delta = rounded - exact
        return np.einsum("ri,ij,rj->", delta, gram, delta)

    codes = [8, 6, search.COLUMN + 6, search.ROW + 5]
    plain, none = search.gptq_round(raw, (h, d), codes, gram[:d, :d])
    assert none is None
    per_row, _ = search.gptq_round(raw, (h, d), codes, np.repeat(gram[None, :d, :d], h, axis=0))
    assert per_row == plain
    compensated, biases = search.gptq_round(raw, (h, d), codes, gram, bias)
    for code, matrix, matrix_b, b in zip(codes, plain, compensated, biases, strict=True):
        nearest = search._round_code(raw, code, (h, d))
        assert error(matrix) < error(nearest)
        assert error(matrix_b, b) < error(matrix)
        for out in (matrix, matrix_b):
            values = np.frombuffer(out, "<f2").astype(np.float64).reshape(h, d)
            assert not values[:, 5].any()
            if code < search.COLUMN:
                steps = values * 2.0 ** search._grid_shift(np.abs(w).max(), code)
            elif code >= search.ROW:
                shifts = search._grid_shift(np.abs(w.astype("<f2")).max(axis=1), code % 100)
                steps = values * np.exp2(shifts)[:, None]
            else:
                _shifts, slices = coder.slice_grids(values.ravel(), (h, d), coder.COLUMNS)
                assert np.abs(slices).max() <= 2 ** (code % 100 - 1) - 1
                continue
            assert np.array_equal(steps, np.rint(steps))


@pytest.mark.parametrize(("n", "d", "rows"), [(3000, 7, 4), (1001, 97, 5), (777, 48, 30)])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_weighted_gram_matches_the_definition(modules, n, d, rows, device):
    import torch

    if device == "mps" and not _mps():
        pytest.skip("needs MPS")
    _trainer, search, _coder = modules
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(n, d, generator=generator)
    weights = torch.rand(n, rows, generator=generator)
    xa = torch.cat([x, torch.ones(n, 1)], dim=1).double()
    expected = torch.einsum("nr,ni,nj->rij", weights.double(), xa, xa).numpy()
    x, weights = x.to(device), weights.to(device)
    atol = 1e-6 * np.abs(expected).max()
    np.testing.assert_allclose(
        search._weighted_gram(x, weights, True), expected, rtol=1e-4, atol=atol
    )
    np.testing.assert_allclose(
        search._weighted_gram(x, weights, False), expected[:, :d, :d], rtol=1e-4, atol=atol
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
@pytest.mark.parametrize("hidden", [6, 12, 48])
def test_fused_output_gradients_match_autograd(modules, hidden):
    if not _mps():
        pytest.skip("needs MPS")
    trainer, search, coder = modules
    dimensions = [coder.feature_dimensions(c, True, 1)[0] for c in (1, 0, 2)]
    x, y, a = _data(dimensions, 3000)
    x = x.half().float()
    reference = search.Estimator(x, y, a, dimensions, 4, trainer.mixture_bits)
    fused = search.Estimator(
        x.half().to("mps"), y.to("mps"), a.to("mps"), dimensions, 4, trainer.mixture_bits
    )
    fused.GRADIENT_ROWS = 1000
    for g, net in enumerate(_nets(trainer, dimensions, hidden)):
        state = {k: v.detach().numpy().astype("<f2").tobytes() for k, v in net.state_dict().items()}
        expected = reference.output_gradients(g, state, hidden)
        got = fused._fused_gradients(g, state, hidden)
        assert got is not None and got.keys() == expected.keys()
        for name, value in expected.items():
            assert got[name].shape == value.shape
            scale = float(value.max())
            np.testing.assert_allclose(
                got[name].cpu().numpy(), value.numpy(), rtol=1e-3, atol=1e-5 * scale
            )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
@pytest.mark.parametrize(
    ("hidden", "mixtures"), [(5, 1), (8, 4), (24, 4), (32, 3), (52, 2), (64, 4)]
)
@pytest.mark.parametrize("half", [False, True])
def test_fused_gradients_match_autograd(modules, hidden, mixtures, half):
    if not _mps():
        pytest.skip("fused training needs Metal")
    import torch
    from metal_training import FusedGroupedTrainer

    trainer, _search, _coder = modules
    dimensions = [39, 57, 76]
    nets = [net.to("mps") for net in _nets(trainer, dimensions, hidden, mixtures)]
    x, y, a = (t.to("mps") for t in _data(dimensions, 3000))
    # The fused trainer keeps inputs in binary16.
    x = x.half().float()
    index = torch.randperm(3000, device="mps")[:2000].to(torch.int32)
    fused = FusedGroupedTrainer(nets, dimensions, x, y, a, lr=0.01, half_scratch=half)
    g1, g2, g3, bits = fused.gradients(index)
    grouped = trainer.GroupedNet(nets, dimensions).to("mps")
    rows = index.long()
    reference = trainer.mixture_bits(*grouped(x[:, rows]), y[:, rows], a[:, rows])
    reference.mean(1).sum().backward()
    d = max(dimensions)
    pairs = [
        (grouped.w1.grad, g1[:, :d, :hidden]),
        (grouped.affine.grad[..., 0], g1[:, :d, hidden]),
        (grouped.b1.grad[:, 0], g1[:, d, :hidden]),
        (grouped.w2.grad, g2[:, :hidden]),
        (grouped.b2.grad[:, 0], g2[:, hidden]),
        (grouped.w3.grad, g3[:, :hidden]),
        (grouped.b3.grad[:, 0], g3[:, hidden]),
    ]
    assert torch.allclose(bits, reference.detach(), rtol=1e-4, atol=1e-3)
    for want, got in pairs:
        assert float((want - got).norm() / want.norm()) < 1e-4


def test_fused_estimates_match_the_reference_estimator(modules):
    if not _mps():
        pytest.skip("fused evaluation needs Metal")
    trainer, search, coder = modules
    channels = (1, 0, 2)
    dimensions = [coder.feature_dimensions(c, True, 1)[0] for c in channels]
    hidden = 12
    nets = _nets(trainer, dimensions, hidden)
    x, y, a = (t.to("mps") for t in _data(dimensions, 1500))
    x = x.half()
    states = [
        {k: v.detach().numpy().astype("<f2").tobytes() for k, v in net.state_dict().items()}
        for net in nets
    ]
    requests = [(g, search.round_state(states[g], bits)) for g in (0, 1, 2) for bits in (12, 7)]
    # Means far below the range, where the coder's logistic saturates.
    bias = np.frombuffer(states[0]["head.bias"], "<f2").copy()
    bias[4:8] = -1000.0
    requests.append((0, {**states[0], "head.bias": bias.astype("<f2").tobytes()}))
    fused = search.Estimator(x, y, a, dimensions, 4, trainer.coded_bits)
    reference = search.Estimator(x, y, a, dimensions, 4, trainer.coded_bits)
    got = fused.many(requests, hidden)
    want = [reference.bits(g, state, hidden) for g, state in requests]
    assert np.allclose(got, want, rtol=1e-5)
    assert fused.many(requests[:1], hidden) == got[:1]


@pytest.mark.parametrize("hidden", [6, 12, 48])
def test_staged_estimates_equal_full_estimates(modules, hidden):
    if not _mps():
        pytest.skip("fused evaluation needs Metal")
    trainer, search, coder = modules
    dimensions = [coder.feature_dimensions(c, True, 1)[0] for c in (1, 0, 2)]
    nets = _nets(trainer, dimensions, hidden)
    x, y, a = (t.to("mps") for t in _data(dimensions, 1500))
    x = x.half()
    staged = search.Estimator(x, y, a, dimensions, 4, trainer.coded_bits)
    for g, net in enumerate(nets):
        state = {k: v.detach().numpy().astype("<f2").tobytes() for k, v in net.state_dict().items()}
        base = search.round_state(state, 12)
        changes = [
            {"hidden.2.weight": 7},
            {"hidden.2.weight": search.COLUMN + 6, "hidden.2.bias": 8},
            {"hidden.2.bias": 9},
            {"head.weight": 6},
            {"head.weight": search.ROW + 7, "head.bias": 8},
            {"affine.weight": 9},
            {"head.weight": 7, "affine.weight": 9},
            {"hidden.0.weight": 6},
        ]
        states = [search.round_state(base, change) for change in changes]
        full = search.Estimator(x, y, a, dimensions, 4, trainer.coded_bits)
        want = full.many([(g, s) for s in states], hidden)
        assert staged.many_after(g, base, states, hidden) == want
        # One at a time, each from base's cached outputs.
        single = search.Estimator(x, y, a, dimensions, 4, trainer.coded_bits)
        assert [single.many_after(g, base, [s], hidden)[0] for s in states] == want


def test_quantized_snapshot_is_on_the_post_training_grid(modules):
    if not _mps():
        pytest.skip("fused training needs Metal")
    import copy

    import torch
    from metal_training import FusedGroupedTrainer

    trainer, search, _coder = modules
    dimensions = [39, 57, 76]
    nets = [net.to("mps") for net in _nets(trainer, dimensions, 16)]
    with torch.no_grad():
        for net in nets:
            for value in net.parameters():
                value.copy_(value.half().float())
    x, y, a = (t.to("mps") for t in _data(dimensions, 512))
    fused = FusedGroupedTrainer(nets, dimensions, x, y, a, lr=0.01)
    before = trainer._states(nets)
    snapshot = [copy.deepcopy(net) for net in nets]
    fused.export(snapshot, fused.quantized(8))
    for plane, state in zip(before, trainer._states(snapshot), strict=True):
        assert state == search.round_state(plane, 8)


def test_pruned_inputs_stay_zero_and_folding_keeps_the_outputs(modules):
    if not _mps():
        pytest.skip("fused training needs Metal")
    import torch
    from metal_training import FusedGroupedTrainer

    trainer, _search, _coder = modules
    dimensions = [39, 57, 76]
    x, y, a = (t.to("mps") for t in _data(dimensions, 1024))
    x[:, :, 0] = 1 / 32
    x = x.half().float()
    nets = [net.to("mps") for net in _nets(trainer, dimensions, 8)]
    fused = FusedGroupedTrainer(nets, dimensions, x, y, a, lr=0.01)
    index = torch.arange(1024, device="mps", dtype=torch.int32)
    before = fused.gradients(index)[3]
    fused.fold(0, 1 / 32)
    assert torch.allclose(fused.gradients(index)[3], before, rtol=1e-5, atol=1e-4)
    pruner = trainer.Pruner(fused, 4, 1024)
    assert pruner.epochs == {1, 2}
    torch.manual_seed(11)
    trainer._train_fused(fused, 1024, 256, 4, 0.01, "cosine", prune=pruner)
    assert all(0 in plane and len(plane) > 1 for plane in pruner.pruned)
    trainer._train_fused(fused, 1024, 256, 1, 0.002, "cosine", bits=8)
    fused.export(nets)
    for net, plane in zip(nets, pruner.pruned, strict=True):
        weight = net.hidden[0].weight.detach().cpu()
        assert torch.all(weight[:, sorted(plane)] == 0)
        live = [j for j in range(weight.shape[1]) if j not in plane]
        assert torch.all(weight[:, live].abs().sum(0) > 0)


def test_race_checkpoint_does_not_change_training(modules):
    if not _mps():
        pytest.skip("fused training needs Metal")
    import torch
    from metal_training import FusedGroupedTrainer

    trainer, _search, _coder = modules
    dimensions = [39, 57, 76]
    x, y, a = (t.to("mps") for t in _data(dimensions, 1024))
    x = x.half().float()
    finals, scores = [], []

    def score(mean_bits):
        assert len(mean_bits) == 3 and all(bits > 0 for bits in mean_bits)
        scores.append(trainer.race_score(fused, nets, mean_bits, 1024))
        return True

    for checkpoint in (None, (2, score)):
        nets = [net.to("mps") for net in _nets(trainer, dimensions, 8)]
        fused = FusedGroupedTrainer(nets, dimensions, x, y, a, lr=0.01)
        torch.manual_seed(11)
        trainer._train_fused(fused, 1024, 256, 4, 0.01, "cosine", checkpoint=checkpoint)
        fused.export(nets)
        finals.append(trainer._states(nets))
    assert finals[0] == finals[1]
    assert len(scores) == 1 and scores[0] > 0
    nets = [net.to("mps") for net in _nets(trainer, dimensions, 8)]
    fused = FusedGroupedTrainer(nets, dimensions, x, y, a, lr=0.01)
    with pytest.raises(trainer.Dropped):
        trainer._train_fused(fused, 1024, 256, 4, 0.01, "cosine", checkpoint=(2, lambda _: False))
