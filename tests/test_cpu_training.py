"""The fused trainer's CPU copy (cpu_training) against autograd, Adam and the Metal trainer."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("imagecodecs") is None,
    reason="optional trainer requires ml and hybrid extras",
)
DIMENSIONS = [39, 57, 76]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import cpu_training
    import model_search
    import neural_probability_probe

    return neural_probability_probe, model_search, cpu_training


def _mps() -> bool:
    import torch

    return bool(torch.backends.mps.is_available())


def _nets(trainer, hidden, mixtures=4, seed=3):
    import torch

    torch.manual_seed(seed)
    nets = []
    for d in DIMENSIONS:
        net = trainer.ProbabilityNet(d, hidden, mixtures, np.linspace(-1, 1, d))
        with torch.no_grad():
            net.head.weight.normal_(0, 0.3)
            for value in net.parameters():
                value.copy_(value.half().float())
        net.register_buffer("guide_coeff", torch.zeros(3))
        nets.append(net)
    return nets


def _data(count, seed=5):
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(DIMENSIONS), count, max(DIMENSIONS))
    for g, d in enumerate(DIMENSIONS):
        x[g, :, :d] = torch.randn(count, d, generator=generator) * 0.5
    a = torch.randint(0, 256, (len(DIMENSIONS), count), generator=generator).float()
    y = torch.randint(-30, 31, (len(DIMENSIONS), count), generator=generator).float()
    # The fused trainers keep inputs in binary16.
    return x.half().float(), torch.clamp(y, -a, 255 - a), a


def _cpu(trainer, cpu, nets, x, y, a, max_norm=10.0):
    return cpu.TorchGroupedTrainer(nets, DIMENSIONS, x, y, a, 0.01, max_norm)


@pytest.mark.parametrize(("hidden", "mixtures"), [(5, 1), (12, 4), (32, 3)])
def test_cpu_gradients_match_autograd(modules, hidden, mixtures):
    import torch

    trainer, _search, cpu = modules
    nets = _nets(trainer, hidden, mixtures)
    x, y, a = _data(3000)
    index = torch.randperm(3000)[:2000].to(torch.int32)
    fused = _cpu(trainer, cpu, nets, x, y, a)
    g1, g2, g3, bits = fused.gradients(index)
    grouped = trainer.GroupedNet(nets, DIMENSIONS)
    rows = index.long()
    reference = trainer.mixture_bits(*grouped(x[:, rows]), y[:, rows], a[:, rows])
    reference.mean(1).sum().backward()
    d = max(DIMENSIONS)
    pairs = [
        (grouped.w1.grad, g1[:, :d, :hidden]),
        (grouped.affine.grad[..., 0], g1[:, :d, hidden]),
        (grouped.b1.grad[:, 0], g1[:, d, :hidden]),
        (grouped.w2.grad, g2[:, :hidden]),
        (grouped.b2.grad[:, 0], g2[:, hidden]),
        (grouped.w3.grad, g3[:, :hidden]),
        (grouped.b3.grad[:, 0], g3[:, hidden]),
    ]
    assert torch.allclose(bits, reference.detach(), rtol=1e-4, atol=1e-4)
    for want, got in pairs:
        assert float((want - got).norm() / want.norm()) < 1e-5
    # The unused w1a cell (bias row, affine column) gets no gradient.
    assert torch.all(g1[:, d, hidden] == 0)


def test_cpu_steps_are_adam_with_per_plane_clipping(modules):
    import torch

    trainer, _search, cpu = modules
    nets = _nets(trainer, 12)
    x, y, a = _data(1024)
    fused = _cpu(trainer, cpu, nets, x, y, a, max_norm=0.5)
    grouped = trainer.GroupedNet(copy.deepcopy(nets), DIMENSIONS)
    optimizer = torch.optim.Adam(grouped.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(7)
    for _ in range(5):
        index = torch.randperm(1024, generator=generator)[:256]
        lost = fused.step(index.to(torch.int32), 0.01)
        optimizer.zero_grad()
        reference = trainer.mixture_bits(*grouped(x[:, index]), y[:, index], a[:, index])
        reference.mean(1).sum().backward()
        trainer.clip_per_plane(grouped.parameters(), 0.5)
        optimizer.step()
        assert torch.allclose(lost, reference.detach(), rtol=1e-4, atol=1e-4)
    fused.export(nets)
    for g, net in enumerate(nets):
        d = DIMENSIONS[g]
        np.testing.assert_allclose(
            net.hidden[0].weight.detach().numpy(),
            grouped.w1[g, :d].T.detach().numpy(),
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            net.head.bias.detach().numpy(), grouped.b3[g, 0].detach().numpy(), atol=1e-6
        )


def test_cpu_quantized_snapshot_is_on_the_post_training_grid(modules):
    trainer, search, cpu = modules
    nets = _nets(trainer, 16)
    x, y, a = _data(512)
    fused = _cpu(trainer, cpu, nets, x, y, a)
    before = trainer._states(nets)
    for bits in (12, 8):
        snapshot = [copy.deepcopy(net) for net in nets]
        fused.export(snapshot, fused.quantized(bits))
        for plane, state in zip(before, trainer._states(snapshot), strict=True):
            assert state == search.round_state(plane, bits)


def test_cpu_evaluator_matches_the_reference_estimator(modules):
    trainer, search, cpu = modules
    hidden = 12
    nets = _nets(trainer, hidden)
    x, y, a = _data(1500)
    states = [
        {k: v.detach().numpy().astype("<f2").tobytes() for k, v in net.state_dict().items()}
        for net in nets
    ]
    requests = [(g, search.round_state(states[g], bits)) for g in (2, 0, 1) for bits in (12, 7)]
    # Means far below the range, where the coder's logistic saturates.
    bias = np.frombuffer(states[0]["head.bias"], "<f2").copy()
    bias[4:8] = -1000.0
    requests.append((0, {**states[0], "head.bias": bias.astype("<f2").tobytes()}))
    reference = search.Estimator(x.half(), y, a, DIMENSIONS, 4, trainer.coded_bits)
    want = [reference.bits(g, state, hidden) for g, state in requests]
    estimator = search.Estimator(x.half(), y, a, DIMENSIONS, 4, trainer.coded_bits)
    blocks = np.stack([estimator._pack(g, state, hidden) for g, state in requests])
    evaluator = cpu.TorchEvaluator(x, y, a, hidden, 4, DIMENSIONS, loss=trainer.coded_bits)
    evaluator.GROUP, evaluator.CHUNK = 2, 512
    got = evaluator.bits([g for g, _ in requests], blocks)
    assert np.allclose(got, want, rtol=1e-5)


def test_cpu_pruning_and_race_checkpoints(modules):
    import torch

    trainer, _search, cpu = modules
    x, y, a = _data(1024)
    x[:, :, 0] = 1 / 32
    nets = _nets(trainer, 8)
    fused = _cpu(trainer, cpu, nets, x, y, a)
    index = torch.arange(1024, dtype=torch.int32)
    before = fused.gradients(index)[3]
    fused.fold(0, 1 / 32)
    assert torch.allclose(fused.gradients(index)[3], before, rtol=1e-5, atol=1e-4)
    pruner = trainer.Pruner(fused, 4, 1024)
    torch.manual_seed(11)
    trainer._train_fused(fused, 1024, 256, 4, 0.01, "cosine", prune=pruner)
    assert all(0 in plane and len(plane) > 1 for plane in pruner.pruned)
    trainer._train_fused(fused, 1024, 256, 1, 0.002, "cosine", bits=8)
    fused.export(nets)
    for net, plane in zip(nets, pruner.pruned, strict=True):
        weight = net.hidden[0].weight.detach()
        assert torch.all(weight[:, sorted(plane)] == 0)

    finals, scores = [], []

    def score(mean_bits):
        scores.append(trainer.race_score(fused, nets, mean_bits, 1024))
        return True

    for checkpoints in ((), [(2, score)]):
        nets = _nets(trainer, 8)
        fused = _cpu(trainer, cpu, nets, x, y, a)
        torch.manual_seed(11)
        trainer._train_fused(fused, 1024, 256, 4, 0.01, "cosine", checkpoints=checkpoints)
        fused.export(nets)
        finals.append(trainer._states(nets))
    assert finals[0] == finals[1]
    assert len(scores) == 1 and scores[0] > 0
    with pytest.raises(trainer.Dropped):
        trainer._train_fused(
            fused, 1024, 256, 2, 0.01, "cosine", checkpoints=[(1, lambda _: False)]
        )


def test_cpu_training_follows_the_metal_trainer(modules):
    if not _mps():
        pytest.skip("comparison needs Metal")
    import torch
    from metal_training import FusedEvaluator, FusedGroupedTrainer

    trainer, _search, cpu = modules
    x, y, a = _data(2048)
    metal = FusedGroupedTrainer(
        [n.to("mps") for n in _nets(trainer, 12)],
        DIMENSIONS,
        x.to("mps"),
        y.to("mps"),
        a.to("mps"),
        lr=0.01,
    )
    fused = _cpu(trainer, cpu, _nets(trainer, 12), x, y, a)
    generator = torch.Generator().manual_seed(9)
    for _ in range(20):
        index = torch.randperm(2048, generator=generator)[:512].to(torch.int32)
        want = metal.step(index.to("mps"), 0.01).cpu()
        got = fused.step(index, 0.01)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3)
    params = metal.params.cpu()
    assert float((fused.params - params).norm() / params.norm()) < 1e-4
    # Same parameters, same grid.
    metal.params.copy_(fused.params.to("mps"))
    assert torch.equal(fused.quantized(10), metal.quantized(10).cpu())
    blocks = np.stack([params[g].numpy() for g in range(3)])
    reference = FusedEvaluator(x.to("mps"), y.to("mps"), a.to("mps"), 12, 4, DIMENSIONS)
    evaluator = cpu.TorchEvaluator(x, y, a, 12, 4, DIMENSIONS, loss=trainer.coded_bits)
    assert np.allclose(
        evaluator.bits([0, 1, 2], blocks), reference.bits([0, 1, 2], blocks), rtol=1e-5
    )


def test_torch_trainer_on_cuda_follows_the_cpu(modules):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("comparison needs a CUDA or ROCm GPU")
    trainer, _search, cpu = modules
    x, y, a = _data(2048)
    gpu = _cpu(trainer, cpu, [n.cuda() for n in _nets(trainer, 12)], x.cuda(), y.cuda(), a.cuda())
    fused = _cpu(trainer, cpu, _nets(trainer, 12), x, y, a)
    generator = torch.Generator().manual_seed(9)
    for _ in range(20):
        index = torch.randperm(2048, generator=generator)[:512].to(torch.int32)
        want = fused.step(index, 0.01)
        got = gpu.step(index.cuda(), 0.01).cpu()
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3)
    assert float((gpu.params.cpu() - fused.params).norm() / fused.params.norm()) < 1e-4
    blocks = np.stack([fused.params[g].numpy() for g in range(3)])
    evaluator = cpu.TorchEvaluator(x, y, a, 12, 4, DIMENSIONS, loss=trainer.coded_bits)
    on_gpu = cpu.TorchEvaluator(
        x.cuda(), y.cuda(), a.cuda(), 12, 4, DIMENSIONS, loss=trainer.coded_bits
    )
    assert np.allclose(on_gpu.bits([0, 1, 2], blocks), evaluator.bits([0, 1, 2], blocks), rtol=1e-5)


def test_cuda_graph_steps_match_eager_steps(modules):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    trainer, _search, cpu = modules
    x, y, a = (t.cuda() for t in _data(2048))
    runs = []
    for graphs in (False, True):
        fused = _cpu(trainer, cpu, [n.cuda() for n in _nets(trainer, 12)], x, y, a)
        fused.graphs = graphs
        generator = torch.Generator().manual_seed(9)
        lost = []
        for i in range(40):
            if i == 15:
                fused.prune([[3], [5, 7], []])
            if i == 25:
                saved = fused.state()
            size = 300 if i % 9 == 8 else 512
            index = torch.randperm(2048, generator=generator)[:size].to(torch.int32).cuda()
            lost.append(fused.step(index, 0.01, bits=12 if i >= 30 else None).cpu())
        fused.restore(saved)
        index = torch.arange(512, dtype=torch.int32, device="cuda")
        lost.append(fused.step(index, 0.01).cpu())
        runs.append((fused.params.cpu(), lost))
    assert torch.equal(runs[0][0], runs[1][0])
    assert all(torch.equal(p, q) for p, q in zip(runs[0][1], runs[1][1], strict=True))


@pytest.mark.parametrize(("hidden", "mixtures"), [(12, 4), (48, 4)])
def test_cuda_gradients_match_autograd(modules, hidden, mixtures):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    trainer, _search, cpu = modules
    nets = [n.cuda() for n in _nets(trainer, hidden, mixtures)]
    x, y, a = (t.cuda() for t in _data(3000))
    # 2000 rows: seven chunks of 256 and a remainder.
    index = torch.randperm(3000)[:2000].to(torch.int32).cuda()
    fused = _cpu(trainer, cpu, nets, x, y, a)
    g1, g2, g3, bits = fused.gradients(index)
    fused.split = 0
    whole = fused.gradients(index)
    grouped = trainer.GroupedNet(nets, DIMENSIONS)
    rows = index.long()
    reference = trainer.mixture_bits(*grouped(x[:, rows]), y[:, rows], a[:, rows])
    reference.mean(1).sum().backward()
    d = max(DIMENSIONS)
    pairs = [
        (grouped.w1.grad, g1[:, :d, :hidden]),
        (grouped.affine.grad[..., 0], g1[:, :d, hidden]),
        (grouped.b1.grad[:, 0], g1[:, d, :hidden]),
        (grouped.w2.grad, g2[:, :hidden]),
        (grouped.b2.grad[:, 0], g2[:, hidden]),
        (grouped.w3.grad, g3[:, :hidden]),
        (grouped.b3.grad[:, 0], g3[:, hidden]),
    ]
    assert torch.allclose(bits, reference.detach(), rtol=1e-4, atol=1e-4)
    for want, got in pairs:
        assert float((want - got).norm() / want.norm()) < 1e-5
    for split, one in zip((g1, g2, g3), whole[:3], strict=True):
        assert float((split - one).norm() / one.norm()) < 1e-6
