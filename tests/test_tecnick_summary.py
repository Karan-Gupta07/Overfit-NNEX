"""Summary arithmetic for the Tecnick receipts: complete files, smallest baseline."""

from __future__ import annotations

from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import summarize_tecnick

    return summarize_tecnick


def report(name, ours, e10, *, floor, learned, prior=None):
    selected = {"format": "nnex-v3"} if ours in learned else {"format": "scdx"}
    row = {
        "image": name,
        "complete_file_bytes": ours,
        "file_sha256": name + "-hash",
        "exact_rgba": True,
        "selected": selected,
        "baselines": {
            "png": 4 * e10,
            "webp-lossless": 2 * e10,
            "jxl-lossless-e9": e10 + 1,
            "jxl-lossless-e10": e10,
        },
        "candidates": [{"format": "scdx", "complete_file_bytes": floor}]
        + [{"format": "nnex-v3", "complete_file_bytes": size} for size in learned],
        "seconds": 10.0,
        "decode_seconds": 1.0,
        "baseline_seconds": 2.0,
    }
    if prior is not None:
        row["prior_round_reference"] = {"complete_file_bytes": prior, "seconds": 20.0}
    return row


def test_image_row_counts_smallest_baseline_and_prior(summary):
    row = summary.image_row(
        report("a.png", 80, 100, floor=110, learned=[80, 90], prior=95), 400, {"a.png-hash"}
    )
    assert row["best_baseline"] == "jxl-lossless-e10"
    assert row["saving_vs_best_baseline_percent"] == 20.0
    assert row["learned_best_bytes"] == 80
    assert row["learned_vs_prior_round_percent"] == pytest.approx(100 * (1 - 80 / 95), abs=1e-3)
    assert row["bits_per_sample"] == 1.6
    assert row["cross_isa_identical"] is True


def test_aggregate_counts_losses_and_floor(summary):
    rows = [
        summary.image_row(report("a.png", 80, 100, floor=110, learned=[80]), 400, set()),
        summary.image_row(report("b.png", 105, 100, floor=105, learned=[120]), 400, set()),
    ]
    result = summary.aggregate(rows)
    assert (result["wins"], result["losses"], result["selected_learned"]) == (1, 1, 1)
    assert result["complete_file_bytes"] == 185
    assert result["aggregate_saving_vs_best_baseline_percent"] == 7.5
    assert result["median_saving_vs_best_baseline_percent"] == pytest.approx(7.5)
    assert result["saved_bits_per_sample"] == pytest.approx(8 * 15 / 800)
    assert result["scdx_floor_bytes"] == 215
    assert result["selected_learned_cross_isa_identical"] == 0
    assert "learned_vs_prior_round" not in result
