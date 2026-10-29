"""paq8px comparison arithmetic: only verified archives count."""

from __future__ import annotations

from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import summarize_paq8px

    return summarize_paq8px


def receipt(phase, rows):
    return {"phase": phase, "cohort": "c", "images": rows}


def compressed(name, size, sha):
    return {
        "image": name,
        "archive_bytes": size,
        "archive_sha256": sha,
        "compress_seconds": 30.0,
        "compress_peak_rss_bytes": 5,
    }


def checked(name, size, sha, exact=True):
    return {
        "image": name,
        "archive_bytes": size,
        "archive_sha256": sha,
        "exact_rgb": exact,
        "decompress_seconds": 31.0,
        "decompress_peak_rss_bytes": 7,
    }


def ours(name, size):
    return {
        "image": name,
        "complete_file_bytes": size,
        "baselines": {"jxl-lossless-e10": 200, "png": 400},
        "best_baseline": "jxl-lossless-e10",
        "bits_per_sample": round(8 * size / 800, 4),
        "samples": 800,
        "encode_seconds": 10.0,
        "decode_seconds": 1.0,
    }


def test_only_exact_matching_archives_count(compare):
    archives = compare.verified_archives(
        [
            receipt("compress", [compressed("a", 100, "x"), compressed("b", 90, "y")]),
            receipt("verify", [checked("a", 100, "x"), checked("b", 90, "y", exact=False)]),
        ]
    )
    assert set(archives) == {("c", "a")}
    assert archives["c", "a"]["decompress_seconds"] == 31.0


def test_changed_archive_is_rejected(compare):
    with pytest.raises(ValueError):
        compare.verified_archives(
            [
                receipt("compress", [compressed("a", 100, "x")]),
                receipt("verify", [checked("a", 100, "z")]),
            ]
        )


def test_signs_and_aggregate(compare):
    paq = {**compressed("a", 100, "x"), **checked("a", 100, "x")}
    rows = [
        compare.image_row(ours("a", 80), paq),
        compare.image_row(ours("b", 110), paq | {"image": "b"}),
    ]
    assert rows[0]["learned_vs_paq8px_percent"] == 20.0
    assert rows[0]["paq8px_vs_best_baseline_percent"] == 50.0
    assert rows[0]["paq8px_peak_rss_bytes"] == 7
    result = compare.aggregate(rows)
    assert (result["learned_smaller"], result["paq8px_smaller"]) == (1, 1)
    assert result["aggregate_learned_vs_paq8px_percent"] == pytest.approx(5.0)
    assert result["paq8px_bits_per_sample"] == 1.0
    assert result["median_paq8px_decompress_seconds"] == 31.0
