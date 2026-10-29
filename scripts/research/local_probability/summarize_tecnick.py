"""Summarize raw Tecnick validation receipts into one durable results file.

Counts complete files only (all model bytes and headers). A saving is relative
to the smallest verified baseline for that image; positive means smaller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "data/corpora/tecnick-manifest.json"
HERE = Path(__file__).resolve().parent
IMPLEMENTATION = (
    "compress.py",
    "neural_probability_probe.py",
    "tecnick_validation.py",
    "quantize_model.py",
    "neural_bitstream.py",
    "input_image.py",
    "nnproto_integer.cpp",
    "arithmetic.hpp",
)


def saving(ours: int, reference: int) -> float:
    return 100.0 * (1.0 - ours / reference)


def bits_per_sample(size: int, samples: int) -> float:
    return round(8 * size / samples, 4)


def image_row(report: dict, samples: int, verified: set[str]) -> dict:
    baselines = report["baselines"]
    best_name = min(baselines, key=baselines.get)
    best = baselines[best_name]
    learned = [c for c in report["candidates"] if c["format"] != "scdx"]
    floor = [c["complete_file_bytes"] for c in report["candidates"] if c["format"] == "scdx"]
    ours = report["complete_file_bytes"]
    row = {
        "image": report["image"],
        "complete_file_bytes": ours,
        "file_sha256": report["file_sha256"],
        "exact_rgba": report["exact_rgba"],
        "selected": report["selected"],
        "baselines": baselines,
        "best_baseline": best_name,
        "saving_vs_best_baseline_percent": round(saving(ours, best), 3),
        "saving_vs_jxl_e10_percent": round(saving(ours, baselines["jxl-lossless-e10"]), 3),
        "samples": samples,
        "bits_per_sample": bits_per_sample(ours, samples),
        "best_baseline_bits_per_sample": bits_per_sample(best, samples),
        "scdx_floor_bytes": floor[0] if floor else None,
        "learned_best_bytes": min((c["complete_file_bytes"] for c in learned), default=None),
        "encode_seconds": round(report["seconds"], 1),
        "decode_seconds": round(report["decode_seconds"], 2),
        "baseline_seconds": round(report["baseline_seconds"], 2),
        "cross_isa_identical": report["file_sha256"] in verified
        if report["selected"]["format"] != "scdx"
        else None,
    }
    prior = report.get("prior_round_reference")
    if prior is not None:
        row["prior_round_learned_bytes"] = prior["complete_file_bytes"]
        row["prior_round_seconds"] = round(prior["seconds"], 1)
        if row["learned_best_bytes"] is not None:
            row["learned_vs_prior_round_percent"] = round(
                saving(row["learned_best_bytes"], prior["complete_file_bytes"]), 3
            )
    return row


def aggregate(rows: list[dict]) -> dict:
    ours = sum(r["complete_file_bytes"] for r in rows)
    best = sum(r["baselines"][r["best_baseline"]] for r in rows)
    e10 = sum(r["baselines"]["jxl-lossless-e10"] for r in rows)
    savings = [r["saving_vs_best_baseline_percent"] for r in rows]
    samples = sum(r["samples"] for r in rows)
    learned = [r for r in rows if r["selected"]["format"] != "scdx"]
    result = {
        "images": len(rows),
        "complete_file_bytes": ours,
        "best_baseline_bytes": best,
        "jxl_e10_bytes": e10,
        "aggregate_saving_vs_best_baseline_percent": round(saving(ours, best), 3),
        "aggregate_saving_vs_jxl_e10_percent": round(saving(ours, e10), 3),
        "bits_per_sample": bits_per_sample(ours, samples),
        "best_baseline_bits_per_sample": bits_per_sample(best, samples),
        "saved_bits_per_sample": round(8 * (best - ours) / samples, 4),
        "scdx_floor_bytes": sum(r["scdx_floor_bytes"] for r in rows),
        "saving_vs_scdx_floor_percent": round(
            saving(ours, sum(r["scdx_floor_bytes"] for r in rows)), 3
        ),
        "median_saving_vs_best_baseline_percent": round(statistics.median(savings), 3),
        "min_saving_percent": min(savings),
        "max_saving_percent": max(savings),
        "wins": sum(s > 0 for s in savings),
        "losses": sum(s < 0 for s in savings),
        "selected_learned": len(learned),
        "selected_learned_cross_isa_identical": sum(
            bool(r["cross_isa_identical"]) for r in learned
        ),
        "all_exact_rgba": all(r["exact_rgba"] for r in rows),
        "median_encode_seconds": round(statistics.median(r["encode_seconds"] for r in rows), 1),
        "median_decode_seconds": round(statistics.median(r["decode_seconds"] for r in rows), 2),
    }
    prior = [r for r in rows if "learned_vs_prior_round_percent" in r]
    if prior:
        current = sum(r["learned_best_bytes"] for r in prior)
        before = sum(r["prior_round_learned_bytes"] for r in prior)
        result["learned_vs_prior_round"] = {
            "images": len(prior),
            "current_learned_bytes": current,
            "prior_round_learned_bytes": before,
            "aggregate_saving_percent": round(saving(current, before), 3),
            "median_saving_percent": round(
                statistics.median(r["learned_vs_prior_round_percent"] for r in prior), 3
            ),
            "current_smaller": sum(r["learned_vs_prior_round_percent"] > 0 for r in prior),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--cross-isa", type=Path, nargs="*", default=[])
    parser.add_argument("--timing", type=Path, help="supplementary decode_timing.py receipt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    shapes = {
        (name, row["image"]): row["shape"]
        for name, rows in manifest["cohorts"].items()
        for row in rows
    }
    cross = [json.loads(path.read_text()) for path in args.cross_isa]
    verified = {
        row["file_sha256"]
        for receipt in cross
        if receipt["complete"]
        for row in receipt["images"]
        if row["exact_rgb"] and row["identical_encoded_bytes"]
    }
    cohorts: dict[str, list[dict]] = {}
    sources = []
    for path in args.receipts:
        receipt = json.loads(path.read_text())
        cohort = receipt["scope"].rsplit(" ", 1)[-1]
        cohorts.setdefault(cohort, []).extend(
            image_row(r, math.prod(shapes[cohort, r["image"]]), verified) for r in receipt["images"]
        )
        sources.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "complete": receipt["complete"],
                "protocol_sha256": receipt["protocol_sha256"],
                "native_backend_sha256": receipt["native_backend_sha256"],
                "images": len(receipt["images"]),
            }
        )
    if len({s["protocol_sha256"] for s in sources}) != 1:
        raise ValueError("receipts use different protocols")
    summary = {
        "scope": "Fresh frozen Tecnick validation; complete files including every model byte",
        "protocol": "docs/experiments/2026-10-24-tecnick-protocol.json",
        "protocol_sha256": sources[0]["protocol_sha256"],
        "raw_receipts": sources,
        "all_receipts_complete": all(s["complete"] for s in sources),
        "cohorts": {
            name: {"summary": aggregate(rows), "images": rows} for name, rows in cohorts.items()
        },
        "combined": aggregate([r for rows in cohorts.values() for r in rows]),
    }
    summary["manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    summary["implementation_sha256_at_summary"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in IMPLEMENTATION
    }
    summary["cross_isa"] = [
        {key: value for key, value in receipt.items() if key != "images"}
        | {"files": len(receipt["images"])}
        for receipt in cross
    ]
    if args.timing:
        summary["supplementary_decode_timing"] = json.loads(args.timing.read_text())
    text = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(json.dumps({k: v["summary"] for k, v in summary["cohorts"].items()}, indent=2))
    print(json.dumps({"combined": summary["combined"]}, indent=2))


if __name__ == "__main__":
    main()
