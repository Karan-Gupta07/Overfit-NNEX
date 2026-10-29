"""Compare verified paq8px archives with the frozen Tecnick results, per image.

Supplementary and post-hoc: the learned results are read unchanged from the
durable Tecnick summary. A paq8px size counts only once its verify receipt
shows an exact round trip of the same archive bytes. Positive savings mean the
learned file is smaller than the paq8px archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "docs/results/2026-10-24-tecnick.json"


def saving(ours: int, reference: int) -> float:
    return 100.0 * (1.0 - ours / reference)


def verified_archives(receipts: list[dict]) -> dict[tuple[str, str], dict]:
    compressed: dict[tuple[str, str], dict] = {}
    checked: dict[tuple[str, str], dict] = {}
    for receipt in receipts:
        target = compressed if receipt["phase"] == "compress" else checked
        for row in receipt["images"]:
            target[receipt["cohort"], row["image"]] = row
    archives = {}
    for key, row in compressed.items():
        check = checked.get(key)
        if check is None or not check["exact_rgb"]:
            continue
        if check["archive_sha256"] != row["archive_sha256"]:
            raise ValueError(f"verified archive differs from compressed archive for {key}")
        archives[key] = row | check
    return archives


def image_row(ours: dict, paq: dict) -> dict:
    size = paq["archive_bytes"]
    best = ours["baselines"][ours["best_baseline"]]
    return {
        "image": ours["image"],
        "learned_bytes": ours["complete_file_bytes"],
        "paq8px_bytes": size,
        "best_baseline_bytes": best,
        "learned_vs_paq8px_percent": round(saving(ours["complete_file_bytes"], size), 3),
        "paq8px_vs_best_baseline_percent": round(saving(size, best), 3),
        "learned_bits_per_sample": ours["bits_per_sample"],
        "paq8px_bits_per_sample": round(8 * size / ours["samples"], 4),
        "samples": ours["samples"],
        "learned_encode_seconds": ours["encode_seconds"],
        "learned_decode_seconds": ours["decode_seconds"],
        "paq8px_compress_seconds": paq["compress_seconds"],
        "paq8px_decompress_seconds": paq["decompress_seconds"],
        "paq8px_peak_rss_bytes": max(
            paq["compress_peak_rss_bytes"], paq["decompress_peak_rss_bytes"]
        ),
        "paq8px_exact_rgb": paq["exact_rgb"],
    }


def aggregate(rows: list[dict]) -> dict:
    ours = sum(r["learned_bytes"] for r in rows)
    paq = sum(r["paq8px_bytes"] for r in rows)
    best = sum(r["best_baseline_bytes"] for r in rows)
    samples = sum(r["samples"] for r in rows)
    savings = [r["learned_vs_paq8px_percent"] for r in rows]

    def median(key: str) -> float:
        return round(statistics.median(r[key] for r in rows), 2)

    return {
        "images": len(rows),
        "learned_bytes": ours,
        "paq8px_bytes": paq,
        "best_baseline_bytes": best,
        "aggregate_learned_vs_paq8px_percent": round(saving(ours, paq), 3),
        "median_learned_vs_paq8px_percent": round(statistics.median(savings), 3),
        "min_learned_vs_paq8px_percent": min(savings),
        "max_learned_vs_paq8px_percent": max(savings),
        "learned_smaller": sum(s > 0 for s in savings),
        "paq8px_smaller": sum(s < 0 for s in savings),
        "aggregate_paq8px_vs_best_baseline_percent": round(saving(paq, best), 3),
        "learned_bits_per_sample": round(8 * ours / samples, 4),
        "paq8px_bits_per_sample": round(8 * paq / samples, 4),
        "all_paq8px_exact_rgb": all(r["paq8px_exact_rgb"] for r in rows),
        "median_learned_encode_seconds": median("learned_encode_seconds"),
        "median_learned_decode_seconds": median("learned_decode_seconds"),
        "median_paq8px_compress_seconds": median("paq8px_compress_seconds"),
        "median_paq8px_decompress_seconds": median("paq8px_decompress_seconds"),
        "max_paq8px_peak_rss_bytes": max(r["paq8px_peak_rss_bytes"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="+", type=Path, help="paq8px_baseline.py receipts")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    loaded = [json.loads(path.read_text()) for path in args.receipts]
    if len({r["binary_sha256"] for r in loaded}) != 1:
        raise ValueError("receipts use different paq8px binaries")
    if {r["level"] for r in loaded if r["phase"] == "compress"} != {"-9L"}:
        raise ValueError("expected every compress receipt at -9L")
    archives = verified_archives(loaded)
    cohorts = {}
    missing = []
    for name, cohort in results["cohorts"].items():
        rows = []
        for ours in cohort["images"]:
            paq = archives.get((name, ours["image"]))
            if paq is None:
                missing.append(f"{name}/{ours['image']}")
            else:
                rows.append(image_row(ours, paq))
        if rows:
            cohorts[name] = {"summary": aggregate(rows), "images": rows}
    summary = {
        "scope": "Supplementary post-hoc paq8px comparison; outside the frozen Tecnick protocol",
        "learned_results": str(args.results.relative_to(ROOT)),
        "learned_results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "paq8px": {key: loaded[0][key] for key in ("binary_sha256", "source", "machine")}
        | {"level": "-9L", "input": "binary PPM (P6) of the exact RGB samples"},
        "raw_receipts": [
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "phase": receipt["phase"],
                "complete": receipt["complete"],
                "images": len(receipt["images"]),
            }
            for path, receipt in zip(args.receipts, loaded, strict=True)
        ],
        "missing_verified_images": missing,
        "cohorts": cohorts,
        "combined": aggregate([r for c in cohorts.values() for r in c["images"]]),
        "timing_note": "paq8px ran four single-threaded processes concurrently on one laptop; "
        "learned timings come from the frozen run, which was also concurrent",
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(json.dumps({k: v["summary"] for k, v in cohorts.items()}, indent=2))
    print(json.dumps({"combined": summary["combined"], "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
