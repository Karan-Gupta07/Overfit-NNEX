"""Summarize the fresh Tecnick cohort into one durable results file.

Reads fresh_validation.py receipts and paq8px_baseline.py receipts. A paq8px
size counts only once its verify receipt shows an exact round trip of the same
archive bytes. Savings are 1 - ours / reference in percent; positive means our
complete file is smaller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from summarize_paq8px import verified_archives

ROOT = Path(__file__).resolve().parents[3]
CODECS = ("png", "webp-lossless", "jxl-lossless-e9", "jxl-lossless-e10")


def saving(ours: int, reference: int) -> float:
    return round(100.0 * (1.0 - ours / reference), 3)


def image_row(report: dict, paq: dict) -> dict:
    baselines = {name: report["baselines"][name] for name in CODECS}
    best_name = min(baselines, key=baselines.get)
    ours, samples = report["complete_file_bytes"], report["samples"]
    selected = report["selected"]
    return {
        "image": report["image"],
        "complete_file_bytes": ours,
        "file_sha256": report["file_sha256"],
        "format": selected["format"],
        "hidden": selected.get("hidden"),
        "model_bytes": selected.get("model_bytes"),
        "exact_rgba": report["exact_rgba"] and report["exact_decode"],
        "samples": samples,
        "baselines": baselines,
        "best_baseline": best_name,
        "saving_vs_best_baseline_percent": saving(ours, baselines[best_name]),
        "bits_per_sample": round(8 * ours / samples, 4),
        "best_baseline_bits_per_sample": round(8 * baselines[best_name] / samples, 4),
        "encode_seconds": round(report["seconds"], 2),
        "decode_seconds": round(report["decode_seconds"], 3),
        "phase_seconds": report["phase_seconds"],
        "paq8px_bytes": paq["archive_bytes"],
        "paq8px_archive_sha256": paq["archive_sha256"],
        "saving_vs_paq8px_percent": saving(ours, paq["archive_bytes"]),
        "paq8px_compress_seconds": paq["compress_seconds"],
        "paq8px_decompress_seconds": paq["decompress_seconds"],
    }


def aggregate(rows: list[dict]) -> dict:
    ours = sum(r["complete_file_bytes"] for r in rows)
    samples = sum(r["samples"] for r in rows)
    best = sum(r["baselines"][r["best_baseline"]] for r in rows)
    paq = sum(r["paq8px_bytes"] for r in rows)
    vs_best = [r["saving_vs_best_baseline_percent"] for r in rows]
    vs_paq = [r["saving_vs_paq8px_percent"] for r in rows]

    def median(key: str) -> float:
        return round(statistics.median(r[key] for r in rows), 2)

    return {
        "images": len(rows),
        "complete_file_bytes": ours,
        "codec_bytes": {name: sum(r["baselines"][name] for r in rows) for name in CODECS},
        "saving_vs_codec_percent": {
            name: saving(ours, sum(r["baselines"][name] for r in rows)) for name in CODECS
        },
        "best_baseline_bytes": best,
        "saving_vs_best_baseline_percent": saving(ours, best),
        "median_saving_vs_best_baseline_percent": round(statistics.median(vs_best), 3),
        "min_saving_vs_best_baseline_percent": min(vs_best),
        "max_saving_vs_best_baseline_percent": max(vs_best),
        "wins_vs_best_baseline": sum(s > 0 for s in vs_best),
        "bits_per_sample": round(8 * ours / samples, 4),
        "best_baseline_bits_per_sample": round(8 * best / samples, 4),
        "paq8px_bytes": paq,
        "paq8px_bits_per_sample": round(8 * paq / samples, 4),
        "saving_vs_paq8px_percent": saving(ours, paq),
        "median_saving_vs_paq8px_percent": round(statistics.median(vs_paq), 3),
        "min_saving_vs_paq8px_percent": min(vs_paq),
        "max_saving_vs_paq8px_percent": max(vs_paq),
        "smaller_than_paq8px": sum(s > 0 for s in vs_paq),
        "all_exact": all(r["exact_rgba"] for r in rows),
        "median_encode_seconds": median("encode_seconds"),
        "median_decode_seconds": median("decode_seconds"),
        "median_paq8px_compress_seconds": median("paq8px_compress_seconds"),
        "median_paq8px_decompress_seconds": median("paq8px_decompress_seconds"),
    }


def cross_isa_block(receipt: dict, cohorts: dict, native_backend: str) -> dict:
    if not receipt["complete"] or receipt["native_backend_sha256"] != native_backend:
        raise ValueError("cross-ISA receipt is incomplete or used another native backend")
    checked = {row["file_sha256"]: row for row in receipt["images"]}
    for cohort in cohorts.values():
        for row in cohort["images"]:
            match = checked.get(row["file_sha256"])
            if match is None or not (match["exact_rgb"] and match["identical_encoded_bytes"]):
                raise ValueError(f"no passing cross-ISA check for {row['image']}")
    return {
        "files_checked": len(checked),
        "all_exact_and_identical": True,
        "native_architecture": receipt["native_architecture"],
        "alternate_backend_sha256": receipt["alternate_backend_sha256"],
        "note": receipt["warning"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned", nargs="+", type=Path, required=True)
    parser.add_argument("--paq8px", nargs="+", type=Path, required=True)
    parser.add_argument("--cross-isa", type=Path, help="verify_cross_isa.py receipt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    learned = [json.loads(path.read_text()) for path in args.learned]
    paq = [json.loads(path.read_text()) for path in args.paq8px]
    if len({r["commit"] for r in learned}) != 1:
        raise ValueError("learned receipts come from different commits")
    if any(not r["complete"] or r["tracked_changes"] for r in learned):
        raise ValueError("learned receipts must be complete and from a clean tree")
    if len({r["binary_sha256"] for r in paq}) != 1:
        raise ValueError("paq8px receipts use different binaries")
    archives = verified_archives(paq)
    cohorts = {}
    for receipt in learned:
        cohort = receipt["scope"].rsplit(" ", 1)[-1]
        rows = []
        for report in receipt["images"]:
            match = archives.get((cohort, report["image"]))
            if match is None:
                raise ValueError(f"no verified paq8px archive for {cohort}/{report['image']}")
            rows.append(image_row(report, match))
        cohorts[cohort] = {"summary": aggregate(rows), "images": rows}
    summary = {
        "scope": "Default encoder on the held-out fresh Tecnick cohort, with paq8px -9L",
        "commit": learned[0]["commit"],
        "manifest_sha256": learned[0]["manifest_sha256"],
        "native_backend_sha256": learned[0]["native_backend_sha256"],
        "paq8px": {key: paq[0][key] for key in ("binary_sha256", "source", "machine")}
        | {"level": "-9L", "input": "binary PPM (P6) of the exact RGB samples"},
        "raw_receipts": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [*args.learned, *args.paq8px, *([args.cross_isa] if args.cross_isa else [])]
        ],
        "cohorts": cohorts,
        "timing_note": "One M4 Pro laptop (12 cores). Our encoder ran one image at a time, "
        "after the paq8px runs finished; paq8px ran several single-threaded processes "
        "concurrently",
    }
    if args.cross_isa:
        summary["cross_isa"] = cross_isa_block(
            json.loads(args.cross_isa.read_text()), cohorts, summary["native_backend_sha256"]
        )
    text = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(json.dumps({name: c["summary"] for name, c in cohorts.items()}, indent=2))


if __name__ == "__main__":
    main()
