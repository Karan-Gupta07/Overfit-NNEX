"""Replay actual cached NNEX candidates against live, complete SCDX encodings.

This measures file sizes/fidelity without retraining. It is not an end-to-end
speed benchmark or fresh validation, and every selected model stays in-band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import compress as experiment
import numpy as np
from input_image import read_rgb

from semanticcodec.encode.pipeline import EncodeOptions, encode_image

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = ROOT / "docs/results/raw"
    original = json.loads((raw / "neural-integer-all.json").read_text())
    small = json.loads((raw / "neural-h16-regression.json").read_text())
    development = json.loads((raw / "neural-quant-development.json").read_text())
    regression = json.loads((raw / "neural-quant-sipi-regression.json").read_text())
    if not all(x["complete"] for x in (original, small, development, regression)):
        raise ValueError("incomplete model receipt")
    h16 = {r["image"]: r for r in small["images"]}
    h32 = {r["image"]: r for r in development["images"] + regression["images"]}
    if args.start < 0 or args.limit <= 0:
        parser.error("invalid selection")
    rows = original["images"][args.start : args.start + args.limit]
    if not rows:
        parser.error("empty selection")
    result = {
        "complete": False,
        "selection": [row["image"] for row in rows],
        "scope": "cached known-input regression replay, not fresh or end-to-end timing",
        "floor": "actual complete SCDX, balanced hybrid, no partitioning",
        "images": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for old in rows:
        name = old["image"]
        corpus = "kodak" if old["role"] == "development" else "sipi-color-holdout"
        source = ROOT / "data/corpora" / corpus / name
        rgb = read_rgb(source)
        rgba = np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)
        if hashlib.sha256(source.read_bytes()).hexdigest() != old["source_sha256"]:
            raise ValueError("changed source image")
        cache = ROOT / "data/research-local-probability"
        stem = Path(name).stem
        choices = []
        for hidden, report in ((16, h16[name]), (32, h32[name])):
            if report["source_sha256"] != old["source_sha256"]:
                raise ValueError("model/source mismatch")
            selected = report["selected"]
            prefix = (
                f"h16-{stem}"
                if hidden == 16
                else f"quant-{stem}"
                if old["role"] == "development"
                else f"{stem}-precision"
            )
            path = cache / f"{prefix}-q{selected['bits']}.nnexp"
            blob = path.read_bytes()
            if len(blob) != selected["complete_file_bytes"]:
                raise ValueError("cached stream length mismatch")
            if hashlib.sha256(blob).hexdigest() != selected["file_sha256"]:
                raise ValueError("cached stream digest mismatch")
            if not np.array_equal(experiment.decompress(blob), rgba):
                raise AssertionError("cached NNEX does not decode to the source")
            choices.append((blob, {"format": "nnex-v2", "hidden": hidden, **selected}))
        floor, _document, report = encode_image(rgba, EncodeOptions(hybrid=True))
        if not np.array_equal(experiment.decompress(floor), rgba):
            raise AssertionError("actual SCDX floor mismatch")
        best = floor
        selected = {"format": "scdx", "codec": report.selected_codec}
        for blob, choice in choices:
            if len(blob) < len(best):
                best, selected = blob, choice
        if not np.array_equal(experiment.decompress(best), rgba):
            raise AssertionError("selected complete file mismatch")
        output = ROOT / "data/probe" / f"unified-{stem}.coded"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(best)
        row = {
            "image": name,
            "role": old["role"],
            "source_sha256": old["source_sha256"],
            "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            "file_sha256": hashlib.sha256(best).hexdigest(),
            "complete_file_bytes": len(best),
            "selected": selected,
            "actual_scdx_bytes": len(floor),
            "adaptive_nn_bytes": min(len(blob) for blob, _ in choices),
            "baselines": old["baselines"],
            "exact_rgba": True,
        }
        result["images"].append(row)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("REPLAY VERIFIED", name, len(best), "selected", selected["format"], flush=True)
    result["complete"] = True
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
