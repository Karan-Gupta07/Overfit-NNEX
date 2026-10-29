"""Compare lpcb_validation.py receipts with LPCB's published per-image sizes.

Receipts from whole PPMs have one file per image. Older receipts coded the
five largest images as separate PNG strips, and their sizes are the sums of
the strips. The published paq8px,
cmix and GraLIC sizes are for the same PPM files (fetch_lpcb.py checks their MD5
sums); the timings were measured on other machines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LPCB = Path(__file__).resolve().parents[3] / "data/corpora/lpcb"
SETS = (
    ("olympus_xz1", "Olympus XZ1"),
    ("sony_a55", "Sony A55"),
    ("fujifilm", "Fujifilm X100"),
    ("canon", "Canon 1100D"),
    ("PIA", "NASA PIA (other)"),
    ("STA", "NASA STA (stars)"),
)
KEYS = ("bytes", "paq8px", "cmix", "gralic", "pixels", "encode_s", "decode_s")
BASELINES = ("png", "webp-lossless", "jxl-lossless-e9", "jxl-lossless-e10")


def published() -> tuple[dict, dict, dict]:
    root = LPCB / "published"
    paq, cmix, gralic = {}, {}, {}
    for line in (root / "paq8px_v216-9L_lpcb_log.txt").read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 9 and fields[0] == "paq8px":
            paq[fields[4]] = int(fields[6])
    for line in (root / "cmix-v16-tsv.txt").read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 3 and fields[0].endswith(".ppm"):
            cmix[fields[0]] = int(fields[2])
    text = (root / "list-gra111.txt").read_text()
    for match in re.finditer(r"([\d,]+)\s+(\S+)\.gra111\s*$", text, re.M):
        gralic[match.group(2) + ".ppm"] = int(match.group(1).replace(",", ""))
    return paq, cmix, gralic


def summarize(rows: list[dict], measured: dict[str, dict] | None = None) -> dict:
    manifest = json.loads((LPCB / "manifest.json").read_text())
    by_name = {row["image"]: row for row in rows}
    paq, cmix, gralic = published()
    images = []
    for item in manifest:
        name = item["ppm"]
        whole = by_name.get(name)
        parts = [whole] if whole else [by_name.get(strip["png"]) for strip in item["strips"]]
        if not all(parts):
            continue
        images.append(
            {
                "ppm": name,
                "pixels": item["width"] * item["height"],
                "strips": whole.get("strips", 1) if whole else len(parts),
                "files": len(parts),
                "bytes": sum(p["bytes"] for p in parts),
                "exact": all(p["exact"] for p in parts),
                "encode_s": sum(p["encode_wall"] for p in parts),
                "decode_s": sum(p["decode_seconds"] for p in parts),
                "paq8px": paq[name],
                "cmix": cmix[name],
                "gralic": gralic[name],
                **({key: measured[name][key] for key in BASELINES} if measured else {}),
            }
        )
        if measured:
            images[-1]["best_baseline"] = min(measured[name][key] for key in BASELINES)
    keys = KEYS + ((*BASELINES, "best_baseline") if measured else ())

    def total(group: list[dict]) -> dict:
        return {
            "images": len(group),
            **{key: sum(i[key] for i in group) for key in keys},
            "smaller_than_paq8px": sum(i["bytes"] < i["paq8px"] for i in group),
            "smaller_than_cmix": sum(i["bytes"] < i["cmix"] for i in group),
            **(
                {"smaller_than_best_baseline": sum(i["bytes"] < i["best_baseline"] for i in group)}
                if measured
                else {}
            ),
        }

    sets = [
        {"set": label, **total([i for i in images if i["ppm"].startswith(prefix)])}
        for prefix, label in SETS
        if any(i["ppm"].startswith(prefix) for i in images)
    ]
    return {
        "images": len(images),
        "all_exact": all(i["exact"] for i in images),
        "total": total(images),
        "camera_photos": total([i for i in images if not i["ppm"].startswith(("PIA", "STA"))]),
        "sets": sets,
        "per_image": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, help="lpcb_baselines.py output")
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    measured = None
    if args.baselines:
        measured = {row["ppm"]: row for row in json.loads(args.baselines.read_text())}
    rows = []
    for path in args.receipts:
        data = json.loads(path.read_text())
        rows += data["images"] if isinstance(data, dict) else data
    summary = summarize(rows, measured)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")

    def pct(a: int, b: int) -> str:
        return f"{(a / b - 1) * 100:+.2f}%"

    print(
        f"{'set':17} {'n':>3} {'ours':>12} {'paq8px':>12} {'':>8} {'cmix':>12} {'':>8}"
        f" {'GraLIC':>12} {'':>8} {'< paq8px':>8}"
    )
    totals = [
        {"set": "camera photos", **summary["camera_photos"]},
        {"set": "all", **summary["total"]},
    ]
    for row in summary["sets"] + totals:
        print(
            f"{row['set']:17} {row['images']:>3} {row['bytes']:>12,} {row['paq8px']:>12,}"
            f" {pct(row['bytes'], row['paq8px']):>8} {row['cmix']:>12,}"
            f" {pct(row['bytes'], row['cmix']):>8} {row['gralic']:>12,}"
            f" {pct(row['bytes'], row['gralic']):>8}"
            f" {row['smaller_than_paq8px']:>4}/{row['images']}"
        )
    if measured:
        for row in summary["sets"] + totals:
            print(
                f"{row['set']:17}",
                *(f"{key} {pct(row['bytes'], row[key])}" for key in (*BASELINES, "best_baseline")),
                f"< best {row['smaller_than_best_baseline']}/{row['images']}",
            )
    print("all exact:", summary["all_exact"])


if __name__ == "__main__":
    main()
