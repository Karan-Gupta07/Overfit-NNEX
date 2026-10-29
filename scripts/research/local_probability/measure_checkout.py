"""Measure the default encoder of another checkout, such as an old commit.

Run from that checkout's root, for example a worktree of an older commit:

    git worktree add tmp/old f53cb0f
    cd tmp/old && python ../../scripts/research/local_probability/measure_checkout.py \
        OUT.json IMAGE.png ...

It imports compress.py from the current directory's research folder, so it works
on commits older than fresh_validation.py. Each image is encoded, decoded and
compared with its source. Resumes from OUT.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts/research/local_probability"))


def main() -> None:
    import compress as experiment
    import neural_bitstream as coder
    import numpy as np
    from input_image import read_rgb

    out = Path(sys.argv[1])
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(row["cohort"], row["image"]) for row in rows}
    for name in sys.argv[2:]:
        path = Path(name)
        if (path.parent.name, path.name) in done:
            continue
        pixels = read_rgb(path, max_pixels=coder.MAX_PIXELS)
        start = time.perf_counter()
        blob, report = experiment.compress(path)
        encode_wall = time.perf_counter() - start
        start = time.perf_counter()
        decoded = experiment.decompress(blob)
        decode_seconds = time.perf_counter() - start
        exact = bool(np.array_equal(np.asarray(decoded)[..., :3], np.asarray(pixels)[..., :3]))
        rows.append(
            {
                "image": path.name,
                "cohort": path.parent.name,
                "bytes": len(blob),
                "seconds": report.get("seconds"),
                "encode_wall": encode_wall,
                "decode_seconds": decode_seconds,
                "exact": exact,
                "selected": report.get("selected"),
                "phase_seconds": report.get("phase_seconds"),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(
            path.name,
            len(blob),
            f"{encode_wall:.1f}s",
            f"dec {decode_seconds:.2f}s",
            "exact" if exact else "MISMATCH",
            flush=True,
        )


if __name__ == "__main__":
    main()
