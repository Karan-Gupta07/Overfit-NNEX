"""Run a command; write its wall seconds and peak summed RSS of its process tree as JSON."""

import contextlib
import json
import subprocess
import sys
import time

import psutil


def tree_rss(root: psutil.Process) -> int:
    total = 0
    for proc in [root, *root.children(recursive=True)]:
        with contextlib.suppress(psutil.Error):
            total += proc.memory_info().rss
    return total


out, command = sys.argv[1], sys.argv[2:]
start = time.perf_counter()
child = subprocess.Popen(command)
root = psutil.Process(child.pid)
peak = 0
while child.poll() is None:
    with contextlib.suppress(psutil.Error):
        peak = max(peak, tree_rss(root))
    time.sleep(0.5)
wall = time.perf_counter() - start
report = {
    "command": command,
    "exit": child.returncode,
    "wall_seconds": round(wall, 1),
    "peak_tree_rss_gb": round(peak / 2**30, 2),
    "cpus": psutil.cpu_count(),
    "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
}
with open(out, "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report))
sys.exit(child.returncode)
