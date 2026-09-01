"""Score every cached experiment on the selection half. Cache-driven, so it
works regardless of whether the batch script that produced a vector has
finished, flushed, or crashed."""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

groups = defaultdict(list)
for p in sorted(lab.CACHE.glob("*.npy")):
    m = re.match(r"^(.*)_([0-9a-f]{16})_s(\d+)\.npy$", p.name)
    if m:
        groups[m.group(1)].append(p)

rows = []
for name, paths in groups.items():
    vs = [np.load(p) for p in paths]
    singles = np.array([lab.score_on(v, "sel")["primary"] for v in vs])
    ens = lab.score_on(np.mean([lab.ranks(v) for v in vs], axis=0), "sel")["primary"]
    rows.append((singles.mean(), name, singles, ens))

base = next((r for r in rows if r[1] == "base"), None)
print(f"{'config':26s} {'n':>2s} {'single mean':>12s} {'std':>8s} {'ens':>9s}"
      f" {'paired vs base':>16s}")
print("-" * 80)
for mean, name, singles, ens in sorted(rows, reverse=True):
    tag = ""
    if base is not None and name != "base":
        n = min(len(singles), len(base[2]))
        if n > 1:
            d = singles[:n] - base[2][:n]
            se = d.std(ddof=1) / np.sqrt(n)
            verdict = "WIN" if d.mean() > 2 * se else ("LOSS" if d.mean() < -2 * se else "null")
            tag = f"{d.mean():+.5f} {verdict}"
    std = singles.std(ddof=1) if len(singles) > 1 else float("nan")
    print(f"{name:26s} {len(singles):2d} {singles.mean():12.5f} {std:8.5f} "
          f"{ens:9.5f} {tag:>16s}")
