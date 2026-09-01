"""Batch 3 — heterogeneous blending with fitted weights, done honestly.

This is the direction that most often produces a real gain in a competition,
and it is also the easiest place to fool yourself: fitting blend weights on the
same rows you then report is how a +0.03 that isn't there gets published.

PROTOCOL. The selection half is split AGAIN, by user hash under a different
salt, into:

    selA  — the only rows blend weights are ever fitted on
    selB  — the only rows a blend result is ever reported on

The confirmation half is not touched by this file. That leaves the final,
already-budgeted confirmation look available for exactly one candidate.

METHOD. Caruana-style greedy ensemble selection with replacement, which is
preferred here over unconstrained weight fitting for two reasons: it can only
ever select models that help on selA, and selecting with replacement produces
integer weights that are far harder to overfit than continuous ones.

Everything blends RANKS, not scores: only within-user order is scored, ranks
are scale-free, and no single model's score distribution can dominate.
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

BLEND_SALT = b"kuairand-blend-fit-v1"   # distinct from the eval-protocol salt


def sub_half(user_ids, salt=BLEND_SALT):
    """Deterministic 50/50 user split, same construction as eval_protocol."""
    cut = int(0.5 * (2 ** 32))
    out = np.empty(len(user_ids), dtype=bool)
    for i, u in enumerate(user_ids):
        h = hashlib.blake2b(str(u).encode(), key=salt, digest_size=4).digest()
        out[i] = int.from_bytes(h, "big") < cut
    return out


def load_pool():
    """Rank-average each config across its seeds, then return one vector per
    config. Averaging within a config first removes seed noise, so the greedy
    search chooses between ARCHITECTURES rather than between lucky seeds."""
    groups = defaultdict(list)
    for p in sorted(lab.CACHE.glob("*.npy")):
        m = re.match(r"^(.*)_([0-9a-f]{16})_s(\d+)\.npy$", p.name)
        if m:
            groups[f"{m.group(1)}"].append(p)
    pool = {}
    for name, paths in sorted(groups.items()):
        vs = [lab.ranks(np.load(p)) for p in paths]
        pool[name] = (np.mean(vs, axis=0), len(vs))
    return pool


def greedy(cands, names, fit_mask, n_iter=40, init_best=3):
    """Caruana greedy selection with replacement, scored on fit_mask only."""
    users, labels, _ = lab.geometry()
    from pipeline.eval_protocol import aggregate, per_user_stats

    def sc(v):
        st = per_user_stats(users[fit_mask], labels[fit_mask], v[fit_mask])
        return aggregate(st)["primary"]

    solo = sorted(((sc(c), n) for c, n in zip(cands, names)), reverse=True)
    print("  top solo on selA:")
    for s, n in solo[:8]:
        print(f"    {n:28s} {s:.5f}")

    idx = {n: i for i, n in enumerate(names)}
    chosen = [idx[n] for _, n in solo[:init_best]]
    acc = np.sum([cands[i] for i in chosen], axis=0)
    best = sc(acc / len(chosen))
    print(f"  init with {init_best} best: {best:.5f}")

    for _ in range(n_iter):
        gains = []
        for j in range(len(cands)):
            trial = (acc + cands[j]) / (len(chosen) + 1)
            gains.append((sc(trial), j))
        s, j = max(gains)
        if s <= best + 1e-6:
            break
        best, acc = s, acc + cands[j]
        chosen.append(j)
    return chosen, acc / len(chosen), best


def main():
    users, labels, sel = lab.geometry()
    sub = sub_half(users)
    fit = sel & sub          # selA
    hold = sel & ~sub        # selB
    print(f"selA (fit) {fit.sum():,} rows | selB (report) {hold.sum():,} rows "
          f"| confirmation untouched", flush=True)

    pool = load_pool()
    pool = {k: v for k, v in pool.items() if not k.startswith("probe_")}
    print(f"pool: {len(pool)} configs\n", flush=True)
    names = list(pool)
    cands = [pool[n][0] for n in names]

    def on(mask, v):
        from pipeline.eval_protocol import aggregate, per_user_stats
        return aggregate(per_user_stats(users[mask], labels[mask], v[mask]))["primary"]

    base = pool.get("base")
    if base is not None:
        print(f"REFERENCE base ({pool['base'][1]} seeds)  "
              f"selA {on(fit, base[0]):.5f} | selB {on(hold, base[0]):.5f}\n", flush=True)

    eq = np.mean(cands, axis=0)
    print(f"equal-weight all {len(cands)}   selA {on(fit, eq):.5f} | "
          f"selB {on(hold, eq):.5f}\n", flush=True)

    chosen, blend, fit_score = greedy(cands, names, fit)
    counts = defaultdict(int)
    for j in chosen:
        counts[names[j]] += 1
    print("\n  selected (weight = times picked):")
    for n, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {n:28s} x{c}")

    hold_score = on(hold, blend)
    print(f"\n  greedy blend  selA {fit_score:.5f} (FITTED - optimistic)")
    print(f"  greedy blend  selB {hold_score:.5f} (honest)")
    if base is not None:
        print(f"  base          selB {on(hold, base[0]):.5f}")
        print(f"  --> honest delta vs base on selB: "
              f"{hold_score - on(hold, base[0]):+.5f}")
    print("\nNOTE: selB is ~15.8k rows, so its own noise is larger than the full"
          "\nselection half. Treat anything under +0.002 here as unproven.")


if __name__ == "__main__":
    main()
