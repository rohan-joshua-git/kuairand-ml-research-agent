"""
Phase 1 — how many seeds does the rank-average ensemble need?

Rank averaging, not score averaging: only within-user ORDER is scored, so ranks
are scale-free and no single model's score distribution can dominate the blend.
This is applied at SUBMISSION time only, so it costs nothing in the loop.

Prior evidence: 5-seed rank-average measured +0.0011 on plain FM and NEUTRAL on
DeepFM-lite. This asks whether more members change that, and where the curve
flattens. Variance reduction from averaging is monotone even when no signal is
added, so the honest question is "where does it stop paying", not "does it help".

Protocol. All decisions are read off the SELECTION half. The confirmation half
is computed and written to disk but NOT printed — it stays sealed until the
ensemble size is frozen, at which point scripts/confirm_final.py spends the
single look.

Trains N_MAX seeds once and derives every ensemble size from that pool, so the
whole sweep costs N_MAX training runs rather than sum(sizes).

Run:  python scripts/seed_ensemble.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data.features import build_features   # noqa: E402
from pipeline.data.loader import load_split         # noqa: E402
from pipeline.eval_protocol import (aggregate, bootstrap_delta,  # noqa: E402
                                    per_user_stats, split_stats)
from pipeline.train import run_training             # noqa: E402

N_MAX = 20
SIZES = [1, 3, 5, 10, 15, 20]
SCORES_NPZ = REPO_ROOT / "logs" / "seed_ensemble_scores.npz"
OUT = REPO_ROOT / "logs" / "seed_ensemble.json"


def rank_average(score_matrix: np.ndarray) -> np.ndarray:
    """Mean of per-model global ranks. Ties get average rank, matching the
    tie handling the official AUC uses."""
    ranks = np.empty_like(score_matrix, dtype=np.float64)
    for i, row in enumerate(score_matrix):
        order = np.argsort(row, kind="mergesort")
        r = np.empty(len(row), dtype=np.float64)
        r[order] = np.arange(len(row), dtype=np.float64)
        # average ties
        s = row[order]
        j = 0
        while j < len(s):
            k = j
            while k + 1 < len(s) and s[k + 1] == s[j]:
                k += 1
            if k > j:
                r[order[j:k + 1]] = (j + k) / 2.0
            j = k + 1
        ranks[i] = r
    return ranks.mean(axis=0)


def main() -> None:
    split = load_split()
    val_feat = build_features(split.val)
    users = val_feat["user_id"].to_numpy()

    if SCORES_NPZ.exists():
        blob = np.load(SCORES_NPZ, allow_pickle=True)
        scores, labels = blob["scores"], blob["labels"]
        print(f"reusing {scores.shape[0]} cached seed score vectors from {SCORES_NPZ.name}\n")
    else:
        rows, labels = [], None
        t0 = time.time()
        for seed in range(N_MAX):
            r = run_training(split=split, seed=seed)
            rows.append(r.val_scores)
            labels = r.val_labels
            single = aggregate(per_user_stats(r.val_user_ids, r.val_labels, r.val_scores))["primary"]
            print(f"  seed {seed:>2}  full primary={single:.4f}   [{time.time()-t0:.0f}s]", flush=True)
        scores = np.vstack(rows)
        SCORES_NPZ.parent.mkdir(exist_ok=True)
        np.savez_compressed(SCORES_NPZ, scores=scores, labels=labels, users=users)
        print(f"\nsaved {scores.shape} score matrix to {SCORES_NPZ.name}\n")

    probe = per_user_stats(users, labels, scores[0])
    sel_idx, conf_idx = split_stats(probe)

    single_stats = per_user_stats(users, labels, scores[0])
    records = []
    print(f"{'seeds':>6} {'sel primary':>13} {'vs 1-seed':>11} {'full primary':>14}")
    print("-" * 50)
    for n in SIZES:
        if n > scores.shape[0]:
            continue
        ens = rank_average(scores[:n])
        st = per_user_stats(users, labels, ens)
        sel = aggregate(st, sel_idx)["primary"]
        full = aggregate(st)["primary"]
        conf = aggregate(st, conf_idx)["primary"]   # stored, not shown
        records.append({"seeds": n, "selection": sel, "full": full, "confirmation": conf})
        base = records[0]["selection"]
        print(f"{n:>6} {sel:>13.4f} {sel - base:>+11.4f} {full:>14.4f}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Is the best ensemble actually distinguishable from a single model?
    best = max(records, key=lambda r: r["selection"])
    if best["seeds"] > 1:
        ens_stats = per_user_stats(users, labels, rank_average(scores[:best["seeds"]]))
        bs = bootstrap_delta(single_stats, ens_stats, idx=sel_idx, n_boot=2000, seed=0)
        print(f"\nbootstrap over selection users, {best['seeds']}-seed vs 1-seed:")
        print(f"  delta {bs['delta']:+.5f}   95% CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]"
              f"   excludes zero: {bs['excludes_zero']}")

        # NEGATIVE CONTROL: two disjoint seed groups of the IDENTICAL model.
        half = best["seeds"] // 2
        if scores.shape[0] >= 2 * half and half >= 2:
            a = per_user_stats(users, labels, rank_average(scores[:half]))
            b = per_user_stats(users, labels, rank_average(scores[half:2 * half]))
            nc = bootstrap_delta(a, b, idx=sel_idx, n_boot=2000, seed=1)
            print(f"\nNEGATIVE CONTROL, two disjoint {half}-seed groups of the SAME model:")
            print(f"  delta {nc['delta']:+.5f}   95% CI [{nc['ci_low']:+.5f}, {nc['ci_high']:+.5f}]")
            print("  If this is the same magnitude as the effect above, the effect is noise.")

    print("\nConfirmation half computed and stored in logs/seed_ensemble.json but NOT shown.")
    print("Freeze the ensemble size from the selection column, then spend the single look.")


if __name__ == "__main__":
    main()
