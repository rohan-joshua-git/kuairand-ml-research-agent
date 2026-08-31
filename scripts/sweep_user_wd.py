"""
Step 1 — field-specific weight decay on the `user_id` embedding.

Hypothesis. The user embedding is partly fitting noise while the item
embeddings are not. Three independent measurements motivate this:

  1. user x video affinity is chance (0.4973), yet deleting user_id costs
     -0.0091, so the user embedding contributes through some other channel.
  2. Every two-way user x context encoding scores BELOW the context alone
     (user x tab 0.5545 vs tab 0.5789), so it is not a two-way interaction.
  3. Global capacity reduction helps (k monotonically declining 0.6049 -> 0.6037),
     so the model is over-parameterised — but global weight decay was the only
     regularisation ever tested, and 22,377 user embeddings backed by ~50
     impressions each versus 7,583 video embeddings backed by ~150 have very
     different data-per-parameter ratios.

Protocol (pipeline/eval_protocol.py). Every arm early-stops on the SELECTION
half only, so held-out confirmation users never participate in model selection.
Selection-half primary is the decision metric. The confirmation half is computed
and stored but deliberately NOT printed per-arm — it is revealed once, for the
single arm that wins on selection, so the confirmation look is spent once.

Run:  python scripts/sweep_user_wd.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data.features import build_features           # noqa: E402
from pipeline.data.loader import load_split                 # noqa: E402
from pipeline.eval_protocol import (aggregate, per_user_stats,  # noqa: E402
                                    split_stats, user_half)
from pipeline.train import run_training                     # noqa: E402

USER_WD_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SEEDS = [0, 1, 2]
OUT = REPO_ROOT / "logs" / "sweep_user_wd.json"


def arm_metrics(result, sel_idx, conf_idx):
    stats = per_user_stats(result.val_user_ids, result.val_labels, result.val_scores)
    return (
        aggregate(stats, sel_idx)["primary"],
        aggregate(stats, conf_idx)["primary"],
        aggregate(stats)["primary"],
    )


def main() -> None:
    split = load_split()
    val_feat = build_features(split.val)

    # The early-stop mask must align with the frame run_training evaluates on.
    # build_features sorts by user_id, so build it from that frame, not split.val.
    es_mask = user_half(val_feat["user_id"].to_numpy())
    print(f"validation rows: {len(val_feat):,} | selection rows: {es_mask.sum():,} "
          f"({es_mask.mean():.1%})")

    # Per-user index arrays, resolved once from any scored frame's user ordering.
    probe = per_user_stats(val_feat["user_id"].to_numpy(),
                           np.zeros(len(val_feat)), np.zeros(len(val_feat)))
    sel_idx, conf_idx = split_stats(probe)
    print(f"selection users: {len(sel_idx):,} | confirmation users: {len(conf_idx):,}\n")

    records = []
    t0 = time.time()
    for wd in USER_WD_GRID:
        sels, confs, fulls = [], [], []
        for seed in SEEDS:
            r = run_training(split=split, seed=seed, user_weight_decay=wd,
                             early_stop_mask=es_mask)
            s, c, f = arm_metrics(r, sel_idx, conf_idx)
            sels.append(s); confs.append(c); fulls.append(f)
            records.append({"user_wd": wd, "seed": seed, "selection": s,
                            "confirmation": c, "full": f})
            print(f"  wd={wd:<8g} seed={seed}  selection={s:.4f}", flush=True)
        print(f"wd={wd:<8g}  SELECTION mean={np.mean(sels):.4f} "
              f"std={np.std(sels):.4f}   [{time.time()-t0:.0f}s]\n", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Decision on the selection half only.
    by_wd = {}
    for rec in records:
        by_wd.setdefault(rec["user_wd"], []).append(rec)
    control = float(np.mean([r["selection"] for r in by_wd[0.0]]))

    print("=" * 68)
    print(f"{'user_wd':>10} {'sel mean':>10} {'sel std':>9} {'vs control':>12}")
    print("-" * 68)
    winner, best = None, control
    for wd in USER_WD_GRID:
        vals = [r["selection"] for r in by_wd[wd]]
        m = float(np.mean(vals))
        tag = "  <- control" if wd == 0.0 else ""
        print(f"{wd:>10g} {m:>10.4f} {np.std(vals):>9.4f} {m - control:>+12.4f}{tag}")
        if wd != 0.0 and m > best:
            winner, best = wd, m
    print("=" * 68)

    if winner is None:
        print("\nNo arm beat the control on the selection half. "
              "Confirmation half NOT looked at — the look is preserved.")
        return
    if best - control < 0.001:
        print(f"\nBest arm wd={winner:g} gains only {best - control:+.4f} on selection, "
              "below the 0.001 reporting floor. Confirmation half NOT looked at.")
        return

    conf_win = float(np.mean([r["confirmation"] for r in by_wd[winner]]))
    conf_ctl = float(np.mean([r["confirmation"] for r in by_wd[0.0]]))
    print(f"\nSpending one confirmation look on wd={winner:g}:")
    print(f"  selection    {best - control:+.4f}")
    print(f"  confirmation {conf_win - conf_ctl:+.4f}")
    print("\nStill required before shipping: user-level bootstrap CI excluding zero, "
          "and a negative control (two seed groups of the IDENTICAL model).")


if __name__ == "__main__":
    main()
