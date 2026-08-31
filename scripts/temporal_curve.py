"""
Step 6 — is there temporal distribution shift across the validation window?

This is a different question from the one already closed. The temporal MODEL
family (freshness, EWMA, HAR, GARCH, Hawkes, ARIMA) is dead because `upload_dt`
has three distinct values, so there is no video lifecycle to model. That
remains true. Temporal *distribution shift* is a separate question: does the
model's advantage decay across 4/22 -> 4/28, and therefore further into the
4/29 -> 5/08 test window?

Section 9 measured decay in the mlp-only DELTA (+0.00127 early vs +0.00027
late) but never the absolute curve, which is what would justify reopening
recency weighting.

Critical control. Raw per-day primary conflates two things: the model getting
worse, and the day being harder. So every day is also scored with a model-free
reference — a smoothed train-fit video-quality prior, which by construction
cannot drift because it is frozen from train. Read the GAP, not the level:

  gap shrinking over time  -> the model's learned advantage is decaying
  gap flat, both declining -> later days are simply harder; nothing to fix
  gap flat, both flat      -> no shift; recency weighting stays closed

Run:  python scripts/temporal_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data.features import build_features   # noqa: E402
from pipeline.data.label import resolve_label       # noqa: E402
from pipeline.data.loader import load_split         # noqa: E402
from pipeline.eval_protocol import aggregate, per_user_stats  # noqa: E402
from pipeline.train import run_training             # noqa: E402

SEEDS = [0, 1, 2]
PRIOR_SMOOTHING = 20.0


def video_prior(train_df: pd.DataFrame) -> tuple[dict, float]:
    """Smoothed train long_view rate per video — frozen, so it cannot drift."""
    y = resolve_label(train_df).to_numpy(dtype=np.float64)
    frame = pd.DataFrame({"video_id": train_df["video_id"].to_numpy(), "y": y})
    g = frame.groupby("video_id")["y"].agg(["sum", "count"])
    prior = float(y.mean())
    sm = (g["sum"] + PRIOR_SMOOTHING * prior) / (g["count"] + PRIOR_SMOOTHING)
    return sm.to_dict(), prior


def main() -> None:
    split = load_split()
    val_feat = build_features(split.val)
    users = val_feat["user_id"].to_numpy()
    labels = resolve_label(val_feat).to_numpy()
    dates = val_feat["date"].to_numpy()

    table, prior_mean = video_prior(split.train)
    ref_scores = np.array([table.get(v, prior_mean) for v in val_feat["video_id"].to_numpy()])

    print("training champion (3 seeds) for the temporal curve...\n", flush=True)
    model_scores = []
    for seed in SEEDS:
        r = run_training(split=split, seed=seed)
        # run_training scores build_features(split.val), same frame/order as here.
        model_scores.append(r.val_scores)
        print(f"  seed={seed} full primary={aggregate(per_user_stats(r.val_user_ids, r.val_labels, r.val_scores))['primary']:.4f}", flush=True)

    def block(mask):
        m = float(np.mean([
            aggregate(per_user_stats(users[mask], labels[mask], s[mask]))["primary"]
            for s in model_scores
        ]))
        ref = aggregate(per_user_stats(users[mask], labels[mask], ref_scores[mask]))["primary"]
        return m, ref, m - ref

    uniq_days = sorted(np.unique(dates))
    print("\nPer-day validation curve")
    print(f"  {'date':>10} {'rows':>8} {'model':>9} {'prior ref':>11} {'gap':>9}")
    for d in uniq_days:
        mask = dates == d
        m, ref, gap = block(mask)
        print(f"  {int(d):>10} {int(mask.sum()):>8,} {m:>9.4f} {ref:>11.4f} {gap:>+9.4f}")

    print("\nThree-block view")
    blocks = np.array_split(np.array(uniq_days), 3)
    gaps = []
    for name, days in zip(["early", "middle", "late"], blocks):
        mask = np.isin(dates, days)
        m, ref, gap = block(mask)
        gaps.append(gap)
        span = f"{int(days[0])}-{int(days[-1])}"
        print(f"  {name:>7} {span:>19} rows={int(mask.sum()):>7,}  model={m:.4f}  ref={ref:.4f}  gap={gap:+.4f}")

    drift = gaps[-1] - gaps[0]
    print(f"\ngap change early -> late: {drift:+.4f}")
    if drift < -0.002:
        print("The model's advantage is DECAYING across the window. Recency weighting is")
        print("reopened — but note the prior attempt measured null (half-life 14d/7d both")
        print("0.6389 vs uniform 0.6387, 2d HURTS at 0.6342), so it needs a real margin.")
    else:
        print("No material decay in the model's advantage. Recency weighting stays CLOSED;")
        print("any late-window shortfall is day difficulty, which no reweighting fixes.")


if __name__ == "__main__":
    main()
