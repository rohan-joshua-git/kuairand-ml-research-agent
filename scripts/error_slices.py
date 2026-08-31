"""
Phases 3 and 4 — what does the primary metric actually reward, and where does
the model lose?

Two questions, one pass, no training (reuses logs/seed_ensemble_scores.npz).

Q1 WHERE IS THE SCORE MASS? The primary is not uniform over users. GAUC is a
   positive-count-weighted mean over ELIGIBLE users only (0 < npos < n), so
   single-impression users contribute exactly zero to it. nDCG@5 is a plain
   mean over ALL users, so a 1-impression user counts as much as a 40-
   impression one. A slice can be large and still barely move the score.

Q2 WHERE IS THE RECOVERABLE HEADROOM? Model-minus-reference says how much the
   model already adds; oracle-minus-model says how much is left. Neither alone
   is actionable. What matters is oracle-minus-model WEIGHTED by the slice's
   share of the score — a big gap in a slice carrying 2% of the mass is worth
   less than a small gap in a slice carrying 40%.

The oracle ranks by the TRUE label. It is a per-slice ceiling for diagnosis
only and is never used for selection or submission.

Run:  python scripts/error_slices.py   (after scripts/seed_ensemble.py)
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

SCORES_NPZ = REPO_ROOT / "logs" / "seed_ensemble_scores.npz"
PRIOR_SMOOTHING = 20.0
MIN_SLICE_USERS = 150


def main() -> None:
    if not SCORES_NPZ.exists():
        sys.exit(f"{SCORES_NPZ} not found — run scripts/seed_ensemble.py first.")

    blob = np.load(SCORES_NPZ, allow_pickle=True)
    scores, labels, users = blob["scores"], blob["labels"], blob["users"]
    model_scores = scores[0] if scores.ndim > 1 else scores

    split = load_split()
    val_feat = build_features(split.val)
    assert len(val_feat) == len(users), "cached scores do not match the val frame"

    # Model-free reference: frozen train-fit smoothed video quality prior.
    y_tr = resolve_label(split.train).to_numpy(dtype=np.float64)
    grouped = pd.DataFrame(
        {"v": split.train["video_id"].to_numpy(), "y": y_tr}
    ).groupby("v")["y"].agg(["sum", "count"])
    p0 = float(y_tr.mean())
    prior = ((grouped["sum"] + PRIOR_SMOOTHING * p0) / (grouped["count"] + PRIOR_SMOOTHING)).to_dict()
    ref_scores = np.array([prior.get(v, p0) for v in val_feat["video_id"].to_numpy()])

    st_model = per_user_stats(users, labels, model_scores)
    st_ref = per_user_stats(users, labels, ref_scores)
    st_oracle = per_user_stats(users, labels, labels.astype(float))  # diagnosis only

    gauc_den = float(st_model.npos[st_model.eligible].sum())
    n_users = len(st_model.user_ids)

    print("=" * 86)
    print(f"overall   model={aggregate(st_model)['primary']:.4f}  "
          f"prior={aggregate(st_ref)['primary']:.4f}  "
          f"oracle={aggregate(st_oracle)['primary']:.4f}")
    print(f"users={n_users:,}   GAUC-eligible={int(st_model.eligible.sum()):,} "
          f"({st_model.eligible.mean():.1%})   — the rest move nDCG only")
    print("=" * 86)

    def emit(name: str, bins: list[tuple[str, np.ndarray]]) -> None:
        print(f"\n--- {name}")
        print(f"  {'slice':>14} {'users':>7} {'%GAUC':>7} {'%nDCG':>7} "
              f"{'model':>8} {'prior':>8} {'oracle':>8} {'gap':>8} {'wgt gap':>8}")
        for label, idx in bins:
            if len(idx) < MIN_SLICE_USERS:
                continue
            gs = float(st_model.npos[idx][st_model.eligible[idx]].sum()) / gauc_den
            ns = len(idx) / n_users
            m = aggregate(st_model, idx)["primary"]
            r = aggregate(st_ref, idx)["primary"]
            o = aggregate(st_oracle, idx)["primary"]
            gap = o - m
            print(f"  {label:>14} {len(idx):>7,} {gs:>6.1%} {ns:>6.1%} "
                  f"{m:>8.4f} {r:>8.4f} {o:>8.4f} {gap:>8.4f} {gap * 0.5 * (gs + ns):>8.4f}")

    uid = st_model.user_ids

    n_imp = val_feat.groupby("user_id").size()
    cand = np.array([n_imp.get(u, 0) for u in uid])
    emit("candidate-set size (validation impressions per user)", [
        ("1", np.where(cand == 1)[0]),
        ("2", np.where(cand == 2)[0]),
        ("3", np.where(cand == 3)[0]),
        ("4-5", np.where((cand >= 4) & (cand <= 5))[0]),
        ("6-10", np.where((cand >= 6) & (cand <= 10))[0]),
        ("11+", np.where(cand >= 11)[0]),
    ])

    tc = split.train["user_id"].value_counts()
    freq = np.array([tc.get(u, 0) for u in uid])
    emit("training frequency", [
        ("1", np.where(freq == 1)[0]),
        ("2-5", np.where((freq >= 2) & (freq <= 5))[0]),
        ("6-15", np.where((freq >= 6) & (freq <= 15))[0]),
        ("16-40", np.where((freq >= 16) & (freq <= 40))[0]),
        ("41+", np.where(freq >= 41)[0]),
    ])

    tab_mode = val_feat.groupby("user_id")["tab"].agg(lambda x: x.mode().iloc[0])
    tabs = np.array([tab_mode.get(u, -1) for u in uid])
    top_tabs = sorted(pd.Series(tabs).value_counts().head(6).index.tolist())
    emit("dominant tab", [(f"tab {int(t)}", np.where(tabs == t)[0]) for t in top_tabs])

    n_tab = val_feat.groupby("user_id")["tab"].nunique()
    spans = np.array([n_tab.get(u, 1) for u in uid])
    emit("tab span", [
        ("single-tab", np.where(spans == 1)[0]),
        ("multi-tab", np.where(spans > 1)[0]),
    ])

    print("\nRead the `wgt gap` column: recoverable headroom scaled by the slice's")
    print("share of the score. A large raw gap in a slice carrying little mass is")
    print("not where the remaining points are. Oracle is diagnosis only.")


if __name__ == "__main__":
    main()
