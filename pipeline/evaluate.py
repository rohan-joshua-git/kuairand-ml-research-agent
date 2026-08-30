"""
GAUC and nDCG@5 — the two metrics the challenge actually scores.

This module does NOT reimplement the scoring math. It dynamically loads
`starter_kit/evaluate.py` (the organizer's official, unmodified scorer —
their README says explicitly "不要改", don't touch it) and delegates every
score to it. That's a deliberate choice over a hand-rolled pandas/numpy
version: the organizer's evaluate.py is the sole source of truth for how
GAUC and nDCG@5 are computed (weighting, which users are excluded, the
nDCG gain formula), and any independent reimplementation risks a subtle
mismatch — e.g. GAUC weighting by impressions instead of by positive
count, which is wrong per `starter_kit/evaluate.py::evaluate`. Loading the
one real copy by file path (rather than duplicating its ~50 lines) means
there is exactly one place this logic can drift.

Candidate set: confirmed by the Starter Kit — within-user ranking over
each user's logged impressions ("用户内排序"), not full-catalog retrieval.
`compute_ranking_metrics` ranks over whatever rows are in the input
DataFrame for a given user, which is correct as long as callers only ever
hand it a user's impressed rows (see `pipeline/train.py`).

Reading the numbers (from `starter_kit/baseline_scores.json`): on the
hidden test set, 27.1% of users have no positive label (nDCG@5 = 0 for
any model, included in the mean) and 9.2% are all-positive (AUC
undefined, excluded from GAUC). A perfect (oracle) ranking therefore
tops out at GAUC 1.0000 / nDCG@5 0.7289 / primary 0.8645, not 1.0 — judge
progress against that ceiling. The official FM baseline scores primary
0.5946 on test; that's what the agent needs to beat.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_OFFICIAL_EVALUATE_PATH = Path(__file__).resolve().parents[1] / "starter_kit" / "evaluate.py"


def _load_official_evaluate():
    spec = importlib.util.spec_from_file_location("_official_kuairand_evaluate", _OFFICIAL_EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.evaluate


_official_evaluate = _load_official_evaluate()


@dataclass
class RankingMetrics:
    gauc: float
    ndcg_at_5: float
    n_users: int
    n_users_gauc: int  # users with 0 < positives < impressions, i.e. counted toward GAUC


def compute_ranking_metrics(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "score",
    label_col: str = "label",
    k_ndcg: int = 5,
) -> RankingMetrics:
    """`df` must have one row per (user, candidate) with a predicted `score`
    and a ground-truth binary `label` (the official label is `long_view` —
    see `pipeline/data/label.py::resolve_primary_label`)."""
    result = _official_evaluate(
        df[user_col].to_numpy(),
        df[label_col].to_numpy(),
        df[score_col].to_numpy(),
        k=k_ndcg,
    )

    # n_users_gauc is a diagnostic count, not part of the scoring math above —
    # safe to compute independently without touching the official function.
    n_users_gauc = 0
    for _uid, group in df.groupby(user_col, sort=False):
        npos = int(group[label_col].sum())
        if 0 < npos < len(group):
            n_users_gauc += 1

    return RankingMetrics(
        gauc=float(result["GAUC"]),
        ndcg_at_5=float(result[f"nDCG@{k_ndcg}"]),
        n_users=int(result["users"]),
        n_users_gauc=n_users_gauc,
    )


def score_delta(agent_metrics: RankingMetrics, baseline_metrics: RankingMetrics) -> float:
    """The challenge's primary metric: mean of the absolute delta on GAUC
    and nDCG@5, agent vs. baseline (mirrors `starter_kit/evaluate.py`'s
    `primary = (gauc + ndcg) / 2.0`, just applied to the deltas)."""
    return float(
        np.mean(
            [
                agent_metrics.gauc - baseline_metrics.gauc,
                agent_metrics.ndcg_at_5 - baseline_metrics.ndcg_at_5,
            ]
        )
    )
