"""
GAUC / nDCG@5, delegated to the organizer's vendored evaluate.py
(`starter_kit/evaluate.py`) rather than reimplemented here.

This is deliberate, not laziness: the challenge is graded once, on the
hidden test set, by that exact script. Reimplementing the metric math in
pandas risks a silent numerical divergence (tie handling in AUC, the
zero-positive-user convention, the GAUC weighting) that would make every
validation score during development lie about the real hidden-test score.
Calling the vendored function directly makes that divergence structurally
impossible. Do not reimplement GAUC/nDCG@5 math in this file.

Candidate set: each user is ranked only within their own rows in whatever
DataFrame is passed in (the Starter Kit's "within-user ranking over logged
impressions" task) — this module doesn't construct that set, callers do
(`pipeline/train.py`, `agent/referee.py`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STARTER_KIT_DIR = Path(__file__).resolve().parents[1] / "starter_kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

from evaluate import evaluate as _official_evaluate  # noqa: E402 — path insert above is required first


@dataclass
class RankingMetrics:
    gauc: float
    ndcg_at_5: float
    primary: float
    n_users: int


def compute_ranking_metrics(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "score",
    label_col: str = "label",
) -> RankingMetrics:
    """`df` must have one row per (user, candidate) with a predicted `score`
    and a ground-truth binary `label` (resolved `long_view` — see
    `pipeline/data/label.py`)."""
    result = _official_evaluate(
        df[user_col].to_numpy(),
        df[label_col].to_numpy(),
        df[score_col].to_numpy(),
        k=5,
    )
    return RankingMetrics(
        gauc=float(result["GAUC"]),
        ndcg_at_5=float(result["nDCG@5"]),
        primary=float(result["primary"]),
        n_users=int(result["users"]),
    )


def score_delta(agent_metrics: RankingMetrics, baseline_metrics: RankingMetrics) -> float:
    """The challenge's primary metric: mean of the absolute deltas over GAUC
    and nDCG@5. Since primary = mean(GAUC, nDCG@5) for both sides, this
    collapses to a plain difference of primaries — see 2.6 Judging Criteria.
    """
    return agent_metrics.primary - baseline_metrics.primary
