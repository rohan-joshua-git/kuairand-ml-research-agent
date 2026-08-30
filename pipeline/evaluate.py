"""
NDCG@10 and Recall@50, computed per-user over whatever candidate set is
handed in.

IMPORTANT — candidate-set ambiguity (Day-1 blocker, see README): these
metrics mean different things depending on whether "candidates" is each
user's impressed set from the log, or the full ~7,583-item catalog. This
module does not decide that — it ranks over whatever rows are in the input
DataFrame for a given user. `pipeline/train.py` / the orchestrator is
responsible for constructing the candidate set correctly per
`config.starter_kit.candidate_set` once that's confirmed. Until confirmed,
the default used elsewhere in this pipeline is the impressed-set
interpretation (each user's rows in the val/test log), since that's
computable today without organizer input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RankingMetrics:
    ndcg_at_10: float
    recall_at_50: float
    n_users: int


def _ndcg_at_k(relevance_sorted_by_score: np.ndarray, k: int) -> float:
    rel_k = relevance_sorted_by_score[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel_k) + 2))
    dcg = float(np.sum(rel_k * discounts))

    ideal_rel_k = np.sort(relevance_sorted_by_score)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_rel_k) + 2))
    idcg = float(np.sum(ideal_rel_k * ideal_discounts))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _recall_at_k(relevance_sorted_by_score: np.ndarray, k: int) -> float:
    n_relevant = int(relevance_sorted_by_score.sum())
    if n_relevant == 0:
        return float("nan")  # excluded from the mean — a user with no positives has no recall to measure
    hit = int(relevance_sorted_by_score[:k].sum())
    return hit / n_relevant


def compute_ranking_metrics(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "score",
    label_col: str = "label",
    k_ndcg: int = 10,
    k_recall: int = 50,
) -> RankingMetrics:
    """`df` must have one row per (user, candidate) with a predicted `score`
    and a ground-truth binary `label`. Ranking and metrics are computed
    independently per user, then averaged (macro, over users with at least
    one positive for recall; over all users for NDCG, since NDCG@k is 0 —
    not undefined — for a user with no positives in the candidate set).
    """
    ndcgs = []
    recalls = []

    for _uid, group in df.groupby(user_col, sort=False):
        order = np.argsort(-group[score_col].to_numpy())
        rel = group[label_col].to_numpy()[order]

        ndcgs.append(_ndcg_at_k(rel, k_ndcg))
        r = _recall_at_k(rel, k_recall)
        if not np.isnan(r):
            recalls.append(r)

    n_users = df[user_col].nunique()
    return RankingMetrics(
        ndcg_at_10=float(np.mean(ndcgs)) if ndcgs else float("nan"),
        recall_at_50=float(np.mean(recalls)) if recalls else float("nan"),
        n_users=n_users,
    )


def score_delta(agent_metrics: RankingMetrics, baseline_metrics: RankingMetrics) -> float:
    """The challenge's primary metric: mean of the absolute deltas over the
    two metrics. Recall@50 is typically much larger in magnitude than
    NDCG@10, so it will dominate this average — see README.
    """
    return float(
        np.mean(
            [
                agent_metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10,
                agent_metrics.recall_at_50 - baseline_metrics.recall_at_50,
            ]
        )
    )
