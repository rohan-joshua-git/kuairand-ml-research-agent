"""
GAUC and nDCG@5, computed per-user over whatever candidate set is handed in.
These are the two metrics the challenge actually scores (see the brief's
Judging Criteria: KuaiRand-Pure -> GAUC / nDCG@5, equal-weighted absolute
delta vs. the official baseline on the hidden test set).

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

Reading the numbers (per the brief): the hidden test set has 27.1% of
users with no positive label (nDCG@5 = 0 for any model, included in the
mean below) and 9.2% all-positive (AUC undefined for them, excluded from
GAUC — see `_user_auc`). A perfect ranking therefore tops out at
GAUC 1.0000 / nDCG@5 0.7289, not 1.0/1.0 — judge deltas against that
ceiling, not against a naive 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class RankingMetrics:
    gauc: float
    ndcg_at_5: float
    n_users: int
    n_users_gauc: int  # users with both a positive and a negative label, i.e. AUC-defined


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


def _user_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Per-user AUC, the building block of GAUC. Undefined (returns None)
    when a user has no negatives or no positives — that user is excluded
    from the GAUC average rather than counted as some default value, which
    is what makes GAUC 1.0 (not <1.0) attainable under a perfect ranking
    despite the dataset's all-positive/all-negative users."""
    n_pos = int(labels.sum())
    n = len(labels)
    if n_pos == 0 or n_pos == n:
        return None
    return float(roc_auc_score(labels, scores))


def compute_ranking_metrics(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "score",
    label_col: str = "label",
    k_ndcg: int = 5,
) -> RankingMetrics:
    """`df` must have one row per (user, candidate) with a predicted `score`
    and a ground-truth binary `label`. Both metrics are computed per user,
    then aggregated:
      - nDCG@5: macro-averaged over ALL users (a user with no positives in
        the candidate set contributes 0, per the brief, not "undefined").
      - GAUC: averaged over users whose candidate set contains both a
        positive and a negative label, weighted by that user's number of
        candidates (impressions) — the standard GAUC definition used in
        industry CTR benchmarks.
    """
    ndcgs = []
    gaucs = []
    gauc_weights = []

    for _uid, group in df.groupby(user_col, sort=False):
        scores = group[score_col].to_numpy()
        labels = group[label_col].to_numpy()

        order = np.argsort(-scores)
        rel = labels[order]
        ndcgs.append(_ndcg_at_k(rel, k_ndcg))

        auc = _user_auc(scores, labels)
        if auc is not None:
            gaucs.append(auc)
            gauc_weights.append(len(group))

    n_users = df[user_col].nunique()
    gauc = float(np.average(gaucs, weights=gauc_weights)) if gaucs else float("nan")

    return RankingMetrics(
        gauc=gauc,
        ndcg_at_5=float(np.mean(ndcgs)) if ndcgs else float("nan"),
        n_users=n_users,
        n_users_gauc=len(gaucs),
    )


def score_delta(agent_metrics: RankingMetrics, baseline_metrics: RankingMetrics) -> float:
    """The challenge's primary metric: equal-weighted mean of the absolute
    delta on each of GAUC and nDCG@5, agent vs. baseline. Both metrics sit
    in a comparable [0, ~0.86] range on this dataset, so neither dominates
    the mean by construction (unlike the earlier NDCG@10/Recall@50 stand-in
    this replaced, where Recall@50's larger magnitude swamped NDCG@10).
    """
    return float(
        np.mean(
            [
                agent_metrics.gauc - baseline_metrics.gauc,
                agent_metrics.ndcg_at_5 - baseline_metrics.ndcg_at_5,
            ]
        )
    )
