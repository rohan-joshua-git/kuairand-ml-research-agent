"""
Play 1: Unbiased Referee.

KuaiRand-Pure's standard validation split is itself biased — every row in
it was already selected for exposure by the prior recommendation policy.
An agent that only ever checks its score against that split can, given
enough iterations, learn to exploit quirks of that specific biased sample
rather than genuinely improving (this is the failure mode SpecBench
documented: an agent preferring a 97%-validation/0%-held-out artifact over
a genuine 53%/43% one, because validation score was the only signal it
searched against).

This module scores every candidate checkpoint against a second, unbiased
probe built from `log_random_4_22_to_5_08_pure.csv` (uniformly-random
exposure) and reports the *divergence* between biased-validation score and
unbiased-probe score. A widening divergence — score improving on the
biased split while flat or dropping on the unbiased probe — is the signal
that the agent is fitting the proxy, not the underlying task, and should
trigger a reflect+revise course-correction rather than "keep going."

Respects `config.referee.mode`:
  - "tier_a"    train directly on the random log (requires organizer
                confirmation it's permitted beyond diagnostics)
  - "tier_b"    (default) use it only for propensity estimation / as a
                held-out unbiased scoring probe, never for gradient updates
  - "disabled"  skip entirely (falls back to biased-validation-only, with
                the compression gate as the sole overfitting guard)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.data.loader import load_config, load_random_exposure_log
from pipeline.evaluate import RankingMetrics, compute_ranking_metrics


@dataclass
class RefereeReport:
    biased_metrics: RankingMetrics
    unbiased_metrics: RankingMetrics
    divergence_ndcg: float
    divergence_recall: float
    alert: bool


def compute_video_propensities(random_log: pd.DataFrame) -> pd.Series:
    """Per-video exposure propensity under the uniform-random policy:
    exposures for a video / total random-log rows. Used both as a Tier-B
    diagnostic weight and as the Tier-A importance-weighting input (see
    agent/skill_store/tier3_deep/autodebias.md)."""
    counts = random_log["video_id"].value_counts()
    return counts / counts.sum()


def score_against_unbiased_probe(
    scored_candidates: pd.DataFrame,
    score_col: str = "score",
) -> RankingMetrics:
    """`scored_candidates` must be the model's predictions joined onto the
    random-exposure log rows (i.e. score every (user, video) pair that
    appeared in log_random, using is_click from that log as the label).
    Because exposure was uniform-random, this is an unbiased estimate of
    ranking quality over the true item distribution, not just the
    already-filtered biased-log distribution.
    """
    return compute_ranking_metrics(scored_candidates, score_col=score_col, label_col="is_click")


def build_referee_report(
    biased_val_scored: pd.DataFrame,
    unbiased_probe_scored: pd.DataFrame,
    cfg: dict | None = None,
) -> RefereeReport:
    cfg = cfg or load_config()
    threshold = cfg["referee"]["divergence_alert_threshold"]

    biased = compute_ranking_metrics(biased_val_scored, label_col="label")
    unbiased = score_against_unbiased_probe(unbiased_probe_scored)

    div_ndcg = biased.ndcg_at_10 - unbiased.ndcg_at_10
    div_recall = biased.recall_at_50 - unbiased.recall_at_50

    alert = div_ndcg > threshold or div_recall > threshold

    return RefereeReport(
        biased_metrics=biased,
        unbiased_metrics=unbiased,
        divergence_ndcg=div_ndcg,
        divergence_recall=div_recall,
        alert=alert,
    )


def referee_enabled(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return cfg["referee"]["mode"] != "disabled"


def load_probe_log(cfg: dict | None = None) -> pd.DataFrame:
    """Loads the random-exposure log for use as the unbiased probe. Raises
    if referee.mode == "disabled" so callers don't accidentally spend I/O
    on a file they've been told not to use."""
    cfg = cfg or load_config()
    if not referee_enabled(cfg):
        raise RuntimeError("referee.mode is 'disabled' in config — unbiased probe unavailable.")
    return load_random_exposure_log(cfg)
