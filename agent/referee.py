"""
Play 1: Unbiased Referee — wired into the live per-iteration loop.

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
that the agent is fitting the proxy, not the underlying task. This is
surfaced as a pitfall-store entry (`agent/orchestrator.py`) so it feeds the
next iteration's reflect+revise prompt, rather than blocking acceptance
outright — `compression_gate.py` is the hard accept/reject gate; this is
an earlier-warning diagnostic.

Respects `config.referee.mode`:
  - "tier_a"    train directly on the random log (requires organizer
                confirmation it's permitted beyond diagnostics — see
                README "Open questions"; NOT implemented here)
  - "tier_b"    (default) use it only for propensity estimation / as a
                held-out unbiased scoring probe, never for gradient updates
  - "disabled"  skip entirely (falls back to biased-validation-only, with
                the compression gate as the sole overfitting guard)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch.nn as nn

from pipeline.data.loader import load_config, load_random_exposure_log
from pipeline.evaluate import RankingMetrics, compute_ranking_metrics

PROBE_LABEL_COLUMN = "long_view"  # the same primary label used everywhere else — see pipeline/data/label.py


@dataclass
class RefereeReport:
    biased_metrics: RankingMetrics
    unbiased_metrics: RankingMetrics
    divergence_gauc: float
    divergence_ndcg: float
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
    appeared in log_random, using `long_view` from that log as the label —
    see `score_probe_with_model`, which builds exactly this). Because
    exposure was uniform-random, this is an unbiased estimate of ranking
    quality over the true item distribution, not just the already-filtered
    biased-log distribution.
    """
    return compute_ranking_metrics(scored_candidates, score_col=score_col, label_col=PROBE_LABEL_COLUMN)


def build_referee_report(
    biased_metrics: RankingMetrics,
    unbiased_probe_scored: pd.DataFrame,
    cfg: dict | None = None,
) -> RefereeReport:
    """`biased_metrics` is the standard validation RankingMetrics the
    orchestrator already computed via `run_training` — passed in rather
    than recomputed here, since recomputing it would mean scoring the val
    split a second time for no reason."""
    cfg = cfg or load_config()
    threshold = cfg["referee"]["divergence_alert_threshold"]

    unbiased = score_against_unbiased_probe(unbiased_probe_scored)

    div_gauc = biased_metrics.gauc - unbiased.gauc
    div_ndcg = biased_metrics.ndcg_at_5 - unbiased.ndcg_at_5

    alert = div_gauc > threshold or div_ndcg > threshold

    return RefereeReport(
        biased_metrics=biased_metrics,
        unbiased_metrics=unbiased,
        divergence_gauc=div_gauc,
        divergence_ndcg=div_ndcg,
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


def load_probe_sample(cfg: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Loads and samples the random-exposure log ONCE per run — callers
    (agent/orchestrator.py) should cache the result rather than re-reading
    the ~87MB file every iteration. Sampled (not the full 1.19M rows) so
    the per-iteration referee check stays cheap; `config.referee.probe_sample_size`
    controls the size, and the fixed seed keeps the sample consistent
    across iterations so divergence trend is comparable iteration to
    iteration rather than noisy from a different random slice each time.
    """
    cfg = cfg or load_config()
    df = load_probe_log(cfg)
    sample_size = cfg["referee"].get("probe_sample_size")
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def score_probe_with_model(
    model: nn.Module,
    id_maps: dict,
    probe_df: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Runs `model` over an already-loaded probe sample (see
    `load_probe_sample`) and returns a DataFrame ready for
    `score_against_unbiased_probe`. Imports `build_features`/`score_split`
    inside the function body, not at module level: those are two of the
    modules `agent/code_editor.py` lets the agent rewrite, and
    `agent/orchestrator.py` reloads them in-process after every applied
    patch (see its `_reload_editable_modules`) — a module-level import
    here would capture a stale pre-patch reference for the rest of the
    run, the same staleness bug that motivated that reload logic
    everywhere else.
    """
    from pipeline.data.features import build_features
    from pipeline.train import score_split

    cfg = cfg or load_config()
    ds = cfg["dataset"]
    raw_dir = Path(ds["raw_dir"])
    vf_basic_path = raw_dir / ds["video_features_basic"]
    vf_basic = pd.read_csv(vf_basic_path) if vf_basic_path.exists() else None

    feat = build_features(probe_df, video_features_basic=vf_basic).copy()
    # Unseen ids fall back to index 0, matching pipeline/train.py's val-split
    # handling — the random log spans the same date range as validation but
    # wasn't part of training, so it can contain ids run_training never saw.
    feat["user_id"] = feat["user_id"].where(feat["user_id"].isin(id_maps["user"]), next(iter(id_maps["user"])))
    feat["video_id"] = feat["video_id"].where(feat["video_id"].isin(id_maps["video"]), next(iter(id_maps["video"])))

    scores = score_split(model, feat, id_maps)
    return pd.DataFrame(
        {
            "user_id": feat["user_id"].to_numpy(),
            "score": scores,
            PROBE_LABEL_COLUMN: feat[PROBE_LABEL_COLUMN].to_numpy(),
        }
    )
