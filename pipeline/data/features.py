"""
Feature engineering — this module is the primary surface the agent rewrites
each iteration (Figure 1 "engineer features" stage).

`build_features` is intentionally a single, clearly-named entrypoint so the
agent's code diffs have one obvious place to land: it can add columns,
change encodings, or swap out the whole body, and `ablation.py` can diff
this file's contents across iterations to explain what changed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from pipeline.data.leakage_guard import drop_leaky_columns

# duration_ms is an ITEM property (video length) — known before exposure, safe.
# play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter are all
# OUTCOMES of the impression being predicted (measured during/after the view),
# so feeding them is label leakage, not modeling. Measured on real validation
# data: play_time_ms corr 0.64 with long_view; comment_stay_time corr 0.17
# (staying in the comments implies you long-viewed); profile_stay_time corr ~0
# but same causal class. The official baseline's field list uses duration_ms
# and none of the outcome fields — same conclusion.
NUMERIC_SIGNAL_COLUMNS = [
    "duration_ms",
]

AUXILIARY_LABEL_COLUMNS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
]


@lru_cache(maxsize=1)
def load_video_basic_features() -> pd.DataFrame | None:
    """`video_features_basic_pure.csv` (video_id -> author_id): static
    attributes fixed at upload time, so not leaky — unlike the statistic file
    (see leakage_guard.py). author_id is the one item-side field the official
    baseline uses. Cached: it is read once per process, not once per call."""
    from pipeline.data.loader import load_config

    cfg = load_config()
    path = Path(cfg["dataset"]["raw_dir"]) / cfg["dataset"]["features"]["video_basic"]
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["video_id", "author_id"])


def build_features(
    df: pd.DataFrame,
    video_features_basic: pd.DataFrame | None = None,
    video_features_statistic: pd.DataFrame | None = None,
    allow_leaky_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Joins interaction rows with video-side features from KuaiRand's two
    separate video-feature files, each with different leakage properties, then
    sorts by user_id so each user's rows form one contiguous group (what a
    user-grouped pairwise/listwise loss needs).

    Args:
        df: interaction log rows (train/val slice).
        video_features_basic: `video_features_basic_pure.csv` — static
            per-video attributes (`author_id`, `music_id`, `upload_type`,
            ...) fixed at upload time. Not leaky, merged as-is.
        video_features_statistic: `video_features_statistic_pure.csv` —
            month-long running aggregates spanning train/val/test. Leaky by
            construction; passed through `leakage_guard.drop_leaky_columns`
            before merging.
        allow_leaky_columns: explicit opt-in for otherwise-dropped leaky
            columns — see leakage_guard.py. Empty by default.
    """
    out = df.copy()

    # Auto-load the static video-side table when the caller didn't pass one, so
    # every path builds IDENTICAL features. Training passed it explicitly while
    # pipeline/submit.py and the referee probe did not, which silently turned
    # author_id into UNK at scoring time — a train/serve skew that degrades the
    # submission without failing any check. One code path removes that class of
    # bug entirely. Pass an empty DataFrame to opt out.
    if video_features_basic is None:
        video_features_basic = load_video_basic_features()

    if video_features_basic is not None and not video_features_basic.empty:
        out = out.merge(video_features_basic, on="video_id", how="left")

    if video_features_statistic is not None:
        vf_clean, dropped = drop_leaky_columns(video_features_statistic, allow_columns=allow_leaky_columns)
        out = out.merge(vf_clean, on="video_id", how="left")
        if dropped:
            # Set AFTER merge, not before: pandas' DataFrame.attrs does not
            # reliably survive .merge() (it only propagates when all inputs
            # share identical attrs, which vf_clean's empty attrs breaks).
            # Left visible rather than silently swallowed — the orchestrator's
            # logger should capture this as part of the iteration's diff summary.
            out.attrs["dropped_leaky_columns"] = dropped

    for col in NUMERIC_SIGNAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    # Sort interactions by user_id so user interactions form contiguous groups
    # required for user-grouped pairwise BPR loss and ranking evaluation.
    if "user_id" in out.columns:
        out = out.sort_values(by="user_id", kind="stable").reset_index(drop=True)

    return out


def auxiliary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the non-click feedback signals available for multi-task
    auxiliary transfer (Play 3 / ESMM-PLE style heads). Missing columns are
    tolerated so this stays safe to call before all signals are confirmed
    present in a given log file.
    """
    present = [c for c in AUXILIARY_LABEL_COLUMNS if c in df.columns]
    return df[present].fillna(0)
