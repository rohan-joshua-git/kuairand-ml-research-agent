"""
Feature engineering — this module is the primary surface the agent rewrites
each iteration (Figure 1 "engineer features" stage).

`build_features` is intentionally a single, clearly-named entrypoint so the
agent's code diffs have one obvious place to land: it can add columns,
change encodings, or swap out the whole body, and `ablation.py` can diff
this file's contents across iterations to explain what changed.
"""
from __future__ import annotations

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


def build_features(
    df: pd.DataFrame,
    video_features: pd.DataFrame | None = None,
    allow_leaky_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Joins interaction rows with (leakage-guarded) video/user features and
    sorts by user_id to support user-grouped pairwise loss (e.g. BPR / LambdaRank).

    Args:
        df: interaction log rows (train/val slice).
        video_features: optional per-video feature table to left-join on
            `video_id`. Pass the raw KuaiRand video-features file(s) here;
            leaky columns are dropped automatically.
        allow_leaky_columns: explicit opt-in for otherwise-dropped leaky
            columns — see leakage_guard.py. Empty by default.
    """
    out = df.copy()

    if video_features is not None:
        vf_clean, dropped = drop_leaky_columns(video_features, allow_columns=allow_leaky_columns)
        if dropped:
            out.attrs["dropped_leaky_columns"] = dropped
        out = out.merge(vf_clean, on="video_id", how="left")

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
