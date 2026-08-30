"""
Feature engineering — this module is the primary surface the agent rewrites
each iteration (Figure 1 "engineer features" stage).

`build_features` is intentionally a single, clearly-named entrypoint so the
agent's code diffs have one obvious place to land: it can add columns,
change encodings, or swap out the whole body, and `ablation.py` can diff
this file's contents across iterations to explain what changed.

The starting implementation is deliberately minimal (id features + raw
numeric signals) — it exists to make the pipeline run end-to-end from
iteration 0, not to be a competitive feature set.
"""
from __future__ import annotations

import pandas as pd

from pipeline.data.leakage_guard import drop_leaky_columns

NUMERIC_SIGNAL_COLUMNS = [
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
]

AUXILIARY_LABEL_COLUMNS = [
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
]


def build_features(
    df: pd.DataFrame,
    video_features: pd.DataFrame | None = None,
    allow_leaky_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Joins interaction rows with (leakage-guarded) video/user features.

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
            # Left visible rather than silently swallowed — the orchestrator's
            # logger should capture this as part of the iteration's diff summary.
            out.attrs["dropped_leaky_columns"] = dropped
        out = out.merge(vf_clean, on="video_id", how="left")

    for col in NUMERIC_SIGNAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    return out


def auxiliary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the non-click feedback signals available for multi-task
    auxiliary transfer (Play 3 / ESMM-PLE style heads). Missing columns are
    tolerated so this stays safe to call before all signals are confirmed
    present in a given log file.
    """
    present = [c for c in AUXILIARY_LABEL_COLUMNS if c in df.columns]
    return df[present].fillna(0)
