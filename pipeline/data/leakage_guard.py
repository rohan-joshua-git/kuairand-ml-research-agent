"""
Guards against the `video_features_statistic` leakage trap.

Per the KuaiRand docs, statistical video features are computed as running
averages "over one month" — a window that spans train, validation, AND
hidden test. Feeding these raw injects post-hoc future information into
training: a video's stats used to predict a April 10th interaction may
already reflect how it performed all the way through May 8th.

This module doesn't try to be clever about reconstructing point-in-time
stats (that requires raw per-day aggregation history the dataset may not
expose). It defaults to the safe behavior: drop leaky columns entirely,
and only allow them back in behind an explicit, logged opt-in — so if the
agent ever re-enables them, that decision is visible in the run log rather
than silently baked into a config file.
"""
from __future__ import annotations

import pandas as pd

# Column name patterns known (from the KuaiRand-Pure field spec) to be
# month-long running averages that span the train/val/test boundary.
LEAKY_COLUMN_MARKERS = (
    "_statistic",
)


def find_leaky_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(marker in c for marker in LEAKY_COLUMN_MARKERS)]


def drop_leaky_columns(df: pd.DataFrame, allow_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Drops leaky columns, returning (clean_df, dropped_column_names).

    `allow_columns` is an explicit opt-in list — pass column names here only
    if the agent has a specific, logged reason to use them (e.g. it
    implemented point-in-time reconstruction). Anything not in that list
    gets dropped regardless of how useful it looks in validation, because
    validation is exactly where this leak inflates scores.
    """
    allow_columns = set(allow_columns or [])
    leaky = [c for c in find_leaky_columns(df) if c not in allow_columns]
    return df.drop(columns=leaky), leaky
