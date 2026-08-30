"""
Guards against the `video_features_statistic_pure.csv` leakage trap.

Per the KuaiRand docs, this file's columns (`play_cnt`, `show_cnt`,
`like_cnt`, ...) are running aggregates computed "over one month" — a
window that spans train, validation, AND hidden test. Feeding these raw
injects post-hoc future information into training: a video's stats used to
predict an April 10th interaction may already reflect how it performed all
the way through May 8th.

Unlike a column-name heuristic (e.g. matching a `"_statistic"` substring —
what an earlier, synthetic-data-only version of this module did), this
treats the ENTIRE video_features_statistic table as leaky by construction:
every column in it except the join key (`video_id`) is one of those
running aggregates, regardless of what it's named. That's correct because
`video_features_statistic_pure.csv` is a distinct file from
`video_features_basic_pure.csv` (static per-video attributes like
`author_id`, `music_id`, `upload_type` — set once at upload time, not
leaky) — see `pipeline/data/features.py::build_features`, which merges the
two separately for exactly this reason.

This module doesn't try to be clever about reconstructing point-in-time
stats (that requires raw per-day aggregation history the dataset may not
expose). It defaults to the safe behavior: drop every non-key column, and
only allow one back in behind an explicit, logged opt-in — so if the agent
ever re-enables one, that decision is visible in the run log rather than
silently baked into a config file.
"""
from __future__ import annotations

import pandas as pd

JOIN_KEY = "video_id"


def find_leaky_columns(video_features_statistic: pd.DataFrame, join_key: str = JOIN_KEY) -> list[str]:
    return [c for c in video_features_statistic.columns if c != join_key]


def drop_leaky_columns(
    video_features_statistic: pd.DataFrame,
    join_key: str = JOIN_KEY,
    allow_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drops leaky columns from a video_features_statistic-shaped table,
    returning (clean_df, dropped_column_names).

    `allow_columns` is an explicit opt-in list — pass column names here only
    if the agent has a specific, logged reason to use them (e.g. it
    implemented point-in-time reconstruction). Anything not in that list
    gets dropped regardless of how useful it looks in validation, because
    validation is exactly where this leak inflates scores.
    """
    allow_columns = set(allow_columns or [])
    leaky = [c for c in find_leaky_columns(video_features_statistic, join_key) if c not in allow_columns]
    return video_features_statistic.drop(columns=leaky), leaky
