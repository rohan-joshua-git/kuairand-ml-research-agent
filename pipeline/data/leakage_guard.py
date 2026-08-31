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

IMPORTANT: on the real KuaiRand-Pure `video_features_statistic_pure.csv`,
the leaky columns do NOT follow a `_statistic` naming convention — they're
named things like `show_cnt`, `play_cnt`, `like_cnt`, `follow_cnt` (see the
real header). A pure substring-marker check silently misses every one of
them. `KNOWN_LEAKY_STATISTIC_COLUMNS` below is the file's actual column
list (minus the `video_id` join key); `LEAKY_COLUMN_MARKERS` is kept only
as a fallback for naming conventions that do self-identify (e.g. the
synthetic test fixture's `video_features_statistic_play_count`).
"""
from __future__ import annotations

import pandas as pd

LEAKY_COLUMN_MARKERS = (
    "_statistic",
)

# Exact header of video_features_statistic_pure.csv, minus video_id — all are
# month-long running counts/rates that span train/val/test.
KNOWN_LEAKY_STATISTIC_COLUMNS = {
    "counts", "show_cnt", "show_user_num", "play_cnt", "play_user_num",
    "play_duration", "complete_play_cnt", "complete_play_user_num",
    "valid_play_cnt", "valid_play_user_num", "long_time_play_cnt",
    "long_time_play_user_num", "short_time_play_cnt", "short_time_play_user_num",
    "play_progress", "comment_stay_duration", "like_cnt", "like_user_num",
    "click_like_cnt", "double_click_cnt", "cancel_like_cnt", "cancel_like_user_num",
    "comment_cnt", "comment_user_num", "direct_comment_cnt", "reply_comment_cnt",
    "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt",
    "comment_like_user_num", "follow_cnt", "follow_user_num", "cancel_follow_cnt",
    "cancel_follow_user_num", "share_cnt", "share_user_num", "download_cnt",
    "download_user_num", "report_cnt", "report_user_num", "reduce_similar_cnt",
    "reduce_similar_user_num", "collect_cnt", "collect_user_num",
    "cancel_collect_cnt", "cancel_collect_user_num", "direct_comment_user_num",
    "reply_comment_user_num", "share_all_cnt", "share_all_user_num",
    "outsite_share_all_cnt",
}


def find_leaky_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c in KNOWN_LEAKY_STATISTIC_COLUMNS or any(marker in c for marker in LEAKY_COLUMN_MARKERS)
    ]


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
