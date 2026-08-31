"""
Label resolution.

The primary relevance label is `long_view` — confirmed by the organizer
Starter Kit (`starter_kit/data.py` LABEL, `starter_kit/README.md` "Task
definition"), NOT `is_click`. `resolve_label` below returns it directly.

`is_click` is still relevant, but only as an AUXILIARY signal: the Starter
Kit's own priority-ranked "where headroom is" list (item 3) names
is_click/is_like/is_follow/is_comment/is_forward/play_time_ms as usable
auxiliary tasks to help the long_view main task via multi-task learning
(see `agent/skill_store/tier1_core.md`). Because `is_click` conflates two
different user-behavior constructs depending on which UI the interaction
happened in (the `tab` field, range 0-14):

  - Two-column UI: is_click is a genuine tap/click.
  - Single-column UI (main feed): is_click is actually `valid_play` —
    1 when play_time_ms >= duration_ms for videos under 7000ms, or when
    play_time_ms > 7000ms for longer videos.

...an auxiliary head trained on the raw column conflates two different
constructs. `resolve_auxiliary_click_label` below makes that split
explicit so a multi-task auxiliary head trains on a clean signal, and
`profile_label` reports the finding (this is meant to show up in the
agent's run log as a discovered insight, not just get quietly patched).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PRIMARY_LABEL_COLUMN = "long_view"
DURATION_THRESHOLD_MS = 7000


def resolve_label(df: pd.DataFrame) -> pd.Series:
    """Returns the primary relevance label (long_view, 0/1) as float."""
    return df[PRIMARY_LABEL_COLUMN].astype(float).copy()


@dataclass
class LabelProfile:
    n_rows: int
    n_two_column: int
    n_single_column: int
    click_rate_two_column: float
    click_rate_single_column: float
    tabs_seen: list[int]


def profile_label(df: pd.DataFrame, two_column_tabs: set[int]) -> LabelProfile:
    """Surfaces the tab-conditioned split as a reportable finding.

    `two_column_tabs` must be supplied by the caller (ideally confirmed
    against the Starter Kit / organizer docs) since the challenge brief does
    not enumerate which of the 15 tab values map to which UI. Until
    confirmed, treat this as a documented assumption — see README.
    """
    is_two_col = df["tab"].isin(two_column_tabs)
    two_col_df = df[is_two_col]
    single_col_df = df[~is_two_col]
    return LabelProfile(
        n_rows=len(df),
        n_two_column=len(two_col_df),
        n_single_column=len(single_col_df),
        click_rate_two_column=float(two_col_df["is_click"].mean()) if len(two_col_df) else float("nan"),
        click_rate_single_column=float(single_col_df["is_click"].mean()) if len(single_col_df) else float("nan"),
        tabs_seen=sorted(df["tab"].unique().tolist()),
    )


def resolve_auxiliary_click_label(df: pd.DataFrame, two_column_tabs: set[int], mode: str = "raw") -> pd.Series:
    """Returns a resolved `is_click` column, for use as a multi-task
    AUXILIARY head only (see module docstring) — never as the primary
    training target, which is `resolve_label`/long_view.

    mode:
      "raw"                -> just returns is_click unchanged
      "click_only"         -> zero out rows in the single-column UI (keep only genuine clicks)
      "scenario_conditioned" -> keep is_click as-is but caller should train
                                 separate scenario-conditioned heads keyed on
                                 `tab`, rather than collapsing to one label.
                                 This function just returns the raw label in
                                 that case; the conditioning happens in the
                                 model, not the label.
    """
    if mode == "raw":
        return df["is_click"].copy()

    is_two_col = df["tab"].isin(two_column_tabs)

    if mode == "click_only":
        label = df["is_click"].copy()
        label[~is_two_col] = 0
        return label

    if mode == "scenario_conditioned":
        return df["is_click"].copy()

    raise ValueError(f"Unknown label resolution mode: {mode}")


def valid_play_consistency_check(df: pd.DataFrame, two_column_tabs: set[int]) -> float:
    """Sanity-checks the documented valid_play definition against is_click
    for single-column-UI rows. Returns the fraction of rows where is_click
    matches the derived valid_play rule — should be close to 1.0 if the
    field spec assumption holds for this data.
    """
    single = df[~df["tab"].isin(two_column_tabs)].copy()
    if single.empty:
        return float("nan")
    derived = (
        ((single["duration_ms"] < DURATION_THRESHOLD_MS) & (single["play_time_ms"] >= single["duration_ms"]))
        | ((single["duration_ms"] >= DURATION_THRESHOLD_MS) & (single["play_time_ms"] > DURATION_THRESHOLD_MS))
    ).astype(int)
    return float((derived == single["is_click"]).mean())
