"""
Primary label + Play 3 (Label Archaeology).

The official Starter Kit (`starter_kit/data.py` / `evaluate.py`) fixes the
SCORED label as `long_view` — a clean, native 0/1 column that needs no
resolution. `resolve_primary_label` below is what training and evaluation
must use.

`is_click` is a *different*, separate signal, not the scored label — but
it's still a genuinely useful one, and per the KuaiRand field spec it means
different things depending on which UI the interaction happened in (the
`tab` field, range 0-14):

  - Two-column UI: is_click is a genuine tap/click.
  - Single-column UI (main feed): is_click is actually `valid_play` —
    1 when play_time_ms >= duration_ms for videos under 7000ms, or when
    play_time_ms > 7000ms for longer videos.

Conflating those two constructs would corrupt `is_click` if it's ever used
as an auxiliary multi-task signal (see `starter_kit/README.md`'s headroom
item #3 and `pipeline/data/features.py::AUXILIARY_LABEL_COLUMNS`). The rest
of this module makes that distinction explicit and *reports* the finding
(meant to show up in the agent's run log as a discovered insight, not just
get quietly patched).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PRIMARY_LABEL_COLUMN = "long_view"
DURATION_THRESHOLD_MS = 7000


def resolve_primary_label(df: pd.DataFrame) -> pd.Series:
    """The official scored label. Already clean (0/1) per the Starter Kit —
    no resolution needed, unlike `is_click` below."""
    return (df[PRIMARY_LABEL_COLUMN] != 0).astype(int)


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


def resolve_label(df: pd.DataFrame, two_column_tabs: set[int], mode: str = "raw") -> pd.Series:
    """Returns the resolved positive-label column.

    mode:
      "raw"                -> just returns is_click unchanged (baseline-compatible)
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
