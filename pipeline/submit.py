"""
Generates the final submission artifact.

Schema is confirmed by the Starter Kit (`starter_kit/submit.py` /
`starter_kit/README.md`) — CSV with header `row_id,user_id,video_id,score`:

    row_id    0-indexed, strictly sequential, matching the row order of
              `pipeline.data.loader.load_split(cfg, allow_test=True).test`
              (which itself preserves the source CSV's row order, filtered
              by date — the same determinism the organizer's `data.load()`
              produces).
    user_id / video_id   redundant, used only by the organizer's checker to
              confirm alignment with their own eval-set row order.
    score     any real number; only relative order within a user matters.
              NaN/Inf are rejected by the organizer's checker.

Why row_id and not (user_id, video_id) as the key: those pairs are NOT
unique in the eval set (per the Starter Kit, ~3% of test rows are
duplicate (user_id, video_id) pairs, up to 12x) — row position is the only
valid key.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(scores: np.ndarray, split_df: pd.DataFrame, out_path: str | Path) -> None:
    """`scores` must be aligned 1:1 with `split_df`'s row order (e.g. the
    `test` or `val` DataFrame from `pipeline.data.loader.load_split`)."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(split_df):
        raise ValueError(f"scores has {len(scores)} rows but split_df has {len(split_df)}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contains NaN/Inf — the organizer's checker rejects these")

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(split_df)),
            "user_id": split_df["user_id"].to_numpy(),
            "video_id": split_df["video_id"].to_numpy(),
            "score": scores,
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
