"""
Generates the final submission artifact in the organizer's confirmed
schema (`starter_kit/README.md` "Submission format"):

    row_id,user_id,video_id,score

Row order/`row_id` must match `starter_kit/data.py`'s `load()` exactly —
read `log_standard_4_08_to_4_21_pure.csv` then
`log_standard_4_22_to_5_08_pure.csv`, filtered by date, file order
preserved. Rather than re-derive that ordering independently (and risk a
silent misalignment that `--check` would catch too late), this module
loads the canonical row list straight from the vendored `starter_kit.data`
and writes with the vendored `starter_kit.submit.write_submission` — the
same code the organizer's own `--make` path uses. It cross-checks that
ordering against `pipeline.data.loader`'s pandas-loaded split before
trusting it, since the model scores come from that DataFrame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

STARTER_KIT_DIR = Path(__file__).resolve().parents[1] / "starter_kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import data as _sk_data  # noqa: E402 — path insert above is required first
import submit as _sk_submit  # noqa: E402

from pipeline.data.features import build_features  # noqa: E402
from pipeline.data.loader import load_config, load_split  # noqa: E402


class SubmissionAlignmentError(RuntimeError):
    pass


def write_submission(
    model: nn.Module,
    id_maps: dict,
    out_path: str | Path,
    split: str = "valid",
    allow_test: bool = False,
    cfg: dict | None = None,
) -> None:
    """Writes a submission CSV for `split` ('valid' or 'test'). Generating a
    'test' submission requires `allow_test=True` explicitly — same guard
    philosophy as `pipeline.data.loader.load_split`, since test is hidden
    during development and only the final scoring run should touch it.
    """
    if split == "test" and not allow_test:
        raise ValueError(
            "Generating a 'test' split submission requires allow_test=True. "
            "Test is hidden during development — only the final, designated "
            "submission run should set this."
        )

    cfg = cfg or load_config()
    kit_splits = _sk_data.load(cfg["dataset"]["raw_dir"])
    kit_rows = kit_splits[split]

    our_split = load_split(cfg, allow_test=(split == "test"))
    our_df = our_split.val if split == "valid" else our_split.test

    if len(our_df) != len(kit_rows):
        raise SubmissionAlignmentError(
            f"pipeline.data.loader loaded {len(our_df)} rows for split={split!r}, "
            f"starter_kit.data.load loaded {len(kit_rows)} — row-count mismatch, "
            "would produce a misaligned submission."
        )
    our_uv = list(zip(our_df["user_id"].astype(str), our_df["video_id"].astype(str)))
    kit_uv = [(x[1], x[2]) for x in kit_rows]
    if our_uv != kit_uv:
        first_bad = next(i for i, (a, b) in enumerate(zip(our_uv, kit_uv)) if a != b)
        raise SubmissionAlignmentError(
            f"Row order mismatch at index {first_bad}: pipeline.data.loader gave "
            f"{our_uv[first_bad]}, starter_kit.data.load gave {kit_uv[first_bad]}. "
            "Do not submit — row_id alignment is not guaranteed."
        )

    # build_features is agent-editable and MAY permute rows (it currently
    # sorts by user_id for pairwise-loss grouping; the raw logs are NOT
    # user-sorted — verified on real data). Scores must be written in
    # kit_rows' original file order, so carry each row's original position
    # through the feature build and un-permute the scores afterwards. This
    # is the difference between a valid submission and one that silently
    # passes --check with every score attached to the wrong row.
    from pipeline.train import score_dataframe  # late import: use on-disk code

    tagged = our_df.copy()
    tagged["_orig_row_pos"] = np.arange(len(tagged))
    feat_df = build_features(tagged)
    if len(feat_df) != len(tagged) or set(feat_df["_orig_row_pos"]) != set(range(len(tagged))):
        raise SubmissionAlignmentError(
            "build_features added or dropped rows — cannot align scores to the "
            "submission row order. Fix build_features to be row-preserving."
        )
    permuted_scores = score_dataframe(model, id_maps, feat_df)
    scores = np.empty(len(tagged), dtype=np.float64)
    scores[feat_df["_orig_row_pos"].to_numpy()] = permuted_scores

    _sk_submit.write_submission(str(out_path), kit_rows, scores)


def validate_submission(path: str | Path, split: str = "test", cfg: dict | None = None) -> None:
    """Thin wrapper around the vendored `starter_kit.submit.read_submission`
    — run this before treating any submission as final."""
    cfg = cfg or load_config()
    kit_splits = _sk_data.load(cfg["dataset"]["raw_dir"])
    _sk_submit.read_submission(str(path), kit_splits[split])
