"""
NOT part of the agent pipeline. Ad-hoc verification script — checks the
non-ML logic (label resolution, leakage guard, ranking metrics) against
hand-computable examples and the synthetic data, so these can be trusted
before spending real API calls / GPU time on the full loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data.features import build_features
from pipeline.data.label import profile_label, resolve_label, resolve_primary_label, valid_play_consistency_check
from pipeline.data.leakage_guard import drop_leaky_columns, find_leaky_columns
from pipeline.data.loader import load_config, load_split
from pipeline.evaluate import compute_ranking_metrics, score_delta

TWO_COLUMN_TABS = {0, 1}


def test_label_logic():
    split = load_split()
    profile = profile_label(split.train, TWO_COLUMN_TABS)
    print(f"[label] rows={profile.n_rows} two_col={profile.n_two_column} single_col={profile.n_single_column}")
    print(f"[label] click_rate two_col={profile.click_rate_two_column:.3f} single_col={profile.click_rate_single_column:.3f}")

    consistency = valid_play_consistency_check(split.train, TWO_COLUMN_TABS)
    print(f"[label] valid_play consistency on single-col rows: {consistency:.3f}")
    # A loose sanity bound, not a strict check: `is_click` isn't the primary
    # scored label (that's `long_view`, resolved separately and needing no
    # derivation), and `TWO_COLUMN_TABS` is a documented, unconfirmed
    # assumption about which `tab` values map to which UI — real data won't
    # satisfy the hypothesized valid_play rule as cleanly as hand-built
    # synthetic data did. This just catches the derivation being flat-out
    # broken, not real-world noise around the boundary.
    assert consistency > 0.90, "valid_play derivation is far off from is_click even loosely — check the TWO_COLUMN_TABS assumption"

    raw = resolve_label(split.train, TWO_COLUMN_TABS, mode="raw")
    click_only = resolve_label(split.train, TWO_COLUMN_TABS, mode="click_only")
    assert (click_only <= raw).all(), "click_only mode should never exceed raw"
    is_two_col = split.train["tab"].isin(TWO_COLUMN_TABS)
    assert (click_only[~is_two_col] == 0).all(), "click_only should zero out all single-column-UI rows"
    print("[label] PASS")


def test_leakage_guard():
    cfg = load_config()
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    vf_stat = pd.read_csv(raw_dir / cfg["dataset"]["video_features_statistic"])

    leaky = find_leaky_columns(vf_stat)
    print(f"[leakage] detected leaky columns: {len(leaky)} (e.g. {leaky[:3]}...)")
    assert "play_cnt" in leaky
    assert "video_id" not in leaky  # the join key must never be dropped

    clean, dropped = drop_leaky_columns(vf_stat)
    assert "play_cnt" not in clean.columns
    assert "video_id" in clean.columns
    assert dropped == leaky

    clean_allowed, dropped_allowed = drop_leaky_columns(vf_stat, allow_columns=["play_cnt"])
    assert "play_cnt" not in dropped_allowed
    assert "play_cnt" in clean_allowed.columns
    print("[leakage] PASS")


def test_ranking_metrics():
    # Hand-computable case: 2 users, 4 candidates each.
    # User A: perfect ranking (positive ranked first) -> AUC=1.0, nDCG@5=1.0
    # User B: worst ranking (positive ranked last)     -> AUC=0.0, nDCG@5 < 1.0, known value
    df = pd.DataFrame(
        {
            "user_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "score": [4, 3, 2, 1, 1, 2, 3, 4],       # A: descending matches order given; B: ascending
            "label": [1, 0, 0, 0, 1, 0, 0, 0],        # single relevant item each
        }
    )
    metrics = compute_ranking_metrics(df, k_ndcg=5)
    print(f"[metrics] GAUC={metrics.gauc:.4f} nDCG@5={metrics.ndcg_at_5:.4f}")

    # User A: relevant item has the highest score -> ranked 1st -> DCG=IDCG -> NDCG=1.0
    # User B: relevant item has the LOWEST score -> ranked 4th (last) -> DCG = 1/log2(5), IDCG = 1/log2(2)
    expected_ndcg_b = (1 / np.log2(5)) / (1 / np.log2(2))
    expected_mean_ndcg = (1.0 + expected_ndcg_b) / 2
    assert abs(metrics.ndcg_at_5 - expected_mean_ndcg) < 1e-9, f"expected {expected_mean_ndcg}, got {metrics.ndcg_at_5}"

    # A: positive ranked above all 3 negatives -> AUC=1.0. B: positive ranked below all 3 -> AUC=0.0.
    # GAUC is weighted by each user's positive count (1 each here, so a plain mean) -> (1.0 + 0.0) / 2 = 0.5
    assert abs(metrics.gauc - 0.5) < 1e-9, f"expected GAUC 0.5, got {metrics.gauc}"
    assert metrics.n_users_gauc == 2

    # Sanity-check score_delta: identical metrics -> 0 delta
    delta = score_delta(metrics, metrics)
    assert abs(delta) < 1e-9

    # A user with zero positives (single-class labels): included in nDCG@5 as 0.0 (per the
    # challenge brief — a no-positive-label user has nDCG 0 for any model), but excluded from
    # GAUC entirely (AUC is undefined without both classes).
    df_no_pos = pd.DataFrame({"user_id": ["C", "C"], "score": [1, 2], "label": [0, 0]})
    df2 = pd.concat([df, df_no_pos], ignore_index=True)
    metrics2 = compute_ranking_metrics(df2, k_ndcg=5)
    expected_mean_ndcg_with_c = (1.0 + expected_ndcg_b + 0.0) / 3
    assert abs(metrics2.ndcg_at_5 - expected_mean_ndcg_with_c) < 1e-9, "user with no positives should count as 0 in nDCG@5 mean"
    assert abs(metrics2.gauc - metrics.gauc) < 1e-9, "user with no positives should not change GAUC"
    assert metrics2.n_users_gauc == 2, "user with no positives should not be counted toward GAUC"
    print("[metrics] PASS")


def test_real_data_integration():
    """Confirms the primary-label switch (long_view) and the two-file video
    feature merge (basic=safe, statistic=leaky) work end to end against the
    real KuaiRand-Pure data, matching the official Starter Kit's task
    definition (starter_kit/data.py::LABEL, starter_kit/README.md)."""
    cfg = load_config()
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    split = load_split(cfg)

    label = resolve_primary_label(split.train)
    assert set(label.unique()) <= {0, 1}
    click_rate = label.mean()
    print(f"[real-data] long_view positive rate (train): {click_rate:.3f}")
    assert 0.0 < click_rate < 1.0, "long_view shouldn't be degenerate across 1.14M real rows"

    vf_basic = pd.read_csv(raw_dir / cfg["dataset"]["video_features_basic"])
    vf_stat = pd.read_csv(raw_dir / cfg["dataset"]["video_features_statistic"])
    feat = build_features(split.train.head(5000), video_features_basic=vf_basic, video_features_statistic=vf_stat)

    assert "author_id" in feat.columns, "video_features_basic should merge in (not leaky)"
    leaky_cols = [c for c in vf_stat.columns if c != "video_id"]
    assert not any(c in feat.columns for c in leaky_cols), "leaky video_features_statistic columns leaked into build_features output"
    assert feat.attrs.get("dropped_leaky_columns"), "build_features should report which columns it dropped"
    print(f"[real-data] merged features shape={feat.shape}, dropped {len(feat.attrs['dropped_leaky_columns'])} leaky columns")
    print("[real-data] PASS")


if __name__ == "__main__":
    test_label_logic()
    test_leakage_guard()
    test_ranking_metrics()
    test_real_data_integration()
    print("\nALL LOGIC TESTS PASSED")
