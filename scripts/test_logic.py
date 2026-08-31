"""
NOT part of the agent pipeline. Ad-hoc verification script — checks the
non-ML logic (label resolution, leakage guard, ranking metrics) against
hand-computable examples and real data, so these can be trusted before
spending real API calls / GPU time on the full loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.data.label import (
    profile_label,
    resolve_auxiliary_click_label,
    resolve_label,
    valid_play_consistency_check,
)
from pipeline.data.leakage_guard import drop_leaky_columns, find_leaky_columns
from pipeline.data.loader import load_config, load_split
from pipeline.evaluate import RankingMetrics, compute_ranking_metrics, score_delta

TWO_COLUMN_TABS = {0, 1}


def test_label_logic():
    split = load_split()

    primary = resolve_label(split.train)
    assert set(primary.unique().tolist()) <= {0.0, 1.0}, "long_view should be binary"
    assert (primary == split.train["long_view"].astype(float)).all(), "resolve_label must return long_view unchanged"
    print(f"[label] primary=long_view, positive rate={primary.mean():.3f}")

    profile = profile_label(split.train, TWO_COLUMN_TABS)
    print(f"[label] rows={profile.n_rows} two_col={profile.n_two_column} single_col={profile.n_single_column}")
    print(f"[label] aux is_click rate two_col={profile.click_rate_two_column:.3f} single_col={profile.click_rate_single_column:.3f}")

    consistency = valid_play_consistency_check(split.train, TWO_COLUMN_TABS)
    print(f"[label] valid_play consistency on single-col rows: {consistency:.3f}")

    raw = resolve_auxiliary_click_label(split.train, TWO_COLUMN_TABS, mode="raw")
    click_only = resolve_auxiliary_click_label(split.train, TWO_COLUMN_TABS, mode="click_only")
    assert (click_only <= raw).all(), "click_only mode should never exceed raw"
    is_two_col = split.train["tab"].isin(TWO_COLUMN_TABS)
    assert (click_only[~is_two_col] == 0).all(), "click_only should zero out all single-column-UI rows"
    print("[label] PASS")


def test_leakage_guard():
    cfg = load_config()
    raw_dir = cfg["dataset"]["raw_dir"]
    video_stats = pd.read_csv(f"{raw_dir}/{cfg['dataset']['features']['video_statistic']}")

    leaky = find_leaky_columns(video_stats)
    print(f"[leakage] detected {len(leaky)} leaky columns (of {len(video_stats.columns)} total)")
    assert "show_cnt" in leaky and "play_cnt" in leaky and "like_cnt" in leaky, (
        "real video_features_statistic_pure.csv columns (show_cnt/play_cnt/like_cnt, no "
        "'_statistic' substring) must be caught by the exact-name list, not just the marker"
    )
    assert "video_id" not in leaky, "the join key itself must not be flagged as leaky"

    clean, dropped = drop_leaky_columns(video_stats)
    assert "show_cnt" not in clean.columns
    assert dropped == leaky

    clean_allowed, dropped_allowed = drop_leaky_columns(video_stats, allow_columns=leaky)
    assert dropped_allowed == []
    assert "show_cnt" in clean_allowed.columns
    print("[leakage] PASS")


def test_ranking_metrics():
    # Hand-computable case: 2 users, 4 candidates each.
    # User A: perfect ranking (positive ranked first) -> nDCG@5 = 1.0
    # User B: worst ranking (positive ranked last) -> nDCG@5 < 1.0, known value
    df = pd.DataFrame(
        {
            "user_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "score": [4, 3, 2, 1, 1, 2, 3, 4],       # A: descending matches order given; B: ascending
            "label": [1, 0, 0, 0, 1, 0, 0, 0],        # single relevant item each
        }
    )
    metrics = compute_ranking_metrics(df)
    print(f"[metrics] GAUC={metrics.gauc:.4f} nDCG@5={metrics.ndcg_at_5:.4f} primary={metrics.primary:.4f}")

    # User A: relevant item ranked 1st -> DCG=IDCG -> nDCG=1.0
    # User B: relevant item ranked 4th -> DCG = 1/log2(5), IDCG = 1/log2(2)
    expected_ndcg_b = (1 / np.log2(5)) / (1 / np.log2(2))
    expected_mean_ndcg = (1.0 + expected_ndcg_b) / 2
    assert abs(metrics.ndcg_at_5 - expected_mean_ndcg) < 1e-9, f"expected {expected_mean_ndcg}, got {metrics.ndcg_at_5}"

    # Both users have exactly 1 positive out of 4 -> both counted in GAUC (0 < pos < n).
    # A: positive scored highest -> perfectly ranked above all 3 negatives -> AUC=1.0
    # B: positive scored lowest -> ranked below all 3 negatives -> AUC=0.0
    # GAUC weights by positive count (1 each) -> mean(1.0, 0.0) = 0.5
    assert abs(metrics.gauc - 0.5) < 1e-9, f"expected GAUC 0.5, got {metrics.gauc}"

    # Sanity-check score_delta: identical metrics -> 0 delta
    delta = score_delta(metrics, metrics)
    assert abs(delta) < 1e-9

    # A user with zero positives should score nDCG=0 and be INCLUDED in the mean,
    # and be EXCLUDED from GAUC (per starter_kit/evaluate.py's pinned convention).
    df_no_pos = pd.DataFrame({"user_id": ["C", "C"], "score": [1, 2], "label": [0, 0]})
    df2 = pd.concat([df, df_no_pos], ignore_index=True)
    metrics2 = compute_ranking_metrics(df2)
    expected_ndcg2 = (1.0 + expected_ndcg_b + 0.0) / 3
    assert abs(metrics2.ndcg_at_5 - expected_ndcg2) < 1e-9, f"expected {expected_ndcg2}, got {metrics2.ndcg_at_5}"
    assert abs(metrics2.gauc - metrics.gauc) < 1e-9, "an all-negative user must not change GAUC"
    print("[metrics] PASS")


if __name__ == "__main__":
    test_label_logic()
    test_leakage_guard()
    test_ranking_metrics()
    print("\nALL LOGIC TESTS PASSED")
