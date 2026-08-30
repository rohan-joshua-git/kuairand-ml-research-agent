"""
NOT part of the agent pipeline. A test-only tool that generates fake,
KuaiRand-Pure-shaped CSVs so the pipeline can be smoke-tested end-to-end
without the real (194MB, manually-obtained) dataset. Do not use these
numbers for anything except "does the code run" — they're random.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def _date_range_yyyymmdd(start: str, end: str) -> list[int]:
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    n_days = (end_d - start_d).days + 1
    return [int((start_d + dt.timedelta(days=i)).strftime("%Y%m%d")) for i in range(n_days)]

RNG = np.random.default_rng(0)

N_USERS = 200
N_VIDEOS = 150
TWO_COLUMN_TABS = {0, 1}  # arbitrary assumption for this synthetic test
ALL_TABS = list(range(15))


def _gen_log(n_rows: int, date_range: list[int], include_stat_cols: bool = True) -> pd.DataFrame:
    dates = RNG.choice(date_range, size=n_rows)
    tabs = RNG.choice(ALL_TABS, size=n_rows)
    duration_ms = RNG.integers(1000, 20000, size=n_rows)

    # single-column UI (tab not in TWO_COLUMN_TABS): derive is_click from valid_play rule
    # two-column UI: is_click is a genuine, independent-ish click signal
    play_time_ms = (duration_ms * RNG.uniform(0.3, 1.3, size=n_rows)).astype(int)
    is_two_col = np.isin(tabs, list(TWO_COLUMN_TABS))

    valid_play = np.where(
        duration_ms < 7000, play_time_ms >= duration_ms, play_time_ms > 7000
    ).astype(int)
    genuine_click = RNG.binomial(1, 0.15, size=n_rows)
    is_click = np.where(is_two_col, genuine_click, valid_play)

    df = pd.DataFrame(
        {
            "user_id": RNG.integers(0, N_USERS, size=n_rows),
            "video_id": RNG.integers(0, N_VIDEOS, size=n_rows),
            "date": dates,
            "tab": tabs,
            "is_click": is_click,
            "play_time_ms": play_time_ms,
            "duration_ms": duration_ms,
            "profile_stay_time": RNG.integers(0, 5000, size=n_rows),
            "comment_stay_time": RNG.integers(0, 2000, size=n_rows),
            "is_like": RNG.binomial(1, 0.05, size=n_rows),
            "is_follow": RNG.binomial(1, 0.01, size=n_rows),
            "is_comment": RNG.binomial(1, 0.02, size=n_rows),
            "is_forward": RNG.binomial(1, 0.01, size=n_rows),
            "is_hate": RNG.binomial(1, 0.005, size=n_rows),
            "long_view": RNG.binomial(1, 0.2, size=n_rows),
        }
    )

    if include_stat_cols:
        # deliberately-leaky column, spans the whole window — for testing leakage_guard.py
        df["video_features_statistic_play_count"] = RNG.integers(0, 100000, size=n_rows)

    return df


def main() -> None:
    out_dir = "./data/raw"
    import os
    os.makedirs(out_dir, exist_ok=True)

    train_dates = _date_range_yyyymmdd("2022-04-08", "2022-04-21")
    eval_dates = _date_range_yyyymmdd("2022-04-22", "2022-05-08")

    train_df = _gen_log(20000, train_dates)
    eval_df = _gen_log(8000, eval_dates)
    random_df = _gen_log(6000, eval_dates, include_stat_cols=False)

    train_df.to_csv(f"{out_dir}/log_standard_4_08_to_4_21_pure.csv", index=False)
    eval_df.to_csv(f"{out_dir}/log_standard_4_22_to_5_08_pure.csv", index=False)
    random_df.to_csv(f"{out_dir}/log_random_4_22_to_5_08_pure.csv", index=False)

    print(f"Wrote synthetic data to {out_dir}/:")
    print(f"  train: {len(train_df)} rows, dates {train_df['date'].min()}-{train_df['date'].max()}")
    print(f"  eval:  {len(eval_df)} rows, dates {eval_df['date'].min()}-{eval_df['date'].max()}")
    print(f"  random:{len(random_df)} rows")


if __name__ == "__main__":
    main()
