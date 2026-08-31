"""
Loads KuaiRand-Pure interaction logs and applies the official Starter Kit
split (`starter_kit/data.py::SPLITS` — do not deviate from this, it's what
the hidden-test scoring uses):

    train = 2022-04-08 .. 2022-04-21   (log_standard_4_08_to_4_21_pure.csv)
    valid = 2022-04-22 .. 2022-04-28   (log_standard_4_22_to_5_08_pure.csv)
    test  = 2022-04-29 .. 2022-05-08   (log_standard_4_22_to_5_08_pure.csv, HIDDEN — never load during dev)

KuaiRand's standard interaction logs carry a `date` column (int, YYYYMMDD).
This loader splits on that column rather than re-deriving dates from
`time_ms`, since `date` is what the Starter Kit's split is defined on.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_yyyymmdd(date_str: str) -> int:
    return int(dt.date.fromisoformat(date_str).strftime("%Y%m%d"))


@dataclass
class KuaiRandSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame  # DO NOT read/inspect this during agent development


def _read_log(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m pipeline.data.download --check` "
            "for setup instructions."
        )
    return pd.read_csv(path)


def load_split(cfg: dict | None = None, allow_test: bool = False) -> KuaiRandSplit:
    """Load train/val/test per the fixed date split.

    `allow_test=False` (default) returns an empty test DataFrame as a guard
    rail against accidentally touching hidden test during development — the
    agent's orchestrator must never set this to True. Only the final,
    organizer-run scoring step should ever see real test rows.
    """
    cfg = cfg or load_config()
    ds = cfg["dataset"]
    raw_dir = Path(ds["raw_dir"])

    train_df = _read_log(raw_dir, ds["logs"]["standard_train"])
    evalset_df = _read_log(raw_dir, ds["logs"]["standard_evalset"])

    if "date" not in train_df.columns or "date" not in evalset_df.columns:
        raise ValueError(
            "Expected a `date` column (YYYYMMDD int) in the KuaiRand-Pure logs. "
            f"Got columns: {list(evalset_df.columns)}. Schema may have changed — "
            "update loader.py's split logic to match."
        )

    val_lo, val_hi = ds["val_range"]
    test_lo, test_hi = ds["test_range"]
    val_lo, val_hi = _to_yyyymmdd(val_lo), _to_yyyymmdd(val_hi)
    test_lo, test_hi = _to_yyyymmdd(test_lo), _to_yyyymmdd(test_hi)

    val_df = evalset_df[(evalset_df["date"] >= val_lo) & (evalset_df["date"] <= val_hi)].reset_index(drop=True)

    if allow_test:
        test_df = evalset_df[(evalset_df["date"] >= test_lo) & (evalset_df["date"] <= test_hi)].reset_index(drop=True)
    else:
        test_df = evalset_df.iloc[0:0].copy()

    return KuaiRandSplit(train=train_df.reset_index(drop=True), val=val_df, test=test_df)


def load_random_exposure_log(cfg: dict | None = None) -> pd.DataFrame:
    """Loads the uniformly-random-exposure log (Play 1 / referee.py).

    This file sits outside the prescribed train/val/test split. It is
    in-dataset (shipped as part of KuaiRand-Pure) but its use is gated by
    `config.referee.mode` pending organizer confirmation — see README.
    """
    cfg = cfg or load_config()
    ds = cfg["dataset"]
    raw_dir = Path(ds["raw_dir"])
    return _read_log(raw_dir, ds["logs"]["random_unbiased"])


if __name__ == "__main__":
    split = load_split()
    print(f"train: {len(split.train):,} rows")
    print(f"val:   {len(split.val):,} rows")
    print(f"test:  {len(split.test):,} rows (0 expected unless allow_test=True)")
