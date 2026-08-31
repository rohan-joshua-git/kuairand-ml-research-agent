"""
Fast sanity check used by agent/code_editor.py before a code change is
trusted with a full training run: 1 epoch, a small subset of train, one
pass through val. Exists to catch import errors, shape mismatches, and
crashes cheaply (seconds, not minutes) — a full `run_training` call is not
an appropriate smoke test, it's the thing the smoke test exists to protect.

Exits 0 and prints `SMOKE_TEST_METRICS: {...json...}` on success. Any
exception propagates as a non-zero exit so the calling subprocess in
code_editor.py can detect failure without parsing stack traces.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from pipeline.data.features import build_features
from pipeline.data.loader import load_config, load_split
from pipeline.train import run_training, score_dataframe


def main() -> None:
    cfg = load_config()
    split = load_split(cfg)

    # Cap to a small subset purely for smoke-test speed.
    split.train = split.train.sample(n=min(5000, len(split.train)), random_state=0)
    split.val = split.val.sample(n=min(2000, len(split.val)), random_state=0)

    result = run_training(split=split, epochs=1)

    # The scoring contract (used by submit.py and the referee probe) must
    # survive any patch too: one finite score per row, order preserved.
    sample = build_features(split.val)
    scores = score_dataframe(result.model, result.id_maps, sample)
    assert len(scores) == len(sample), (
        f"score_dataframe returned {len(scores)} scores for {len(sample)} rows — contract broken"
    )
    assert np.isfinite(scores).all(), "score_dataframe produced NaN/Inf scores — contract broken"

    metrics = {
        "gauc": result.val_metrics.gauc,
        "ndcg_at_5": result.val_metrics.ndcg_at_5,
        "primary": result.val_metrics.primary,
    }
    print(f"SMOKE_TEST_METRICS: {json.dumps(metrics)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — deliberately broad: any failure here should fail the smoke test
        print(f"SMOKE_TEST_ERROR: {e}", file=sys.stderr)
        sys.exit(1)
