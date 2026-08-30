"""
Fast sanity check before an LLM-authored edit to agent/ablation.py is
trusted — the agent/ablation.py counterpart to pipeline/smoke_test.py.

pipeline/smoke_test.py checks that pipeline training still works; it
doesn't import or exercise agent/ablation.py at all, so it can't catch a
broken ablation-grid rewrite. This module actually runs
default_block_variants() through run_ablation() and
pick_highest_impact_block() against a small real-data sample, and checks
every BlockVariant's editable_target is a real key in
agent.code_editor.EDITABLE_FILES — a variant with a typo'd or
newly-invented target would otherwise fail silently much later, in
agent/orchestrator.py, with a confusing KeyError.

Exits 0 and prints `SMOKE_TEST_METRICS: {...json...}` on success, same
convention as pipeline/smoke_test.py, so agent/code_editor.py's
apply_and_smoke_test can use either interchangeably.
"""
from __future__ import annotations

import json
import sys

from agent.ablation import default_block_variants, pick_highest_impact_block, run_ablation
from agent.code_editor import EDITABLE_FILES
from pipeline.data.loader import load_config, load_split
from pipeline.train import run_training


def main() -> None:
    cfg = load_config()
    split = load_split(cfg)

    # Cap to a small subset purely for smoke-test speed — matches
    # pipeline/smoke_test.py's sampling.
    split.train = split.train.sample(n=min(5000, len(split.train)), random_state=0)
    split.val = split.val.sample(n=min(2000, len(split.val)), random_state=0)

    baseline_metrics = run_training(split=split, epochs=1).val_metrics

    variants = default_block_variants()
    if not variants:
        raise ValueError("default_block_variants() returned an empty grid")

    for v in variants:
        if v.editable_target not in EDITABLE_FILES:
            raise ValueError(
                f"BlockVariant {v.block_name!r} has editable_target={v.editable_target!r}, "
                f"not a key in agent.code_editor.EDITABLE_FILES ({list(EDITABLE_FILES)})"
            )

    results = run_ablation(split, baseline_metrics, variants=variants)
    if not results:
        raise ValueError("run_ablation() returned no results — every variant raised an exception")

    target = pick_highest_impact_block(results)
    if target is None:
        raise ValueError("pick_highest_impact_block() returned None despite non-empty results")

    metrics = {
        "n_variants": len(variants),
        "n_results": len(results),
        "picked_block": target.block_name,
        "picked_target": target.editable_target,
    }
    print(f"SMOKE_TEST_METRICS: {json.dumps(metrics)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — deliberately broad: any failure here should fail the smoke test
        print(f"SMOKE_TEST_ERROR: {e}", file=sys.stderr)
        sys.exit(1)
