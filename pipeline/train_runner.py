"""
Subprocess entrypoint for every scored training run the orchestrator makes.

NOT in the agent's editable allowlist — this file is the fixed measurement
harness around the editable surface (pipeline/train.py, features.py,
label.py). It exists for two load-bearing reasons:

1. **Fresh imports.** The orchestrator is a long-lived process that imported
   the pipeline modules at startup. `agent/code_editor.py` patches files on
   disk; an in-process `run_training` call after that would silently execute
   the STALE pre-patch code (Python binds imported names once). A fresh
   subprocess re-imports whatever is on disk, so what gets scored is exactly
   what the patch wrote. (This was a real bug: the first live run measured
   iteration 1 with the pre-patch feature code.)

2. **Crash isolation.** A patch can pass the 1-epoch smoke test and still
   crash or hang in a full run. A subprocess failure is a recorded iteration
   error + rollback, not a dead orchestrator.

Prints a single machine-readable line on success:
    TRAIN_RUNNER_METRICS: {"gauc": ..., "ndcg_at_5": ..., "primary": ...,
                           "n_users": ..., "epoch_losses": [...],
                           "unbiased_primary": ...|null, ...}

The unbiased_* fields are the Play-1 referee probe: the trained model scores
the uniformly-random-exposure log (restricted to the validation date window,
2022-04-22..04-28 — strictly inside the sanctioned "extra validation" use and
outside the hidden-test date range) and reports GAUC/nDCG@5 there too.
Probe failure is reported in "referee_error" instead of failing the run.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subsample-train", type=int, default=None,
                        help="cap train rows (cheap ablation probes only)")
    parser.add_argument("--no-referee", action="store_true",
                        help="skip the unbiased-probe scoring pass")
    args = parser.parse_args()

    # Imports happen here, inside the fresh subprocess, on purpose.
    from pipeline.data.features import build_features
    from pipeline.data.label import resolve_label
    from pipeline.data.loader import load_config, load_random_exposure_log, load_split
    from pipeline.evaluate import compute_ranking_metrics
    from pipeline.train import run_training, score_dataframe

    cfg = load_config()
    split = load_split(cfg)
    if args.subsample_train is not None and args.subsample_train < len(split.train):
        split.train = split.train.sample(n=args.subsample_train, random_state=args.seed or 0)

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed

    result = run_training(split=split, **overrides)
    m = result.val_metrics

    payload = {
        "gauc": float(m.gauc),
        "ndcg_at_5": float(m.ndcg_at_5),
        "primary": float(m.primary),
        "n_users": int(m.n_users),
        "epoch_losses": [float(x) for x in result.epoch_losses],
        "unbiased_gauc": None,
        "unbiased_ndcg_at_5": None,
        "unbiased_primary": None,
        "referee_error": None,
    }

    referee_mode = cfg.get("referee", {}).get("mode", "disabled")
    if referee_mode != "disabled" and not args.no_referee:
        try:
            probe = load_random_exposure_log(cfg)
            # Validation window only: never a date that belongs to the hidden
            # test period, even in the (separate, sanctioned) random log.
            probe = probe[(probe["date"] >= 20220422) & (probe["date"] <= 20220428)].reset_index(drop=True)
            probe_feat = build_features(probe)
            probe_label = resolve_label(probe_feat)
            probe_scores = score_dataframe(result.model, result.id_maps, probe_feat)
            probe_df = probe_feat[["user_id"]].copy()
            probe_df["score"] = probe_scores
            probe_df["label"] = probe_label.to_numpy()
            um = compute_ranking_metrics(probe_df)
            payload["unbiased_gauc"] = float(um.gauc)
            payload["unbiased_ndcg_at_5"] = float(um.ndcg_at_5)
            payload["unbiased_primary"] = float(um.primary)
        except Exception as e:  # noqa: BLE001 — probe is diagnostic, never fatal
            payload["referee_error"] = f"{type(e).__name__}: {e}"

    print("TRAIN_RUNNER_METRICS: " + json.dumps(payload))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — any failure must be a non-zero exit
        print(f"TRAIN_RUNNER_ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
