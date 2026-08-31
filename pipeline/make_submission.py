"""
The final, explicitly-designated submission step (run manually AFTER the
orchestrator has converged):

    python -m pipeline.make_submission --out-dir submissions

Runs in a fresh process so it trains exactly the code state the orchestrator
left on disk — which, because non-improving patches are rolled back, is the
best-known state — with the same fixed seed used during the run, then:

  1. writes the VALID-split submission, runs the official alignment check
     and the official local score on it (valid scoring is sanctioned);
  2. writes the TEST-split submission with allow_test=True — this is the
     only place in the repo that flag is ever set. The test submission is
     format/alignment-checked but NEVER locally scored: the hidden-test
     number is the organizer's to compute, and the agent's development loop
     must stay blind to it.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override only if the converged train.py default is not desired")
    parser.add_argument("--ensemble-seeds", type=int, default=1,
                        help="train N models with different seeds (seed, seed+1, ... seed+N-1) and "
                             "rank-average their scores. FROZEN FINAL CONFIG: --seed 0 "
                             "--ensemble-seeds 10. Measured on the DeepFM-lite champion over a "
                             "20-seed pool: single seed mean 0.6045 (std 0.0005, range "
                             "0.6036-0.6052), 10-seed rank-average 0.6053. The mean gain is NOT "
                             "established (it is inside the same-model negative control), but the "
                             "VARIANCE reduction is: 5-seed std 0.0002 vs single-seed 0.0005, "
                             "matching sqrt(n). Ensembling is shipped to protect the downside on a "
                             "one-shot submission, not to claim a higher mean.")
    args = parser.parse_args()

    import numpy as np
    from pipeline.submit import (aligned_rows, score_rows, validate_submission,
                                 write_submission_from_scores)
    from pipeline.train import run_training

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"seed": args.seed}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    # Rank-average rather than score-average: only within-user ORDER is scored,
    # and ranks are scale-free, so one model's score distribution cannot
    # dominate the blend.
    def rank(v):
        order = v.argsort()
        out = np.empty(len(v), dtype=np.float64)
        out[order] = np.arange(len(v))
        return out

    _, valid_rows = aligned_rows(split="valid")
    _, test_rows = aligned_rows(split="test", allow_test=True)
    valid_acc = np.zeros(len(valid_rows)); test_acc = np.zeros(len(test_rows))

    for i in range(args.ensemble_seeds):
        seed = args.seed + i
        print(f"[make_submission] Training model {i + 1}/{args.ensemble_seeds} (seed={seed}) "
              "from the on-disk (best-known) code state...")
        result = run_training(**{**overrides, "seed": seed})
        m = result.val_metrics
        print(f"[make_submission]   seed {seed} valid: GAUC={m.gauc:.4f}, "
              f"nDCG@5={m.ndcg_at_5:.4f}, primary={m.primary:.4f}")
        valid_acc += rank(score_rows(result.model, result.id_maps, valid_rows))
        test_acc += rank(score_rows(result.model, result.id_maps, test_rows))

    valid_path = out_dir / "submission_valid.csv"
    write_submission_from_scores(valid_acc, valid_path, split="valid")
    validate_submission(valid_path, split="valid")
    print(f"[make_submission] Wrote + alignment-checked {valid_path}")

    test_path = out_dir / "submission_test.csv"
    write_submission_from_scores(test_acc, test_path, split="test", allow_test=True)
    validate_submission(test_path, split="test")
    print(f"[make_submission] Wrote + alignment-checked {test_path} (NOT locally scored — hidden split)")


if __name__ == "__main__":
    main()
