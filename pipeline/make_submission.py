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
    args = parser.parse_args()

    from pipeline.submit import validate_submission, write_submission
    from pipeline.train import run_training

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"seed": args.seed}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    print("[make_submission] Training final model from the on-disk (best-known) code state...")
    result = run_training(**overrides)
    m = result.val_metrics
    print(
        f"[make_submission] Final model valid metrics: GAUC={m.gauc:.4f}, "
        f"nDCG@5={m.ndcg_at_5:.4f}, primary={m.primary:.4f}"
    )

    valid_path = out_dir / "submission_valid.csv"
    write_submission(result.model, result.id_maps, valid_path, split="valid")
    validate_submission(valid_path, split="valid")
    print(f"[make_submission] Wrote + alignment-checked {valid_path}")

    test_path = out_dir / "submission_test.csv"
    write_submission(result.model, result.id_maps, test_path, split="test", allow_test=True)
    validate_submission(test_path, split="test")
    print(f"[make_submission] Wrote + alignment-checked {test_path} (NOT locally scored — hidden split)")


if __name__ == "__main__":
    main()
