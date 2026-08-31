"""
Reproduces the organizer's official baseline (Task Requirement #1: "Stand
up a working end-to-end pipeline and confirm it reaches the official
baseline's reported validation score").

This runs the vendored `starter_kit/baseline.py`'s FM directly — not a
reimplementation — so there is no risk of a subtly-different model quietly
drifting from the number that's actually published
(`starter_kit/baseline_scores.json`). `agent/orchestrator.py` calls
`reproduce_official_baseline` once at the start of a run, before any
LLM-driven iteration, and uses its result as the fixed reference point for
"are we beating baseline" throughout the loop (separate from
iteration-to-iteration convergence tracking against the agent's own best
checkpoint).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

STARTER_KIT_DIR = Path(__file__).resolve().parents[1] / "starter_kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import baseline as _sk_baseline  # noqa: E402 — path insert above is required first
import data as _sk_data  # noqa: E402

from pipeline.data.loader import load_config  # noqa: E402


@dataclass
class BaselineReproductionResult:
    valid_primary: float
    test_primary: float
    valid_gauc: float
    valid_ndcg_at_5: float
    test_gauc: float
    test_ndcg_at_5: float
    matches_published: bool
    published_valid_primary: float
    published_test_primary: float


def reproduce_official_baseline(cfg: dict | None = None, seed: int = 0, tolerance: float = 0.003) -> BaselineReproductionResult:
    """Trains the organizer's own FM (k=16, lr=0.001) on real data via the
    vendored starter_kit and checks the result against the published
    numbers in config.starter_kit.official_baseline. `tolerance` defaults
    to ~3.75x the published 5-seed std (0.0008) to allow for normal
    single-seed variance.
    """
    cfg = cfg or load_config()
    ref = cfg["starter_kit"]["official_baseline"]

    splits = _sk_data.load(cfg["dataset"]["raw_dir"])
    result = _sk_baseline.run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=seed, verbose=False)

    # starter_kit's vendored evaluate() mixes numpy float32 (from labels) into
    # its arithmetic, so GAUC/nDCG/primary come back as numpy scalars, not
    # plain Python floats — not JSON-serializable as-is. Cast at this
    # boundary rather than touching the vendored evaluate.py.
    valid_primary = float(result["valid"]["primary"])
    test_primary = float(result["test"]["primary"])

    matches = (
        abs(valid_primary - ref["valid"]["primary"]) <= tolerance
        and abs(test_primary - ref["test"]["primary"]) <= tolerance
    )

    return BaselineReproductionResult(
        valid_primary=valid_primary,
        test_primary=test_primary,
        valid_gauc=float(result["valid"]["GAUC"]),
        valid_ndcg_at_5=float(result["valid"]["nDCG@5"]),
        test_gauc=float(result["test"]["GAUC"]),
        test_ndcg_at_5=float(result["test"]["nDCG@5"]),
        matches_published=matches,
        published_valid_primary=ref["valid"]["primary"],
        published_test_primary=ref["test"]["primary"],
    )


if __name__ == "__main__":
    r = reproduce_official_baseline()
    status = "MATCHES" if r.matches_published else "DOES NOT MATCH"
    print(f"valid primary: {r.valid_primary:.4f} (published: {r.published_valid_primary:.4f})")
    print(f"test primary:  {r.test_primary:.4f} (published: {r.published_test_primary:.4f})")
    print(f"{status} published baseline (Task Requirement #1)")
