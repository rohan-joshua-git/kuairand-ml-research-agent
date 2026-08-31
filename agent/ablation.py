"""
Play 2 chassis: block-level ablation.

Instead of rewriting the whole pipeline each iteration (the failure mode
that makes generic MLE agents expensive and slow to converge), this module
runs a small ablation over named pipeline "blocks" each round and reports
which one is actually carrying the current score — so the orchestrator's
next code-diff targets that block specifically. This mirrors MLE-STAR's
approach, which reached the strongest reported medal rate on MLE-bench Lite
via targeted, ablation-guided refinement rather than tree search over whole
solutions.

Probes are genuinely reduced-scale (subsampled train, few epochs) and run
in a subprocess (agent/subprocess_training.py) so they always execute the
code currently on disk. Their absolute deltas vs the full-run best are
therefore pessimistic — only the RELATIVE ordering between variants is
meaningful, which is all pick_highest_impact_block uses. A variant that
crashes (e.g. an agent rewrite of train.py dropped a kwarg a variant
passes) is skipped and reported with its error, never fatal.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.subprocess_training import TrainSubprocessError, run_training_subprocess
from pipeline.evaluate import RankingMetrics

ABLATION_SUBSAMPLE_TRAIN = 300_000
ABLATION_EPOCHS_DEFAULT = 2


@dataclass
class BlockVariant:
    block_name: str
    description: str
    kwargs_override: dict


@dataclass
class AblationResult:
    block_name: str
    description: str
    val_primary: float
    delta_primary: float
    error: str | None = None


def default_block_variants() -> list[BlockVariant]:
    """A starting ablation grid over train.py's exposed knobs. The agent is
    expected to grow/replace this list over iterations as it introduces new
    blocks (e.g. once a pairwise-loss variant, a sequence-model block, or a
    multitask head exists, add variants that toggle them, per the Starter
    Kit's priority list in agent/skill_store/tier1_core.md) — this function
    is a seed, not a fixed set.
    """
    return [
        BlockVariant("training_schedule", "more epochs", {"epochs": 4}),
        BlockVariant("learning_rate", "lower learning rate", {"epochs": ABLATION_EPOCHS_DEFAULT, "lr": 3e-4}),
    ]


def run_ablation(
    baseline_metrics: RankingMetrics,
    variants: list[BlockVariant] | None = None,
) -> list[AblationResult]:
    """One cheap subprocess probe per variant, measured against the current
    best full-run metrics. Deltas are pessimistic (subsampled probes score
    below full runs) — compare variants to each other, not to zero.
    """
    variants = variants or default_block_variants()
    results = []
    for variant in variants:
        try:
            probe = run_training_subprocess(
                subsample_train=ABLATION_SUBSAMPLE_TRAIN,
                no_referee=True,
                timeout_s=900,
                **variant.kwargs_override,
            )
            results.append(
                AblationResult(
                    block_name=variant.block_name,
                    description=variant.description,
                    val_primary=probe.primary,
                    delta_primary=probe.primary - baseline_metrics.primary,
                )
            )
        except TrainSubprocessError as e:
            results.append(
                AblationResult(
                    block_name=variant.block_name,
                    description=variant.description,
                    val_primary=float("nan"),
                    delta_primary=float("-inf"),
                    error=f"{e} | {e.output[-300:]}",
                )
            )
    return results


def pick_highest_impact_block(results: list[AblationResult]) -> AblationResult:
    """Rank by delta on `primary` (mean of GAUC and nDCG@5) — the challenge's
    actual scoring quantity (2.6 Judging Criteria), which for equal-weighted
    metrics collapses to a plain difference of primaries. See
    pipeline/evaluate.py:score_delta.
    """
    return max(results, key=lambda r: r.delta_primary)
