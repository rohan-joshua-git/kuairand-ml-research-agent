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

Blocks are intentionally coarse-grained (feature set / label resolution /
architecture width / training schedule) rather than line-level, because the
point is to cheaply identify *which stage of Figure 1* to focus the next
(expensive) LLM-driven code change on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pipeline.data.loader import KuaiRandSplit
from pipeline.evaluate import RankingMetrics
from pipeline.train import run_training


@dataclass
class BlockVariant:
    block_name: str
    description: str
    kwargs_override: dict


@dataclass
class AblationResult:
    block_name: str
    description: str
    val_metrics: RankingMetrics
    delta_gauc: float
    delta_ndcg: float


def default_block_variants() -> list[BlockVariant]:
    """A starting ablation grid over train.py's exposed knobs. The agent is
    expected to grow/replace this list over iterations as it introduces new
    blocks (e.g. once a debiasing block or multitask head exists, add
    variants that toggle them) — this function is a seed, not a fixed set.
    """
    return [
        BlockVariant("label_resolution", "raw vs click_only label mode", {"label_mode": "click_only"}),
        BlockVariant("training_schedule", "more epochs", {"epochs": 6}),
        BlockVariant("learning_rate", "lower learning rate", {"lr": 3e-4}),
    ]


def run_ablation(
    split: KuaiRandSplit,
    baseline_metrics: RankingMetrics,
    variants: list[BlockVariant] | None = None,
) -> list[AblationResult]:
    """Trains one short run per variant and measures its delta against the
    current baseline_metrics. Callers should use a reduced `epochs` in the
    baseline run for this to stay cheap — ablation is meant to be a fast
    signal, not a full training run per block.
    """
    variants = variants or default_block_variants()
    results = []
    for variant in variants:
        train_result = run_training(split=split, **variant.kwargs_override)
        m = train_result.val_metrics
        results.append(
            AblationResult(
                block_name=variant.block_name,
                description=variant.description,
                val_metrics=m,
                delta_gauc=m.gauc - baseline_metrics.gauc,
                delta_ndcg=m.ndcg_at_5 - baseline_metrics.ndcg_at_5,
            )
        )
    return results


def pick_highest_impact_block(results: list[AblationResult]) -> AblationResult:
    """Per README/Tier-1 guidance: the challenge scores an equal-weighted
    mean of GAUC delta and nDCG@5 delta, so rank on the combined (mean)
    delta rather than favoring either metric.
    """
    return max(results, key=lambda r: (r.delta_gauc + r.delta_ndcg) / 2.0)
