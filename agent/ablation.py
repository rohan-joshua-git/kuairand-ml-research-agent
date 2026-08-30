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
    editable_target: str  # key into agent.code_editor.EDITABLE_FILES — which
    # file the orchestrator should target if this variant wins the ablation


@dataclass
class AblationResult:
    block_name: str
    description: str
    val_metrics: RankingMetrics
    delta_gauc: float
    delta_ndcg: float
    editable_target: str


def default_block_variants() -> list[BlockVariant]:
    """A starting ablation grid over train.py's exposed knobs — a seed, not
    a fixed set. `agent/ablation.py` is itself in `code_editor.EDITABLE_FILES`,
    and `agent/orchestrator.py`'s `_maybe_grow_ablation_grid` periodically
    lets the agent rewrite this function directly (smoke tested by
    `agent/ablation_smoke_test.py`), so the grid growing over time is a
    property of the running agent, not something a human needs to seed by
    hand after the fact.

    All current variants target `editable_target="train"` since they probe
    `pipeline/train.py`'s hyperparameters (epoch count, learning rate, loss
    weighting) — coverage for `features.py`, `label.py`, and
    `model/baseline.py` is exactly the kind of gap grid growth is meant to
    close as the agent edits those files and needs a way to ablate the
    changes it makes there.
    """
    return [
        BlockVariant("training_schedule", "more epochs", {"epochs": 6}, editable_target="train"),
        BlockVariant("learning_rate", "lower learning rate", {"lr": 3e-4}, editable_target="train"),
        BlockVariant(
            "pos_class_weight", "upweight long_view positives via BCE pos_weight", {"pos_weight": 2.0}, editable_target="train"
        ),
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

    Each variant runs behind a try/except: `kwargs_override` is forwarded
    straight to `run_training(**kwargs_override)`, and since the grid can
    now be grown autonomously (agent/orchestrator.py's
    `_maybe_grow_ablation_grid`, via `agent/ablation_smoke_test.py`), a
    newly-added variant may reference a keyword `pipeline/train.py` doesn't
    accept yet (e.g. the agent adds a `use_author_id` variant before it's
    gotten around to teaching `run_training` that kwarg). A crash from one
    bad variant must not take down the whole ablation round — it's simply
    skipped, and `pipeline/train.py` catching up on a later iteration is
    the self-correction, not a hard failure here.
    """
    variants = variants or default_block_variants()
    results = []
    for variant in variants:
        try:
            train_result = run_training(split=split, **variant.kwargs_override)
        except Exception as e:  # noqa: BLE001 — see docstring: one bad variant must not crash the round
            print(f"[ablation] variant {variant.block_name!r} failed, skipping: {e}")
            continue
        m = train_result.val_metrics
        results.append(
            AblationResult(
                block_name=variant.block_name,
                description=variant.description,
                val_metrics=m,
                delta_gauc=m.gauc - baseline_metrics.gauc,
                delta_ndcg=m.ndcg_at_5 - baseline_metrics.ndcg_at_5,
                editable_target=variant.editable_target,
            )
        )
    return results


def pick_highest_impact_block(results: list[AblationResult]) -> AblationResult | None:
    """Per README/Tier-1 guidance: the challenge scores an equal-weighted
    mean of GAUC delta and nDCG@5 delta, so rank on the combined (mean)
    delta rather than favoring either metric.

    Returns None if every variant failed (see `run_ablation`) — callers
    must handle that as a non-event for this round, not a crash.
    """
    if not results:
        return None
    return max(results, key=lambda r: (r.delta_gauc + r.delta_ndcg) / 2.0)
