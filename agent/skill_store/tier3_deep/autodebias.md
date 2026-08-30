# Tier 3 deep dive — AutoDebias-style debiasing

Loaded on demand only when the current ablation target is the debiasing /
referee block. Full method notes for `agent/referee.py` Tier-A mode
(training directly on the random-exposure log, not just using it for
diagnostics).

## Core idea

Standard recommender training data is Missing Not At Random (MNAR): what
gets logged as an impression was already filtered by a prior recommendation
policy, so the training distribution differs from the true target
distribution (all user-item pairs, uniformly). A small uniformly-random
exposure log is Missing At Random (MAR) and, while too small to train a
full model on by itself, is exactly what's needed to correct for the MNAR
bias in the large biased log.

## Bi-level formulation

1. Outer loop: learn debiasing parameters (e.g. per-example or per-group
   loss weights) that make a model trained on *reweighted biased data*
   perform well on the *uniform* (MAR) validation set.
2. Inner loop: given current debiasing parameters, train the recommender
   on the biased data with those weights applied.
3. Alternate / unroll a few inner steps per outer step (full bi-level
   optimization is expensive — a truncated unroll is a reasonable
   compute-budget compromise).

## Cheaper fallback (Tier-B compatible)

If bi-level optimization is unstable or too expensive given the token/GPU
budget: estimate per-video exposure propensity directly as
`(exposures in random log) / (total random log size)`, normalized, and use
`1 / propensity` (clipped to avoid extreme weights) as a static importance
weight per training example. This loses the joint optimization benefit but
is a single extra column in `pipeline/data/features.py` and trains in one
pass — appropriate when iteration budget is tight.

## Evaluation implication

Whichever variant is used, always report metrics on BOTH the standard
validation split AND the unbiased probe (`agent/referee.py`). The gap
between them, not either number alone, is the signal that debiasing is
actually working versus just moving the overfit target.
