# Tier 2 — RecSys domain priors (loaded when relevant to current ablation target)

Loaded selectively by `retriever.py`, keyed to which pipeline block
`agent/ablation.py` identified as the current bottleneck. Not injected
wholesale every iteration.

## Feature / architecture block

- **DeepFM-style**: factorization-machine low-order crosses + deep MLP for
  high-order interactions, sharing the embedding layer. Good default for
  sparse categorical CTR features (user_id, video_id, tab).
- **DIN (Deep Interest Network)**: attention over a user's historical
  interactions weighted by relevance to the candidate item. Per Tier 1, the
  organizers rank user-history sequence modeling as headroom item #2 —
  each user has hundreds-to-thousands of train interactions, and no
  sequential feature exists yet. Worth real investment here, not a
  deprioritized stretch goal.

## Multi-task / auxiliary-signal block

- KuaiRand logs several feedback signals (`is_click`, `is_like`,
  `is_follow`, `is_comment`, `is_forward`, `is_hate`, `play_time_ms`, ...)
  but only `long_view` is scored (Tier 1). Per the organizers (Tier 1,
  headroom item #3), auxiliary-task transfer using these unused signals is
  untested territory worth exploring.
- **ESMM (Entire Space Multi-task Model)**: jointly models two related
  tasks to avoid sample-selection bias between them. Adaptable here as
  `long_view` (primary) + `is_click` or `is_like` (auxiliary) sharing a
  bottom tower.
- **PLE (Progressive Layered Extraction)**: separates task-shared and
  task-specific experts explicitly, reducing negative transfer between
  loosely-related auxiliary tasks compared to a naive shared-bottom MMoE.
  Preferred over plain MMoE when auxiliary signals (e.g. `is_hate`) might
  actively conflict with the click objective.

## Debiasing block (see also `agent/referee.py`, Play 1)

- **AutoDebias**: uses a small uniform (missing-at-random) exposure set to
  learn debiasing weights via bi-level meta-learning, applied on top of a
  standard model trained on biased (logged-policy) data. Reported strong
  gains even with as little as 1% uniform data on benchmark datasets —
  KuaiRand-Pure's random log is a much larger fraction than that.
- Simpler alternative if AutoDebias's bi-level optimization proves unstable
  given the compute budget: inverse-propensity weighting using propensities
  estimated directly from the random log's exposure rate per video, as a
  per-example loss weight during standard training.

## Negative sampling block

- Implicit-feedback CTR data is heavily class-imbalanced (most impressions
  are not clicks). Standard options: random negative sampling, popularity-
  biased negative sampling (harder negatives, but can amplify popularity
  bias — be cautious combining with the debiasing block above), and
  in-batch negatives if moving to a two-tower retrieval-style formulation.
