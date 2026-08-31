# Tier 2 — RecSys domain priors (loaded when relevant to current ablation target)

Loaded selectively by `retriever.py`, keyed to which pipeline block
`agent/ablation.py` identified as the current bottleneck. Not injected
wholesale every iteration.

## Feature / architecture block

- **DeepFM-style**: factorization-machine low-order crosses + deep MLP for
  high-order interactions, sharing the embedding layer. Good default for
  sparse categorical CTR features (user_id, video_id, tab).
- **DIN (Deep Interest Network)**: attention over a user's historical
  interactions weighted by relevance to the candidate item. Relevant if
  building any sequence-of-recent-interactions feature, but remember Tier
  1's warning: KuaiRand-Pure's sequential logs are incomplete (only the
  ~7,583 candidate-pool videos are logged), so a full DIN-style attention
  stack may be more complexity than the data supports.

## Multi-task / auxiliary-signal block

- KuaiRand logs 12 feedback signals total (click, like, follow, comment,
  forward, hate, long_view, ...) but only **long_view** is scored (the
  primary label — see Tier 1). The other signals are usable only as
  auxiliary tasks, per the Starter Kit's own priority-3 lead.
- **ESMM (Entire Space Multi-task Model)**: jointly models CTR and CTCVR to
  avoid sample-selection bias between a click model and a downstream
  conversion model. Adaptable here as click + long_view sharing a bottom
  tower, with long_view as the head that's actually scored — note KuaiRand
  logs long_view on every impression (not only clicked ones), so the
  classic sample-selection-bias motivation for ESMM doesn't apply as
  directly here; the transfer-learning benefit (shared representations)
  is the part that still holds.
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
