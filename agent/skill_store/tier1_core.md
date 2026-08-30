# Tier 1 — Core (always loaded)

This tier is small on purpose. HASTE's ablation found flat-loading a large
skill inventory performs *identically to loading no skills at all*, at
double the token cost — only tiered, targeted loading beats cold-start.
Tier 1 stays under ~1 page; deeper material lives in Tier 2/3 and is loaded
by `retriever.py` only when the current ablation target calls for it.

## Task framing (confirmed by the official Starter Kit — `starter_kit/`)

- Dataset: KuaiRand-Pure. **Positive label: `long_view`** (native 0/1
  column, no resolution needed — see `pipeline/data/label.py`). `is_click`
  is a *different*, unscored signal — useful only as a candidate auxiliary
  task, never as the training target.
- Task: **within-user ranking over logged impressions** — each user's
  candidates are only the videos they were actually shown in the eval
  window, not the full ~7,583-item catalog. No retrieval/candidate-
  generation step needed.
- Metrics: GAUC, nDCG@5. Scored as `primary = mean(GAUC, nDCG@5)` vs. the
  official baseline — **absolute** delta, not relative, on the hidden test
  set. Zero-positive users count as nDCG=0 in the mean; GAUC only counts
  users with `0 < positives < impressions`, weighted by positive count.
- Split: train `2022-04-08..04-21` / valid `2022-04-22..04-28` / test
  `2022-04-29..05-08` (hidden — never load it, see
  `pipeline/data/loader.py`'s `allow_test` guard).
- **Official FM baseline (the target): valid primary 0.6016, test primary
  0.5946.** Full ladder in `starter_kit/baseline_scores.json`: random
  0.4753, item-popularity 0.5715, FM 0.5946, oracle ceiling 0.8645 (test).
  nDCG@5's ceiling is 0.7289, not 1.0 — 27.1% of test users are all-negative
  (nDCG stuck at 0 regardless of model) and 9.2% all-positive. **Judge
  progress against 0.8645, not 1.0** — the FM baseline has already claimed
  ~31% of the attainable headroom.
- Convergence: ε=0.002, N=3 (FM's seed-to-seed std is 0.0008, so this is
  ≈2.5σ — a real plateau, not noise).
- You are scored once, on the validation-best checkpoint at convergence.
  A checkpoint that scores well on validation but doesn't generalize is
  worse than useless — see `agent/compression_gate.py` and use it before
  designating anything final.

## What NOT to waste iterations on (organizer-tested, see `starter_kit/README.md`)

1. **More static features.** Adding CWM's full 13 feature domains
   (music_id, video_type, upload_type + 6 user-side coarse buckets) moved
   test primary from 0.5950 to 0.5940 — noise, if anything slightly worse.
2. **More model capacity.** FM embedding dim k=8/16/32 gave 0.5895/0.5902/
   0.5887 — barely moves. 1.14M training rows can't support much more
   capacity, and `user_id × video_id` crosses already capture most of the
   learnable signal.
3. **Pure user-side features, full stop.** Ranking is *within* a user, so
   any feature that's constant across a user's own candidates contributes
   exactly zero to their ranking (verified empirically: `item_pop × user
   bias` scores bit-identical to plain `item_pop`). User-side signal only
   matters through a **cross term with an item-side feature**.

## Where the organizers believe headroom actually is (priority order, untested by them)

1. **Loss/objective mismatch** — training is pointwise logloss but the
   metrics are ranking metrics (GAUC, nDCG). Pairwise (BPR) or listwise
   (per-user softmax over that user's impressions) loss aligns the
   objective with what's scored. **This is what they'd try first.**
2. **User history sequence** — current features use zero behavioral
   sequence, despite each user having hundreds-to-thousands of train
   interactions. DIN/SIM-style interest modeling is untouched territory.
3. **Multi-task** — `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` are all in the logs but unused; auxiliary
   heads on top of `long_view` (see Tier 2/3) are a natural next step.
4. **Watch-time modeling** — CWM's contribution: treat watch time as
   right-censored (a video that plays to completion has an unknown true
   "would-have-watched" time) and use a one-sided/censored loss rather than
   squared error. Research-depth, higher effort.
5. **Model swap** (DeepFM/DCN/xDeepFM) — deprioritized versus 1-4 since
   capacity is confirmed not to be the bottleneck.
6. **Time features / distribution drift** — `hourmin`, `date`, and drift
   between train and test.
7. **Unbiased validation (advanced)** — `log_random_4_22_to_5_08_pure.csv`
   (1.19M rows, uniform exposure) as an overfitting check. See
   `agent/referee.py` and Tier 3 `autodebias.md`.

## Where to look next

- Need RecSys architecture/method background? -> Tier 2 (`tier2_domain.md`)
- Need a deep dive on a specific method you're about to implement? ->
  Tier 3 (`tier3_deep/`), loaded on demand by `retriever.py` keyed to the
  current ablation target.
