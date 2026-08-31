# Tier 1 — Core (always loaded)

This tier is small on purpose. HASTE's ablation found flat-loading a large
skill inventory performs *identically to loading no skills at all*, at
double the token cost — only tiered, targeted loading beats cold-start.
Tier 1 stays under ~1 page; deeper material lives in Tier 2/3 and is loaded
by `retriever.py` only when the current ablation target calls for it.

## Task framing (confirmed against the organizer Starter Kit — `starter_kit/`)

- Dataset: KuaiRand-Pure. Positive label: **`long_view`** (NOT `is_click` —
  see `pipeline/data/label.py`). Metrics: **GAUC, nDCG@5**. Primary score =
  `mean(GAUC, nDCG@5)`. Scored as the mean of the two metrics' absolute
  deltas vs. the official baseline — which, since both sides are
  equal-weighted means, collapses to a plain difference of primaries
  (`pipeline/evaluate.py:score_delta`).
- Split is date-pinned: train = 4/08-4/21 (1,141,112 rows), val = 4/22-4/28
  (124,909 rows), test = 4/29-5/08 (170,588 rows, hidden — never load it,
  see `pipeline/data/loader.py` `allow_test` guard).
- Official FM baseline (k=16, lr=0.001, fields=user_id/video_id/author_id/
  tab/dur_bucket): test primary **0.5946** (GAUC 0.6610, nDCG@5 0.5282).
  This is what you must beat — reproduced locally via
  `pipeline/official_baseline.py`, confirmed to match within seed noise.
- **Read headroom against the oracle ceiling (test primary 0.8645), not
  against 1.0.** 27.1% of test users are all-negative (nDCG=0 for any
  model, excluded from GAUC) and 9.2% are all-positive (nDCG=1, excluded
  from GAUC) — only the remaining 63.7% are discriminative. The baseline
  has already used ~31% of the *available* range; there's ~0.27 of primary
  left, not ~0.41.
- Convergence: ε=0.002, N=3 (3 iterations without a >0.002 primary
  improvement on the agent's own best checkpoint). Hard cap 50 iterations
  / 6h wall-clock.
- You are scored once, on the validation-best checkpoint at convergence,
  on the hidden test set. A checkpoint that scores well on validation but
  doesn't generalize is worse than useless — see `agent/compression_gate.py`
  and use it before designating anything final.

## Already tried (organizer-run, ~free signal — don't re-spend iterations here)

`starter_kit/ablation_features.py` is a reproducible experiment the
organizers already ran:

- **Adding more features** (CWM's 13 fields: +music_id/video_type/
  upload_type + 6 bucketed user-side fields) moved primary from 0.5950 to
  0.5940 — within noise, if not slightly worse.
- **Growing model capacity** (embedding dim k=8/16/32) was flat: 0.5895 /
  0.5902 / 0.5887.
- Why: `user_id x video_id` crossing already captures most learnable
  signal, and 1.14M rows already carry enough capacity for it.
  **Pure user-side features contribute ~0 on their own** — ranking happens
  *within* each user, so anything constant within a user doesn't change
  that user's order. User-side features only matter through a cross term
  with item-side features.

**Don't burn early iterations on "add more columns" or "bump embedding
size" as a first move — it's a measured dead end, not a guess.**

## Where headroom should be (organizer's own priority order)

1. **Switch the loss function.** Pointwise logloss now; GAUC/nDCG are
   *ranking* metrics — pairwise (BPR) or listwise (softmax over a user's
   candidates) aligns training with eval. Organizer's own top pick.
   Grounding: BPR-OPT (Rendle et al., UAI 2009, arXiv:1205.2618) is a
   differentiable surrogate for per-user AUC — it replaces the Heaviside
   step in the per-user AUC statistic with log-sigmoid over
   (positive, negative) score differences. GAUC here IS a positive-weighted
   mean of per-user AUCs, so this is the matched objective, not a loose
   analogy. Dataset-specific deviation from the textbook version: sample
   negatives j from the SAME USER'S LOGGED IMPRESSIONS with long_view=0
   (not from the full catalog as in retrieval-style BPR) — the metric only
   ranks within the logged candidate set, so full-catalog negatives
   optimize the wrong comparison. For nDCG@5 specifically, a listwise
   softmax cross-entropy over each user's candidates (ListNet-style)
   targets top-of-list quality more directly than pairwise.
2. **User behavior sequence modeling.** Zero behavioral history used today.
   Users have hundreds of train interactions — DIN/SIM-style interest
   modeling is unexplored.
3. **Multi-task auxiliary heads.** `is_click`, `is_like`, `is_follow`,
   `is_comment`, `is_forward` are usable auxiliary signals for the
   `long_view` main task (`pipeline/data/features.py:AUXILIARY_LABEL_COLUMNS`).
   Note `play_time_ms` is EXCLUDED from features on purpose — it's a
   near-direct proxy for `long_view` itself (0.64 correlation, 84.7%
   threshold-accuracy on real data) and using it as an input is leakage,
   not modeling.
4. **Watch-time-aware / censored regression.** CWM's contribution — watch
   time is censored by video length, so a one-sided loss beats squared
   error. Research-depth; treat CWM as an advanced lead, not a starting
   point (needs `torch==1.6.0`, uses its own `long_view2` label).
5. **Swap architectures** (DeepFM/DCN/xDeepFM). Explicitly lower priority
   given capacity scaling was flat (see "Already tried" above) — do 1-4
   first.
6. **Temporal features / distribution shift.** `date`/time-of-day and the
   train-to-test drift.
7. **`log_random_4_22_to_5_08_pure.csv` as EXTRA VALIDATION, never
   training.** Confirmed sanctioned use (`config.referee.mode: tier_b`) —
   an unbiased-exposure probe to check whether a checkpoint is overfitting
   the biased standard-log validation split, not a training source.
   `tier_a` (training on it) is NOT sanctioned.

## Dataset-specific traps

1. **`is_click` is two different constructs**, relevant only for the
   auxiliary head above (not the primary label). In the two-column UI it's
   a genuine tap; in the single-column UI it's `valid_play`
   (`play_time_ms >= duration_ms` for videos under 7000ms, or
   `play_time_ms > 7000ms` for longer). See `pipeline/data/label.py`
   `resolve_auxiliary_click_label`/`profile_label`. On real data this
   assumption holds ~97.2% of the time, not exactly 1.0 — profile it, don't
   assume.
2. **`video_features_statistic_pure.csv` columns leak** (month-long running
   counts spanning train/val/test) — and on real data they do NOT follow a
   `_statistic` naming convention (`show_cnt`, `play_cnt`, `like_cnt`, ...).
   `pipeline/data/leakage_guard.py` drops them by exact name by default.
3. **Every per-impression engagement field is an OUTCOME, not a feature.**
   `play_time_ms` is the known big one (corr 0.64 with long_view), but
   `comment_stay_time` (corr 0.17 on real validation data — staying in the
   comments implies you long-viewed), `profile_stay_time`, and
   `is_profile_enter` are the same causal class: measured during/after the
   impression being predicted. Only ITEM/CONTEXT properties known before
   exposure are legitimate inputs (`duration_ms`, `tab`, ids, `hourmin`,
   `date`). `pipeline/data/features.py:NUMERIC_SIGNAL_COLUMNS` encodes
   this. Do not add any `*_stay_time` / `is_profile_enter` / `play_time_ms`
   column back as an input.
4. **Row order is a submission-correctness invariant.** The raw logs are
   NOT sorted by user_id (verified on real data). `build_features` may
   permute rows (it sorts by user_id for pairwise grouping) —
   `pipeline/submit.py` un-permutes scores back to file order via an
   explicit position column, and `pipeline/train.py:score_dataframe` must
   keep returning one score per input row in input order. Breaking either
   silently produces a misaligned submission that still passes `--check`.
5. **KuaiRand-Pure is the debiasing/multi-task variant**, not the
   sequential-modeling variant (that's 27K/1K) — but per priority item 2
   above, sequence modeling on Pure's own within-split history is still an
   organizer-flagged unexplored lead, not the same thing as "use the
   27K/1K sequential variant."


## Measured on this repo, 2026-08-31 (our own experiments — treat as settled)

These were run against real KuaiRand-Pure and are reproducible. They close off
several directions the organizer listed as "unexplored", so do NOT re-spend
iterations on them without new evidence.

**The task is item-quality estimation, not personalisation.** Within-user GAUC
of single signals, on a 4,000-user validation sample (0.5 = no signal):

| Signal | within-user GAUC |
|---|---|
| Train-derived smoothed video long_view prior | **0.6453** |
| `tab` | 0.5387 |
| Session position (impression order within user-day) | 0.5148 — **usable, now in the pipeline** |
| `duration_ms` | 0.4865 |
| `hourmin` | 0.4867 |
| **user x author affinity** (train-derived) | **0.4981 — none** |
| **user x video affinity** (train-derived) | **0.4970 — none** |

For scale, the whole official FM reaches valid GAUC 0.6674. A single scalar
item-quality prior gets to 0.6453 of that on its own.

**Consequences:**
1. **Behavioural-history / sequence modelling (organizer priority 2) is very
   likely a dead end here.** User-side history has no measurable within-user
   ranking signal, and repeat exposure is only 1.62% of validation rows with
   an identical long_view rate on repeats (0.3072) vs new pairs (0.3134).
   DIN/SIM-style interest modelling needs a personalisation signal that this
   split does not appear to contain.
2. **Loss-function changes did not pay off** (three independent agent
   implementations): pairwise BPR 0.5994, listwise ListNet 0.6004, multi-task
   auxiliary head 0.6013 — all below pointwise logloss at 0.6017.
3. **Not undertrained.** Raising the FM epoch cap 12 -> 40 with patience 4
   early-stops at 13 epochs and returns an identical 0.6017.
4. **LambdaMART (LightGBM `lambdarank`, groups = users) scored 0.5901** on
   engineered prior/context features. It spends its top splits on
   within-user-CONSTANT features (`user_rate`, `user_n`) which cannot affect
   within-user order. If retried, drop every user-constant feature first.
5. **Seed ensembling is a measured win: 5-seed rank-average = 0.6028 vs
   0.6017 single** (+0.0011). Rank-average, not score-average — only order is
   scored. Applied at submission time only, so it does not slow the loop.
6. **Session position is a measured win: adding `pos_bucket` as a 6th
   categorical field moved valid primary 0.6017 -> 0.6024** (GAUC 0.6676 ->
   0.6681, nDCG@5 0.5358 -> 0.5366). It is built in
   `pipeline/data/features.py`. Causal by construction: a cumcount over
   time-ordered rows within a user-day counts only PRECEDING impressions, uses
   no labels, and is known at serving time. This is the general lesson —
   **only signals that vary WITHIN a user can change that user's ranking**, so
   look for context that changes across a user's impressions rather than more
   user attributes.

**Per-user candidate lists are tiny**: median 4 impressions, 63.7% of users
have <= 5. So nDCG@5 is effectively full-list nDCG for most users, and there
is little room for clever re-ordering.

**The random-exposure log cannot be used for training.** 1,186,059 rows, dates
2022-04-22..05-08 — **897,721 of them (75.7%) fall inside the hidden-test
window**, and its long_view rate is 0.0850 versus 0.3133 in the standard log
(uniform exposure shows users mostly irrelevant videos). Training on it means
both test-period contamination and a train/serve distribution mismatch. Its
sanctioned use is the unbiased probe, which is how `agent/referee.py` uses it.
This also explains why the referee's absolute divergence is always large
(~0.19-0.24): the two splits have structurally different label distributions,
so only the CHANGE in divergence across iterations is informative.

## Where to look next

- Need RecSys architecture/method background? -> Tier 2 (`tier2_domain.md`)
- Need a deep dive on a specific method you're about to implement? ->
  Tier 3 (`tier3_deep/`), loaded on demand by `retriever.py` keyed to the
  current ablation target.
