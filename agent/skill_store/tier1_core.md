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
| author AGGREGATE quality (train-derived) | **0.6439 — strong** |
| music_id AGGREGATE quality (train-derived) | **0.6413 — strong** |
| tag aggregate quality | 0.5690 |
| upload_type / video_type aggregate quality | 0.5194 / 0.5023 |
| video freshness (impression date - upload_dt) | 0.5120 |
| **user x video affinity** (train-derived) | **0.4970 — none** |

For scale, the whole official FM reaches valid GAUC 0.6674. A single scalar
item-quality prior gets to 0.6453 of that on its own.

**Do not confuse two different hypotheses about the same field.** "user x
author affinity is dead" (0.4981) does NOT mean "author is useless": author
AGGREGATE quality scores 0.6439, nearly as strong as the video prior. The
first asks whether matching an author to a user helps; the second asks whether
some authors are simply better. Only the first is dead. The same holds for
music_id (0.6413) and tag (0.5690), neither of which is currently in the model
— and note the organizer's flat feature ablation tested music_id as a raw ID
embedding, not as a smoothed target encoding, which is a much stronger
encoding for a high-cardinality field.

**Ranking-invariance trap when designing candidate-relative features.** Within
one user's candidate set, rank(f) is a monotone transform of f, so a model
ranking on rank(age) produces an ORDERING IDENTICAL to one ranking on age. A
candidate-relative version of a single feature adds nothing by itself. Its
value is putting features on a comparable scale ACROSS candidate sets for a
global model, and enabling interactions — so encode several features'
ranks/z-scores jointly, never one alone. The same trap applies to calibration:
a global monotone recalibration cannot change GAUC or nDCG at all; only
group-wise calibration can, and only for users whose candidates span groups.

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

## Measured 2026-08-31 (second pass — dataset STRUCTURE, treat as settled)

**Check the granularity of a field before believing its aggregate score.**
`video_features_basic_pure.csv` has 7,583 videos and:

| field | distinct | videos per entity |
|---|---|---|
| author_id | 6,510 | 87% own exactly ONE video |
| music_id | 7,202 | 98% own exactly ONE video |
| tag | 110 (multi-valued) | median 10 — a REAL pooling level |
| upload_type | 14 | real, weak |
| video_type / music_type / visible_status | 3 / 5 / 1 | dead |

This CORRECTS the entry above. Author and music aggregate quality really do
score ~0.64, but they are the **video identity in disguise**: target-encoded on
train and scored on val, `corr(video_te, author_te) = 0.985` and
`corr(video_te, music_te) = 0.987`. They add nothing over the `video_id`
embedding the model already has. Only `tag1` is independent —
`corr(video_te, tag1_te) = 0.458` at GAUC 0.5604. Adding `tag1` + `upload_type`
as encoded fields moved valid primary 0.6045 -> 0.6048, i.e. **neutral**
(seed std is 0.0004). Kept, but it is not a win.

**`upload_dt` has THREE distinct values** (2022-04-09/10/11): every video in
KuaiRand-Pure was uploaded inside a 3-day window. There is no video lifecycle
here, so the whole temporal-dynamics family has no substrate — freshness/decay
curves, momentum, rolling/EWMA statistics, HAR, GARCH, Hawkes/self-exciting
intensity, Kalman/state-space, change-point detection, ARIMA-family forecasting.
Do not spend iterations on any of them. The earlier "video freshness 0.5120"
entry is really a 3-level categorical, not an age.

**`tab` is the master context variable.** Its long_view base rate runs from
0.004 (tab 3) to 0.489 (tab 4) across 15 tabs, and every item feature gains
sharply when crossed with it (smoothed target encoding, fit on train, scored on
val): video_id 0.6387 -> 0.6479, tag1 0.5604 -> 0.6153, upload_type 0.5214 ->
0.5938. `video_id x user_id` is 0.4973 — noise, re-confirming no personalisation.

**But a cross's standalone GAUC overstates its INCREMENTAL value.** Feeding
`video x tab` to the model as a 5-fold out-of-fold smoothed target encoding,
quantile-bucketed into 32 bins, scored **0.6047 vs 0.6045 — flat**. The FM's
second-order term already approximates `<e_video, e_tab>`. General lesson:
before building a cross feature, check whether the model already contains both
parent fields; if it does, the standalone number is not the expected gain.
(The machinery is in `pipeline/train.py` — `CROSS_TE_SPECS`, `_fit_cross_te` —
and is correct and reusable; it is the hypothesis that failed, not the code.)

**Oracle ceilings, priors fit ON VALIDATION (diagnosis only, never submitted):**

| oracle | primary |
|---|---|
| video quality | 0.6146 |
| video x tab | 0.6351 |
| video x tab x date | 0.6913 |
| user x video (memorises the label) | 0.8477 |
| **our trained model** | **0.6047** |

We are at **98% of the pure item-quality ceiling**. That is the single most
important number for planning: gains from better item-quality estimation are
capped at roughly +0.010, and the split's remaining headroom lives in
user x video, which we have measured three times as noise (0.4970 / 0.4973 /
0.4981). Treat large remaining gaps to the 0.8645 oracle as unreachable.

**`user_id` is NOT a pure overfitting engine — hypothesis tested and killed.**
Reasoning from our own numbers (user x video affinity 0.4970, user x author
0.4981, user x video cross 0.4973 — all noise), the 22k x 16 user embedding
looked like 350k parameters fitting nothing, and its LINEAR term provably
cannot change a within-user ranking. Deleting the field costs **-0.0091**
(0.5955 vs 0.6046, 3 seeds each, std 0.0004 / 0.0001). Keep `user_id`.

The mechanism is worth understanding, because it is the exception to
"personalisation is dead here". Two-way `user x context` target encodings do
NOT explain the gain — every one of them scores BELOW the context alone
(user x tab 0.5545 vs tab 0.5789; user x dur_bucket 0.5106 vs 0.5323;
user x tag1 0.5229; user x pos 0.5035), because splitting by user just makes
the cell sparse. So the user embedding is not acting as an affinity term at
all: it enters the MLP branch as a per-user offset that RESHAPES the function
applied to the item and context features — group-wise modulation, which is
exactly the one channel that can change GAUC (see the ranking-invariance trap
above; a per-user constant added to the LOGIT cannot, but a per-user input to a
nonlinearity can). Caveat on the size of the effect: dropping a field also
shrinks the MLP input (9 x 16 -> 8 x 16) and removes parameters, so part of the
-0.0091 is capacity rather than user signal. The sign is not in doubt.

**User METADATA adds nothing — the `user_features_pure.csv` question is closed.**
The file (27,285 users, 31 columns) was genuinely unused by the pipeline, and
the argument for it is sound: these fields are constant within a user so they
cannot rank rows alone, but through the FM's second-order term they could give
a prior SHARED across users that a sparse per-user embedding cannot learn for
rare users. Tested properly, it is empty:

| arm | encoded fields | 3-seed primary |
|---|---|---|
| control | 8 | **0.6047** (std 0.0001) |
| + 7 core user fields | 15 | 0.6045 (std 0.0005) |
| + all 22 user fields | 30 | 0.6048 (std 0.0001) |

`is_lowactive_period` has ONE distinct value. The raw counts duplicate the
dataset's own `*_range` buckets. The loader stays in `features.py` but is NOT
registered by default.

**The CONDITIONAL probe — use this before building any feature.** Marginal
within-user GAUC answers the wrong question; what matters is signal remaining
AFTER controlling for video quality. Method: bucket the train-fit video prior
into 20 quantile bins (`qb`), target-encode `qb x C` on train, score on val,
compare to `qb` alone (0.6383). It costs seconds instead of a training run:

| conditional on video quality | delta |
|---|---|
| **tab** | **+0.0172** |
| session position | +0.0011 |
| hour / duration / tag1 / upload_type | +0.0002 .. -0.0003 |
| all 7 user-metadata fields | -0.0004 .. +0.0001 |
| `qb x tab x pos` | +0.0011 vs `qb x tab` |
| `qb x tab x` hour / duration / tag1 / user_active_degree | **-0.0008 .. -0.0032** |

`tab` is the ONLY context carrying real conditional signal. Note the power
asymmetry when reading nulls: the user-metadata null is well powered
(user_active_degree gives 9 x 20 = 180 cells over 1.1M rows), but the 3-way
nulls are weaker, because a cell-count encoding cannot represent what a learned
low-rank cross can. Do not quote the 3-way rows as proof against DCN-V2.

**Candidate-relative features are empty, exactly as the invariance trap
predicts.** Within-set percentile of the video prior scores 0.6382 alone vs
0.6387 for the raw prior — ranking-identical. Conditional versions are all
negative: `qb x` percentile -0.0181, deviation-from-set-mean -0.0061,
set size -0.0049, gap-to-best -0.0005, set spread -0.0004.

**Multi-hot `tag` is not better than primary-tag.** Mean-pooling per-tag target
encodings over each video's full tag set scores 0.5607 marginal, against 0.5604
for `tag1` alone and 0.5631 for the full tag string. Conditional on video
quality all are +0.0001..+0.0003, and `qb x tab x pooled-tag` (0.6534) is BELOW
`qb x tab` (0.6554). There is no combinatorial tag structure to recover, so
attention pooling has nothing to find that mean pooling missed.

**Embedding width: smaller is better, not larger.** 2-seed sweep at
weight_decay 1e-6: k=8 **0.6050**, k=16 0.6046, k=32 0.6046; weight decay
1e-5 0.6047 and 1e-4 0.6045 at k=16. The model is over-parameterised, not
rank-limited. (Full 3-seed k in {4,8,16,32,64,128} curve — see results table.)

**The `from ... import <list>` trap — this repo's most expensive recurring bug.**
`pipeline/train.py` does `from pipeline.data.features import
EXTRA_CATEGORICAL_FIELDS`, which binds the LIST OBJECT into train.py's
namespace at import time. Two consequences, both of which have already cost
real experiments:

1. Rebinding the name in `features.py` (`features.EXTRA_CATEGORICAL_FIELDS =
   [...]`, or an agent patch that reassigns the literal) is INVISIBLE to
   `resolve_fields`, which reads train.py's own global. An A/B test written
   that way silently measures the same configuration twice — the giveaway is
   scores identical to 4+ decimals SEED BY SEED, which is not what a real tie
   looks like. This is the same root cause as the earlier "two feature patches
   both scored exactly 0.6024" incident.
2. Therefore: to change the encoded field list at runtime, patch
   `pipeline.train.EXTRA_CATEGORICAL_FIELDS`. To change it in code, MUTATE the
   list (`.append(...)`) or edit the literal in features.py — and always print
   `id_maps["fields"]` to confirm the field actually reached the encoder.

**Verify a leakage guard, do not trust it.** The out-of-fold target encoding in
`_fit_cross_te` was checked directly on train rows: `corr(encoding, label)` is
0.3817 in-fold vs 0.3609 out-of-fold overall, but on cells with <= 3 rows it is
**0.7531 in-fold vs 0.1748 OOF** — in-fold, the encoding of a small cell very
nearly IS that row's label. Any future target encoding must be checked the same
way, on the small-cell subset, where the leak actually lives.

**Exposure counts are weak, and the transductive version is WORSE.** Video
impression count in train 0.5396, the same count computed on the evaluation
split itself 0.5189, against the train quality prior at 0.6387. There is
therefore no reason to go near transductive eval-set statistics: the causal
feature strictly dominates the gray-area one.

**Temporal drift is not exploitable.** The `x date` oracle above looks like
drift but is the oracle memorising validation labels in tiny cells.
Exponentially recency-weighting the video prior over the 13 train days is flat
(half-life 14d 0.6389, 7d 0.6389, uniform 0.6387) and a 2-day half-life HURTS
(0.6342). Sliding-window / exponentially-decayed training data is dead here.

## Where to look next

- Need RecSys architecture/method background? -> Tier 2 (`tier2_domain.md`)
- Need a deep dive on a specific method you're about to implement? ->
  Tier 3 (`tier3_deep/`), loaded on demand by `retriever.py` keyed to the
  current ablation target.
