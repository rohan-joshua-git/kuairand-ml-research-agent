# Research process — KuaiRand-Pure within-user ranking

A complete record of what was considered, what was dropped, what was measured,
and the reasoning at each step. Written so a reader can check the logic, not
just the conclusions. Negative results are reported at the same weight as
positive ones — there are far more of the former, and they are the substance of
this project.

**Status vocabulary** used throughout (a plain tested/untested flag loses the
distinction that matters most here):

| status | meaning |
|---|---|
| `SUPPORTED` | measured improvement beyond seed noise |
| `WEAK` | nominal improvement, inside noise |
| `NULL` | tested, no effect |
| `REJECTED_STRUCTURALLY` | the dataset cannot express the hypothesis; not a verdict on the method |
| `NOT_IDENTIFIABLE` | confounded — cannot be measured cleanly with this design |
| `NOT_APPLICABLE` | requires data or an environment this benchmark does not have |

---

## 1. The benchmark

- Task: within-user ranking over logged impressions. Each user is ranked only
  within their own rows.
- Label: `long_view`. Metrics: GAUC and nDCG@5; primary = mean.
- Splits: train 2022-04-08..04-21 (1,141,112 rows), valid 04-22..04-28
  (124,909 rows, 22,377 users), test 04-29..05-08 (hidden).
- Official FM baseline: valid primary **0.6016**. Published oracle ceiling:
  test primary 0.8645.
- Our accepted pipeline: valid primary **0.6047** (3 seeds, std 0.0001).
  Submitted artifact scored 0.6044, i.e. **+0.0028** over baseline.

Seed standard deviation is **0.0001–0.0006**. Nothing below ~0.001 is a result.
This single fact invalidated several apparently-positive readings below, and it
is the reason every claim here is quoted at 3 seeds.

---

## 2. Method: two probes we built, and why

Most of the efficiency in this project came from replacing "implement it and
train" with two cheap tests.

### 2.1 The conditional probe

Marginal within-user GAUC answers the wrong question. What matters is signal
remaining **after controlling for video quality**, because item quality
dominates everything here.

> Bucket the train-fit video-quality prior into 20 quantile bins (`qb`).
> Target-encode `qb × C` on train, score on validation, compare against `qb`
> alone (0.6383).

Cost: seconds. It killed several feature families that would each have cost a
training run, and it caught the case where a marginal number was strong and the
conditional number was zero.

### 2.2 The residual probe

> Train the model. Compute its residual on **train**. Fit a smoothed encoding of
> that residual on feature X. Add `alpha * enc_X` to the **validation** logit,
> sweep alpha, keep the best.

This is a direct test of "does X explain what the model gets wrong", rather than
an ablation of "does removing X hurt". Its limitation is important and is
recorded in section 6.3.

### 2.3 Leakage verification, not leakage assumption

Every target encoding is fit out-of-fold (5 folds, not leave-one-out — LOO is
invertible on small cells). We verified the guard rather than trusting it:
on `(video, tab)` cells with 3 or fewer training rows, `corr(encoding, label)`
is **0.7531 in-fold vs 0.1748 out-of-fold**. In-fold, a small cell's encoding
very nearly *is* that row's label. Any future encoding must be checked on the
small-cell subset, where the leak actually lives.

---

## 3. Structural audit — done before any modelling

Four facts about the data closed more hypotheses than any experiment.

| field | distinct | consequence |
|---|---|---|
| `upload_dt` | **3** (Apr 9/10/11) | no video lifecycle exists |
| `author_id` | 6,510 for 7,583 videos | 87% own exactly one video |
| `music_id` | 7,202 for 7,583 videos | 98% own exactly one video |
| `tag` | 110, multi-valued | median 10 videos — the one real pooling level |
| `video_type` / `music_type` / `visible_status` | 3 / 5 / 1 | dead |
| `is_lowactive_period` | 1 | dead |

**Author and music are the video prior in disguise.** Target-encoded,
`corr(video_te, author_te) = 0.985` and `corr(video_te, music_te) = 0.987`.
Their strong marginal scores (0.6367 / 0.6365) carry no information the
`video_id` embedding lacks. Only `tag1` is independent (corr 0.458).

This corrected an earlier recommendation of our own that had ranked
author/music target encoding as the single highest-value experiment. The error
was reasoning from a marginal score without checking entity granularity.

**`user_id` target-encoded scores exactly 0.5000.** Anything constant within a
user cannot change that user's ranking. This is a structural constraint, not an
empirical finding, and it disqualifies every purely user-side feature.

---

## 4. What we dropped, and on what grounds

`REJECTED_STRUCTURALLY` — the data cannot express these. This is **not** a claim
that the methods are bad. A Hawkes process is a fine model; it is simply not
estimable when the relevant temporal dimension has three values.

| family | ground |
|---|---|
| Freshness/decay curves, momentum, rolling and EWMA statistics, volatility, HAR, GARCH-family, Hawkes/self-exciting, Kalman/state-space, change-point, ARIMA-family, HMM/regime-switching, extreme-value theory, reservoir computing | `upload_dt` has 3 values |
| Hierarchical pooling `global -> tag -> author -> video`, author/music latent factors | author and music are video identity |
| Sequence models (DIN/DIEN/SASRec/BERT4Rec), multi-interest (MIND/ComiRec), graph CF (LightGCN/NGCF/SGL), user-item metric learning | `user x video` = 0.4973, measured three times |
| User-side features, user clustering, CLV/RFM, IRT user ability | within-user-constant |
| NLP, CV, multimodal, semantic IDs | no text or frames in KuaiRand-Pure |
| Bandits, RL, submodular feed selection | no online environment; candidate sets are fixed |
| IPS / causal deconfounding | the random-exposure log is 75.7% inside the hidden-test window |

`NOT_APPLICABLE`: repeat-view modelling — repeat exposure is 1.62% of validation
rows with an identical long_view rate (0.3072 repeat vs 0.3134 new).

---

## 5. Experiment ledger

### 5.1 Features

| hypothesis | result | status |
|---|---|---|
| `tag1` + `upload_type` as encoded fields | 0.6047 vs 0.6046 | `WEAK` |
| `video x tab` out-of-fold smoothed target encoding, 32 buckets | 0.6046 vs 0.6046 | `NULL` |
| User metadata, 7 core fields | 0.6045, variance tripled | `NULL` |
| User metadata, all 22 fields | 0.6048 vs 0.6047 | `NULL` |
| Multi-hot tag (mean-pooled per-tag encodings) | 0.5607 marginal vs 0.5604 for `tag1` | `NULL` |
| Candidate-relative features | percentile alone 0.6382 vs raw prior 0.6387 | `NULL` |
| Transductive eval-set exposure counts | 0.5189 vs causal 0.5396 | `NULL` |
| Recency-weighted training data | 0.6389 vs 0.6387 uniform; 2-day half-life 0.6342 | `NULL` |

**Why the `video x tab` null is the most instructive.** Its marginal score is
excellent — `video x tab` = 0.6479 against `video` alone at 0.6387 — and `tab`
crossed with anything gains sharply (`tag1` 0.5604 to 0.6153, `upload_type`
0.5214 to 0.5938). It still added nothing, because **the FM's second-order term
already approximates the video-tab interaction**. The lesson generalises: a
cross's standalone GAUC overstates its incremental value whenever the model
already contains both parent fields.

**Why the user-metadata null is trustworthy.** `user_active_degree` gives
9 x 20 = 180 conditional cells over 1.1M rows. That is a well-powered null, not
a sparsity artifact. Conditional deltas for all seven core fields fall in
-0.0004..+0.0001, and the model-level test agrees.

**Why candidate-relative features cannot work.** Within one user's candidate
set, `rank(f)` is a monotone transform of `f`, so ranking on the percentile is
*identical* to ranking on the raw value. Measured: 0.6382 vs 0.6387. Conditional
versions are all negative (`qb x` percentile -0.0181, deviation-from-mean
-0.0061, set size -0.0049). The same argument shows a global monotone
recalibration cannot change GAUC or nDCG at all; only group-wise can.

### 5.2 Capacity and regularisation

Embedding width, 3 seeds each:

| k | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| primary | 0.6049 | 0.6047 | 0.6047 | 0.6041 | 0.6041 | 0.6037 |

Monotonically **declining**. The hypothesis that k=16 imposes a limiting rank on
the interaction matrix is refuted, not merely unsupported. Weight decay: 1e-6
(the existing default) is at least as good as 1e-5, which beats 1e-4.

#### 5.2.1 Field-specific weight decay on `user_id` — tested, negative

The strongest remaining regularisation hypothesis, and the first experiment run
under the selection/confirmation protocol (section 2.4). Motivation was three
converging measurements: user x video affinity is chance yet deleting `user_id`
costs -0.0091; every two-way `user x context` encoding scores below the context
alone; and global capacity reduction helps. Global weight decay applies the same
shrinkage to a video embedding backed by ~150 impressions and a user embedding
backed by ~50, which is almost certainly wrong.

Implemented as an explicit L2 on the `user_id` rows of the shared embedding
table, in Adam's own units (penalty `0.5 * wd * ||W_user||^2`, gradient
`wd * W_user`). Only the k-dim rows are penalised, not the first-order `linear`
rows — a per-user constant added to the logit provably cannot change a
within-user ranking. 3 seeds per arm, early-stopped on the selection half:

| user_wd | 0 | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|---|---|---|
| selection primary | 0.6078 | 0.6078 | 0.6077 | 0.6070 | 0.6052 | 0.6021 |
| vs control | — | -0.0000 | -0.0001 | -0.0008 | -0.0026 | -0.0058 |

Std <= 0.0004 throughout. **Monotonically harmful across five orders of
magnitude.** The confirmation half was not looked at, because nothing beat the
control on selection — the look is preserved.

This is a directional result, not a noisy null. The prediction was that shrinking
a noise-fitting embedding should help; it does the reverse, monotonically. So the
global-`k` anomaly does **not** transfer to user-specific regularisation, and
section 6.4's conclusion sharpens: whatever `user_id` contributes is not
excess capacity that regularisation can trim. What it *is* remains the open
question — see 5.2.2.

#### 5.2.2 What `user_id` actually contributes — identity, not capacity

Section 6.4 left the mechanism explicitly `NOT_IDENTIFIABLE`: deleting `user_id`
costs -0.0091 while every affinity measurement reads chance, and two explanations
(per-user function reshaping versus simple parameter count) were not separated.
They are now separated, by a three-arm control at 3 seeds:

| arm | full primary | vs real |
|---|---|---|
| real | 0.6046 | — |
| shuffled | 0.5953 | **-0.0093** |
| removed | 0.5960 | -0.0086 |

`shuffled` permutes the user codes ACROSS ROWS in every split. It holds parameter
count, MLP input width and the exact code-frequency distribution fixed, and
destroys only the correspondence between a row and its true user. Design note:
the permutation must be row-level. A *bijective* remap of user -> embedding row
is a symmetry of the model and trains to a numerically identical result, so it
would have produced a spurious null.

Attribution: identity **+0.0093 (108%)**, capacity **-0.0007 (-8%)**. Random user
embeddings are marginally *worse* than no user embedding, which is what injecting
noise into the MLP input should do. **The capacity explanation is refuted.**

This also resolves the caveat recorded in section 6.4 and in tier1_core, that
"dropping a field also shrinks the MLP input, so part of the -0.0091 is capacity
rather than user signal". `shuffled` holds the input width fixed and still loses
the whole effect, so the capacity share is -0.0007 — negligible and, if anything,
negative.

**Is it memorisation?** Stratified by training impressions per user, against
`removed` as the clean reference:

| train impressions | users | vs removed | noise band |
|---|---|---|---|
| 1 | 708 | +0.0062 | +/-0.0045 |
| 2-5 | 1,218 | +0.0076 | +/-0.0034 |
| 6-15 | 3,519 | +0.0048 | +/-0.0020 |
| 16-40 | 7,158 | +0.0080 | +/-0.0014 |
| 41+ | 9,774 | +0.0102 | +/-0.0012 |

Partly. The gain does rise toward the frequent users (+0.0102 at 41+, the most
precisely measured bin), which is memorisation-like. But it is not monotone —
the 6-15 bin dips to +0.0048 — and, more importantly, there is a substantial
**floor at a single training impression** (+0.0062, above its own noise band).
Pure per-user rate memorisation cannot produce that: one observation does not
estimate a rate. So the honest reading is a mixture, with a transferable
component that survives at minimal per-user data.

Method note: the first run of this stratification used `shuffled` as the
reference and reported +0.0097 in the 1-impression bin. That was confounded —
shuffling preserves the frequency distribution while reassigning rows, so a rare
user can receive a heavily-trained embedding belonging to a frequent user, which
is misleading input rather than absent input and inflates exactly the
low-frequency bins the conclusion rests on. Re-run against `removed`, those bins
fell 30-40%. The corrected numbers are the ones above.

#### 5.2.3 Temporal distribution shift — looked for, not present

The temporal MODEL family is dead because `upload_dt` has three distinct values
(section 4). Temporal *distribution shift* is a separate question and had never
been tested on the absolute curve — only on the mlp-only delta in section 9.

The naive version of this diagnostic is a trap. Model primary really does fall
across the validation week, 0.5464 on 4/22 to 0.5330 on 4/28, which reads as
drift. It is not. Every day was also scored with a model-free reference: a
smoothed train-fit video-quality prior, frozen from train and therefore
incapable of drifting. It falls in lockstep.

| block | dates | rows | model | prior ref | gap |
|---|---|---|---|---|---|
| early | 04-22..04-24 | 67,168 | 0.5833 | 0.5635 | +0.0198 |
| middle | 04-25..04-26 | 29,441 | 0.5493 | 0.5304 | +0.0189 |
| late | 04-27..04-28 | 28,300 | 0.5504 | 0.5298 | **+0.0206** |

Gap change early -> late: **+0.0008**, i.e. flat to slightly widening. The
declining level is day difficulty; the model's learned advantage does not decay.
**Recency weighting stays closed** — it was already null (half-life 14d/7d both
0.6389 vs uniform 0.6387, 2d hurts at 0.6342), and there is now no drift for it
to correct.

Reading note: per-day primaries (~0.53) are far below the full-week 0.6047
because slicing by day leaves most users 1-2 impressions, so GAUC excludes them
and nDCG@5 degenerates. The levels are not comparable across slicings; only the
model-vs-reference gap within a slice is.

### 5.3 Architecture and ensembling

| model | single | 3-seed rank-avg |
|---|---|---|
| DeepFM-lite k=16 (control) | 0.6047 | 0.6048 |
| DeepFM-lite k=8 | 0.6047 | 0.6047 |
| DCN-V2, 3 cross layers | 0.6047 | 0.6049 |
| DCN-V2, 2 cross layers | 0.6050 | 0.6049 |
| DCN-V2, 3 layers, rank 32 | 0.6048 | 0.6052 |

All pairwise **rank correlations are 0.998 or higher** (DCN3 vs DCNlr = 0.9996).
Ensembles of these families reach 0.6048–0.6049 against 0.6047 single. When five
architectures compute the same function to three decimal places, ensembling has
nothing to exploit. `NULL` for both DCN-V2 and homogeneous ensembling.

### 5.4 Objectives (earlier sessions)

BPR 0.5994, ListNet 0.6004, multi-task auxiliary head 0.6013, LambdaMART
0.5901 — all below pointwise logloss at 0.6017. `NULL`.

The LambdaMART figure was later diagnosed: it spent its top splits on
within-user-constant features. Re-run as a plain GBDT with those dropped and an
out-of-fold quality prior, it reaches 0.5961 (section 6.2) — so the original
number measured a setup error, not the method.

---

## 6. Where the signal actually is

### 6.1 `tab` is a reordering signal that exists for 40% of users

`tab` is the only context with substantial conditional signal (+0.0172 over the
quality bucket). Decomposed by whether a user's impressions span tabs:

| segment | rows | `qb` | `qb x tab` | delta | model |
|---|---|---|---|---|---|
| single-tab users | 53,339 | 0.6007 | 0.6008 | **+0.0001** | 0.6197 |
| multi-tab users | 71,570 | 0.6695 | 0.7008 | **+0.0314** | 0.7147 |

For the 60% of users inside one tab it contributes **nothing** — the
within-user-constant rule, now confirmed empirically rather than argued. Within
a tab, long_view rate is almost perfectly monotone in quality bucket
(corr +0.989 / +0.993 / +0.990 for tabs 1/4/2), so `tab` never re-ranks *inside*
itself. Its whole effect is the level gap **between** tabs: tab 3 runs 0.004,
tab 0 runs 0.042, tab 1 runs 0.386, tab 4 runs 0.489.

### 6.2 The nested ceiling — and the claim it refutes

| model | GAUC | nDCG@5 | primary | increment |
|---|---|---|---|---|
| M0 video quality prior only | 0.6387 | 0.5227 | 0.5807 | — |
| M1 + tab | 0.6479 | 0.5275 | 0.5877 | +0.0070 |
| M2 + all clean features (LightGBM) | 0.6593 | 0.5329 | 0.5961 | +0.0084 |
| M3 neural model | 0.6716 | 0.5378 | 0.6048 | +0.0087 |

The increments are **roughly equal thirds**. This refutes the tempting summary —
which we had been drifting toward — that *ranking is approximately f(video
quality, tab) plus noise*. Quality + tab reaches 0.5877; the model reaches
0.6048, and that 0.0171 gap is over five times our entire margin over the
official baseline.

The defensible statement is narrower: **item quality is the largest single
factor, `tab` is the only other context with substantial conditional signal, and
a comparable third is captured only by learned structure that no named feature
reproduces.**

### 6.3 The residual is structureless with respect to named features

Against a 0.6716 baseline, best alpha for each:

| residual ~ X | delta |
|---|---|
| `tab` / `pos_bucket` / `tag1` / `user_id` | +0.0000 |
| `upload_type` | +0.0002 |
| `author_id` / `video_id` / duration | +0.0001 |
| `qb x tab` | -0.0001 |
| `qb x tab x pos_bucket` | -0.0002 |

**This does not mean the model is at its ceiling.** The probe can only test
features we can *name* and encode as cell counts. Section 6.2 shows the model
beats a GBDT over those same features by +0.0087, so it is exploiting something
the probe is structurally unable to represent. A null residual probe is evidence
about named features only.

### 6.4 The GBDT comparison was confounded — and fixing it points at `user_id`

M2's GBDT saw the video as one smoothed scalar; the neural model has a free
embedding across 7,583 videos. Giving the GBDT the same identity information:

| model | primary |
|---|---|
| M2 scalar quality only | 0.5959 |
| M2a + `video_id` | 0.5931 |
| M2b + `video_id` + `author_id` | 0.5923 |
| M2c + `video_id` + `author_id` + `user_id` | **0.6017** |
| neural DeepFM | 0.6048 |

That closes 65% of the gap (+0.0089 down to +0.0031). Note the shape: video and
author identity made the GBDT **worse**; the entire gain came from **`user_id`**.

This converges with an independent result. Deleting `user_id` from the neural
model costs **-0.0091**, despite `user x video` affinity measuring 0.4973 and
every two-way `user x context` encoding scoring *below* the context alone
(`user x tab` 0.5545 vs `tab` 0.5789). So user identity matters through
something that is neither affinity nor a two-way interaction — most plausibly
group-wise modulation of the item/context function, which is the one channel
that can change a within-user ranking when a per-user constant cannot.

`NOT_IDENTIFIABLE` so far: the precise mechanism. Two candidate explanations
(per-user function reshaping versus simple parameter count) are not yet
separated.

---

### 6.5 The label-oracle ceiling is not an attainable model ceiling

Terminology first, because the distinction is the whole point. 0.8645 is the
organizer's published **label-oracle ceiling**: the score obtained by ranking
with the true labels. It is a real and correctly computed ceiling for the
metric. What it is NOT is an attainable ceiling for any *probabilistic model*,
and conflating the two invites a natural but wrong reading: we sit at ~0.605,
the ceiling is 0.865, therefore a quarter of the metric is sitting unclaimed.
That inference is testable rather than arguable.

Method. Calibrate the champion's averaged probabilities with isotonic
regression, giving `q` = P(long_view | features) (calibration check: mean
predicted 0.3133 vs actual rate 0.3133). Then SIMULATE a world in which the
model is exactly right: draw `y ~ Bernoulli(q)`. In that world there is no
learnable structure left by construction. Score both the model and a
label-revealing oracle against those simulated labels.

| | primary |
|---|---|
| perfect-probability model vs simulated labels | 0.5915 |
| label ORACLE vs simulated labels | 0.8471 |
| **irreducible gap from pure coin-flip noise** | **0.2556** |
| | |
| our model vs real labels | 0.6053* |
| oracle vs real labels | 0.8484 |
| **observed gap** | **0.2431** |

\* Run against the superseded full-validation-early-stop ensemble (see `superseded_artifact_sha256` in `FROZEN_CONFIG.json`). The shipped artifact scores 0.6049. The 0.0004 difference does not move the 0.2431-vs-0.2556 comparison this section rests on, so the analysis was not re-run.

**The observed gap is smaller than the gap a perfect model faces in a world
with nothing left to learn.** The oracle sees each impression's realised label;
a long_view is a coin flip, and no probability model can order a 0.31-chance
item that came up 1 above a 0.30-chance item that came up 0. That single fact
accounts for the entire 0.2556.

What this does and does not establish. It does NOT prove no signal remains — if
the model is missing structure then `q` is not the true probability and the
simulation is optimistic about how little is left. Nor does it impugn the
organizer's figure, which is exactly what it claims to be. What it establishes
is narrower and still useful: **the label-oracle ceiling is not an attainable
probabilistic-model ceiling, so the size of the gap to it is not evidence for
remaining headroom.** Anyone arguing from "0.8645 minus 0.605 is huge" has to
make the case some other way.

Scale note: 0.4753 / 0.5946 / 0.8645 are TEST-set figures; 0.6053 (above) is a
VALIDATION figure. Mixing them inflates the apparent progress. On the
validation scale (random 0.4841, oracle 0.8484, both measured here) the
baseline captures 32.3% of the attainable range and this model 33.3% — about
one point, not the ~2.7 that mixing scales suggests.

---

### 6.6 User feature-sensitivities: real, decaying, and harmful to expose

The last open question from 5.2.2. `user_id` carries identity information that
is not affinity and not a two-way interaction. The untested candidate was
SENSITIVITY: users differ in how strongly their outcome responds to item and
context features. This is the one class of per-user quantity that can move a
within-user ranking, since a per-user constant provably cannot (user historical
long_view rate scores GAUC exactly 0.500000) while a per-user slope multiplies a
feature that varies across the candidates.

Four sensitivities were built from train only: response to video quality, to
session position, to duration, and spread across tabs. Each is an
empirical-Bayes shrunk covariance, quantile-banded into 8 levels, entering the
shared embedding table as an ordinary categorical field.

**Run 1 was invalid, and the controls are what revealed it.** The video-quality
regressor was out-of-fold but the per-user statistic computed from it was not:
for a training row of user u, the band was built using that row's own label.
Arms: real -0.0048 (CI excluding zero), permuted +0.0003, randomized -0.0002.
Identical fields carrying no user information were free, while the label-derived
values did real damage — the signature of a leak rather than of a failed
hypothesis. With only a control and a treatment arm this would have read as
"sensitivities are harmful". Trap #2 in tier1_core, applied one level too
shallow. Fixed by estimating each training row's band from that user's rows in
the other folds; the fix recovered 75% of the drop.

**Run 2, leak-free.** Still negative:

| arm | selection primary | vs control |
|---|---|---|
| control | 0.6078 | — |
| real | 0.6066 | **-0.0012** (CI [-0.00319, -0.00044]) |
| permuted | 0.6075 | -0.0003 (CI includes 0) |
| randomized | 0.6077 | -0.0001 (CI includes 0) |

Both null controls are exactly neutral, so four extra categorical fields of this
cardinality cost nothing. Only the real values hurt.

**The estimator is not noise.** Split-half reliability, same users, disjoint
halves: quality 0.580, tab 0.598, position 0.444, duration 0.310 (Spearman-Brown
0.47-0.75). Median 31 train impressions per user is enough. So the model is
being handed a genuine, reproducible user property and is made worse by it.

**The property does decay over time.** Matched-size random split versus temporal
split of train, identical estimator and sample size, differing only in temporal
separation:

| field | random r | temporal r | retention |
|---|---|---|---|
| sens_quality | 0.3633 | 0.2501 | 68.8% |
| sens_pos | 0.2641 | 0.1021 | 38.7% |
| sens_dur | 0.1610 | 0.1251 | 77.7% |
| sens_tab | 0.4493 | 0.2904 | 64.6% |

The first version of this comparison was confounded — the train window is
front-loaded, so the late half had 5x fewer rows and a noisier estimate.
Subsampling early to match (190,802 rows each) is what the table above reports.

**Run 3 tested whether that decay is the mechanism. It is not.** Estimating the
sensitivity only from the late window (adjacent to validation) versus only from
the early window, sample sizes matched:

| arm | selection primary | vs control |
|---|---|---|
| control | 0.6078 | — |
| early only | 0.6071 | -0.0007 (CI includes 0) |
| late only | 0.6073 | -0.0005 (CI includes 0) |
| full train | 0.6066 | -0.0012 (CI excludes 0) |

late minus early = **+0.00017**, CI [-0.00102, +0.00148]. Recency is irrelevant.

**The unpredicted result is the ordering.** The full-train arm, built from SIX
TIMES more data, is worse than either subset, and is the only arm whose CI
excludes zero. More precise measurement produces more harm. That inverts what a
useful-but-noisy feature would do.

Interpretation, labelled as inference rather than measurement: harm appears to
scale with how much the model trusts the band. The user embedding already learns
this modulation from the same training data; an explicit precomputed summary
adds a redundant, coarser pathway competing for capacity, and the sharper that
summary, the more the model leans on the pathway that generalises worse.

**Disposition: CLOSED.** What is established is narrow and worth stating
exactly. Per-user feature sensitivities are real, reliably measurable, and decay
across roughly a week. Exposing them explicitly to this model is harmful
regardless of recency, and increasingly so with better estimation. This does NOT
establish that no user representation could help — it establishes that this one,
at this quantisation, does not, and that the failure is not noise, not
cardinality, and not staleness.

---

### 6.7 Conditional blending: the disagreement is not exploitable

DeepFM and GBDT genuinely disagree, unlike the five neural families at >=0.998
rank correlation, and a GLOBAL blend of them gave only +0.0002. The untested
question was whether the disagreement is PREDICTABLE — does one model
systematically win for identifiable users or contexts, so that a gate could
route between them?

This was answered with a diagnostic BEFORE building any gate, because a gate
cannot exploit heterogeneity that is not there.

Setup. A LightGBM on item/context features only (within-user-constant features
excluded: they cannot reorder a user's candidates, and the earlier LambdaMART
attempt wasted its top splits on exactly those). Selection-half primary: DeepFM
0.6083, GBDT 0.5959, rank correlation **0.7426** — i.e. MORE disagreement than
the 0.9427 in the record, so this ran with more raw material than the original
finding had.

**The global blend sweep picks w = 1.0, pure DeepFM.** The GBDT adds nothing at
any weight.

Oracle ceilings, using validation labels for diagnosis only:

| | primary | headroom over best global blend |
|---|---|---|
| best global blend (w=1.0) | 0.6083 | — |
| per-TAB oracle (15 groups) | 0.6085 | +0.0002 |
| per-USER oracle (22,377 groups) | 0.6397 | +0.0314 |

The per-user number is a trap, and the two controls say so.

**Control 1, equal quality.** A per-user oracle between two SEEDS of the same
DeepFM — statistically identical models with nothing to gate on — yields
+0.0146 of "headroom" across four disjoint pairs. Selecting the better of two
noisy per-user estimates on a median of 4 impressions is winner's curse.

**Control 2, matched on the quality gap.** DeepFM and the GBDT differ by 0.0124,
so control 1 is not matched. Building a DEGRADED DeepFM (DeepFM plus calibrated
noise, tuned to the GBDT's exact score level, containing no information DeepFM
lacks) and running the same per-user oracle gives **+0.0291**. Against the real
GBDT the figure is +0.0314. The excess attributable to the GBDT being a
genuinely different model is therefore **+0.0023** — and that is an ORACLE with
the labels in hand, so a train-only gate would capture a fraction of it against
a +0.001 reporting floor.

The per-tab oracle corroborates independently: at the one granularity where
per-group estimates are reliable, heterogeneity is +0.0002.

**Disposition: CLOSED.** The DeepFM/GBDT disagreement is real but not
predictably exploitable at any granularity tested. This also retro-explains the
+0.0002 global blend: there was never anything for a weight to capture.

Method note worth keeping: a per-group oracle over noisy per-group estimates is
upward-biased by construction, and the bias here (+0.0291) was TWELVE TIMES the
real signal (+0.0023). Any future "upper bound" of this shape must be run
against a quality-matched control before it is believed.

---

## 7. Process failures, and what they cost

Recorded because they changed conclusions.

1. **The `from ... import <list>` trap, twice.** `train.py` does
   `from ...features import EXTRA_CATEGORICAL_FIELDS`, binding the list object
   at import. Rebinding the name in `features.py` is invisible to
   `resolve_fields`. A keep/revert A/B silently measured the same configuration
   twice; the tell was scores identical to four decimals **seed by seed**, which
   is not what a real tie looks like. Same root cause as an earlier incident
   where two different feature patches both scored exactly 0.6024. Mitigation:
   mutate the list, and always print `id_maps["fields"]`.
2. **A 2-seed result that did not reproduce.** k=8 read 0.6050 at 2 seeds and
   0.6047 at 3. Had the sweep stopped at 2 seeds it would have shipped a
   nonexistent win.
3. **An oracle misread as drift.** A `video x tab x date` oracle scored 0.6913
   and looked like exploitable temporal drift. It was the oracle memorising
   validation labels in tiny cells; recency weighting is flat (0.6389 vs
   0.6387) and aggressive decay hurts (0.6342).
4. **Conditional permutation importance is confounded here.** Permuting within
   `(qb, tab)` strata gives `video_id` -0.0091, `author_id` -0.0074, `tag1`
   -0.0073. But author, tag and upload_type are **deterministic functions of
   `video_id`**, so permuting them manufactures impossible video/tag pairs; the
   drop measures out-of-distribution sensitivity, not information. Only
   `pos_bucket` (-0.0029) is cleanly interpretable.
5. **Two of our own headline hypotheses were wrong.** Author/music target
   encoding was ranked the top experiment before granularity was checked. And
   "`user_id` is a pure overfitting engine" was refuted at -0.0091. Our
   hypotheses failed at about the same rate as the externally proposed ones.

---

## 8. Where this leaves the project

Confirmed levers: none beyond the accepted pipeline. Validation primary is
**0.6047**, unchanged across this entire investigation.

That is the honest headline, and the reason it is worth recording: roughly 90
method families were triaged, six substantive hypotheses were tested at three
seeds each, and the benchmark did not move. The value is the map — which
directions are closed, on what evidence, and which of the closures are
structural (permanent, dataset-level) rather than empirical (contingent on our
implementation).

**Still open**

- The +0.0031 the neural model holds over an identity-equipped GBDT.
- The mechanism by which `user_id` contributes +0.0091 without any measurable
  affinity signal.
- Whether a DeepFM + GBDT rank ensemble helps: every *neural* pair correlates
  at 0.998 or above, but a tree model partitions rather than embeds, so it is
  the only remaining candidate for decorrelated errors.
- A 5–10 seed bootstrap pass before any of the ~0.0003 differences above is
  treated as an ordering rather than a tie.

---

## 9. The one candidate, and why it was rejected

The architecture decomposition (section 5.3) produced the session's only
positive-looking result: dropping the FM pairwise term entirely (`mlp`) scored
**0.6051 vs 0.6047** for the shipped DeepFM, at identical parameter count. It
had everything a real finding should have — a mechanism (the pairwise term is
redundant with what the MLP computes), an independent corroboration (DCN-V2
ties DeepFM at 0.6047 with 8% more parameters), and a monotone decomposition
(linear 0.5981 -> fm 0.6020 -> mlp 0.6051) rather than an isolated spike.

It was still rejected. Four threats were separated, and only the first is
addressed by seed repetition:

| threat | instrument | outcome |
|---|---|---|
| (a) seed noise | 8-seed **paired** test | **passed**: +0.00044, wins 8/8, t = 4.49, 95% CI [+0.00025, +0.00064] |
| (b) validation sampling error | bootstrap over 22,377 users, 2000 resamples | **FAILED**: +0.00028, 95% CI **[-0.00056, +0.00110]** — includes zero |
| (c) temporal transfer | split-half of validation by date | **FAILED**: +0.00127 early (4/22-4/25) vs **+0.00027 late** (4/26-4/28) |
| (d) selection over ~30 configurations | none available | **irreducible** — only data never used for selection could, i.e. the hidden test set |

**The negative control is what settled it.** Two different seed groups of the
IDENTICAL model, pushed through the same bootstrap, differ by |0.00028| with CI
[-0.00110, +0.00053] — the same magnitude as the "effect". When comparing a
model against itself yields the same number as comparing it against its rival,
there is no rival. Every future candidate at this scale must be run against this
control before it is believed.

**Reconciling (a) with (b).** Both are correct; they answer different questions.
The paired design cancels seed variance by construction, so it asks "given THESE
users, is mlp reliably ahead?" — yes, overwhelmingly. The bootstrap asks "would
this hold on a different sample of users?" — not established. Seed noise was
never the binding constraint: user-sampling noise (+/-0.0008) is roughly twice
the effect size. **A tight paired interval is not evidence of generalisation.**

(c) independently condemns it: the effect shrinks 5x across the seven validation
days, and the hidden test window is later still, so the expected test effect is
approximately zero.

Verdict: `WEAK`. The champion remains DeepFM at **0.6047** and the submission
was not regenerated. Recorded because the near-miss is the point — the paired
t of 4.49 looked conclusive and would have shipped a nonexistent gain.

Method note: the bootstrap is exact, not approximate. GAUC decomposes as a
positive-count-weighted mean of per-user AUCs and nDCG@5 as a plain mean over
users, so resampling users and re-aggregating reproduces the metric exactly; we
verified our decomposition against the official `starter_kit/evaluate.py` at
**9.6e-15** before trusting it.

---

# Appendix A — Literature considered

Every paper that entered the decision process, what we took from it, and the
disposition. "Rejected" almost always means *the assumption the paper needs is
absent from this dataset*, not that the paper is wrong.

## A.1 Adopted — these shaped the pipeline

| Work | What we took | Where it lives |
|---|---|---|
| **Rendle et al., Factorization Machines (2010)** | Shared embedding table with per-field offsets; the second-order term is exactly the user×item crossing that within-user ranking needs | `pipeline/train.py`, the starting model |
| **Guo et al., DeepFM (2017)** | Parallel FM + MLP over a shared embedding | `FactorizationMachineWithMLP` — the accepted model, and the only architecture change that ever produced a win (+0.0021) |
| **Wang et al., DCN-V2 (arXiv:2008.13535)** | Explicit bounded-degree crosses, `x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l`, plus the low-rank variant | Implemented as `CrossNetV2` / `DCNv2`. **Tested, NULL** — ties the control at every setting |
| **Micci-Barreca (2001), target encoding; CatBoost ordered statistics (Prokhorenkova et al. 2018)** | Empirical-Bayes smoothing, and the insight that in-fold encoding leaks; we use K-fold rather than leave-one-out because LOO is invertible on small cells | `_fit_cross_te` in `pipeline/train.py`; leak verified at 0.7531 in-fold vs 0.1748 OOF |
| **Gao et al., KuaiRand (2022)** | Dataset semantics: `long_view` definition, the `tab` field, the random-exposure log's intended use | Throughout; `pipeline/data/label.py`, `agent/referee.py` |
| **Ke et al., LightGBM (2017)** | The M2 rung of the nested ceiling — "what a strong non-neural model extracts from all clean features" | Nested ceiling §6.2, confound test §6.4 |

## A.2 Tested and rejected

| Work | Hypothesis | Result |
|---|---|---|
| **Rendle et al., BPR (arXiv:1205.2618)** | Pairwise ranking loss should beat pointwise, since the metric is a ranking metric | **0.5994 vs 0.6017 pointwise.** `NULL` |
| **Cao et al., ListNet (2007)** | Listwise loss targets top-of-list quality, which nDCG@5 rewards | **0.6004.** `NULL` |
| **Ma et al., ESMM (arXiv:1804.07931); Ma et al., MMoE (2018); Tang et al., PLE (2020)** | Auxiliary feedback signals (click, like, follow) share a bottom tower and transfer to `long_view` | **0.6013.** `NULL`. Note the classic ESMM motivation does not even apply here — KuaiRand logs `long_view` on *every* impression, not only clicked ones, so there is no sample-selection bias to correct |
| **Burges, LambdaMART / LambdaRank** | Direct nDCG optimisation with users as groups | **0.5901.** `NULL` — and later diagnosed: it spent top splits on within-user-constant features. Re-run correctly as a plain GBDT it reaches 0.5961, so the original number measured our setup error, not the method |
| **Zhao et al., D2Q duration deconfounding (arXiv:2206.06003)** | Watch-time bias by video duration; deconfound by duration quantile | `duration_ms` is already bucketed by train quantiles for exactly this reason. Conditional on video quality, duration contributes **−0.0003**. `NULL` |
| **CWM / play-time modelling** | Use continuous play ratio as an auxiliary regression target rather than the thresholded label | Subsumed by the multi-task result above. `NULL` |

## A.3 Rejected structurally — the assumption is absent from this data

| Work | Assumption it requires | Why it fails here |
|---|---|---|
| **Covington et al., YouTube DNN (2016)** | Large catalog, retrieval-then-rank | Candidate sets are given and tiny (median 4). There is no retrieval stage to model |
| **Zhou et al., DIN (2018) / DIEN (2019)** | User history predicts affinity to the candidate | `user × video` = 0.4973, measured three times. There is no affinity signal to attend over |
| **Kang & McAuley, SASRec (2018); Sun et al., BERT4Rec (2019)** | Sequential order of consumption carries signal | Same null, plus repeat exposure is 1.62% of rows with an identical `long_view` rate (0.3072 vs 0.3134) |
| **Li et al., MIND (2019); Cen et al., ComiRec (2020)** | Users have multiple separable interests worth disentangling | Requires a personalisation signal that measures at chance |
| **He et al., LightGCN (2020); Wu et al., SGL (2021)** | User–item bipartite graph propagation improves collaborative signal | The collaborative signal itself is absent; propagating noise does not create signal |
| **Wei et al., MMGCN (2019) / MMSSL (2023)** | Multimodal item content (frames, audio, text) | KuaiRand-Pure ships no frames, audio or captions. `NOT_APPLICABLE` |
| **Chen et al., Top-K off-policy correction (2019)**; **Schnabel et al., unbiased learning (2016)**; **AutoDebias** | A uniform-exposure sample usable for training or propensity estimation | The random log is **75.7% inside the hidden-test window** and its `long_view` rate is 0.085 vs 0.313. Using it means test contamination plus train/serve mismatch. Kept for the referee probe only |
| **Liu et al., Monolith (arXiv:2209.07663)** | Online/streaming training with collision-free hashing at industrial scale | Offline fixed benchmark, 7,583 items. The engineering it solves is not our bottleneck |
| **TikTok HCI / attention-economics literature** | Behavioural constructs (novelty seeking, choice overload, attention decay) observable in the log | Only `pos_bucket` is measurable, and it is already in the model (+0.0011 conditional). The rest have no observable proxy |

## A.4 Agent-design literature (methodology, not modelling)

| Work | Use |
|---|---|
| **AutoRecLab (arXiv:2510.18104)** | Autonomous RecSys research-agent framing: hypothesis → code → experiment → reflect, with an experiment log as the artifact |
| **AgentX (arXiv:2606.26859)** | Agent-orchestration patterns; informed the checkpoint/resume and rollback design |

---

# Appendix B — The full method backlog, item by item

All ~95 families that were formally considered, with disposition. Status
vocabulary as defined at the top of this document. The single most common
outcome is `REJECTED_STRUCTURALLY`, which is a statement about **this
benchmark**, not about the method.

### Items 1–5 — the mainstream candidates

| # | Family | Status | Reason |
|---|---|---|---|
| 1 | Feature engineering (frequency/count/log-count encodings, ratios, differences, normalisation, quantile transforms, rank/percentile, binning, crosses, reliability counts, historical aggregates, recency, momentum, trend, rolling stats, EWMA, volatility, z-scores, residual features) | mixed | Binning and crosses **adopted** (`dur_bucket`, `pos_bucket`). Count/popularity **NULL** (train exposure 0.5396, transductive 0.5189). Rank/percentile **NULL** (ranking-invariant, §5.1). Momentum/rolling/EWMA/volatility **REJECTED_STRUCTURALLY** (`upload_dt` has 3 values) |
| 2 | Candidate-relative modelling (percentile, rank in set, minus set mean/median, z-score, relative freshness/popularity/duration, set variance, entropy, diversity, difficulty) | `NULL` | Percentile alone 0.6382 vs raw prior 0.6387 — ranking-identical. Conditional: −0.0181, −0.0061, −0.0049, −0.0005, −0.0004 |
| 3 | Feature interactions (DCN, **DCN-V2**, DeepFM, xDeepFM, AutoInt, FiBiNET, AFM, FFM, FwFM, NFM, explicit crosses, learned interaction matrices, low-rank) | `NULL` | DeepFM adopted. DCN-V2 implemented and tested at 2/3 layers, full and low rank — ties control (0.6047/0.6050/0.6048 vs 0.6047). Explicit `video × tab` cross also flat. Decomposition shows the pairwise term contributes **less** than the MLP and the two do not stack, so xDeepFM/AutoInt/FiBiNET were not pursued — same mechanism, already falsified |
| 4 | Learning-to-rank (pointwise logistic/MSE, RankNet, BPR, pairwise logistic/hinge, LambdaRank, LambdaMART, ListNet, ListMLE, ApproxNDCG, SoftRank) | `NULL` | Three independent implementations: BPR 0.5994, ListNet 0.6004, LambdaMART 0.5901 — all below pointwise 0.6017. The remaining variants are interpolations of the same three ideas |
| 5 | GBDTs (CatBoost, LightGBM, XGBoost, ExtraTrees, RF, HistGradientBoosting) | `NULL` | LightGBM reaches 0.5961 over all clean features, 0.6023 with identity, versus the neural model's 0.6048. Used as the M2 rung and as an ensemble member, not as a champion. CatBoost's specific selling point — ordered target statistics — was adopted directly in `_fit_cross_te` instead |

### Items 6–15 — the temporal / quantitative-finance family

**All `REJECTED_STRUCTURALLY` on one measurement: `upload_dt` has three
distinct values** (2022-04-09/10/11). Every video in KuaiRand-Pure was uploaded
inside a 3-day window, so there is no video lifecycle, no age distribution, and
no virality curve to model. Independently confirmed: exponentially
recency-weighting the quality prior across the 13 training days is flat (0.6389
vs 0.6387) and a 2-day half-life hurts (0.6342).

| # | Family |
|---|---|
| 6 | Temporal feature engineering (upload age, log age, age buckets, cyclical hour/day, rolling mean/median/std/count/target-rate, EWMA, momentum, acceleration, trend, slope, time since peak, multi-horizon 1h/3h/6h/12h/1d/3d) |
| 7 | Freshness transforms (`age`, `log(1+age)`, `exp(−λ·age)`, `(1+age)^−α`, learned/piecewise decay, splines, age percentile, age × position/tag/type/popularity) |
| 8 | Quant-finance temporal (momentum, mean reversion, volatility, EWMA vol, vol-of-vol, z-score shocks, drawdown, distance from max, time since max, Sharpe-like ratios) |
| 9 | Multi-timescale forecasting / HAR |
| 10 | HMM / latent regimes (2/3/4-state, Gaussian HMM, Markov-switching regression and AR) |
| 11 | Kalman / state-space (local level, local trend, DLM, exponential state-space) |
| 12 | Change-point detection (CUSUM, Page-Hinkley, PELT, Bayesian online change-point, binary segmentation) |
| 13 | Hawkes / point processes (Poisson, non-homogeneous, Hawkes, multivariate, marked, self-exciting intensity) |
| 14 | Time-series forecasting (AR, MA, ARMA, ARIMA, SARIMA, ARIMAX, SARIMAX, ETS, Holt, Holt-Winters, TBATS, VAR, VARMA, VECM) |
| 15 | GARCH family (ARCH, GARCH, EGARCH, GJR, TGARCH, APARCH, FIGARCH, stochastic volatility, component GARCH, GARCH-MIDAS) |

Hour-of-day and day-of-week *were* tested separately, since they do not depend
on `upload_dt`: marginal 0.5126 and 0.5104, conditional on video quality
**+0.0002** and effectively zero. `NULL`.

### Items 16–20 — statistical modelling

| # | Family | Status | Reason |
|---|---|---|---|
| 16 | PCA / SVD / NMF / factor analysis / ICA / random projection on video×behaviour matrices | `NOT_APPLICABLE` | 7,583 videos with ~150 impressions each. The `video_id` embedding already learns a dense low-rank item representation from ample data; a separate decomposition of the same counts adds no information |
| 17 | Target encoding (video, author, tag, music, video-type, upload-type, interactions) with Bayesian smoothing, empirical Bayes, hierarchical shrinkage, ordered statistics | mixed | **Adopted** as machinery (verified leak-safe). As a *feature*: author/music `REJECTED_STRUCTURALLY` (87%/98% single-video — corr 0.985/0.987 with the video prior), tag `NULL` (conditional +0.0001), video×tab `NULL` (0.6046 vs 0.6046) |
| 18 | Hierarchical Bayesian partial pooling `global → tag → author → video` | `REJECTED_STRUCTURALLY` | The hierarchy has two real levels, not four: author and music are ~1 video each, so those levels are the leaf |
| 19 | Confidence-weighted learning (sample weights, soft/ordinal labels, auxiliary regression on play ratio, hard-negative and uncertainty weighting) | `NULL` | Covered by the multi-task auxiliary result (0.6013). `play_time_ms` is legitimate as a training-time target but not as a feature |
| 20 | Multi-task (shared bottom, MMoE, PLE, task towers, cross-stitch, uncertainty weighting, GradNorm, PCGrad) | `NULL` | Auxiliary head 0.6013 vs 0.6017 pointwise |

### Items 21–50 — broader ML and methodology

| # | Family | Status | Reason |
|---|---|---|---|
| 21 | Supervised learning (logistic, GLM, regularised linear, MLP, boosting, SVM, RF, ensembles) | tested | This *is* the model family. Decomposed in §5.3: linear 0.5981, fm 0.6020, mlp 0.6051, deepfm 0.6047 |
| 22 | Unsupervised clustering (k-means, GMM, hierarchical, DBSCAN/HDBSCAN, spectral, UMAP, t-SNE, NMF, autoencoders) as cluster-ID features | `REJECTED_STRUCTURALLY` | Video clusters are a coarsening of `video_id`, which the model already has at full resolution; user clusters are within-user-constant |
| 23 | Self-supervised (masked feature prediction, contrastive, temporal contrastive, next-event, denoising) | `NOT_APPLICABLE` | Pretraining objectives need either sequence structure (absent) or content (absent) |
| 24 | Metric learning (Siamese, triplet, contrastive, InfoNCE, proxy-based) | `REJECTED_STRUCTURALLY` | Learns a user–item similarity space; that similarity measures at chance here |
| 25 | Sequence modelling (Transformer, GRU, LSTM, TCN, temporal attention, state-space, Mamba) | `REJECTED_STRUCTURALLY` | No personalisation signal; organizer priority-2 lead closed by measurement |
| 26 | Graph learning (GNN, GAT, GraphSAGE, heterogeneous GNN, hypergraph) and graph-derived features (degree, PageRank, centrality, community, neighbour quality) | `REJECTED_STRUCTURALLY` | The video→author and video→music graphs are near-perfect matchings (87%/98% single-video), so degree/centrality are constants. video→tag is the only real edge set and tag is conditionally null |
| 27 | Network science (centrality, PageRank, community detection, assortativity, k-core, graph entropy) | `REJECTED_STRUCTURALLY` | Same degenerate graph |
| 28 | Information theory (entropy, mutual information, conditional MI, information gain, KL/JS divergence, surprisal, novelty `−log P(x)`) | partially adopted | The **conditional probe** (§2.1) is a practical estimator of conditional MI and became our primary instrument. Surprisal as a *feature* is a monotone function of popularity, measured at 0.5396 marginal. `NULL` |
| 29 | Signal processing (moving averages, EMA, Savitzky-Golay, Butterworth, low/high-pass, wavelets, FFT, spectral entropy, autocorrelation) | `REJECTED_STRUCTURALLY` | Requires a time series per entity; `upload_dt` has 3 values and per-video impression counts are ~150 over 13 days |
| 30 | Survival analysis (Cox, Weibull, exponential, AFT, discrete hazard, competing risks) | `NOT_APPLICABLE` | The natural target is watch-time hazard, i.e. `play_time_ms` — an outcome of the impression being predicted, so leakage as a feature. As a training target it reduces to the auxiliary-regression result (`NULL`) |
| 31 | Discrete choice (multinomial/conditional/nested/mixed logit, multinomial probit) | `NOT_IDENTIFIABLE` | Conceptually the closest formal match to "choose among a small candidate set". But conditional logit with item fixed effects **is** our linear model (0.5981), and random coefficients over users is what the user embedding already provides (+0.0091). Not separately testable with this design |
| 32 | Marketing science (choice modelling, uplift, persuasion, repeat exposure, diminishing returns, RFM, heterogeneity) | `REJECTED_STRUCTURALLY` | Repeat exposure is 1.62% of rows at an identical rate; RFM constructs are within-user-constant |
| 33 | CLV (BG/NBD, Pareto/NBD, Gamma-Gamma, survival CLV) | `REJECTED_STRUCTURALLY` | Purely user-level; cannot reorder a user's candidates |
| 34 | Behavioural economics (recency bias, availability, novelty seeking, familiarity, diminishing utility, choice overload, anchoring, loss aversion, scarcity, attention allocation) | `NULL` | The measurable proxies are position (in the model, +0.0011 conditional) and candidate-set size/spread (conditional −0.0049 / −0.0004) |
| 35 | Psychometrics / IRT (Rasch, 2PL, 3PL, multidimensional) | `NOT_IDENTIFIABLE` | Maps to user ability × item difficulty — which is a rank-1 user×item model, i.e. exactly the affinity term measured at 0.4973 |
| 36 | Attention economics (limited attention, salience, position bias, attention decay, information foraging) | partially adopted | Position bias is real and captured: `pos_bucket` is in the model, long_view falls 0.337→0.195 from first to twelfth impression |
| 37 | Decision theory (expected utility, Bayesian decision theory, risk-sensitive ranking `E[u] − λ·Var`) | `NOT_APPLICABLE` | Scoring is a fixed pointwise metric; there is no decision-cost asymmetry to exploit |
| 38 | Uncertainty quantification (ensemble variance, Bayesian, MC dropout, conformal, temperature scaling, isotonic) | `NULL` | Calibration cannot change GAUC or nDCG: a **global monotone** transform preserves every within-user ordering. Only group-wise recalibration can, and per-tab quality curves are already monotone (corr ≈ +0.99) |
| 39 | Optimal transport (Wasserstein, Sinkhorn, OT embeddings, distributional alignment) | `NOT_APPLICABLE` | No domain-adaptation problem: train and eval are the same distribution modulo a 7-day shift |
| 40 | Distribution shift (PSI, KS, JS, Wasserstein, MMD, adversarial validation) | diagnostic | Ran the temporal split-half in that spirit; used for validation of findings, not as a score lever |
| 41 | Continual learning (replay, rehearsal, temporal fine-tuning, EWC, online SGD, sliding windows, exponential decay) | `NULL` | Directly tested. Recency weighting flat (0.6389 vs 0.6387); 2-day half-life harmful (0.6342) |
| 42 | Online learning (online logistic, FTRL, online boosting, passive-aggressive) | `NOT_APPLICABLE` | Fixed offline benchmark; nothing arrives incrementally |
| 43 | Contextual bandits (LinUCB, Thompson sampling, NeuralUCB/TS, bootstrapped) | `NOT_APPLICABLE` | Candidate sets are pre-determined and logged. No exploration is possible |
| 44 | Reinforcement learning (DQN, policy gradient, actor-critic, offline RL, CQL, IQL, Decision Transformer) | `NOT_APPLICABLE` | Same — no environment, no action space, no long-horizon return |
| 45 | Causal inference (causal forests, treatment effects, doubly robust, propensity models, causal representation, discovery) | `REJECTED_STRUCTURALLY` | Needs the random-exposure log, which is 75.7% inside the hidden-test window |
| 46 | Granger causality | `REJECTED_STRUCTURALLY` | Needs time series per entity; see items 6–15 |
| 47 | Experiment design (controlled experiments, factorial, ablations, bootstrap, repeated seeds, paired tests, sequential testing, multiple-comparison correction, power analysis, effect sizes) | **adopted** | This became central. 3-seed minimum; 8-seed **paired** test on the one live candidate (t = 4.49, 95% CI [+0.00025, +0.00064]); user-level bootstrap; a **negative control** comparing two seed groups of the same model; explicit acknowledgement of selection bias over ~30 comparisons |
| 48 | Counterfactual testing (controlled perturbations, re-score, measure change) | `NOT_IDENTIFIABLE` here | Implemented as conditional permutation within `(quality, tab)` strata. **Confounded**: author, tag and upload_type are deterministic functions of `video_id`, so permuting them manufactures impossible inputs. Only `pos_bucket` (−0.0029) is interpretable. See §7.4 |
| 49 | Residual analysis | **adopted** | Became the residual probe (§2.2). Result: nothing named predicts the residual (−0.0002..+0.0002) |
| 50 | Adversarial validation | not run | Superseded — the temporal split-half answers the same question directly for the one candidate that mattered |

### Items 51–81 — unconventional and content-dependent

| # | Family | Status | Reason |
|---|---|---|---|
| 51 | Evolutionary / genetic feature construction, architecture and hyperparameter search | `NOT_APPLICABLE` | Search needs a signal gradient. The conditional probe shows the feature space is flat: every candidate lands within ±0.0004 of zero. Evolution over a flat landscape is a random walk |
| 52 | Symbolic regression | same | Same reason |
| 53 | Game theory (Nash, strategic recommendation, creator incentives, congestion) | `NOT_APPLICABLE` | No strategic agents in a logged offline benchmark |
| 54 | Operations research (assignment, integer/stochastic/robust programming) | `NOT_APPLICABLE` | Scoring is per-row and independent; there is no allocation constraint |
| 55 | Queueing theory | `NOT_APPLICABLE` | No arrival process is modelled or scored |
| 56 | Control theory (PID, Kalman control, MPC, adaptive control) | `NOT_APPLICABLE` | Needs a closed loop |
| 57 | Dynamical systems (attractors, stability, phase transitions, Lyapunov) | `REJECTED_STRUCTURALLY` | Needs trajectories; see items 6–15 |
| 58 | Complex systems (emergence, cascades, criticality, preferential attachment) | `REJECTED_STRUCTURALLY` | Virality dynamics require a lifecycle; 3 upload dates |
| 59 | Agent-based modelling | `NOT_APPLICABLE` | Would produce synthetic data, which the rules forbid for training |
| 60 | Synthetic user simulation (statistical, LLM-agent, behavioural) | `NOT_APPLICABLE` | Same — no external or generated training data permitted |
| 61 | Extreme value theory (GEV, GPD, peaks-over-threshold, tail index) | `REJECTED_STRUCTURALLY` | Models tail events over time; no lifecycle |
| 62 | Copulas | `NOT_APPLICABLE` | Dependence structure between item attributes is not what a within-user ranking scores |
| 63 | Topological data analysis (persistent homology, Mapper, Betti numbers) | `NOT_APPLICABLE` | Exploratory only; no route to a ranking score |
| 64 | Hyperbolic geometry (Poincaré embeddings, hyperbolic NNs) | `REJECTED_STRUCTURALLY` | Needs a hierarchy. The available one (`tag → video`) is two levels with 110 nodes |
| 65 | Tensor methods (CP, Tucker, tensor-train on user×video×time) | `REJECTED_STRUCTURALLY` | The user×video mode is the affinity signal, measured at chance; the time mode has 3 upload dates |
| 66 | Compressed sensing / robust PCA | `NOT_APPLICABLE` | The interaction matrix is not sparse in the relevant sense — 7,583 items, ~150 impressions each |
| 67 | Reservoir computing (echo state networks, liquid state machines) | `REJECTED_STRUCTURALLY` | Sequence models over a signal measured at chance |
| 68 | Diffusion models | `NOT_APPLICABLE` | Generative; would require synthetic data |
| 69 | Generative models (VAE, GAN, autoregressive, flows) | `NOT_APPLICABLE` | Same |
| 70 | Normalizing flows (density estimation, anomaly, uncertainty) | `NOT_APPLICABLE` | Same |
| 71 | Anomaly detection (Isolation Forest, One-Class SVM, LOF, autoencoder) | `NULL` by proxy | An item-anomaly score is a function of item attributes the model already encodes; conditional on quality, every attribute is ≤ +0.0003 |
| 72 | Novelty modelling (`−log P(video/tag/type)`, distance from historical distribution) | `NULL` | Monotone in popularity; measured 0.5396 marginal, and popularity conditional on quality adds nothing |
| 73 | Diversity / serendipity as a feature | `NULL` | Candidate-set diversity and spread tested: conditional −0.0004 |
| 74 | Submodular optimisation | `NOT_APPLICABLE` | Requires selecting a set; we score a fixed set |
| 75 | Multi-objective optimisation | `NOT_APPLICABLE` | Single scored label |
| 76 | Computational advertising (click/conversion prediction, position bias, delayed feedback, attribution, auctions, ad fatigue) | partially adopted | Position bias adopted (`pos_bucket`). Delayed feedback and attribution have no analogue: `long_view` is logged immediately per impression |
| 77 | Market basket analysis (Apriori, FP-Growth, association rules, lift) | `REJECTED_STRUCTURALLY` | Co-occurrence within a user's ~4 impressions is far too sparse to mine rules |
| 78 | Knowledge graphs (path features, node2vec, TransE, RotatE, ComplEx) | `REJECTED_STRUCTURALLY` | The graph is video→author (matching), video→music (matching), video→tag (110 nodes). No path structure |
| 79 | NLP (TF-IDF, BM25, embeddings, topic models, BERTopic, sentiment) | `NOT_APPLICABLE` | No text in KuaiRand-Pure. Zenodo caption/category supplements exist but were never organizer-sanctioned and would be external data |
| 80 | Computer vision (CLIP, ViT, ResNet, scene/object/aesthetic features) | `NOT_APPLICABLE` | No frames |
| 81 | Multimodal fusion (CLIP-style alignment, cross-attention, multimodal contrastive) | `NOT_APPLICABLE` | Needs items 79–80 |

### Items 82–95 — future architecture and agent design

| # | Family | Status | Reason |
|---|---|---|---|
| 82 | Semantic IDs (RQ-VAE, VQ, product quantisation, generative recommendation) | `NOT_APPLICABLE` | Semantic IDs compress *content* into tokens; there is no content here. And 7,583 items need no compression |
| 83 | Unified embeddings (shared item representation across retrieval and ranking) | `NOT_APPLICABLE` | No retrieval stage |
| 84 | Spotify-style unified search/recommendation | `NOT_APPLICABLE` | No queries in the dataset |
| 85 | Etsy-style unified retrieval/ranking | `NOT_APPLICABLE` | Same as 83 |
| 86 | LLM-generated synthetic labels (topic, emotion, novelty, humour) | `NOT_APPLICABLE` | Requires content to label, and would constitute external data |
| 87 | LLM synthetic data (users, preferences, interactions, hard negatives) | `NOT_APPLICABLE` | External training data is prohibited |
| 88 | Autonomous recommender agents (research agent + critic + deterministic evaluator + experiment DB) | **adopted — this is the project** | `agent/orchestrator.py`, `agent/referee.py`, `agent/compression_gate.py`, `logs/iterations.jsonl` |
| 89 | Autonomous feature discovery | **adopted** | The agent proposes feature hypotheses against `pipeline/data/features.py`; the `EXTRA_CATEGORICAL_FIELDS` registry exists specifically so its patches take effect (an earlier bug made four of them silent no-ops) |
| 90 | Autonomous model selection across a model zoo | partially adopted | `model_type` now selects among linear / fm / mlp / deepfm / dcnv2. Not yet agent-driven |
| 91 | Autonomous scientific falsification (propose → challenge → control experiments) | **adopted** | `agent/compression_gate.py` rejects a checkpoint a fresh context cannot justify. Extended manually this session with negative controls and paired tests |
| 92 | Autonomous experiment prioritisation (expected gain × P(success) ÷ cost) | **adopted informally** | The conditional probe *is* a cost-reduction device: seconds instead of a training run, which is what made triaging 95 families affordable |
| 93 | Autonomous literature mining | partially adopted | The skill store (`tier1_core.md`, `tier2_domain.md`, `tier3_deep/`) is a hand-curated version of this |
| 94 | Autonomous experiment memory (hypothesis, result, confidence, failure reason, verdict) | **adopted** | `tier1_core.md` is exactly this, and it prevented several re-tests. The status vocabulary in this document is its formalisation |
| 95 | Multi-agent research system (researcher / critic / planner / experiment manager) | future work | Single-agent with a compression gate acting as critic. A full split was judged higher risk than value before the deadline |

---

# Appendix C — The reasoning chain, in order

How the conclusions actually developed, including the wrong turns. Each arrow is
a decision point where the evidence changed what we did next.

1. **Start:** accepted pipeline at 0.6047, +0.0031 over baseline. The question
   posed was "what method extracts more signal?"
2. **First instinct — wrong.** We ranked author/music/tag target encoding as the
   top experiment, from marginal within-user GAUC (author 0.6439, music 0.6413).
   → **Checked entity granularity and it collapsed:** 87% of authors and 98% of
   music_ids own exactly one video; `corr(video_te, author_te) = 0.985`. The
   marginal score was the video prior wearing a different label. *Lesson: check
   the granularity of an entity before believing its aggregate score.*
3. → **Audited the raw files before modelling.** Found `upload_dt` has 3 values.
   That single fact closed items 6–15 plus 57, 61, 67 — roughly a quarter of the
   backlog — without an experiment.
4. → **Found `tab` crossed with everything looks powerful** (video 0.6387 →
   0.6479; tag1 0.5604 → 0.6153). Built it properly as a leak-safe out-of-fold
   encoding. **Flat: 0.6046.** *Lesson: a cross's standalone score overstates
   its incremental value when the model already holds both parents — the FM's
   second-order term had it already.*
5. → **Proposed that `user_id` is a pure overfitting engine**, reasoning from
   `user × video` = 0.4970 and the fact that its linear term cannot reorder.
   **Refuted at −0.0091.** *Lesson: a field can matter through a channel the
   marginal probes do not measure.*
6. → **Built the conditional probe** because marginal GAUC kept misleading us.
   Controlling for video quality, only `tab` survives (+0.0172); everything else
   including all user metadata lands in −0.0004..+0.0001.
7. → **An external critique identified a real hole:** `user_features_pure.csv`
   was genuinely unused. Tested it properly — **null at model level** (−0.0002
   and +0.0001), matching the conditional probe.
8. → **Decomposed `tab`.** It is a *reordering* signal that is inert for the 60%
   of users inside one tab (+0.0001) and worth +0.0314 for those who span tabs.
   Within a tab, quality is monotone (corr ≈ +0.99) — so its entire effect is
   the level gap between tabs.
9. → **Built the nested ceiling** and it corrected the story we were converging
   on: 0.5807 → 0.5877 → 0.5961 → 0.6048, i.e. **roughly equal thirds**, not
   "quality + tab explains everything". A third of the signal is captured only
   by learned structure.
10. → **The residual probe came back null on every named feature** — while the
    model still beat a GBDT by +0.0087. Both are true because the probe can only
    test features we can *name*. A null residual probe bounds named features, not
    the model.
11. → **Caught a confound in that GBDT comparison:** it saw one scalar per video
    where the network sees an embedding. Equalising identity closed 65% of the
    gap — and the entire gain came from `user_id`, converging with (5).
12. → **DCN-V2 tied the control at every setting**, and all five neural families
    correlate ≥ 0.998, so ensembling them is futile. Higher-order interaction
    structure is not the missing mechanism.
13. → **Decomposed the architecture instead:** linear 0.5981 → fm 0.6020 → mlp
    **0.6051** → deepfm 0.6047. The nonlinearity does the work; the pairwise term
    is redundant with it and mildly harmful alongside it. This *predicted* (12)
    rather than being fitted to it.
14. → **Subjected the one candidate to adversarial testing** rather than
    accepting it: 8-seed paired test (t = 4.49, CI [+0.00025, +0.00064]), a
    user-level bootstrap, a temporal split-half, and a negative control of two
    seed groups of the same model. With the standing caveat that none of these
    can correct selection over ~30 validation comparisons.

---

# 10. The final night: four more directions, four more nulls

Run on 2026-09-01, 00:40–08:30 SGT, after the submission was already frozen and
byte-verified. Prompted by a report that another competitor had reached a much
higher number through "tuning, then adjusting weights and model blending."

Protocol: every arm seed-matched against `base` on the **selection half only**;
the confirmation half was not read by any script in this batch. Comparisons are
paired per-seed differences reported with a standard error, and nothing is
called a win below 2 SE. 38 cached score vectors across 14 arms.

## 10.1 A ranking objective loses, monotonically

GAUC is a within-user AUC, and AUC is exactly `P(s_pos > s_neg)` for a pair
drawn from one user. The shipped model trains pointwise `BCEWithLogitsLoss`,
which optimises calibration and reaches the ordering only indirectly. Closing
that mismatch is the most principled untested lever in the repo, so we
implemented within-user BPR (`loss_mode="bpr"`) and a hybrid
(`BCE + alpha * BPR`), sampling one negative per positive from the same user and
resampling every epoch.

| arm | selection primary | paired vs base |
|---|---|---|
| base | 0.60781 | — |
| hybrid, alpha 0.1 | 0.60682 | **−0.00099** LOSS |
| hybrid, alpha 0.25 | 0.60646 | **−0.00135** LOSS |
| hybrid, alpha 0.5 | 0.60612 | **−0.00169** LOSS |
| hybrid, alpha 1.0 | 0.60516 | — |
| pure BPR | 0.60488 | — |

**Monotone in alpha.** A scatter of nulls is weak evidence; a dose-response
curve is strong evidence, and this one runs the wrong way at every dose.

The mechanism is visible in the sampler's own log line: `382,579 pairable
positives across 24,290 two-class users (33.5% of train rows)`. A pairwise loss
can only consume rows belonging to users who have **both** a positive and a
negative. It discards the other 66.5% — but those rows still teach the model
what a good *video* looks like, which transfers to every user. **The metric
ignores single-class users; the representation must not.** Matching the loss to
the metric threw away two thirds of the training signal to do it.

## 10.2 Capacity is already at its optimum

| k | 8 | **16** | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| selection primary | 0.60758 | **0.60781** | 0.60703 | 0.60676 | 0.60661 | 0.60627 |
| verdict | null | shipped | LOSS | LOSS | LOSS | LOSS |

Also monotone above k=16, and k=8 is statistically indistinguishable from k=16.
This independently reproduces the earlier `sweep_user_wd` finding that this
model wants *less* capacity, not more — the opposite of the usual intuition that
a bigger embedding table is a free win.

## 10.3 Auxiliary multi-task heads: null

`long_view` fires on 31.3% of training rows; `is_click` fires on 46.3% of the
same rows and is a strictly related outcome. We added a head predicting
engagement outcomes from the shared embedding, as training **targets only** —
the head is absent from every inference path, so no outcome field is ever an
input (`play_time_ms` and friends remain banned as features).

| arm | selection primary | paired vs base |
|---|---|---|
| aux `is_click`, w=0.3 | 0.60819 | +0.00019 null |
| aux `is_click`, w=0.1 | 0.60736 | −0.00045 null |

The w=0.3 arm has the highest single-seed mean of anything we tried all night,
and it is **not a result**: two seeds, std 0.00080, well inside noise, and its
10-seed rank-averaged ensemble (0.60779) is *worse* than base's (0.60826). This
is what the maximum of fourteen noisy arms looks like.

## 10.4 Fitted blend weights: overfitting, caught in the act

The competitor's reported method was weight-tuned model blending. We ran it
under a protocol that can detect the failure mode: split the selection half
again by user hash under a **different salt** into `selA` (the only rows weights
are fitted on) and `selB` (the only rows results are reported on), leaving the
confirmation half untouched. Caruana-style greedy selection with replacement
over 13 configs.

| | selA (fitted) | selB (honest) |
|---|---|---|
| base | 0.61174 | **0.60481** |
| equal-weight all 13 | 0.61159 | 0.60442 |
| **greedy fitted blend** | **0.61194** | **0.60393** |

**The fitted blend beats base where its weights were fitted and loses by
−0.00089 where they were not.** The +0.0002 on `selA` is not a small real gain;
it is the search finding noise, and the sign flips the moment it is asked to
generalise 30,000 rows to the left.

This is the same mechanism as the winner's curse in §9, arriving through a
different door, and it is the direct answer to "would tuned blending have helped
us?" — no, and the version that appears to help is the version that was measured
wrong.

## 10.5 What the night establishes

Fourteen arms, zero wins at 2 SE. Two monotone dose-response curves pointing
away from the two most plausible remaining levers. The shipped configuration —
DeepFM-lite, k=16, pointwise BCE, 10-seed equal-weight rank average — was not
beaten by any variant tried.

We did not change the submission. `submissions/submission_valid.csv` and
`submission_test.csv` still hash to the values frozen in `FROZEN_CONFIG.json`.

The honest summary is that this model sits in a local optimum that the levers
available to us do not move, which is what §8's oracle simulation predicted:
the gap to 0.8645 is mostly irreducible label noise, not headroom waiting to be
taken.
