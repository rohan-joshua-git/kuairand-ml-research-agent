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
