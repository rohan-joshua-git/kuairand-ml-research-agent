# Method-backlog triage (KuaiRand-Pure, Track 2)

A ~95-category method backlog was proposed for this benchmark. This document
records which parts survive contact with the dataset, and **why** — every kill
below is backed by a measurement in this repo, not by an opinion.

Baseline context: official FM valid primary **0.6016**; our accepted pipeline
**0.6045**; oracle ceiling 0.8645 (test).

---

## 1. Killed by dataset structure

These are not "low priority" — the data cannot support them at all.

### `upload_dt` has 3 distinct values

Every one of the 7,583 videos in KuaiRand-Pure was uploaded on 2022-04-09, -10
or -11. There is no video lifecycle, no age distribution, no virality curve.

**Therefore dead:** freshness transforms (`log(1+age)`, `exp(-λ·age)`,
piecewise/spline decay, age percentile, age × anything), momentum, mean
reversion, volatility / vol-of-vol, drawdown, rolling and EWMA statistics,
multi-horizon (1h/3h/6h/12h/1d/3d) aggregates, HAR, ARCH/GARCH/EGARCH/GJR/
TGARCH/APARCH/FIGARCH, Hawkes and self-exciting point processes, Kalman /
local-level / local-trend / DLM state-space, CUSUM / Page-Hinkley / PELT /
Bayesian change-point, AR/ARMA/ARIMA/SARIMA/VAR/VECM/ETS/Holt-Winters/TBATS,
HMM and Markov-switching regimes, extreme-value theory, reservoir computing.

That is backlog items **6-15**, plus 57, 61, 67 — roughly a quarter of the list.

**Independent confirmation:** exponentially recency-weighting the video quality
prior across the 13 training days is flat (half-life 14d 0.6389, 7d 0.6389,
uniform 0.6387) and a 2-day half-life *hurts* (0.6342). Item 41's
"exponentially decayed training data / sliding windows" is dead too.

### `author_id` and `music_id` are video identity in disguise

87% of authors and 98% of music_ids own exactly one video. Target-encoded,
`corr(video_te, author_te) = 0.985` and `corr(video_te, music_te) = 0.987`.
Their strong-looking standalone scores (0.6367 / 0.6365) are the video prior.

**Therefore dead:** author/music hierarchical pooling, author/music latent
factors, and the author/music half of item 17's target-encoding list. The
hierarchy `global -> tag -> author -> video` (item 18) has only two real
levels, not four.

### The ranking group is the user

`user_id` target-encoded scores **exactly 0.5000** — anything constant within a
user cannot move that user's ranking, by construction.

**Therefore dead:** all purely user-side features, user clustering as a ranking
feature, CLV/RFM-style user models (33), psychometrics/IRT user ability (35),
and any global-only regime or platform-level time series.

### No personalisation signal

`user × video` 0.4970 / 0.4973, `user × author` 0.4981 — three independent
measurements at chance.

**Therefore dead:** sequence models (25 — DIN/DIEN/SASRec/BERT4Rec/Mamba),
multi-interest (MIND/ComiRec), graph CF (26 — LightGCN/NGCF/SGL), metric
learning for user-item similarity (24), collaborative latent factors.
Repeat exposure is 1.62% of validation rows with an *identical* long_view rate
(0.3072 repeat vs 0.3134 new).

---

## 2. Killed by prior measurement on this repo

| Backlog item | Result |
|---|---|
| 4 — learning-to-rank losses | BPR **0.5994**, ListNet **0.6004**, multi-task auxiliary head **0.6013**, all below pointwise logloss **0.6017** |
| 5 — GBDTs | LambdaMART (LightGBM `lambdarank`) **0.5901**; it spends top splits on within-user-constant features |
| 20 — multi-task (MMoE/PLE) | the auxiliary-head result above |
| 45 — IPS / causal deconfounding | random-exposure log is 75.7% inside the hidden-test window; unusable for training |
| 73-75 — diversity / serendipity / multi-objective | single-label benchmark; not the scored objective |
| — | Model is **not undertrained**: epochs 12 -> 40 early-stops at 13 and returns an identical score |

---

## 3. Tested this session

All numbers are 3-seed means on validation; seed std is 0.0001-0.0004, so
anything under ~0.001 is noise.

| Hypothesis | Backlog items | Result |
|---|---|---|
| `tag1` + `upload_type` as encoded fields | 1, 17 | 0.6047 vs 0.6046 — **+0.0001, noise** (kept: 2 lines, genuinely independent signal) |
| `video × tab` out-of-fold smoothed target encoding, 32 quantile buckets | 1, 2, 3, 17 | 0.6046 vs 0.6046 — **-0.0001, noise** (machinery kept but DISABLED) |
| `user_id` is a pure overfitting engine — delete it | — | **falsified, -0.0091** |
| Transductive eval-set exposure counts | 1 | 0.5189 vs causal 0.5396 — the gray-area feature is *worse*; not pursued |
| Recency-weighted training data | 41 | flat (0.6389 vs 0.6387); 2-day half-life harmful (0.6342) |
| Regularisation / capacity sweep (embed dim x weight decay) | 21 | *pending* |

### Two process failures worth recording

1. **The first keep/revert test measured nothing.** It patched
   `features.EXTRA_CATEGORICAL_FIELDS`, but `train.py` does
   `from ...features import EXTRA_CATEGORICAL_FIELDS`, binding the list object
   at import time. The tell was scores identical to 4 decimals *seed by seed* —
   not what a real tie looks like. Same root cause as this repo's earlier
   "two patches both scored exactly 0.6024" incident. Always print
   `id_maps["fields"]` to prove a field reached the encoder.
2. **The leakage guard was verified, not trusted.** On (video, tab) cells with
   <= 3 training rows, `corr(encoding, label)` is **0.7531 in-fold vs 0.1748
   out-of-fold** — in-fold, a small cell's encoding very nearly *is* that row's
   label. Any future target encoding must be checked on the small-cell subset,
   where the leak actually lives.

---

## 4. What actually survives

Ranked by expected gain per unit of compute, given everything above.

1. **Capacity and regularisation of the existing DeepFM-lite.** The only two
   changes that ever moved this pipeline were architectural (FM -> DeepFM-lite,
   +0.0021) and variance-reducing (5-seed rank-average, +0.0011). Embedding
   width and weight decay have never been swept.
2. **Explicit interaction architectures** (item 3 — DCN-V2, AutoInt, xDeepFM).
   Same family as the one change that worked. Note the caveat below.
3. **Heterogeneous ensembling** (items 21, 38). Rank-average, not
   score-average — only order is scored.
4. **`tag`-level modelling done properly** (multi-hot rather than primary-tag).
   `tag` is the one genuine pooling level: 110 values, median 10 videos each,
   `corr` with the video prior only 0.458.

### The caveat that governs items 1-3

`tab` is the master context variable — long_view base rate runs 0.004 to 0.489
across 15 tabs — and every item feature gains sharply when crossed with it
(video 0.6387 -> 0.6479, tag1 0.5604 -> 0.6153, upload_type 0.5214 -> 0.5938).
That makes explicit crosses look extremely promising. **They are not**, and the
reason generalises:

> A cross's *standalone* within-user GAUC overstates its *incremental* value
> whenever the model already contains both parent fields. The FM's second-order
> term already approximates `<e_video, e_tab>`.

Measured: feeding `video × tab` as a leakage-safe out-of-fold target encoding
scored **0.6047 vs 0.6045 — flat.** Before building any cross from items 1-3,
check whether both parents are already in the model.

### The number that bounds all of it

Oracle priors fit **on validation itself** (diagnosis only, never submitted):

| oracle | primary |
|---|---|
| video quality | 0.6146 |
| video × tab | 0.6351 |
| user × video (memorises the label) | 0.8477 |
| **our model** | **0.6045** |

We are at **98% of the pure item-quality ceiling**. Better item-quality
estimation is worth at most ~+0.010, and everything beyond it lives in
`user × video`, which is measured noise three times over. Plan against that,
not against the 0.8645 headline oracle.
