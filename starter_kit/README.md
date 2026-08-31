# KuaiRand-Pure Starter Kit (vendored, organizer-provided)

Translated/reformatted from the original Chinese README for this repo. Content and numbers
are preserved exactly; only prose is translated. **This directory is the authoritative
reference for scoring — `evaluate.py` in particular must not be modified.**

## Dependencies

Python 3.9+ and numpy. **No other dependencies.** No torch/pandas/sklearn required for the
starter kit itself.

## Data

Download from https://kuairand.com (Zenodo link, no registration required):

```bash
# run from this directory, unpacks to ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Run

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; pass it explicitly if you unpacked elsewhere.

`--model` is one of `fm` (official baseline) / `pop` (trivial baseline) / `random` (sanity check
only, for self-testing your eval code).
FM takes ~40s on CPU (single core).

## Task definition (scoring is locked — do not change this)

| | |
|---|---|
| Task | **Within-user ranking** — each user is ranked only within their own rows in the eval split, not against a global item pool |
| Relevance label | `long_view` (binary, 0/1) |
| Metrics | `GAUC`, `nDCG@5` — **primary = mean of the two** |
| Split | train `20220408-20220421` / valid `20220422-20220428` / test `20220429-20220508` |
| Zero-positive users | nDCG recorded as 0.0 and included in the mean; GAUC only counts users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel - 1` (equivalent to identity for binary labels) |

Implementation is in `evaluate.py` — read the header comment there for the exact contract.

## Baseline ladder

Scores on the test split. **The FM row is what you need to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity check only) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### Read the numbers against the real ceiling: nDCG@5's ceiling is 0.729, not 1.0

The test split has 23,875 users:

| | share | effect on scoring |
|---|---|---|
| All-negative users (no `long_view` at all) | **27.1%** | nDCG is **0** for every model; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is **1** for every model; excluded from GAUC |
| Discriminative users | **63.7%** | the real sample for GAUC |

So even a perfect ranker (oracle: score = the true label) can only reach:

| | random | FM baseline | **oracle ceiling** | FM's share of headroom used |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Judge headroom against the oracle ceiling, not against 1.0.** 0.5946 looking "far from 1.0" is
misleading — the baseline has already used about a third of the *available* headroom, and there's
0.27 of primary-score room left, not 0.41.

FM's std across 5 random seeds is **0.0008**. Convergence is set from this: **epsilon = 0.002
(~2.5sigma), N = 3** — 3 consecutive iterations without a validation-primary improvement greater
than 0.002 means converged.

> Sanity check: if your eval code gets primary != ~0.475 (+/- 0.001) for `--model random`, your
> harness has a bug — fix that before anything else.

## Submission format

CSV with header, one row per eval-split row:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| field | meaning |
|---|---|
| `row_id` | 0-based, strictly increasing; corresponds to `data.load()[split]` row order (deterministic: read `log_standard_4_08_to_4_21_pure.csv` then `log_standard_4_22_to_5_08_pure.csv`, filter by date, keep file order) |
| `user_id` / `video_id` | redundant fields, used only to verify alignment |
| `score` | your model's score for that row (any real number, only relative order matters, NaN/Inf not allowed) |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** within the eval split —
> 3.06% of test rows are repeated pairs, up to 12 times — so it can't serve as a key.

Generate and validate:

```bash
python3 submit.py --make  --split test  submission.csv    # generate an example submission with the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format + alignment
python3 submit.py --score --split valid submission.csv    # validate + score (valid split only, locally)
```

`--check` fails on: a wrong header, a row-count mismatch, a `row_id` gap, `user_id`/`video_id`
misaligned against the eval split, or a non-numeric score. **Run `--check` yourself before
submitting.**

## Where to start

The list below is **already run** — these are not guesses, they're dead ends and open leads the
organizers already tested. Don't burn iterations re-discovering them.

### Already tried (these two moves earned nothing — don't waste iterations)

| Tried | Result |
|---|---|
| **Wired in more features** — added CWM's 13 fields (+`music_id`/`video_type`/`upload_type` + 6 bucketed user-side fields) | primary **0.5940** vs the 5-field kit's **0.5950** — within noise, if not slightly worse |
| **Grew model capacity** — embedding dim k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887 — flat |

Why: `user_id x video_id` crossing already captures most of the learnable signal. Bucket-style
features like `follow_user_num_range` are near-constant within a `user_id`, and 1.14M rows already
carry enough capacity for that crossing. **The headroom is not in feature count or model capacity.**

Also note: **pure user-side features contribute ~0 to the score on their own**, because ranking
happens *within* each user — anything constant within a user doesn't change that user's ranking
order (empirically: `item_pop x user-bucket` and plain `item_pop` score the same). User-side
features can only matter through a **cross term with item-side features**.

### Unexplored — headroom should be here (priority order, per the organizers)

1. **Switch the loss function.** Currently pointwise logloss, but the metrics (GAUC / nDCG) are
   **ranking metrics**. Switching to pairwise (BPR) or listwise (softmax over a user's candidates)
   aligns the training objective with the eval objective — **this is what we think is most likely
   to help.**
2. **User behavior sequence modeling.** Current features use **zero** behavioral history. KuaiRand
   users have hundreds of interactions in train — DIN/SIM-style interest modeling is a completely
   unexplored direction.
3. **Multi-task.** The logs also carry `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` — usable as auxiliary tasks to help the `long_view` main task.
4. **Watch-time-aware modeling.** This is exactly [CWM](https://github.com/hyz20/CWM)'s
   contribution: watch time is a **censored regression** problem (a completed play means the true
   watch time was truncated by video length, so use a one-sided loss instead of squared error).
   This is a research-depth direction.
5. **Swap architectures.** DeepFM / DCN / xDeepFM. Given capacity scaling was flat, **prioritize
   1-4 before this.**
6. **Temporal features and distribution shift.** `hourmin`, `date`, and the drift between train and
   test.
7. **`log_random_4_22_to_5_08_pure.csv` as extra validation (not training).** This is a random-
   exposure log (1.18M rows) usable as an additional held-out validation set, to check whether the
   model is only overfitting on the biased-exposure eval set.

## Using your own model (including CWM)

`evaluate.py` doesn't know or care about models — it needs one flat array:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the eval-split row's `user_id`
- `labels`: that row's `long_view` (0/1)
- `scores`: your model's score for that row (any real number, only relative order matters)

So you can drop `baseline.py` entirely, swap in PyTorch, LightGBM, or CWM's xDeepFM — as long as
you hand the final `scores` to `evaluate()`. **Scoring is decided by `evaluate.py` alone.**

> Note on CWM: it depends on `torch==1.6.0` (2020, may not build against a modern GPU stack), and
> its main loss innovation is counterfactual watch time. Its own eval label is a self-built
> `long_view2`, not this kit's `long_view`. It's a research-depth reference — legitimate as an
> **advanced lead**, not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metrics implementation + the scoring contract. **Do not modify.** |
| `data.py` | Data loading, official split, feature encoding. Adding features starts here. |
| `baseline.py` | Baselines. FM is the one you need to beat. |
| `baseline_scores.json` | Officially published numbers + seed std + convergence parameters. |
| `submit.py` | Generate / validate the submission file. |
| `ablation_features.py` | The feature-scaling experiment, reproducible. Adding features earned nothing — see the numbers. |
