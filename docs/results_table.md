# Results

**Every metric below is on the VALIDATION split** (2022-04-22..04-28), scored by the vendored organizer script (`starter_kit/evaluate.py`). Metrics are GAUC / nDCG@5; primary = mean of the two.

Comparisons are against the official FM baseline's **validation** figures: GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016 (`starter_kit/baseline_scores.json`).

> The same baseline's published **hidden-test** primary is 0.5946. It appears here once, as the organizer's own reference figure, and is never compared against a validation number. The two are different splits on different scales; mixing them overstates progress.

| Iteration | Hypothesis | GAUC | nDCG@5 | Primary | Delta vs prev best | Delta vs baseline | Unbiased probe | Accepted | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (baseline reproduction, not an LLM hypothesis) | 0.6671 | 0.5358 | 0.6015 | - | - | - | False | - |
| 0 | (initial editable-pipeline training, not an LLM hypothesis) | 0.6681 | 0.5366 | 0.6024 | - | +0.0008 | 0.3646 | False | - |
| 1 | We propose to add a smoothed out-of-fold target-encoded feature for `video_id` r | 0.6681 | 0.5366 | 0.6024 | +0.0000 | +0.0008 | 0.3646 | False | - |
| 2 | We propose to implement a DeepFM architecture in `pipeline/train.py` by adding a | 0.6681 | 0.5366 | 0.6024 | - | - | - | False | smoke test exited 1 |
| 3 | We propose to introduce an embedding dropout layer with a rate of 0.1 applied di | 0.6692 | 0.5370 | 0.6031 | +0.0008 | +0.0015 | 0.3632 | False | - |
| 4 **<- SUBMITTED (accepted best)** | We propose to implement a simple two-layer feed-forward Multi-Layer Perceptron ( | 0.6710 | 0.5380 | 0.6045 | +0.0021 | +0.0029 | 0.3689 | True | - |
| 5 | We propose to add LayerNorm and a dropout rate of 0.2 to the hidden layer of the | 0.6693 | 0.5369 | 0.6031 | -0.0014 | +0.0015 | 0.3672 | False | - |
| 6 *(scored higher but rejected -> rolled back)* | We propose to add a weight decay of 1e-5 to the Adam optimizer in `pipeline/trai | 0.6718 | 0.5381 | 0.6050 | +0.0005 | +0.0034 | 0.3675 | False | - |
| 7 | We propose to add a mild dropout rate of 0.05 to the hidden layer of the newly i | 0.6702 | 0.5375 | 0.6039 | -0.0006 | +0.0023 | 0.3698 | False | - |

## Resource usage

- Total tokens: 91,430
- Wall-clock hours: 0.196
- GPU hours: 0.000 (CPU-only pipeline)
- Manual interventions: 2
- Automatic process restarts after a crash: 0 (recorded in `logs/restarts.jsonl`). These are NOT manual interventions: the organizer confirmed in the Track 2 workshop Q&A (2026-08-31) that only changing the agent's behaviour counts, and `agent/supervisor.py` re-executes an identical command while the orchestrator resumes from its own checkpoint.

### Token usage by model

- gemini-3.5-flash: 23,354 in / 17,504 out
- gemini-3.5-flash-lite: 35,285 in / 15,287 out

## Convergence rule

- epsilon = 0.002, N = 3 (organizer default, not redeclared)
- Hard caps: 50 iterations, 6 h wall-clock
- Logged iterations: 9
- Iterations that crash or produce no validation score count toward the iteration cap but do not advance or reset the convergence window (FAQ 2.9.1).

## Final submission (frozen)

**DeepFM-lite, 10-seed rank-average** (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).

| Validation metric | Official baseline | This submission | Absolute delta |
|---|---|---|---|
| GAUC | 0.6674 | **0.6724** | **+0.0050** |
| nDCG@5 | 0.5357 | **0.5382** | **+0.0025** |
| primary | 0.6016 | **0.6053** | **+0.0037** |

The scoring formula is `score_dataset = mean over m of delta(m)`. Because primary is itself the mean of the two metrics, that equals the primary delta.

### Hidden test

NOT CLAIMED. The hidden test is scored once by the organizer. No test score is asserted anywhere in this repository. The official baseline's published test primary is 0.5946, quoted only as the organizer's own reference figure and never compared against a validation number.

### Task Requirement #1 - baseline reproduction

Reproduced the organizer's FM on validation: GAUC 0.6671 / nDCG@5 0.5358 / primary 0.6015, against a published 0.6016. The test split is never evaluated - see the hidden-test guard in `pipeline/official_baseline.py`.

### Why an ensemble

Single-seed primary over 20 seeds: mean 0.60448, std 0.00049, range 0.6036-0.6052.

The ensemble's **mean** gain is NOT established: 5-seed vs 1-seed is +0.00027, while same-model negative controls reach 0.00077. It ships for **variance**, which is established: 5-seed std 0.00020 against single-seed 0.00049, matching sqrt(n). On a one-shot submission the floor is what matters - worst single seed 0.6036, worst 5-seed ensemble 0.6046.

### Selection / confirmation protocol

Validation was split by user hash into 11,270 selection and 11,107 confirmation users. All exploration and early stopping used selection only; the confirmation half was looked at 1 time, after the config was frozen.

Raw confirmation (0.6023) sits below selection (0.6083), but a frozen model-free video prior drops nearly identically, so that gap is population difficulty rather than overfitting. The model's advantage over that reference is +0.0250 on selection and +0.0242 on confirmation - it transfers to users never used for any decision.

### Provenance

The reported validation primary was produced twice, by independent paths:

1. cached score matrix -> `pipeline/eval_protocol.py` decomposition -> 0.6053
2. `pipeline.make_submission` -> `aligned_rows`/`score_rows` -> CSV -> `starter_kit/evaluate.py` via `submit.py --score` -> 0.6053

Agreement to four decimals on all three metrics establishes that the number reported here is the number the submitted artifact actually scores.

- `submissions/submission_valid.csv` sha256 `1fd3f7c0ef47d1dd9fa1792b610f8c512e70765db02cd89098db178c559a46e2`
- `submissions/submission_test.csv` sha256 `7f460171bbe0710244e3486c79d9a2e4342333203ade2242be5887abb4c46f30`
