# Results

**Note:** metrics are GAUC / nDCG@5 (primary = mean of the two), scored by the vendored organizer evaluate.py (`starter_kit/evaluate.py`). Official FM baseline: test GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 — see `starter_kit/baseline_scores.json`.

| Iteration | Hypothesis | GAUC | nDCG@5 | Primary | Delta vs prev best | Delta vs baseline | Unbiased probe | Accepted | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (baseline reproduction, not an LLM hypothesis) | 0.6671 | 0.5358 | 0.6015 | - | - | - | False | - |
| 0 | (initial editable-pipeline training, not an LLM hypothesis) | 0.6681 | 0.5366 | 0.6024 | - | +0.0008 | 0.3646 | False | - |
| 1 | We propose to add a smoothed out-of-fold target-encoded feature for `video_id` r | 0.6681 | 0.5366 | 0.6024 | +0.0000 | +0.0008 | 0.3646 | False | - |
| 2 | We propose to implement a DeepFM architecture in `pipeline/train.py` by adding a | 0.6681 | 0.5366 | 0.6024 | - | - | - | False | smoke test exited 1 |
| 3 | We propose to introduce an embedding dropout layer with a rate of 0.1 applied di | 0.6692 | 0.5370 | 0.6031 | +0.0008 | +0.0015 | 0.3632 | False | - |
| 4 **<- SUBMITTED (accepted best)** | We propose to implement a simple two-layer feed-forward Multi-Layer Perceptron  | 0.6710 | 0.5380 | 0.6045 | +0.0021 | +0.0029 | 0.3689 | True | - |
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
