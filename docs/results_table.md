# Results

**Note:** metrics are GAUC / nDCG@5 (primary = mean of the two), scored by the vendored organizer evaluate.py (`starter_kit/evaluate.py`). Official FM baseline: test GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 — see `starter_kit/baseline_scores.json`.

| Iteration | Hypothesis | GAUC | nDCG@5 | Primary | Delta vs prev best | Delta vs baseline | Unbiased probe | Accepted | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (baseline reproduction, not an LLM hypothesis) | 0.6671 | 0.5358 | 0.6015 | - | - | - | False | - |
| 0 **<- BEST (submitted)** | (initial editable-pipeline training, not an LLM hypothesis) | 0.6676 | 0.5358 | 0.6017 | - | +0.0001 | 0.3629 | False | - |
| 1 | We will implement a per-user pairwise Bayesian Personalized Ranking (BPR) loss i | 0.6646 | 0.5341 | 0.5994 | -0.0024 | -0.0022 | 0.3590 | False | - |
| 2 | We will implement a multi-task learning architecture in the training pipeline by | 0.6670 | 0.5356 | 0.6013 | -0.0004 | -0.0003 | 0.3636 | False | - |
| 3 | We will replace the pointwise Binary Cross Entropy loss with a listwise ListNet  | 0.6655 | 0.5353 | 0.6004 | -0.0013 | -0.0012 | 0.3697 | False | - |

## Resource usage

- Total tokens: 38,361
- Wall-clock hours: 0.118
- GPU hours: 0.000 (CPU-only pipeline)
- Manual interventions: 1
- Automatic process restarts after a crash: 0 (recorded in `logs/restarts.jsonl`). These are NOT manual interventions: the organizer confirmed in the Track 2 workshop Q&A (2026-08-31) that only changing the agent's behaviour counts, and `agent/supervisor.py` re-executes an identical command while the orchestrator resumes from its own checkpoint.

### Token usage by model

- gemini-3.5-flash: 20,366 in / 17,995 out
