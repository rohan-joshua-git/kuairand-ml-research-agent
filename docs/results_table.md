# Results

**Note:** metrics are GAUC / nDCG@5 (primary = mean of the two), scored by the vendored organizer evaluate.py (`starter_kit/evaluate.py`). Official FM baseline: test GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 — see `starter_kit/baseline_scores.json`.

| Iteration | Hypothesis | GAUC | nDCG@5 | Primary | Delta vs prev best | Delta vs baseline | Unbiased probe | Accepted | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (baseline reproduction, not an LLM hypothesis) | 0.6671 | 0.5358 | 0.6015 | - | - | - | False | - |
| 0 **<- BEST (submitted)** | (initial editable-pipeline training, not an LLM hypothesis) | 0.6676 | 0.5358 | 0.6017 | - | +0.0001 | 0.3629 | False | - |

## Resource usage

- Total tokens: 5,923
- Wall-clock hours: 0.174
- GPU hours: 0.174 (CPU-only pipeline)
- Manual interventions: 0
- Automatic process restarts after a crash: 0 (recorded in `logs/restarts.jsonl`). These are NOT manual interventions: the organizer confirmed in the Track 2 workshop Q&A (2026-08-31) that only changing the agent's behaviour counts, and `agent/supervisor.py` re-executes an identical command while the orchestrator resumes from its own checkpoint.

### Token usage by model

- gemini-3.6-flash: 4,041 in / 1,882 out
