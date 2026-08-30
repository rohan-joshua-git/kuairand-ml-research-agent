# Autonomous ML Research Agent - KuaiRand-Pure

An LLM agent that runs the ML engineering loop (read the problem, inspect data, engineer features, train, tune, evaluate, reflect, revise) on the KuaiRand-Pure recommendation dataset, aiming to beat a baseline on GAUC and nDCG@5 without a human in the iteration loop.

Status tags used below: **Implemented** (exists in this repo, verifiable by the file cited next to it), **Placeholder** (a working stand-in, not the final version), **Planned** (designed or partially built, not yet wired into the live loop).

## Overview

Recommendation models like the one behind a short-video feed are built by repeatedly testing changes against held-out data. A common failure mode when automating this loop: the training data itself is biased, since it only contains items a prior recommender already chose to show users. An agent optimizing against that data can learn to exploit the bias instead of genuinely improving, and then underperform on unseen data.

Three things in this build address that:

- **Unbiased referee.** KuaiRand-Pure includes a slice of interactions collected under uniformly random exposure instead of the production recommender. The agent scores candidates against this random-exposure log as well as the standard validation split, and flags divergence between the two.
- **Compression gate.** Before a checkpoint is finalized, the agent summarizes the approach and hands the summary to a fresh LLM context with no access to validation scores. If that second pass can't reproduce the result from the summary alone, the checkpoint is rejected.
- **Pre-seeded domain knowledge.** The agent starts with a tiered knowledge base about the dataset, including two undocumented data quirks found by reading the raw field spec and file manifest (details below), instead of rediscovering them through trial and error.

## Current state

This is a working, runnable scaffold, not a finished benchmarked agent.

| Piece | Status | Note |
|---|---|---|
| End-to-end loop | Implemented | Runs on real KuaiRand-Pure data, tracked against the vendored official FM baseline (`starter_kit/`, `config.starter_kit.official_baseline`). See [Confirmed by the Starter Kit](#confirmed-by-the-starter-kit). |
| Self-editing code | Implemented | Real API calls, scoped to `pipeline/data/features.py` and `pipeline/data/label.py`, with automatic rollback on a failed smoke test. See [Limitations](#limitations) — this is still narrower than the Starter Kit's top headroom suggestion (a loss-function change), which lives in `pipeline/train.py`, not yet editable. |
| Unbiased referee | Planned | Scoring, propensity estimation, and divergence tracking are implemented and tested in isolation, not yet wired into the live per-iteration loop. |
| Submission writer | Implemented | `pipeline/submit.py` writes the confirmed `row_id,user_id,video_id,score` CSV format (see `starter_kit/submit.py`). |

## Architecture

```
                    orchestrator.py loop
        read -> inspect -> engineer -> train -> evaluate -> reflect/revise -> ...

        Skill store              Ablation                Referee + gate
        (warm start)              (targeted search)        (unbiased scoring,
        tier1/2/3                 which block moves         overfit rejection)
                                   the score?
```

**Ablation-first refinement.** Instead of rewriting the whole pipeline each round, `agent/ablation.py` runs a cheap ablation over pipeline blocks (training schedule, learning rate, positive-class weighting) to find which one is moving the score, and `agent/orchestrator.py` targets that block for the next LLM-driven code change. `agent/skill_store/` holds domain knowledge in three tiers: Tier 1 (always loaded, dataset-specific quirks), Tier 2 (RecSys method priors, loaded when relevant), Tier 3 (deep dives, loaded on demand via keyword match in `agent/skill_store/retriever.py`). This follows results from MLE-STAR and HASTE showing ablation-guided, tiered-knowledge search outperforms flat knowledge-dumping and whole-pipeline rewrites.

**Unbiased referee.** `log_random_4_22_to_5_08_pure.csv` contains ~1.19M interactions from uniformly random exposure, undocumented in the challenge brief. `agent/referee.py` scores candidates against this log alongside standard validation and tracks the gap between the two; a widening gap signals the agent is fitting the biased proxy rather than improving generally.

**Compression gate.** `agent/compression_gate.py` compresses a winning approach into a short summary and hands it to a fresh LLM context with no memory of the search and no access to validation scores. If that reproducer can't get behind the approach, the checkpoint is rejected and the previous best is kept. This targets a known failure mode where agents pick a validation-overfit artifact over a genuinely weaker-looking but real solution.

## Dataset findings

1. **The scored label is `long_view`, not `is_click`.** Confirmed by the official Starter Kit (`starter_kit/data.py::LABEL`) — a clean, native 0/1 column needing no resolution. `pipeline/data/label.py::resolve_primary_label` is what training and evaluation use. `is_click` remains available as a candidate auxiliary multi-task signal (see below), never as the training target.
2. **`is_click` means two different things**, independent of the finding above — not documented in the public challenge brief, found by reading the raw KuaiRand field spec. In the two-column UI it's a genuine tap. In the single-column UI it's actually `valid_play`: `play_time_ms >= duration_ms` for videos under 7 seconds, or `play_time_ms > 7000ms` for longer ones, keyed on the `tab` field. `pipeline/data/label.py` resolves this explicitly (`profile_label`, `resolve_label`) so that if `is_click` is ever used as an auxiliary signal, it isn't a conflated one.
3. **`video_features_statistic_pure.csv`'s columns all leak**, and it's a *separate file* from `video_features_basic_pure.csv` (static, safe attributes like `author_id`) — the two must be merged in differently. They're month-long running averages spanning train, validation, and test. `pipeline/data/leakage_guard.py` drops every column of the statistic file by default (by construction, not by name pattern — see the module docstring for why a name-pattern approach doesn't work against the real file); re-enabling one requires an explicit, logged opt-in.

## File map

```
starter_kit/              the organizer's official Starter Kit, vendored unmodified —
                           canonical task def, FM baseline, evaluate.py, submit.py, baseline_scores.json

pipeline/                 the RecSys ML pipeline; what the agent edits
  data/
    download.py            dataset setup/verification instructions
    loader.py               date-pinned train/val/test split, matching starter_kit/data.py::SPLITS exactly
                             (test is guarded, never loaded unless explicitly allowed)
    label.py                 primary label (long_view) + tab-conditioned is_click resolution (auxiliary-only)
    leakage_guard.py          drops every video_features_statistic column by construction
    features.py                feature engineering, the agent's main edit surface; merges the two
                                 separate video-feature files (basic=safe, statistic=leaky) differently
  model/
    baseline.py               this pipeline's own starting model (small embedding+MLP CTR net) —
                               NOT the scored baseline, see starter_kit/baseline.py for that
    architectures/             where agent-proposed model variants would land
  train.py                     training entrypoint (used by orchestrator + smoke tests)
  evaluate.py                  loads starter_kit/evaluate.py by file path and delegates to it — one
                                 copy of the GAUC/nDCG@5 scoring logic in the whole repo
  smoke_test.py                fast sanity check before a patch is trusted
  submit.py                    writes the confirmed row_id,user_id,video_id,score submission CSV

agent/                    the autonomous agent
  llm_client.py             Anthropic API wrapper + token accounting
  orchestrator.py            main loop
  ablation.py                 block-level ablation
  code_editor.py               applies LLM-written patches with smoke-test rollback
  referee.py                    unbiased scoring via the random-exposure log
  compression_gate.py            overfit-rejection check before finalizing
  pitfall_store.py                structured failure/recovery log (feeds Tier-1 context)
  logger.py                        per-iteration run log, intervention counter, resource usage
  report.py                         generates docs/results_table.md from run logs
  skill_store/
    tier1_core.md                    always loaded: task framing + dataset quirks
    tier2_domain.md                   RecSys method priors, loaded when relevant
    tier3_deep/                        deep dives, loaded on demand
    retriever.py                        tiered retrieval logic

config/agent_config.yaml   all tunables and organizer-dependent values, isolated in one place
docs/                       results_table.md (generated), devpost_writeup.md (draft)
logs/                       generated at runtime: iterations.jsonl, interventions.jsonl, pitfalls.json, resource_usage.json
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # required, the agent makes real API calls, no offline mode

python -m pipeline.data.download --check   # tells you what to place where
# KuaiRand-Pure isn't behind a stable direct-download URL; get it from
# https://kuairand.com/ (or the Zenodo record linked from starter_kit/README.md)
# and place the CSVs listed by --check into config.dataset.raw_dir (default
# ./data/raw). Real data is ~200MB and gitignored (data/raw/*.csv) — each
# teammate downloads it locally, it's never committed to this repo.
```

The organizer's official Starter Kit is vendored, unmodified, at [`starter_kit/`](starter_kit/) — its `README.md` is the canonical task definition (label, split, metrics, submission format, baseline ladder). `pipeline/evaluate.py` doesn't reimplement its scoring; it loads `starter_kit/evaluate.py` directly by file path, so there's exactly one copy of the scoring logic in this repo. Run the official baseline standalone with:

```bash
cd starter_kit && python3 baseline.py --data_dir ../data/raw --model fm   # ~20s CPU, numpy-only
```

## Reproduction steps

| Command | Purpose |
|---|---|
| `python -m pipeline.train` | Sanity-check the pipeline's own model runs end-to-end |
| `python -m agent.orchestrator` | Run the full autonomous agent loop |
| `python -m agent.report` | Generate `docs/results_table.md` and a resource-usage summary from the run log |
| `python -m scripts.test_logic` | Non-ML logic checks (label resolution, leakage guard, metrics parity) against real data |

`config/agent_config.yaml` controls which Claude models run which role, iteration budget, convergence thresholds, and the referee mode toggle.

## Confirmed by the Starter Kit

Answered — no longer open questions:

1. **Candidate set**: within-user ranking over each user's logged impressions ("用户内排序"), not full-catalog retrieval. `config.starter_kit.candidate_set: "impressed_set"`.
2. **Convergence/budget**: ε=0.002, N=3, 50-iteration cap, 6h wall-clock ceiling — all set in `config.starter_kit` and enforced as real loop-breaking conditions in `agent/orchestrator.py`. No token budget specified.
3. **Official baseline, eval script, submission schema**: `starter_kit/baseline.py` (FM), `starter_kit/evaluate.py` (vendored, delegated to by `pipeline/evaluate.py`), and CSV `row_id,user_id,video_id,score` (implemented in `pipeline/submit.py`). FM's official validation score — the one Task Requirement #1 asks this pipeline to reproduce — is primary 0.6016 (`config.starter_kit.official_baseline.valid`, sourced from `starter_kit/baseline_scores.json`).

## Open questions

1. Is `log_random_4_22_to_5_08_pure.csv` permitted for training (`referee.mode: tier_a`), for propensity estimation/diagnostics only (`tier_b`, current default), or not at all (`disabled`)? Not addressed by the Starter Kit's own baseline (which never uses it), but it lists using this log as unbiased-validation as headroom item #7 — unclear whether that extends to training on it.
2. The Zenodo supplementary files (video captions, category taxonomy) aren't referenced anywhere in the Starter Kit — treat as out of scope unless organizers say otherwise. Still gated in `pipeline/data/download.py` if that changes.

## Limitations

- **Editable surface is narrow by design.** `agent/code_editor.py` only lets the agent rewrite `pipeline/data/features.py` and `pipeline/data/label.py`, not model architecture or the training loop directly. This was a deliberate scope cut to ship a safe, rollback-capable loop first; extending `EDITABLE_FILES` to `pipeline/model/architectures/` and `pipeline/train.py` is the natural next step (the smoke-test/rollback mechanism already generalizes), and is needed for the agent to act on the Starter Kit's own top headroom suggestion (a pairwise/listwise ranking loss, since the current loss lives in `pipeline/train.py`, not yet an editable file).
- **Referee's live per-iteration wiring is partial.** The scoring logic and divergence math (`agent/referee.py`) are implemented and tested in isolation, but `orchestrator.py` doesn't yet run inference over the random-exposure log every iteration; the integration point is marked explicitly in the code (`referee_note` in `orchestrator.py`).
- **No GPU-hour tracking beyond wall-clock.** `logger.py` reports wall-clock time as a proxy; real GPU-hour accounting (e.g. via `nvidia-smi` polling) isn't wired in — not that it matters much, since the Starter Kit's own reference pipeline needs no GPU at all.
- **This pipeline's own model (`pipeline/model/baseline.py`) is a starting point, not the scored baseline.** Per the challenge brief, "any starter pipeline the agent builds for itself is an internal step, not the reference it is scored against" — the reference is the vendored `starter_kit/baseline.py` (FM) and its numbers in `config.starter_kit.official_baseline`. This pipeline's own model exists so the agent has something to iterate on from iteration 0.

---

In one line: an ablation-guided agent that edits its own feature and label code against KuaiRand-Pure, checked against an unbiased random-exposure log and a fresh-context reproduction gate before any result counts as real.
