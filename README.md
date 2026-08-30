# Autonomous ML Research Agent - KuaiRand-Pure

An LLM agent that runs the ML engineering loop (read the problem, inspect data, engineer features, train, tune, evaluate, reflect, revise) on the KuaiRand-Pure recommendation dataset, aiming to beat a baseline on NDCG@10 and Recall@50 without a human in the iteration loop.

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
| End-to-end loop | Implemented | Runs against a placeholder baseline model, not the official organizer one; Starter Kit isn't available yet. See [Open questions](#open-questions). |
| Self-editing code | Implemented | Real API calls, scoped to `pipeline/data/features.py` and `pipeline/data/label.py`, with automatic rollback on a failed smoke test. See [Limitations](#limitations). |
| Unbiased referee | Planned | Scoring, propensity estimation, and divergence tracking are implemented and tested in isolation, not yet wired into the live per-iteration loop. |
| Submission writer | Implemented (guarded) | `pipeline/submit.py` raises rather than guessing a schema, until the organizer's format is known. |

## Architecture

```
                    orchestrator.py loop
        read -> inspect -> engineer -> train -> evaluate -> reflect/revise -> ...

        Skill store              Ablation                Referee + gate
        (warm start)              (targeted search)        (unbiased scoring,
        tier1/2/3                 which block moves         overfit rejection)
                                   the score?
```

**Ablation-first refinement.** Instead of rewriting the whole pipeline each round, `agent/ablation.py` runs a cheap ablation over pipeline blocks (label resolution, training schedule, learning rate) to find which one is moving the score, and `agent/orchestrator.py` targets that block for the next LLM-driven code change. `agent/skill_store/` holds domain knowledge in three tiers: Tier 1 (always loaded, dataset-specific quirks), Tier 2 (RecSys method priors, loaded when relevant), Tier 3 (deep dives, loaded on demand via keyword match in `agent/skill_store/retriever.py`). This follows results from MLE-STAR and HASTE showing ablation-guided, tiered-knowledge search outperforms flat knowledge-dumping and whole-pipeline rewrites.

**Unbiased referee.** `log_random_4_22_to_5_08_pure.csv` contains ~1.19M interactions from uniformly random exposure, undocumented in the challenge brief. `agent/referee.py` scores candidates against this log alongside standard validation and tracks the gap between the two; a widening gap signals the agent is fitting the biased proxy rather than improving generally.

**Compression gate.** `agent/compression_gate.py` compresses a winning approach into a short summary and hands it to a fresh LLM context with no memory of the search and no access to validation scores. If that reproducer can't get behind the approach, the checkpoint is rejected and the previous best is kept. This targets a known failure mode where agents pick a validation-overfit artifact over a genuinely weaker-looking but real solution.

## Dataset findings

Neither of these is documented in the public challenge brief; both came from reading the raw KuaiRand-Pure field spec and file manifest.

1. **`is_click` means two different things.** In the two-column UI it's a genuine tap. In the single-column UI it's actually `valid_play`: `play_time_ms >= duration_ms` for videos under 7 seconds, or `play_time_ms > 7000ms` for longer ones, keyed on the `tab` field. `pipeline/data/label.py` resolves this explicitly (`profile_label`, `resolve_label`) instead of training on a conflated signal.
2. **`video_features_statistic` columns leak.** They're month-long running averages spanning train, validation, and test. `pipeline/data/leakage_guard.py` drops them by default; re-enabling one requires an explicit, logged opt-in.

## File map

```
pipeline/                 the RecSys ML pipeline; what the agent edits
  data/
    download.py            dataset setup/verification instructions
    loader.py               date-pinned train/val/test split (test is guarded, never loaded unless explicitly allowed)
    label.py                 tab-conditioned is_click resolution
    leakage_guard.py          drops video_features_statistic by default
    features.py                feature engineering, the agent's main edit surface
  model/
    baseline.py               placeholder reference model (small embedding+MLP CTR net)
    architectures/             where agent-proposed model variants would land
  train.py                     training entrypoint (used by orchestrator + smoke tests)
  evaluate.py                  NDCG@10 / Recall@50, computed per-user
  smoke_test.py                fast sanity check before a patch is trusted
  submit.py                    final submission writer, raises until schema is known

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
# https://kuairand.com/ and place the CSVs listed by --check into
# config.dataset.raw_dir (default ./data/raw).
```

## Reproduction steps

| Command | Purpose |
|---|---|
| `python -m pipeline.train` | Sanity-check the pipeline runs against the placeholder baseline |
| `python -m agent.orchestrator` | Run the full autonomous agent loop |
| `python -m agent.report` | Generate `docs/results_table.md` and a resource-usage summary from the run log |

`config/agent_config.yaml` controls which Claude models run which role, iteration budget, convergence thresholds, and the referee mode toggle. Every value that depends on the organizer's Starter Kit is called out there explicitly.

## Open questions

Facts only the challenge organizers can supply. The codebase isolates each one behind a config value or an explicit exception rather than guessing.

1. Is `log_random_4_22_to_5_08_pure.csv` permitted for training (`referee.mode: tier_a`), for propensity estimation/diagnostics only (`tier_b`, current default), or not at all (`disabled`)?
2. Are the Zenodo supplementary files (video captions, category taxonomy) in scope? Gated in `pipeline/data/download.py`.
3. What is the candidate set at scoring time: each user's impressed set, or the full ~7,583-item catalog? Changes what NDCG@10/Recall@50 mean; `pipeline/evaluate.py` currently ranks over whatever's in the input DataFrame (defaults to impressed-set, computable without organizer input).
4. What are epsilon and N (convergence rule) and the compute budget? Placeholders live in `config.starter_kit`.
5. The actual organizer baseline, eval script, and submission schema. Until these land, `pipeline/model/baseline.py` is a stand-in and `pipeline/submit.py` raises rather than guessing a format.

## Limitations

- **Editable surface is narrow by design.** `agent/code_editor.py` only lets the agent rewrite `pipeline/data/features.py` and `pipeline/data/label.py`, not model architecture or the training loop directly. This was a deliberate scope cut to ship a safe, rollback-capable loop first; extending `EDITABLE_FILES` to `pipeline/model/architectures/` is the natural next step (the smoke-test/rollback mechanism already generalizes).
- **Referee's live per-iteration wiring is partial.** The scoring logic and divergence math (`agent/referee.py`) are implemented and tested in isolation, but `orchestrator.py` doesn't yet run inference over the random-exposure log every iteration; the integration point is marked explicitly in the code (`referee_note` in `orchestrator.py`).
- **No GPU-hour tracking beyond wall-clock.** `logger.py` reports wall-clock time as a proxy; real GPU-hour accounting (e.g. via `nvidia-smi` polling) isn't wired in.
- **Baseline is a placeholder.** Every "beats baseline" claim from this codebase today is against a small reference model, not the organizer's official one.

---

In one line: an ablation-guided agent that edits its own feature and label code against KuaiRand-Pure, checked against an unbiased random-exposure log and a fresh-context reproduction gate before any result counts as real.
