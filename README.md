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
| Self-editing code | Implemented | Real API calls, scoped to `pipeline/data/features.py`, `pipeline/data/label.py`, `pipeline/train.py`, `pipeline/model/baseline.py`, and `agent/ablation.py` itself, with automatic rollback on a failed smoke test and in-process module reloading so a scored patch is actually the code that gets scored (`agent/orchestrator.py::_reload_editable_modules`). |
| Autonomous ablation targeting | Implemented | The agent periodically rewrites its own ablation block-variant grid (`agent/orchestrator.py::_maybe_grow_ablation_grid`), not just a fixed human-seeded set — see [Limitations](#limitations) for what's still a fixed seed. |
| Unbiased referee | Implemented | Wired into the live per-iteration loop — every accepted patch's trained model is scored against a cached sample of the random-exposure log, with divergence alerts feeding the pitfall store (`agent/referee.py`, `agent/orchestrator.py::_referee_check`). |
| Crash recovery | Implemented | `agent/checkpoint.py` snapshots editable files + run state (iteration, best metrics, elapsed wall-clock, token usage) on every accepted patch; `orchestrator.py` resumes from the last checkpoint on startup instead of restarting from iteration 0. This also fixed a real correctness bug — a patch that scored worse or failed the compression gate previously stayed on disk unreverted, so later iterations silently trained against it while `best_metrics` drifted out of sync with the actual code. |
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

**Ablation-first refinement.** Instead of rewriting the whole pipeline each round, `agent/ablation.py` runs a cheap ablation over pipeline blocks (training schedule, learning rate, positive-class weighting) to find which one is moving the score, and `agent/orchestrator.py` targets that block for the next LLM-driven code change. `agent/ablation.py` is itself an editable file — every `agent.ablation_grid_growth_interval` iterations (config, default 5), the agent gets a chance to rewrite its own block-variant grid (`Orchestrator._maybe_grow_ablation_grid`, smoke tested by `agent/ablation_smoke_test.py`), so grid coverage isn't capped at whatever a human seeded it with. A variant that references a training knob that doesn't exist yet is skipped, not a crash (`run_ablation`'s per-variant exception handling) — the self-correction is a later patch teaching `pipeline/train.py` that knob. `agent/skill_store/` holds domain knowledge in three tiers: Tier 1 (always loaded, dataset-specific quirks), Tier 2 (RecSys method priors, loaded when relevant), Tier 3 (deep dives, loaded on demand via keyword match in `agent/skill_store/retriever.py`). This follows results from MLE-STAR and HASTE showing ablation-guided, tiered-knowledge search outperforms flat knowledge-dumping and whole-pipeline rewrites.

**Unbiased referee.** `log_random_4_22_to_5_08_pure.csv` contains ~1.19M interactions from uniformly random exposure, undocumented in the challenge brief. After every accepted patch's training run, `agent/orchestrator.py::_referee_check` scores that run's model against a cached sample of this log (`config.referee.probe_sample_size`, default 20k rows) and tracks the gap against standard validation; a widening gap signals the agent is fitting the biased proxy rather than improving generally, and gets recorded as a pitfall so it feeds the next iteration's reflect+revise prompt.

**Checkpointing.** `agent/checkpoint.py` snapshots every editable file plus run state (iteration count, best metrics, elapsed wall-clock, token usage) whenever a patch is accepted as the new best. Every other outcome (not a new best, or a new best that fails the compression gate) reverts the on-disk files to that snapshot — so the pipeline the next iteration's ablation and training run against always matches `best_metrics` exactly, and a crashed run can resume from the same snapshot on restart instead of losing the whole run.

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
  llm_client.py             Anthropic API wrapper + token accounting (TokenLedger, resumable)
  orchestrator.py            main loop — reload discipline, checkpointing, referee + grid-growth wiring
  checkpoint.py                editable-file + run-state snapshots; crash resume
  ablation.py                 block-level ablation; itself agent-editable (autonomous grid growth)
  ablation_smoke_test.py       dedicated smoke test for agent/ablation.py edits
  code_editor.py               applies LLM-written patches with smoke-test rollback
  referee.py                    unbiased scoring via the random-exposure log, wired into the live loop
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
logs/                       generated at runtime: iterations.jsonl, interventions.jsonl, pitfalls.json, resource_usage.json, checkpoint/ (crash-resume state)
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
| `python -m agent.orchestrator` | Run the full autonomous agent loop — resumes from `logging.checkpoint_dir` automatically if a previous run left one |
| `python -m agent.orchestrator --fresh` | Same, but discards any existing checkpoint first (see Limitations re: what `--fresh` doesn't reset) |
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

- **Ablation grid growth is still constrained by what `run_training` already accepts.** `agent/ablation.py`'s seed grid only probes `train.py` hyperparameters (epochs, learning rate, positive-class weight); the agent can add coverage for `features.py`/`label.py`/`model/baseline.py` via grid growth, but a new BlockVariant can only forward keyword arguments `run_training` already understands — a variant that invents a new one is skipped gracefully (not a crash) until a *separate* patch to `train.py` teaches it that keyword. Growing the grid and extending `train.py`'s knob surface aren't coordinated in one step; closing that gap across iterations is left to the agent's own reflect+revise loop, not guaranteed by construction.
- **`pipeline/model/architectures/` (new architecture variants as separate per-iteration files) isn't wired in.** `code_editor.py`'s backup/restore-one-path mechanism only supports rewriting a fixed existing path, not creating new files — an agent-proposed new architecture has to land inside the single `pipeline/model/baseline.py` file for now.
- **Tier-A referee mode (training directly on the random-exposure log) isn't implemented.** What's wired in (`agent/referee.py`, `Orchestrator._referee_check`) is Tier-B: diagnostic scoring against a cached probe sample, alerting on divergence. Using the random log as actual training data remains gated on organizer confirmation (see Open questions).
- **No GPU-hour tracking beyond wall-clock.** `logger.py` reports wall-clock time as a proxy; real GPU-hour accounting (e.g. via `nvidia-smi` polling) isn't wired in — not that it matters much, since the Starter Kit's own reference pipeline needs no GPU at all.
- **This pipeline's own model (`pipeline/model/baseline.py`) is a starting point, not the scored baseline.** Per the challenge brief, "any starter pipeline the agent builds for itself is an internal step, not the reference it is scored against" — the reference is the vendored `starter_kit/baseline.py` (FM) and its numbers in `config.starter_kit.official_baseline`. This pipeline's own model exists so the agent has something to iterate on from iteration 0.
- **`--fresh` only clears checkpoint bookkeeping, not the editable files themselves.** If a previous run left `pipeline/train.py` etc. mid-experiment and you want a truly clean slate (not just a fresh iteration counter), restore those files via git (e.g. `git checkout -- pipeline agent/ablation.py`) before starting.

---

In one line: an ablation-guided agent that edits its own feature, label, training, and model code (and its own ablation grid) against real KuaiRand-Pure data, checked against an unbiased random-exposure log and a fresh-context reproduction gate before any result counts as real, with crash-resumable checkpointing so a 6-hour run surviving to convergence doesn't depend on nothing going wrong.
