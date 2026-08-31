# Autonomous ML Research Agent - KuaiRand-Pure

An LLM agent that runs the ML engineering loop (read the problem, inspect data, engineer features, train, tune, evaluate, reflect, revise) on the KuaiRand-Pure recommendation dataset, aiming to beat the official FM baseline on GAUC and nDCG@5 without a human in the iteration loop.

Status tags used below: **Implemented** (exists in this repo, verifiable by the file cited next to it), **Placeholder** (a working stand-in, not the final version), **Planned** (designed or partially built, not yet wired into the live loop).

## Overview

Recommendation models like the one behind a short-video feed are built by repeatedly testing changes against held-out data. A common failure mode when automating this loop: the training data itself is biased, since it only contains items a prior recommender already chose to show users. An agent optimizing against that data can learn to exploit the bias instead of genuinely improving, and then underperform on unseen data.

Three things in this build address that:

- **Unbiased referee.** KuaiRand-Pure includes a slice of interactions collected under uniformly random exposure instead of the production recommender. The agent scores candidates against this random-exposure log as well as the standard validation split, and flags divergence between the two. The organizer Starter Kit (`starter_kit/README.md`) independently confirms this log's sanctioned use as exactly this: extra validation, never training.
- **Compression gate.** Before a checkpoint is finalized, the agent summarizes the approach and hands the summary to a fresh LLM context with no access to validation scores. If that second pass can't reproduce the result from the summary alone, the checkpoint is rejected.
- **Pre-seeded domain knowledge.** The agent starts with a tiered knowledge base about the dataset — including two undocumented data quirks found by reading the raw field spec and file manifest, and the organizer's own already-run feature-ablation findings (details below) — instead of rediscovering all of it through trial and error.

## Current state

The organizer's KuaiRand-Pure Starter Kit (vendored verbatim in [`starter_kit/`](starter_kit/)) is now in the repo, along with the real dataset (downloaded via the working Zenodo link — see [Setup](#setup)). Every value that was previously a placeholder pending the Starter Kit — metrics, label, official baseline, convergence rule, submission schema — is now confirmed and wired in.

| Piece | Status | Note |
|---|---|---|
| Official baseline reproduction | Implemented | `pipeline/official_baseline.py` runs the vendored FM baseline directly; verified locally to reproduce the published test primary (0.5946) within the 0.0008 seed std — see [Task Requirement #1](#reproducing-the-official-baseline). |
| End-to-end loop | Implemented | Scores through the vendored `starter_kit/evaluate.py` (GAUC/nDCG@5), not a reimplementation — see [Architecture](#architecture). Hasn't yet been run against a live `ANTHROPIC_API_KEY` in this environment; pipeline mechanics (training, smoke test, rollback, submission) are verified end-to-end against real data. |
| Self-editing code | Implemented | Real API calls, scoped to `pipeline/data/features.py` and `pipeline/data/label.py`, with automatic rollback on a failed smoke test. See [Limitations](#limitations). |
| Unbiased referee | Planned | Scoring, propensity estimation, and divergence tracking are implemented and tested in isolation, not yet wired into the live per-iteration loop. |
| Submission writer | Implemented | `pipeline/submit.py` writes the confirmed `row_id,user_id,video_id,score` schema via the vendored `starter_kit/submit.py`, with row-order alignment cross-checked against `pipeline/data/loader.py`. Verified end-to-end (write + official `--check`/`--score` validation) against real data. |

## Architecture

```
                    orchestrator.py loop
        read -> inspect -> engineer -> train -> evaluate -> reflect/revise -> ...

        Skill store              Ablation                Referee + gate
        (warm start)              (targeted search)        (unbiased scoring,
        tier1/2/3                 which block moves         overfit rejection)
                                   the score?
```

**Ablation-first refinement.** Instead of rewriting the whole pipeline each round, `agent/ablation.py` runs a cheap ablation over pipeline blocks (training schedule, learning rate, and whatever new blocks the agent introduces — pairwise loss, sequence features, multi-task heads) to find which one is moving the score, and `agent/orchestrator.py` targets that block for the next LLM-driven code change. `agent/skill_store/` holds domain knowledge in three tiers: Tier 1 (always loaded, dataset-specific quirks + the organizer's own already-run findings), Tier 2 (RecSys method priors, loaded when relevant), Tier 3 (deep dives, loaded on demand via keyword match in `agent/skill_store/retriever.py`). This follows results from MLE-STAR and HASTE showing ablation-guided, tiered-knowledge search outperforms flat knowledge-dumping and whole-pipeline rewrites.

**Official baseline reproduction.** Every orchestrator run starts by training the organizer's own vendored FM baseline (`pipeline/official_baseline.py`, calling `starter_kit/baseline.py` directly rather than reimplementing it) and confirming it lands within tolerance of the published validation score (0.6016) before any LLM-driven iteration begins — Task Requirement #1. The test-split number is computed too (the vendored function returns both) but is logged for evidence only and never surfaced to the reflect/iterate prompts, since the agent's decisions must never be informed by hidden-test scores.

**Faithful scoring.** `pipeline/evaluate.py` doesn't reimplement GAUC/nDCG@5 — it imports and calls `starter_kit/evaluate.py` directly, so there's no risk of a validation-time number quietly diverging from what the organizer's script would actually report on the hidden test set.

**Unbiased referee.** `log_random_4_22_to_5_08_pure.csv` contains ~1.19M interactions from uniformly random exposure, undocumented in the challenge brief but since confirmed by the Starter Kit README as a sanctioned extra-validation-only signal. `agent/referee.py` scores candidates against this log alongside standard validation and tracks the gap between the two; a widening gap signals the agent is fitting the biased proxy rather than improving generally.

**Compression gate.** `agent/compression_gate.py` compresses a winning approach into a short summary and hands it to a fresh LLM context with no memory of the search and no access to validation scores. If that reproducer can't get behind the approach, the checkpoint is rejected and the previous best is kept. This targets a known failure mode where agents pick a validation-overfit artifact over a genuinely weaker-looking but real solution.

## Dataset findings

Findings 1-2 came from reading the raw KuaiRand-Pure field spec and file manifest, before the Starter Kit was available. Findings 3-4 came from cross-checking that reading against the real, vendored Starter Kit and real data once both arrived.

1. **`is_click` means two different things.** In the two-column UI it's a genuine tap. In the single-column UI it's actually `valid_play`: `play_time_ms >= duration_ms` for videos under 7 seconds, or `play_time_ms > 7000ms` for longer ones, keyed on the `tab` field. `pipeline/data/label.py` resolves this explicitly (`profile_label`, `resolve_auxiliary_click_label`) instead of training on a conflated signal — relevant only to the multi-task auxiliary head, since the primary label is `long_view`, not `is_click` (see below). On real data the derivation holds ~97.2% of the time, not exactly 1.0.
2. **`video_features_statistic` columns leak**, and on real data they do **not** follow a `_statistic` naming convention. The actual header (`show_cnt`, `play_cnt`, `like_cnt`, `follow_cnt`, ...) doesn't self-identify at all — a pure substring check would silently let every one of them through. `pipeline/data/leakage_guard.py` now flags them by an exact name list built from the real file's header, with the substring check kept only as a fallback for naming conventions that do self-identify. Re-enabling one requires an explicit, logged opt-in.
3. **`play_time_ms` is the label in disguise, not a feature.** The primary label is `long_view` (confirmed by the Starter Kit), and on real data a bare threshold on `play_time_ms/duration_ms` predicts it at 84.7% accuracy (corr=0.64) — matching why the official baseline's own field list uses `duration_ms` (video length) but never `play_time_ms` (watch time, i.e. the outcome). `pipeline/data/features.py` excludes it from `NUMERIC_SIGNAL_COLUMNS` for this reason; an earlier version of this pipeline (built before the Starter Kit confirmed `long_view` as the label) had this backwards.
4. **The organizer already ran the "add more features" and "grow model capacity" experiments** (`starter_kit/ablation_features.py`) and found both flat (~0 gain, sometimes slightly negative) — because ranking is within-user, so anything constant within a user (most user-side features) can't move that user's order. This is now Tier-1 knowledge (`agent/skill_store/tier1_core.md`) specifically so the agent doesn't re-spend early iterations rediscovering it.

## File map

```
starter_kit/               organizer-provided Starter Kit, vendored verbatim (evaluate.py: do not modify)
  evaluate.py                 GAUC / nDCG@5, the actual scoring contract
  data.py                      official split + 5-field encoding
  baseline.py                   the official FM baseline (and pop/random references)
  baseline_scores.json           published numbers, seed std, convergence params
  submit.py                       official submission writer/validator
  ablation_features.py             organizer's own "does more features help" experiment (they don't)
  KuaiRand-Pure/data/               real dataset (downloaded, gitignored — see Setup)

pipeline/                 the RecSys ML pipeline; what the agent edits
  data/
    download.py            dataset fetch (--fetch, working direct URL) / verification (--check)
    loader.py               date-pinned train/val/test split (test is guarded, never loaded unless explicitly allowed)
    label.py                 primary label (long_view) + auxiliary is_click resolution
    leakage_guard.py          drops video_features_statistic by exact real-column-name list
    features.py                feature engineering, the agent's main edit surface
  model/
    baseline.py               the agent's OWN editable model (small embedding+MLP), not the organizer baseline
    architectures/             where agent-proposed model variants would land
  official_baseline.py         reproduces the organizer's FM baseline via starter_kit/, for Task Requirement #1
  train.py                     training entrypoint (used by orchestrator + smoke tests)
  evaluate.py                  GAUC / nDCG@5, delegated to starter_kit/evaluate.py (not reimplemented)
  smoke_test.py                fast sanity check before a patch is trusted
  submit.py                    final submission writer, matches the official row_id,user_id,video_id,score schema

agent/                    the autonomous agent
  llm_client.py             Anthropic API wrapper + token accounting
  orchestrator.py            main loop; reproduces the official baseline once, then iterates
  ablation.py                 block-level ablation
  code_editor.py               applies LLM-written patches with smoke-test rollback
  referee.py                    unbiased scoring via the random-exposure log
  compression_gate.py            overfit-rejection check before finalizing
  pitfall_store.py                structured failure/recovery log (feeds Tier-1 context)
  logger.py                        per-iteration run log, intervention counter, resource usage
  report.py                         generates docs/results_table.md from run logs
  skill_store/
    tier1_core.md                    always loaded: task framing, convergence rule, dataset quirks,
                                       organizer's already-tried findings, priority-ranked headroom list
    tier2_domain.md                   RecSys method priors, loaded when relevant
    tier3_deep/                        deep dives, loaded on demand
    retriever.py                        tiered retrieval logic

config/agent_config.yaml   all tunables, confirmed against the Starter Kit (see below)
docs/                       results_table.md (generated), devpost_writeup.md (draft)
logs/                       generated at runtime: iterations.jsonl, interventions.jsonl, pitfalls.json, resource_usage.json
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # required, the agent makes real API calls, no offline mode

python -m pipeline.data.download --fetch   # downloads + unpacks the real KuaiRand-Pure dataset (~47MB, no registration)
python -m pipeline.data.download --check   # verifies everything required is present in ./data/raw
```

## Reproducing the official baseline

Task Requirement #1 ("confirm the pipeline reaches the official baseline's reported validation score") is checked directly:

```bash
python -m pipeline.official_baseline
```

This trains the vendored `starter_kit/baseline.py` FM (k=16, lr=0.001) on real data and compares against `starter_kit/baseline_scores.json`. Verified locally: valid primary 0.6015 vs published 0.6016, test primary 0.5953 vs published 0.5946 (well within the published 0.0008 seed std).

## Reproduction steps

| Command | Purpose |
|---|---|
| `python -m pipeline.official_baseline` | Confirm the official baseline reproduces (Task Requirement #1) |
| `python -m pipeline.train` | Sanity-check the agent's own editable pipeline (not the official baseline) |
| `python -m agent.orchestrator` | Run the full autonomous agent loop — reproduces the baseline once, then iterates |
| `python -m agent.report` | Generate `docs/results_table.md` and a resource-usage summary from the run log |
| `python -m scripts.test_logic` / `python -m scripts.test_code_editor` | Regression tests for label/leakage/metric logic and the patch-rollback safety mechanism |

`config/agent_config.yaml` controls which Claude models run which role, iteration budget, convergence thresholds, and the referee mode toggle. Every value that previously depended on the organizer's Starter Kit is now confirmed and cited against `starter_kit/baseline_scores.json` / `starter_kit/README.md` directly in the config comments.

## Resolved (previously "Open questions")

All of these were unknowns pending the Starter Kit; all are now confirmed and wired in.

1. `log_random_4_22_to_5_08_pure.csv` is confirmed sanctioned for extra validation only, never training (`referee.mode: tier_b`, the default and only sanctioned setting — `tier_a` is kept in code as a documented, non-default option).
2. Candidate set: each user's impressed set within the eval split ("within-user ranking over logged impressions") — matches this pipeline's existing default.
3. Convergence rule: ε=0.002, N=3, 50-iteration cap, 6h wall-clock ceiling — all wired into `config.starter_kit` and `agent/orchestrator.py`.
4. Official baseline, eval script, and submission schema: all vendored in `starter_kit/` and wired into `pipeline/official_baseline.py`, `pipeline/evaluate.py`, `pipeline/submit.py` respectively.

**Not resolved / out of scope:** whether the Zenodo supplementary files (video captions, category taxonomy) are usable — the official problem statement and Starter Kit never mention them, so they're treated as out of scope rather than pursued further.

## Limitations

- **Editable surface is narrow by design.** `agent/code_editor.py` only lets the agent rewrite `pipeline/data/features.py` and `pipeline/data/label.py`, not model architecture or the training loop directly. This was a deliberate scope cut to ship a safe, rollback-capable loop first; extending `EDITABLE_FILES` to `pipeline/model/architectures/` and `pipeline/train.py` (loss function, in particular — the organizer's own #1-ranked lead is a pairwise/listwise loss) is the natural next step, and the highest-value one given the Starter Kit's priority list.
- **Referee's live per-iteration wiring is partial.** The scoring logic and divergence math (`agent/referee.py`) are implemented and tested in isolation, but `orchestrator.py` doesn't yet run inference over the random-exposure log every iteration; the integration point is marked explicitly in the code (`referee_note` in `orchestrator.py`).
- **No GPU-hour tracking beyond wall-clock.** `logger.py` reports wall-clock time as a proxy; real GPU-hour accounting (e.g. via `nvidia-smi` polling) isn't wired in — moot for CPU-only runs, which this pipeline is by default.
- **The live orchestrator loop hasn't been run against a real `ANTHROPIC_API_KEY` in this environment.** Every non-LLM component (data loading, real-data training, GAUC/nDCG@5 scoring via the vendored evaluator, patch rollback, submission writing + validation) is verified end-to-end against real KuaiRand-Pure data; the LLM-driven reflect/iterate loop itself still needs a real run to produce actual run logs, hypotheses, and a final scored delta.

---

In one line: an ablation-guided agent that edits its own feature and label code against real KuaiRand-Pure, scored through the organizer's own vendored evaluator, checked against an unbiased random-exposure log and a fresh-context reproduction gate before any result counts as real.
