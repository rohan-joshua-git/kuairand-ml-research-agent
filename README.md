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
| End-to-end loop | Implemented | Scores through the vendored `starter_kit/evaluate.py` (GAUC/nDCG@5), not a reimplementation — see [Architecture](#architecture). Run live against the Gemini backend; every scored run happens in a fresh subprocess (`pipeline/train_runner.py`) so the code being measured is exactly what the last patch wrote. |
| Self-editing code | Implemented | Real API calls with automatic rollback on a failed smoke test — proven both for a syntactically broken patch and for one that imports cleanly then crashes at run time (`scripts/test_code_editor.py`). `EDITABLE_FILES` allows `features.py`, `label.py`, `train.py`, `model/baseline.py` and `ablation.py`; note the hypothesis router (`route_target_file`) currently only ever selects the first three, so the last two are reachable config rather than surface the agent actually edits today. |
| Unbiased referee | Implemented | Wired into every scored iteration: the trained model scores a validation-window slice of the random-exposure log, and the biased-vs-unbiased divergence is logged per iteration with an alert threshold. |
| Final submission | Implemented | `pipeline/make_submission.py` is the only place `allow_test=True` is set. Supports `--ensemble-seeds N`, which rank-averages N seeds (rank, not score — only within-user order is scored). Measured: 1 seed 0.6017, 5 seeds 0.6028. |
| Submission writer | Implemented | `pipeline/submit.py` writes the confirmed `row_id,user_id,video_id,score` schema via the vendored `starter_kit/submit.py`, with row-order alignment cross-checked against `pipeline/data/loader.py`. Verified end-to-end (write + official `--check`/`--score` validation) against real data. |

## Results (measured, not projected)

**Submitted result: validation primary 0.6044 vs the official baseline's
0.6016 — a delta of +0.0028, with both metrics up** (GAUC 0.6674 -> 0.6710,
nDCG@5 0.5357 -> 0.5379). Scored by the organizer's own
`starter_kit/submit.py --score`, not by our code.

**The agent produced the largest single improvement, autonomously.** Its
DeepFM-lite hypothesis (a parallel [32, 16] MLP branch alongside the FM's
linear and pairwise terms) scored 0.6045, cleared ε=0.002 against its own
best, passed the compression gate, and was checkpointed. It proposed this
*after* its larger DeepFM patch crashed the smoke test and was rolled back
automatically.

We still separate human from agent contributions below, because the autonomy
criterion is scored on what the agent did rather than on the final number.

| Step | Valid primary | Author |
|---|---|---|
| Official FM baseline (published) | 0.6016 | organizer |
| Our editable pipeline, torch FM over the official 5 fields | 0.6017 | human |
| + `pos_bucket` session-position feature | 0.6024 | human |
| **\+ agent's DeepFM-lite MLP branch (accepted, gate-passed)** | **0.6045** | **agent** |
| + 5-seed rank-averaged ensemble (submitted) | **0.6044** | human |

The ensemble is **neutral on the agent's architecture** (0.6044 against a
single-seed range of 0.6040–0.6047) even though it was worth +0.0012 on the
plain FM (0.6024 -> 0.6036). We kept it for variance reduction on unseen data,
not because it measured better here — reporting it as a win would be
overclaiming.

Run 1 (from a weaker starting pipeline) and run 2 both converged without an
accepted improvement; run 3 is the one that found the MLP branch. All three
trajectories are in `logs/`, not just the successful one.


Run of 2026-08-31, Gemini backend, scored by the vendored
`starter_kit/evaluate.py` on the validation split:

| Iteration | Change | GAUC | nDCG@5 | Primary | vs prev best |
|---|---|---|---|---|---|
| 0 | Official FM baseline, reproduced via `starter_kit/baseline.py` | 0.6671 | 0.5358 | 0.6015 | — |
| 0 | **Our starting pipeline (torch FM, human-authored)** | **0.6676** | **0.5358** | **0.6017** | — |
| 1 | Agent: per-user pairwise BPR loss | 0.6646 | 0.5341 | 0.5994 | −0.0024 |
| 2 | Agent: multi-task auxiliary `is_click` head (weight 0.2) | 0.6670 | 0.5356 | 0.6013 | −0.0004 |
| 3 | Agent: listwise ListNet cross-entropy loss | 0.6655 | 0.5353 | 0.6004 | −0.0013 |

Every non-improving patch was rolled back automatically, so the working tree
always holds the best-known state — which is what the submission step
re-trains from. Note iteration 6 scored 0.6050, *higher* than the submitted
0.6045 checkpoint, but only +0.0005 over the accepted best: below ε, so it was
rejected and rolled back. The results table flags it explicitly rather than
quietly reporting the higher number.

Reproduce the submission with:

```bash
python -m pipeline.make_submission --out-dir submissions --ensemble-seeds 5
```

Cost: 38,361 tokens, 0.118 wall-clock hours, 0 GPU-hours (CPU-only).
**Manual interventions: 1** — see [Autonomy accounting](#autonomy-accounting).

### What the agent got right, and what it cost us

The agent independently proposed **switching the loss to a ranking objective
on its first iteration, in three separate runs** — which is the organizer's
own #1-ranked lead in `starter_kit/README.md`. It then progressed pointwise ->
pairwise -> listwise across iterations, reading its own failure history rather
than repeating itself. The search behaviour is sound.

The finding is that the lead did not pay off here. Pairwise BPR, a listwise
ListNet objective, and a multi-task auxiliary head all landed 0.0004–0.0024
*below* pointwise logloss on this data. Three independent implementations
moving the same direction is weak evidence that the ceiling on this dataset is
not objective misalignment.

Two things we checked so we would not misattribute the plateau:
- **Not undertraining.** Raising the epoch cap from 12 to 40 with patience 4
  early-stops at 13 epochs and returns the identical 0.6017.
- **Not the known dead ends.** Adding features and growing embedding capacity
  were already measured flat by the organizer
  (`starter_kit/ablation_features.py`), which is why they are Tier-1 knowledge
  the agent is told not to re-test.

### Autonomy accounting

- **Manual interventions: 1.** Mid-run we changed `agent/llm_client.py`'s
  failover policy (a daily-quota 429 now fails over immediately instead of
  walking a 15/30/60/120s backoff ladder) and relaunched. That is a change to
  the agent's *behaviour*, so it counts. Recorded with its reasoning in
  `logs/interventions.jsonl`.
- **Process restarts after a crash: not counted, and here is why.** In the
  Track 2 workshop Q&A (2026-08-31) the organizer was asked directly and
  answered that restarting a crashed process is not a manual intervention —
  "we only consider the manual intervention if you change the agent's
  behaviour" — and suggested a second session do the restarting.
  `agent/supervisor.py` is that, automated: it re-executes an identical
  command and touches no code, config, or checkpoint selection, while the
  orchestrator resumes from `agent/checkpoint.py` with its wall-clock budget
  already charged. Restarts are logged separately in `logs/restarts.jsonl`.

### Known weakness in our own instrumentation

The unbiased-referee divergence alert fired on **every** iteration (+0.19 to
+0.24 against a 0.05 threshold) and therefore carried no information. The
random-exposure probe has a structurally different label distribution than the
biased split (probe primary ~0.36 vs biased ~0.60), so the absolute gap is
large by construction. The informative signal is the *change* in divergence
across iterations, not its level; the threshold should be set on that instead.
We are reporting this rather than presenting the referee as having caught
something.

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
# The agent makes real API calls — there is no offline/mock mode. Set the key
# for whichever provider config/agent_config.yaml selects (default: gemini).
export GEMINI_API_KEY=your_key_here      # default provider, free tier
# export ANTHROPIC_API_KEY=your_key_here # if agent.llm.provider is "anthropic"

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

- **New-file creation is out of scope for the editor.** `agent/code_editor.py` rewrites whole files at fixed paths, so the agent can change the loss, the model and its own ablation grid, but cannot add a new module under `pipeline/model/architectures/`. That needs a create-a-file flow the current backup/restore-one-path mechanism does not support.
- **The agent's feature hypotheses have a narrow path into the model, and we
  measured this the hard way.** Across two runs, four separate `features.py`
  patches adding a video-quality prior all returned *bit-identical* scores
  (0.6024 to four decimals). Different patches cannot produce identical
  metrics unless the new column is never used — and it wasn't. Originally the
  encoded field list lived in `train.py`, a different file, and a patch is one
  whole-file rewrite, so the agent could not close the loop. We added an
  `EXTRA_CATEGORICAL_FIELDS` registry in `features.py` (verified: the same
  hypothesis then moves the score to 0.6027). A second constraint remains: the
  FM consumes **categorical fields only**, so a continuous feature — like the
  out-of-fold target encoding the agent later proposed — still has no path in
  unless it is bucketed. We stopped intervening at that point deliberately:
  the categorical version of that exact idea measured +0.0003, below the
  0.0008 seed-noise floor, so adding a numeric path would have cost an
  intervention to buy nothing measurable.
- **Cross-file changes cost an iteration.** Each patch is one whole-file rewrite, so a hypothesis needing a model constructor change *and* a matching call-site change in `train.py` must either fit in one file or take two iterations. A broken interface is caught by the smoke test and rolled back — it costs an iteration, it does not corrupt state.
- **GPU-hours are reported as 0.0 because the pipeline is CPU-only.** Wall-clock is reported separately and honestly; there is no `nvidia-smi` accounting because there is no GPU in the loop.
- **The compression gate reasons about a summary, it does not re-train from it.** A stronger version would have the fresh reproducer actually re-run training from the compressed description and compare scores; as built, it is an LLM judgement over a deliberately terse summary with no access to validation scores.

---

In one line: an ablation-guided agent that edits its own feature and label code against real KuaiRand-Pure, scored through the organizer's own vendored evaluator, checked against an unbiased random-exposure log and a fresh-context reproduction gate before any result counts as real.
