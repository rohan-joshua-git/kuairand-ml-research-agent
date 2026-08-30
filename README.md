# Autonomous ML Research Agent — KuaiRand-Pure

An LLM agent that runs the machine-learning-engineer iteration loop —
read the problem, inspect the data, engineer features, train and tune,
evaluate, reflect and revise — on its own, on the KuaiRand-Pure
recommendation dataset, and tries to beat a baseline on NDCG@10 and
Recall@50 without a human doing the modeling work.

This README explains what's built twice: once in plain terms, once with
the technical detail. Skip to whichever you need.

---

## 1. What this is — in plain terms

Video apps like Kuaishou decide what clip to show you next using a
recommendation model. Someone — a machine learning engineer — built and
keeps improving that model by repeating a loop: look at the data, try
something, measure whether it helped, try something else.

This project builds a robot that does that job by itself. You point it at
a dataset and two scores to optimize, and it writes its own code, trains
its own models, checks its own results, and decides what to try next —
without a person in the loop for the day-to-day iteration.

The interesting part isn't just "make a robot that tries things." It's
that most robots built this way fall into a specific trap: they practice
against a rigged practice test (because the training data only contains
videos a previous recommender already chose to show people) and end up
gaming that practice test instead of getting genuinely better. When
they're finally graded on a real, unseen exam, they can disappoint badly.

Three things in this build exist specifically to avoid that:

- **An honest referee.** Buried in the dataset is a small slice of data
  collected by showing people *completely random* videos instead of
  recommended ones. That's a fair sample — auditions open to everyone, not
  just the acts the old show already booked. The agent checks its work
  against that fair sample too, not just the rigged one, and pays
  attention when the two disagree.
- **A "explain it to a stranger" check.** Before the agent finalizes an
  answer, it has to summarize what it learned in a short note and hand
  that note to a second, fresh copy of itself that's never seen the
  practice test. If the second copy can reproduce the result from the
  note, the discovery was real. If it can't, the first copy was probably
  faking it — and the answer gets rejected.
- **A briefing before it starts.** The agent doesn't go in cold. It's
  handed a short, organized packet of what's already known about this
  specific dataset — including two data quirks we found by digging into
  the file structure ourselves (details in the technical section) that
  aren't mentioned anywhere in the public challenge description.

## 1a. What's actually implemented right now

Being direct about where this stands: this is a working, runnable
scaffold, not a finished, benchmarked agent. Concretely:

- The full loop runs end-to-end against a **placeholder baseline** (a
  small model we wrote), not yet the official organizer baseline — that
  requires the organizer's Starter Kit, which we don't have access to yet
  (see [Open Questions](#4-open-questions--day-1-blockers)).
- The agent genuinely writes and applies its own code each iteration (via
  real Claude API calls) to a defined, safe surface (feature engineering
  and label logic), with automatic rollback if a change breaks. It does
  not yet freely rewrite arbitrary files across the whole pipeline — see
  [Limitations](#6-limitations--whats-not-done).
- The "unbiased referee" (Play 1) has the data-loading, propensity
  estimation, and divergence-tracking logic built, but is not yet fully
  wired into the live per-iteration scoring loop — see Limitations.
- Final submission generation intentionally **raises an error** rather
  than guessing a schema, until the organizer's actual submission format
  is known.

## 2. What this is — technical

### Architecture: three layers, one loop

```
                    ┌─────────────────────────────────────┐
                    │         orchestrator.py loop         │
                    │  read → inspect → engineer → train   │
                    │  → evaluate → reflect+revise → …      │
                    └───────────────┬───────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                             │                             │
┌───────▼────────┐          ┌─────────▼─────────┐         ┌─────────▼─────────┐
│  Skill store    │          │   Ablation          │         │  Referee +          │
│  (warm start)   │          │  (targeted search)   │         │  compression gate    │
│  tier1/2/3      │          │  which block is       │         │  Play 1               │
│  Play 2/3 knowledge│         │  carrying the score?  │         │  unbiased scoring +    │
└─────────────────┘          └───────────────────────┘         │  overfit rejection      │
                                                                  └───────────────────────┘
```

- **Layer 1 — Warm-started, ablation-first refinement (Play 2, the
  chassis).** Instead of rewriting the whole pipeline each round,
  `agent/ablation.py` runs a cheap ablation over pipeline blocks
  (label resolution, training schedule, learning rate, ...) each
  iteration to find which one is actually moving the score, and
  `agent/orchestrator.py` targets only that block for the next (expensive)
  LLM-driven code change. `agent/skill_store/` pre-seeds domain knowledge
  in three tiers so the agent isn't rediscovering RecSys basics from
  scratch every run — Tier 1 (always loaded, KuaiRand-specific quirks),
  Tier 2 (RecSys architecture/method priors, loaded when relevant), Tier 3
  (deep dives, loaded on demand via keyword match in
  `agent/skill_store/retriever.py`). This mirrors published results
  (MLE-STAR, HASTE) showing ablation-guided, tiered-knowledge search
  beats both flat knowledge-dumping and whole-pipeline rewrites, and cuts
  iterations-to-best roughly in half.

- **Layer 2 — Unbiased referee (Play 1).** KuaiRand-Pure ships a file,
  `log_random_4_22_to_5_08_pure.csv`, that the challenge brief doesn't
  mention: ~1.19M interactions collected by uniformly-random exposure
  instead of the normal recommendation policy. It's in-dataset (not
  external data) and gives a genuinely unbiased evaluation channel,
  unlike the standard validation split — which was already filtered by
  whatever recommender was running at the time it was logged.
  `agent/referee.py` scores candidates against this probe alongside
  standard validation and tracks the divergence between the two; a
  widening gap is the signal that the agent is fitting the biased proxy
  rather than genuinely improving, and should trigger course-correction.

- **Layer 3 — Compression gate.** Before any checkpoint is designated
  final, `agent/compression_gate.py` compresses the winning approach into
  a short, honest summary and hands it to a *second, fresh* LLM context
  with no memory of the search that found it and no access to validation
  scores. If that fresh "reproducer" can't get behind the approach as
  genuine and reproducible, the checkpoint is rejected and the previous
  best is kept. This targets a documented, real failure mode: agents that
  pick a validation-overfit artifact (in one published case, a lookup
  table scoring 97% on validation and 0% on held-out data) over a
  genuinely weaker-looking but real solution, because validation score was
  the only thing being searched against.

### Two dataset findings baked into the pipeline

Neither of these is mentioned in the public challenge brief — both came
from reading the actual KuaiRand-Pure field spec and file manifest:

1. **`is_click` is two different things.** In the two-column UI it's a
   genuine tap. In the single-column UI it's actually `valid_play`:
   `play_time_ms >= duration_ms` for videos under 7 seconds, or
   `play_time_ms > 7000ms` for longer ones — keyed on the `tab` field.
   `pipeline/data/label.py` makes this explicit (`profile_label`,
   `resolve_label`) instead of silently training on a conflated signal.
2. **`video_features_statistic` columns leak.** They're month-long
   running averages that span train, validation, and test.
   `pipeline/data/leakage_guard.py` drops them by default; re-enabling one
   requires an explicit, logged opt-in.

### File map

```
pipeline/                 the RecSys ML pipeline — what the agent edits
  data/
    download.py             dataset setup/verification instructions
    loader.py                date-pinned train/val/test split (test is guarded — never loaded unless explicitly allowed)
    label.py                  Play 3: tab-conditioned is_click resolution
    leakage_guard.py           drops video_features_statistic by default
    features.py                 feature engineering — agent's main edit surface
  model/
    baseline.py                 PLACEHOLDER reference model (small embedding+MLP CTR net)
    architectures/                where agent-proposed model variants would land
  train.py                      training entrypoint (used by orchestrator + smoke tests)
  evaluate.py                    NDCG@10 / Recall@50, computed per-user
  smoke_test.py                   fast (~seconds) sanity check before a patch is trusted
  submit.py                       final submission writer — raises until schema is known

agent/                    the autonomous agent itself
  llm_client.py              Anthropic API wrapper + token accounting
  orchestrator.py             the main loop (see Architecture above)
  ablation.py                  Play 2: block-level ablation
  code_editor.py                applies LLM-written patches with smoke-test rollback
  referee.py                     Play 1: unbiased scoring via the random-exposure log
  compression_gate.py             Play 1: overfit-rejection check before finalizing
  pitfall_store.py                  structured failure+recovery log (feeds Tier-1 context)
  logger.py                          per-iteration run log, intervention counter, resource usage
  report.py                           generates docs/results_table.md from the run logs
  skill_store/
    tier1_core.md                      always loaded: task framing + dataset quirks
    tier2_domain.md                     RecSys method priors, loaded when relevant
    tier3_deep/                          deep dives, loaded on demand
    retriever.py                          tiered retrieval logic

config/agent_config.yaml   all tunables + every organizer-dependent value, isolated in one place
docs/                       results_table.md (generated), devpost_writeup.md (draft)
logs/                       generated at runtime: iterations.jsonl, interventions.jsonl, pitfalls.json, resource_usage.json
```

## 3. Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # required — the agent makes real API calls, no offline mode

python -m pipeline.data.download --check   # tells you what to place where
# KuaiRand-Pure isn't behind a stable direct-download URL; get it from
# https://kuairand.com/ and place the CSVs listed by --check into
# config.dataset.raw_dir (default ./data/raw).
```

## 4. Reproduction steps

```bash
# 1. Sanity-check the pipeline runs at all, against the placeholder baseline:
python -m pipeline.train

# 2. Run the full autonomous agent loop:
python -m agent.orchestrator

# 3. Generate the results table + resource usage summary from the run log:
python -m agent.report
# -> writes docs/results_table.md
```

`config/agent_config.yaml` controls everything: which Claude models run
which role, iteration budget, convergence thresholds, and the Play 1
Tier-A/Tier-B toggle. Read it before a long run — every value that depends
on the organizer's Starter Kit is called out there explicitly.

## 5. Open Questions / Day-1 blockers

These aren't implementation details we skipped — they're facts only the
challenge organizers can supply, and the codebase is deliberately built so
each one is isolated behind a config value or an explicit exception rather
than silently guessed:

1. **Is `log_random_4_22_to_5_08_pure.csv` permitted** for training
   (`referee.mode: tier_a`), for propensity-estimation/diagnostics only
   (`tier_b`, current default), or not at all (`disabled`)?
2. **Are the Zenodo supplementary files** (video captions, category
   taxonomy) in scope? Gated in `pipeline/data/download.py`.
3. **What is the candidate set at scoring time** — each user's impressed
   set, or the full ~7,583-item catalog? Changes what NDCG@10/Recall@50
   mean; `pipeline/evaluate.py` currently ranks over whatever's in the
   input DataFrame (defaults to impressed-set, since that's computable
   without organizer input).
4. **What are ε and N** (convergence rule) **and the compute budget?**
   Placeholders live in `config.starter_kit`.
5. **The actual organizer baseline, eval script, and submission schema.**
   Until these land, `pipeline/model/baseline.py` is a stand-in and
   `pipeline/submit.py` raises rather than guessing a format.

## 6. Limitations / what's not done

- **Editable surface is narrow by design.** `agent/code_editor.py` only
  lets the agent rewrite `pipeline/data/features.py` and
  `pipeline/data/label.py` end-to-end right now, not the model
  architecture or training loop directly. This was a deliberate scope cut
  to ship a safe, rollback-capable loop first; extending
  `EDITABLE_FILES` to cover `pipeline/model/architectures/` is the most
  natural next step (the smoke-test/rollback mechanism already generalizes).
- **The referee's live per-iteration wiring is partial.** The scoring
  logic and divergence math (`agent/referee.py`) are implemented and
  tested in isolation, but `orchestrator.py` doesn't yet run inference
  over the random-exposure log every iteration — that integration point is
  marked explicitly in the code (`referee_note` in `orchestrator.py`).
- **No GPU-hour tracking beyond wall-clock.** `logger.py` reports
  wall-clock time as a proxy; real GPU-hour accounting (e.g. via
  `nvidia-smi` polling) isn't wired in.
- **Baseline is a placeholder.** Every "beats baseline" claim from this
  codebase today is against our own small reference model, not the
  organizer's official one.

## 7. Contributions

Built as a single push: dataset pipeline (split/label/leakage handling),
the three-layer agent (ablation chassis, unbiased referee, compression
gate), tiered skill store, and run-logging/reporting infrastructure.
