# Autonomous ML Research Agent — KuaiRand-Pure

An LLM agent that runs the ML engineering loop — read the problem, inspect data, engineer features, train, tune, evaluate, reflect, revise — on the KuaiRand-Pure within-user ranking benchmark, without a human in the iteration loop.

## Result

**Validation primary 0.6053** against the official FM baseline's **0.6016**.

| Validation metric | Official baseline | This submission | Absolute delta |
|---|---|---|---|
| GAUC | 0.6674 | **0.6724** | **+0.0050** |
| nDCG@5 | 0.5357 | **0.5382** | **+0.0025** |
| primary | 0.6016 | **0.6053** | **+0.0037** |

Scored by the organizer's own `starter_kit/evaluate.py` via `submit.py --score`, not by our code. Final model: DeepFM-lite, 10-seed rank-average, seeds 0–9.

**No hidden-test score is claimed anywhere in this repository.** The hidden test is scored once by the organizer. The baseline's published test primary (0.5946) appears only as the organizer's reference figure and is never compared against a validation number — they are different splits on different scales.

| Resource | Value |
|---|---|
| Iterations | 9 of a 50 cap |
| Wall-clock | 0.1965 h of a 6 h ceiling |
| LLM tokens | 91,430 |
| GPU-hours | 0 (CPU-only) |
| Manual interventions | 2, both logged with reasoning in `logs/interventions.jsonl` |

Full provenance, including artifact SHA-256s, is in [`docs/results_table.md`](docs/results_table.md) and `submissions/FROZEN_CONFIG.json`.

## What this project actually is

The score moved +0.0037. The interesting part is not that number — it is that **every alternative explanation for it was tested and eliminated**, and that several apparently promising signals were killed by controls rather than accepted.

The methodology is the contribution:

```
strong baseline
      ↓
observe an unexplained signal
      ↓
form a mechanism hypothesis
      ↓
build a control that would expose a false positive
      ↓
measure
      ↓
result survives? → keep     result dies? → record why, close it
      ↓
repeat
```

The final model was not selected because it won a validation sweep. It survived systematic attempts to falsify it.

### The research record

| Hypothesis | Test | Result | Decision |
|---|---|---|---|
| `user_id` helps via affinity or capacity | row-level shuffle holding parameters fixed | identity **108%**, capacity **−8%** | closed — capacity refuted |
| The user embedding overfits, so shrink it | field-specific weight decay, 6 arms × 3 seeds | monotone **negative** over 5 orders of magnitude | closed |
| Performance decays across the eval window | frozen model-free reference on the same days | gap **+0.0008** — decline is day difficulty | closed |
| The 0.8645 gap means signal is missing | simulate `y ~ Bernoulli(q)` from the calibrated model | irreducible gap **0.2556** > observed **0.2431** | closed |
| Users differ in feature *sensitivity* | 4-arm OOF + permuted + randomized controls | **−0.0012**, CI excludes zero | closed |
| Staleness explains that failure | early vs late estimation, sample sizes matched | **+0.00017**, CI includes zero | refuted |
| DeepFM/GBDT disagreement is exploitable | per-group oracle + quality-matched control | **+0.0023** of an apparent +0.0314 | closed |

Detail for each is in [`docs/research_process.md`](docs/research_process.md); the last two live on the research branches (`research/user-sensitivity` @ `9e971bf`, `research/conditional-blend` @ `b5ae63d`), which were deliberately not merged so the frozen artifact could not be disturbed.

### Three findings worth reading

**1. `user_id` encodes identity, not capacity.** Deleting `user_id` costs −0.0091, yet user×video affinity measures at chance three times over. The mechanism was undetermined until a control that permutes the row-to-user link *while holding parameter count, MLP input width and code-frequency distribution fixed* lost the entire effect: identity +0.0093 (108%), capacity −0.0007 (−8%). The permutation has to be row-level — a bijective user→row remap is a symmetry of the model and trains to an identical result, which would have produced a convincing false null.

**2. The published ceiling is not reachable by any model.** 0.8645 is the organizer's **label-oracle** ceiling and is correctly computed. But simulating a world where the model *is* the true conditional probability — so nothing remains to learn by construction — an oracle still beats it by **0.2556**, while the real observed gap is **0.2431**, smaller. A long_view is a coin flip; an oracle that sees the realised label wins by that margin no matter how good the model is. So the distance to 0.8645 is not evidence of remaining headroom.

**3. A per-group oracle is upward-biased, and here the bias was 12× the signal.** A per-user oracle over DeepFM and GBDT showed +0.0314 of apparent headroom. A quality-matched control — DeepFM degraded with calibrated noise to the GBDT's exact score level, containing no information DeepFM lacks — reproduced **+0.0291** of it. Real excess: **+0.0023**. Picking the luckier of two noisy per-user estimates on a median of 4 impressions manufactures most of an "upper bound".

### A leak we introduced, found, and fixed

The first user-sensitivity experiment produced a clean, significant result: real sensitivities scored **−0.0048** with a CI excluding zero, while permuted and randomized versions of the identical fields were neutral (+0.0003, −0.0002).

Noise being free while real values did damage is not how a failed hypothesis behaves — it is how a leak behaves. The per-user statistic had been computed from that user's own training labels, so a training row's feature contained that row's label: an in-fold target encoding one level above where we had applied the out-of-fold rule. Rebuilding it so each row's band comes from that user's *other* folds recovered 75% of the drop.

**With only a control and a treatment arm this would have been recorded as "sensitivities are harmful."** The two null controls are what made it diagnosable. This is why every user-derived feature here ships with both a permuted and a randomized control.

### Measured vs interpreted

Kept separate deliberately.

**Measured:** sensitivities have split-half reliability 0.31–0.60 (Spearman-Brown 0.47–0.75), so they are not noise. They lose 22–61% of their correlation across ~1 week, against a matched-size random split. Exposing them costs −0.0012. Both null controls are neutral. Estimating from 6× more data makes the harm *worse*, not better.

**Interpretation, not established:** harm scales with how much the model trusts the band — the user embedding already learns this modulation from the same data, and an explicit precomputed summary adds a redundant, coarser pathway that generalises worse.

## How results are protected from ourselves

**Selection/confirmation split.** ~30 configurations had been compared against the whole validation split, so the winner's curse was unbounded. [`pipeline/eval_protocol.py`](pipeline/eval_protocol.py) partitions validation by user hash into 11,270 selection and 11,107 confirmation users. All exploration and early stopping use selection only. The confirmation half was looked at **once**, after the config was frozen.

Its per-user decomposition of GAUC/nDCG@5 is verified against `starter_kit/evaluate.py` to **1.6e-14**, including a tie-heavy case, which is what makes the user-level bootstrap exact rather than approximate.

**Confirmation result.** Raw confirmation (0.6023) sits below selection (0.6083) — but a frozen model-free video prior drops by nearly the same amount (+0.0053 against the model's +0.0059), so that gap is population difficulty, not overfitting. The model's advantage over that reference is **+0.0250 on selection and +0.0242 on confirmation**: the learned advantage transfers to users never used for any decision.

**Why the submission is an ensemble.** The mean gain is **not** established — 5-seed vs 1-seed is +0.00027, while same-model negative controls reach 0.00077. It ships for **variance**, which is established: single-seed std 0.00049 over 20 seeds against 5-seed std 0.00020, matching √n. On a one-shot submission the floor is what matters — worst single seed 0.6036, worst 5-seed ensemble 0.6046.

**Reproducibility.** The reported 0.6053 was produced twice by independent paths — cached score matrix → `eval_protocol` decomposition, and `make_submission` → `aligned_rows`/`score_rows` → CSV → `starter_kit/evaluate.py`. They agree to four decimals on all three metrics, which establishes the reported number is the number the submitted artifact actually scores.

## Agent design

```
                    orchestrator.py loop
     read -> inspect -> engineer -> train -> evaluate -> reflect/revise -> ...

     Skill store            Ablation              Referee + gate
     (warm start)           (targeted search)     (unbiased scoring,
     tier1/2/3              which block moves      overfit warning)
                            the score?
```

**Ablation-first refinement.** Rather than rewriting the pipeline each round, `agent/ablation.py` runs a cheap ablation over named blocks to find which is moving the score, and the orchestrator targets that block for the next LLM code change. `agent/skill_store/` holds domain knowledge in three tiers, loaded on demand — following MLE-STAR and HASTE, where tiered retrieval beat flat knowledge-dumping.

**Bounded blast radius.** LLM rewrites can only land in five allowlisted files (`EDITABLE_FILES`), behind a subprocess smoke test that rolls back on failure and records the reason as a pitfall for the next prompt. Every scored run trains in a fresh interpreter (`pipeline/train_runner.py`), so a stale import can never score the previous iteration's code.

**Faithful scoring.** `pipeline/evaluate.py` imports and calls `starter_kit/evaluate.py` rather than reimplementing GAUC/nDCG@5, so a validation number cannot quietly diverge from what the organizer's script reports.

**Unbiased referee.** `agent/referee.py` scores candidates against the uniformly-random-exposure log alongside standard validation and tracks the divergence. The probe is restricted to the **validation window** — 75.7% of that file falls inside the hidden-test window, and while it never drove checkpoint selection, its divergence is surfaced to the reflect step, so the test-window rows are kept out of the loop entirely.

**Compression gate, advisory.** `agent/compression_gate.py` hands a terse summary of a winning approach to a fresh LLM context with no access to validation scores. It does **not** veto: FAQ 2.9.1(c) requires the validation-best checkpoint to ship, so a failed gate is logged as a pitfall and fed to the next reflect step.

## Dataset findings

1. **`is_click` means two different things.** A genuine tap in the two-column UI; `valid_play` in the single-column UI, keyed on `tab`. `pipeline/data/label.py` resolves this explicitly. Relevant only to the auxiliary head — the primary label is `long_view`.
2. **`video_features_statistic` columns leak** and do *not* self-identify by name (`show_cnt`, `play_cnt`, …), so a substring check would let all of them through. `pipeline/data/leakage_guard.py` flags them by exact name.
3. **`play_time_ms` is the label in disguise** — corr 0.64 with `long_view`, 84.7% threshold accuracy. Excluded as an input; it is an outcome of the impression being predicted.
4. **The organizer already ran "add more features" and "grow capacity"** and found both flat. Pre-seeded as Tier-1 knowledge so the agent doesn't re-spend iterations on it.
5. **`upload_dt` has three distinct values.** No video lifecycle exists, which structurally rules out the entire temporal-dynamics family. Confirmed independently: recency-weighting is flat and a 2-day half-life hurts.
6. **`author_id` and `music_id` are the video prior in disguise** — 87% and 98% own exactly one video, target-encoded corr 0.985 / 0.987. Only `tag` is a genuine pooling level. This corrected one of our own top-ranked recommendations.
7. **`tab` is a reordering signal for only 40% of users.** For the 60% inside a single tab it contributes +0.0001 — a within-user constant cannot reorder anything. Its entire effect is the level gap *between* tabs (0.004 to 0.489).
8. **The signal decomposes into roughly equal thirds.** Quality 0.5807 → +tab 0.5877 → GBDT over clean features 0.5961 → neural 0.6048. A third is captured only by learned structure no named feature reproduces.
9. **17.5% of the metric is structurally unmovable.** 3,917 validation users have one impression: model, prior and oracle all score identically, and GAUC excludes them entirely.

## File map

```
agent/
  orchestrator.py        the loop: ablation -> hypothesis -> patch -> smoke -> score -> checkpoint
  code_editor.py         whole-file rewrites behind an allowlist, with rollback
  eval_protocol.py       (in pipeline/) selection/confirmation split + exact bootstrap
  referee.py             unbiased random-exposure probe, validation window only
  compression_gate.py    advisory overfit check; never vetoes the validation-best
  logger.py              iteration + intervention logs (graded evidence)
  skill_store/           tier1 always-loaded findings; tier2/3 retrieved on demand
pipeline/
  train.py               model + training. AGENT-EDITABLE
  data/features.py       feature surface. AGENT-EDITABLE
  data/loader.py         official date split, allow_test guard
  data/leakage_guard.py  drops the leaky statistic columns
  evaluate.py            delegates to starter_kit/evaluate.py
  eval_protocol.py       per-user metric decomposition, verified to 1.6e-14
  submit.py              alignment-checked submission writing
  make_submission.py     the ONLY place allow_test=True is set
scripts/                 the experiments behind the research table
docs/
  research_process.md    full account: every hypothesis, control and disposition
  backlog_triage.md      ~95-item method backlog mapped to evidence
  results_table.md       generated by agent/report.py — never hand-edited
starter_kit/             organizer code, vendored verbatim, never modified
```

## Setup

```bash
pip install -r requirements.txt
python -m pipeline.data.download          # fetch/verify KuaiRand-Pure

# The agent makes real API calls; there is no offline mode. Set the key for
# whichever provider config/agent_config.yaml selects (default: gemini).
export GEMINI_API_KEY=your_key_here
# export ANTHROPIC_API_KEY=...            # if agent.llm.provider is "anthropic"
```

## Reproduction

| Command | Purpose |
|---|---|
| `python -m pipeline.official_baseline` | Task Requirement #1 — reproduce the baseline on validation |
| `python -m pipeline.smoke_test` | Fast contract check (fingerprint `0.4498908751631786`) |
| `python -m pipeline.train` | Train the champion once — expect val primary 0.6047 ± 0.0005 |
| `python -m pipeline.eval_protocol` | Verify the metric decomposition against the official script |
| `python -m agent.orchestrator` | The full autonomous loop |
| `python -m agent.report` | Regenerate `docs/results_table.md` from the run log |
| `python -m pipeline.make_submission --out-dir submissions --seed 0 --ensemble-seeds 10` | Regenerate the frozen submission |

Research scripts (`scripts/seed_ensemble.py`, `error_slices.py`, `sweep_user_wd.py`, `user_id_mechanism.py`, `temporal_curve.py`) reproduce the experiments in the table above.

**On Windows, `starter_kit/submit.py --check` crashes on success** — its success line prints U+2713, which cp1252 cannot encode, so the traceback appears *after* every check has passed. Run it as `PYTHONIOENCODING=utf-8 python submit.py --check ...` and confirm exit 0.

## Limitations, and what we'd do with more time

- **The agent cannot create new files.** `agent/code_editor.py` rewrites whole files at fixed paths, so it can change the loss, the model and its own ablation grid, but cannot add a module. A create-a-file flow needs a different backup/restore mechanism.
- **Continuous features have no path into the model.** The FM consumes categorical fields only, so a continuous feature must be bucketed. We stopped intervening here deliberately: the categorical version of the agent's proposal measured +0.0003, below the noise floor.
- **Cross-file changes cost an iteration.** One patch is one whole-file rewrite. A broken interface is caught by the smoke test and rolled back — it costs an iteration, not state.
- **The recorded trajectory predates three post-run compliance fixes.** Those fixes do not alter the model: every added parameter defaults to prior behaviour, and the smoke-test fingerprint is byte-identical across all of them.
- **The compression gate reasons about a summary, it does not retrain from it.** A stronger version would have the fresh reproducer re-run training from the compressed description and compare scores.
- **Given more time**, the one thread we would reopen is *why* an explicit user representation degrades a model whose implicit user representation demonstrably carries signal. We measured that it does, and eliminated noise, cardinality and staleness as explanations. The redundant-pathway account is currently interpretation, not measurement.

---

In one line: an ablation-guided agent that edits its own pipeline code, scored through the organizer's own evaluator, with every candidate improvement forced past a negative control before it counts as real.
