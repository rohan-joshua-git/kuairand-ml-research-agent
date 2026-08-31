# Autonomous ML Research Agent — KuaiRand-Pure

An LLM agent that runs the ML engineering loop — read the problem, inspect data, engineer features, train, tune, evaluate, reflect, revise — on the KuaiRand-Pure within-user ranking benchmark, without a human in the iteration loop.

## Result

**Validation primary 0.6049** against the official FM baseline's **0.6016**.

| Validation metric | Official baseline | This submission | Absolute delta |
|---|---|---|---|
| GAUC | 0.6674 | **0.6720** | **+0.0046** |
| nDCG@5 | 0.5357 | **0.5378** | **+0.0021** |
| primary | 0.6016 | **0.6049** | **+0.0033** |

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

## Where to verify each claim

| Claim | Evidence |
|---|---|
| Beats the baseline by the stated delta | [`docs/results_table.md`](docs/results_table.md), `submissions/submission_valid.csv` |
| Submission is in the Starter Kit schema | `submissions/submission_valid.csv`, `submissions/submission_test.csv` (committed) |
| Artifact reproduces byte-identically | `submissions/FROZEN_CONFIG.json` SHA-256s + the clean-checkout run below |
| No hidden-test labels used | [`docs/COMPLIANCE_NOTE.md`](docs/COMPLIANCE_NOTE.md), `pipeline/official_baseline.py`, `pipeline/data/loader.py` |
| 2 manual interventions, instrumented not asserted | `logs/interventions.jsonl`, `agent/logger.py` `persisted_intervention_count()` |
| Tokens and wall-clock measured, not estimated | `logs/resource_usage.json`, `agent/llm_client.py` `TokenLedger` |
| The agent drove its own loop | `logs/iterations.jsonl` + `logs/iterations_pre_*.jsonl`, `agent/orchestrator.py` |
| Every research claim has a negative control | [`docs/research_process.md`](docs/research_process.md) §6.5-6.7 |
| Metric matches the organizer's script | `pipeline/evaluate.py` delegates to `starter_kit/evaluate.py`; decomposition verified to 1.6e-14 |

## Strategy

The task is **within-user ranking**: each user is ranked only among their own
logged impressions. That single fact drove every decision.

1. **Reproduce the official baseline first**, using the organizer's own vendored
   code rather than a reimplementation, so the starting point is not in doubt.
2. **Let the agent edit a bounded surface.** LLM rewrites land only in an
   allowlist of five files, behind a smoke test that rolls back on failure.
   Every scored run trains in a fresh interpreter so a stale import can never
   score the previous iteration's code.
3. **Exploit what within-user ranking implies.** Anything constant within a user
   cannot reorder that user's candidates — which rules out most user-side
   features a priori and redirects effort to item and context signals that vary
   across a user's own impressions.
4. **Score through the organizer's evaluator**, never a local reimplementation,
   so a validation number cannot quietly diverge from what the graded script
   reports.
5. **Make every candidate improvement survive a negative control** before it
   counts. This is what the second half of the project became.

## What this project actually is

The score moved +0.0033. The interesting part is not that number — it is that **every alternative explanation for it was tested and eliminated**, and that several apparently promising signals were killed by controls rather than accepted.

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

### What the agent proposed across all runs

The graded run's seven hypotheses are dominated by regularization scalars, and on
their own they understate the search. Six earlier runs are preserved in
`logs/iterations_pre_*.jsonl`, and across them the agent targeted the **loss
function, the architecture, and the feature surface** — not just hyperparameters:

| Run log | Hypothesis | Validation primary |
|---|---|---|
| `_082207` it1 | per-user pairwise BPR loss | 0.5056 |
| `_085518` it1 | per-user pairwise BPR loss (second implementation) | 0.5994 |
| `_085518` it3 | listwise ListNet objective | 0.6004 |
| `_082207` it2 | auxiliary BCE heads on secondary feedback signals | 0.5610 |
| `_085518` it2 | multi-task architecture over auxiliary labels | 0.6013 |
| `_235428` it1 | replace pointwise BCE with a ranking loss | 0.5921 |
| `_090423` it1-3 | global item-quality prior as a feature | 0.6024-0.6026 |
| graded it2 | DeepFM `[64,32]` (crashed the smoke test, rolled back) | - |
| **graded it4** | **DeepFM-lite `[32,16]` MLP branch (accepted)** | **0.6045** |

Every one below the champion was rolled back automatically. The organizer's own
#1-ranked lead — switch to a pairwise or listwise ranking objective — was
proposed by the agent unprompted on the first iteration of three separate runs,
and refuted three times by independent implementations.

**Attribution, stated plainly.** The dataset findings in `agent/skill_store/`
are *human* research, written before any run and injected into the agent's
reflect prompt by design. The hypotheses above are the agent's. The results
table separates human-authored from agent-authored steps for the same reason.

### A rule deviation we are disclosing

Our acceptance policy used epsilon for two jobs, and it cost us. Iteration 6
scored **0.60496** against the accepted best **0.60449** — genuinely better —
but the gain (+0.00046) did not clear epsilon = 0.002, so it was rolled back.
The run therefore stopped holding the **second-best** checkpoint, and the shipped
model is about **+0.0005** below the validation-best.

FAQ 2.9.1(c) asks for the validation-best checkpoint at the point the run stops.
`agent/orchestrator.py` now separates the two rules — any improvement is accepted
as the new best, while epsilon governs only the plateau window — but that change
postdates the graded run and we have not re-run to exploit it. The affected row
is flagged in [`docs/results_table.md`](docs/results_table.md).

## How results are protected from ourselves

**Selection/confirmation split.** ~30 configurations had been compared against the whole validation split, so the winner's curse was unbounded. [`pipeline/eval_protocol.py`](pipeline/eval_protocol.py) partitions validation by user hash into 11,270 selection and 11,107 confirmation users. All exploration and early stopping use selection only. The confirmation half was looked at **once**, after the config was frozen.

Its per-user decomposition of GAUC/nDCG@5 is verified against `starter_kit/evaluate.py` to **1.6e-14**, including a tie-heavy case, which is what makes the user-level bootstrap exact rather than approximate.

**Early stopping uses the selection half only.** `pipeline/make_submission.py` builds the mask with `eval_protocol.user_half` and passes it to every shipped model, so no confirmation user influences epoch choice. An earlier artifact scored **0.6053** but early-stopped over *all* validation rows; it was replaced because the higher number was not a clean held-out estimate. Its hashes are retained in `FROZEN_CONFIG.json` under `superseded_artifact_sha256`.

**Confirmation result.** Raw confirmation (0.6016) sits below selection (0.6081), but a frozen model-free video prior drops similarly on the same users, so that gap is population difficulty rather than overfitting. The model's advantage over that reference is **+0.0248 on selection and +0.0235 on confirmation** — a difference of 0.0013, at the edge of the ~0.0011 user-sampling band rather than comfortably inside it. The learned advantage substantially transfers to users never used for any decision; we do not claim it transfers exactly.

**Why the submission is an ensemble.** The mean gain is **not** established — 5-seed vs 1-seed is +0.00027, while same-model negative controls reach 0.00077. It ships for **variance**, which is established: single-seed std 0.00049 over 20 seeds against 5-seed std 0.00020, matching √n. On a one-shot submission the floor is what matters — worst single seed 0.6036, worst 5-seed ensemble 0.6046.

**Clean-checkout verification.** The repository was cloned fresh from GitHub, data staged, and the whole pipeline re-run. Both submission artifacts regenerate **byte-identical** to their recorded SHA-256s, and the organizer's scorer returns the documented figure:

```
smoke test                  0.4498908751631786   identical fingerprint
baseline reproduction       0.6015 vs published 0.6016   MATCHES
metric decomposition        1.599e-14 vs starter_kit/evaluate.py
submission_valid.csv        1fd3f7c0...46e2   HASH MATCH
submission_test.csv         7f460171...4f30   HASH MATCH
official checker            124,909 / 170,588 rows, both pass
official score (valid)      GAUC 0.6720 | nDCG@5 0.5378 | primary 0.6049
```

**Reproducibility.** The reported 0.6049 was produced twice by independent paths — cached score matrix → `eval_protocol` decomposition, and `make_submission` → `aligned_rows`/`score_rows` → CSV → `starter_kit/evaluate.py`. They agree to four decimals on all three metrics, which establishes the reported number is the number the submitted artifact actually scores.

## Data integrity and compliance

Every rule, how it is enforced, and where to check. Full disclosure of one
closed exposure is in [`docs/COMPLIANCE_NOTE.md`](docs/COMPLIANCE_NOTE.md),
including a correction to one of our own commit messages.

| Rule | Enforcement | Where to verify |
|---|---|---|
| Hidden-test labels never used for training, selection, early stopping or feature stats | `allow_test=False` default; the only `allow_test=True` is the submission writer; `official_baseline.py` starves the vendored baseline of test rows | `pipeline/data/loader.py`, `pipeline/make_submission.py`, `pipeline/official_baseline.py` |
| Test split never locally scored | The test submission is written and alignment-checked, never passed to `--score` | `pipeline/make_submission.py` |
| Feature statistics fit on train only | Vocabularies, `dur_bucket` quantiles and cross target-encodings all built from `split.train`; target encoding is 5-fold out-of-fold | `pipeline/train.py` `_build_id_maps`, `_fit_cross_te` |
| Random-exposure log never used for training | Loader defaults to `window="val"` and **fails closed** if it cannot prove the test window is excluded; the live probe filters dates again | `pipeline/data/loader.py`, `pipeline/train_runner.py` |
| KuaiRand-1k / 27k not used | No loader, no reference anywhere | `grep -rn "kuairand-1k\|kuairand-27k"` returns nothing |
| No external training data | Zenodo supplements declared out of scope, never loaded | `config/agent_config.yaml` `supplements:` |
| Same-row outcome signals never used as inputs | Encoder iterates an explicit allowlist; `play_time_ms`, `*_stay_time`, `is_profile_enter` are excluded by name | `pipeline/train.py` `resolve_fields`, `pipeline/data/features.py` |
| Statistic-file leakage | Dropped by exact column name, not a naming convention | `pipeline/data/leakage_guard.py` |
| No time-travel features | `pos_bucket` is a `cumcount` over time-sorted rows, counting only preceding same-day impressions | `pipeline/data/features.py` |
| Row alignment preserved | `_orig_row_pos` tagged before the feature build and scattered back; `aligned_rows` proves the id sequence matches the starter kit element-for-element | `pipeline/submit.py` |
| No NaN/Inf in scores | Explicit `np.isfinite` guard that raises | `pipeline/submit.py` |

Official checker output on the submitted files:

```
$ PYTHONIOENCODING=utf-8 python submit.py --check ../submissions/submission_valid.csv --split valid
格式与对齐校验通过：124,909 行，split=valid
$ PYTHONIOENCODING=utf-8 python submit.py --check ../submissions/submission_test.csv --split test
格式与对齐校验通过：170,588 行，split=test
```

(On Windows the checker raises `UnicodeEncodeError` *after* every check passes,
because its success line prints U+2713. Set `PYTHONIOENCODING=utf-8` and confirm
exit 0.)

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
  COMPLIANCE_NOTE.md     disclosed hidden-test exposure, closed, with a correction
                         to one of our own commit messages
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

## Team

| Member | Contribution |
|---|---|
| **Rohan Joshua** | Agent architecture and loop, evaluation protocol, research experiments and controls, submission pipeline and provenance |
| **Thaddus Lee** | Referee integration, crash checkpointing and resume, autonomous ablation targeting, Starter Kit and dataset integration, metric/convergence alignment |
| **Waseem Akram** | Audit of all technical files and documentation — research audit and submission audit |

## How this was built: a human-relayed adversarial loop

Development used two AI assistants in opposing roles, with a human relaying
between them:

```
   Claude Code (implementer)  --output-->  human  --relay-->  OpenAI (adversary)
            ^                                                        |
            |________________  human  <--critique--  ________________|
```

**Claude Code** implemented, ran experiments and reported results. **OpenAI**
received those reports and attacked them — challenging conclusions, demanding
controls, proposing alternative explanations, and setting stopping rules. A
human passed messages in both directions and made the calls. This was
**not** an autonomous multi-agent system; the relay was manual.

It is worth documenting because it changed outcomes we can point at:

| The adversary's intervention | What it changed |
|---|---|
| "more seeds cannot hurt" is not true of a ranking metric | corrected an overstatement about variance reduction before it reached the writeup |
| run the oracle diagnostic *before* building a gate | killed the conditional-blending branch in one GBDT fit instead of a full gate pipeline |
| add a *randomized* arm as a stronger negative control | gave the 4-arm design that made an in-fold leak diagnosable instead of reading as a failed hypothesis |
| test early-vs-late estimation directly | refuted staleness, which we had been treating as the likely mechanism |
| 0.8645 is the *label-oracle* ceiling, not "not a ceiling" | sharpened a claim that was loosely stated and would not have survived review |
| freeze the submission before any further research | produced the frozen, hash-verified artifact that every later experiment was measured against |

The pattern is that the adversary rarely proposed better *models* — it
proposed better *tests*, and repeatedly stopped work that would have produced
a confident wrong answer.

**This describes the development process, not the submitted system.** The agent
itself runs on Google Gemini; no OpenAI or Anthropic model is called by the
pipeline at scoring time. See below.

## Development tools, APIs and assets

**Tools:** VS Code, Claude Code (used as an AI coding assistant during development — disclosed for transparency; it is not part of the submitted system).

**APIs used by the agent:** Google Gemini — `gemini-3.5-flash` and `gemini-3.5-flash-lite`. Token usage for the scored run is in `logs/resource_usage.json` (91,430 total). An Anthropic backend is implemented and selectable in `config/agent_config.yaml` but was **not** used for the scored run.

**Libraries:** PyTorch, pandas, numpy, scikit-learn, PyYAML, tqdm, google-genai, anthropic. LightGBM is used only by the blending diagnostic on a research branch and is not a submission dependency.

**Data:** KuaiRand-Pure only, via the organizer's Starter Kit. No external training data, no pretrained weights, and the Zenodo caption/category supplements were treated as out of scope since neither the problem statement nor the Starter Kit sanctions them.

## Limitations

- **The agent cannot create new files.** `agent/code_editor.py` rewrites whole files at fixed paths, so it can change the loss, the model and its own ablation grid, but cannot add a module. A create-a-file flow needs a different backup/restore mechanism.
- **Continuous features have no path into the model.** The FM consumes categorical fields only, so a continuous feature must be bucketed. We stopped intervening here deliberately: the categorical version of the agent's proposal measured +0.0003, below the noise floor.
- **Cross-file changes cost an iteration.** One patch is one whole-file rewrite. A broken interface is caught by the smoke test and rolled back — it costs an iteration, not state.
- **The recorded trajectory predates three post-run compliance fixes.** Those fixes do not alter the model: every added parameter defaults to prior behaviour, and the smoke-test fingerprint is byte-identical across all of them.
- **The compression gate reasons about a summary, it does not retrain from it.** A stronger version would have the fresh reproducer re-run training from the compressed description and compare scores.

## Future work

- **Multi-provider adversarial auditing, automated.** The adversarial loop
  described above was human-relayed. Wiring a second provider in as an
  automated adversary — one model proposes, a different model attacks the
  claim and demands controls — is the natural next version, and would make the
  critique a measurable part of the agent rather than a manual step.
- **Understand why an explicit user representation hurts.** We measured that it
  does, and eliminated noise, cardinality and staleness. The redundant-pathway
  account is interpretation, not measurement — the one thread genuinely worth
  reopening.
- **Make the compression gate retrain rather than reason.** It currently judges
  a terse summary; the stronger version re-runs training from that summary and
  compares scores.
- **Alert the referee on the change in divergence, not its level.** The two
  splits have structurally different label distributions, so the absolute gap
  is uninformative.
- **Persist "this scored worse" across runs.** The pitfall store records crashes
  and gate warnings, so a fresh run would happily re-propose an approach already
  measured as worse.
- **New-file creation in the editor**, so the agent can add an architecture
  module rather than only rewriting existing files.

---

In one line: an ablation-guided agent that edits its own pipeline code, scored through the organizer's own evaluator, with every candidate improvement forced past a negative control before it counts as real.
