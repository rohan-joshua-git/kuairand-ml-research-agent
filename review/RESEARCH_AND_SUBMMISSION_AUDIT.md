# TikTok TechJam 2026 — PS2 Adversarial Judge Report

**Repo:** `rohan-joshua-git/Tiktok-Tech-Jam` @ `a61d2ca` (clean tree)
**Reviewed:** 2026-08-31 21:50 SGT · **Deadline:** 2026-09-01 12:00 SGT · **14.2 hours remaining**
**Reviewer posture:** hostile. No fixes applied. Nothing outside `review/` touched.

---

## READ THIS FIRST

The protocol asks me to write the strongest available case that this submission
misses the top twelve, and to say so at the top if that case is easy to write.

**It is easy to write, and it took me under an hour.** Not because the
engineering is weak — parts of it are the most careful I have seen today — but
because the repository's own artifacts contradict each other on the four things
a judge checks first: the headline score, the intervention count, the token
count, and whether the pipeline ever touched hidden-test labels. Two documents
in this repo state "no test score is asserted anywhere in this repository."
Two other documents in the same repo assert a hidden-test score. A judge who
notices that stops trusting everything else, and that is the correct response.

The team's problem is not capability. It is that the artifact a rushed reviewer
actually reads — `README.md` — describes a worse, older, and partly nonexistent
submission than the one they built.

---

## 1. Phase 0 — Evidence inventory

### 1.1 Required deliverables (PS §2.5)

| # | Deliverable | Expected path | Status | Actual path / evidence |
|---|---|---|---|---|
| 1 | Written project description | Devpost, mirrored in repo | **PRESENT** | `docs/devpost_writeup.md` (13,531 B) |
| 2 | Public repo, commented code, all components | GitHub | **PRESENT** | `origin https://github.com/rohan-joshua-git/Tiktok-Tech-Jam.git`; 4,211 LOC across `agent/` + `pipeline/`; comment density high |
| 3 | README: overview, setup, repro, limitations, **team contributions** | `README.md` | **PARTIAL** | Overview/setup/repro/limitations present. **Team member contributions: ABSENT** — `grep -ni "team\|contribut\|member" README.md` returns only human-vs-agent attribution, no people |
| 4 | Run & iteration logs (hypothesis, diff, GAUC/nDCG, errors, recovery) | `logs/` | **PRESENT** | `logs/iterations.jsonl` (9 records) + 6 rotated `iterations_pre_*.jsonl`. All five required fields present per record |
| 5 | Manual intervention summary | anywhere | **PRESENT, CONTRADICTORY** | `logs/interventions.jsonl` = 2 records; `logs/resource_usage.json` = 2; `docs/results_table.md` = 2; **`README.md` = 1** |
| 6 | Final model output in Starter Kit schema | `artifacts/` | **ABSENT** | `submissions/` contains only `FROZEN_CONFIG.json`. Both CSVs excluded by `.gitignore:30` (`submissions/*.csv`) and `.gitignore:52` (`submissions/**/*.csv`). Not on disk either |
| 7 | Results table + **absolute delta over baseline** | `docs/RESULTS.md` | **PRESENT, WRONG PATH** | `docs/results_table.md`. Content is complete and correct: GAUC +0.0050, nDCG@5 +0.0025, primary +0.0037 |
| 8 | Resource report (tokens, wall-clock, iterations/50, GPU-h) | `docs/RESULTS.md` | **PRESENT, WRONG PATH** | `docs/results_table.md` §"Resource usage": 91,430 tokens, 0.196 h, 0 GPU-h, 2 interventions, 9 logged iterations |

Three of eight deliverables are not where the problem statement says they will
be (`docs/RESULTS.md`, `artifacts/`, and the journal is `iterations.jsonl` not
`journal.jsonl`). One is genuinely absent (#6). One is contradictory (#5).

### 1.2 Commands run

```
$ git log --oneline | wc -l
25
$ git log --pretty=format:'%h|%ad|%s' --date=iso | tail -1
34d674a|2026-08-30 14:22:41 +0800|Initial commit
$ git log --pretty=format:'%h|%ad|%s' --date=iso | head -1
a61d2ca|2026-08-31 20:33:09 +0800|Lock the results table: separate validation from hidden test, add provenance
```
Entire project history spans **30h 10m**.

```
$ wc -l logs/iterations.jsonl
       9 logs/iterations.jsonl
```
Nine records: two `iteration: 0` bookkeeping rows + **seven LLM iterations**, out of a 50 cap.

```
$ ls data/
ls: data/: No such file or directory
$ python -m pipeline.official_baseline
ModuleNotFoundError: No module named 'yaml'
$ python -c "import torch"
ModuleNotFoundError
```
**No dataset, no submission CSV, dependencies not vendored.** `python3 submit.py --check`
and `python3 evaluate.py` could not be run: there is no submission file to check
and no data to evaluate against.

**I regenerated zero of this submission's reported numbers.** Every metric in
this report is quoted from the team's own logs, not verified.

### 1.3 Graded run — timeline (converted from Unix epochs in the logs)

| Event | Time (SGT) | Source |
|---|---|---|
| Intervention #1 (LLM failover policy) | 08-31 08:26:46 | `logs/interventions.jsonl` |
| Intervention #2 (`EXTRA_CATEGORICAL_FIELDS` plumbing) | 08-31 09:04:12 | `logs/interventions.jsonl` |
| **Graded run starts** | **08-31 09:05:16** | `logs/iterations.jsonl` rec 0 |
| **Graded run ends (converged)** | **08-31 09:16:10** | `logs/iterations.jsonl` rec 8 |
| `resource_usage.json` regenerated | 08-31 20:11:32 | `generated_at` field |
| Hidden-test exposures closed | 08-31 20:27:01 | commit `3afb112` |

**Graded run wall-clock: 653.5 seconds (10 min 53 s)** of a 6-hour ceiling.

---

## 2. Phase 1 — Disqualification review

### Verdict: **AT RISK**

Not DISQUALIFIED — I could not trace any hidden-test quantity into model
weights or checkpoint selection. But this repository *did* read hidden-test
labels during the graded run, *did* consume the result in a gate whose output
is recorded in the graded log, and *does* still assert a locally-computed
hidden-test score in two files while asserting in two others that it never
does. Under the instruction to take the less favourable reading, that is AT
RISK, and it is the first thing the team must fix.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Test labels touched | **VIOLATION (closed 20:27, 11h after the graded run)** | See §2.1 |
| 2 | Feature stats fit over val/test | **CLEAR** | `pipeline/train.py:409` `train_feat = build_features(split.train)`; `_fit_cross_te(train_feat, y_np)` at `:415`; `_build_id_maps(train_feat, ...)` at `:419`. Vocabs and `dur_bucket` quantiles (`train.py:295`) are train-only. `CROSS_TE_SPECS = {}` (`train.py:94`) — target encoding is disabled in the frozen config, and when enabled is 5-fold OOF (`train.py:134-137`) |
| 3 | `log_random_4_22_to_5_08_pure.csv` in training | **CLEAR** | Two call sites only. `pipeline/train_runner.py:87-90` loads it then filters `date >= 20220422 & date <= 20220428` — and that filter was **already present at `9f5a8d8`**, the HEAD during the graded run (verified via `git show 9f5a8d8:pipeline/train_runner.py`). Scoring-only, post-training. `agent/referee.py:110` is unfiltered but is dead code (§ Finding 9) |
| 4 | KuaiRand-1k/27k as auxiliary | **CLEAR** | `grep -rni "kuairand-1k\|kuairand-27k"` over all `.py`/`.yaml` → no matches |
| 5 | External training data | **CLEAR** | `config/agent_config.yaml` `supplements:` (Zenodo captions/categories) declared out of scope; no loader references them |
| 6 | Same-row auxiliary signals as input features | **CLEAR** | `_encode` (`train.py:330`) iterates `id_maps["fields"]` — an explicit allowlist resolved by `resolve_fields` (`train.py:54`) = `OFFICIAL_FIELDS + EXTRA_CATEGORICAL_FIELDS + CROSS_TE_FIELDS` = `user_id, video_id, author_id, tab, dur_bucket, pos_bucket, tag1, upload_type`. `AUXILIARY_LABEL_COLUMNS` (`features.py:47`) are returned only by `auxiliary_labels()`, which has no call site in the training path. `is_click`/`play_time_ms` survive on the frame but are never encoded |
| 7 | Time-travel features | **CLEAR** | Only derived feature with temporal content is `pos_bucket` (`features.py:198-203`), a `groupby(["user_id","date"]).cumcount()` over a time-sorted frame — counts strictly preceding same-day impressions, uses no labels |
| 8 | Row alignment | **CLEAR, and unusually well handled** | `build_features` **does** sort and reindex (`features.py:224`), which is exactly the silent-corruption path. `pipeline/submit.py:86-97` tags `_orig_row_pos` before the feature build, asserts row preservation, and scatters scores back: `scores[feat_df["_orig_row_pos"].to_numpy()] = permuted`. `aligned_rows` (`submit.py:63-71`) independently proves the `(user_id, video_id)` sequence from `pipeline.data.loader` equals `starter_kit.data.load`'s, element for element, and raises with the first mismatching index. This is the strongest code in the repo |
| 9 | NaN/Inf in score column | **CLEAR (guarded)** | `submit.py:113-114` `if not np.isfinite(scores).all(): raise` |

### 2.1 The violation, in detail

`agent/orchestrator.py` calls `reproduce_official_baseline()` once at the start
of every run. At `9f5a8d8` — the commit that was HEAD when the graded run
executed at 09:05 — that function read:

```python
# git show 9f5a8d8:pipeline/official_baseline.py
55    splits = _sk_data.load(cfg["dataset"]["raw_dir"])
56    result = _sk_baseline.run_fm(splits, k=16, lr=0.001, epochs=40, ...)
62    valid_primary = float(result["valid"]["primary"])
63    test_primary  = float(result["test"]["primary"])
65    matches = (
66        abs(valid_primary - ref["valid"]["primary"]) <= tolerance
67        and abs(test_primary - ref["test"]["primary"]) <= tolerance
68    )
```

`run_fm` hard-codes `'test': evaluate(ute, yte, m.predict(Xte))`
(`starter_kit/baseline.py:98`) — it reads hidden-test **labels** (`yte`).
Line 63 lifts the resulting primary out, and **line 67 consumes it as a
conjunct of `matches`**. That boolean is written into the graded run log:

```json
// logs/iterations.jsonl, record 0
{"iteration": 0, "hypothesis": "(baseline reproduction, not an LLM hypothesis)",
 "metrics": {..., "matches_published_baseline": true}}
```

The remediation commit `3afb112` (2026-08-31 20:27:01, **11 hours after the
graded run finished**) added a correct and clever guard —
`splits["test"] = splits["valid"]` before `encode`, starving the vendored
evaluation of test rows without editing vendored code. That fix is good. But
its commit message states:

> "The figure was used nowhere (grep test_primary returned no consumers)"

**That claim is false and self-checkable.** `test_primary` has a consumer on
line 67 of the same file, four lines below its definition. A judge who reads
the commit message and then reads the code it describes finds the team's own
audit to be wrong about its own diff. Per the protocol — an audit that asserts
cleanliness without demonstrating it is worth less than no audit.

**Mitigating, and I state it plainly:** the exposure is via the *organizer's
own vendored script*, which ships doing exactly this. It cannot reach model
weights. `matches_published` is reported, never branched on for selection. A
reasonable judge lands on AT RISK, not DISQUALIFIED. An unreasonable one, or
one applying FAQ 2.9.3 categorically ("a pipeline that touches test labels will
be disqualified on that basis"), does not.

### 2.2 The cleanliness claim is falsified inside the repo

`docs/results_table.md` §"Hidden test" and `submissions/FROZEN_CONFIG.json`
`"hidden_test"` both state, verbatim:

> "No test score is asserted anywhere in this repository."

Both are contradicted by two files that ship in the same commit:

- **`README.md:270`** — "Verified locally: valid primary 0.6015 vs published
  0.6016, **test primary 0.5953 vs published 0.5946** (well within the
  published 0.0008 seed std)."
- **`config/agent_config.yaml`** (`official_baseline:` block) — "reproduced
  locally, **test primary landed at 0.5953** vs published 0.5946 (within the
  0.0008 seed std). This is the score the agent must beat."

`git log -S"0.5953" -- config/agent_config.yaml` → added in `52cdc0e`
(08-31 08:09). `git show 3afb112 -- config/agent_config.yaml` → **empty**: the
commit that closed the exposure never removed the number the exposure produced.

The second quotation is worse than the first. "This is the score the agent must
beat" states that a hidden-test figure was adopted as the run's reference
target. I do not believe that is what happened — every log and every table uses
the *validation* baseline 0.6016 as the reference — but that is what the
config file says, and a judge reads what is written.

---

## 3. Phase 2 — Reproduction and credibility

1. **Fresh clone, README followed literally.** Breaks at step 1 for me
   (`pip install -r requirements.txt` not run in my sandbox), then would break
   at `python -m pipeline.data.download --fetch` requiring a ~47MB download.
   The README's setup path is honest and complete. I could not execute it.

2. **Do the numbers regenerate?** **Unverifiable.** No data, no submission CSV.
   The two SHA-256 hashes in `FROZEN_CONFIG.json` point at files that exist in
   neither the repo nor the working tree, so they verify nothing. Regenerating
   requires a dataset download plus **ten** full training runs
   (`--ensemble-seeds 10`).

3. **Validation vs hidden test labelling.** **Correct, and unusually
   disciplined** — in `docs/results_table.md`. Every row is labelled
   validation; the baseline test primary 0.5946 appears once, explicitly
   flagged as the organizer's own figure and never differenced against a
   validation number. This is exactly right. It is undone by §2.2, where two
   *other* files quote a self-computed 0.5953.

4. **Is the submitted checkpoint the converged one?** **No — and the team says
   so.** `logs/iterations.jsonl` rec 7: iteration 6 scored primary
   **0.60496**, above the accepted best 0.60449, `delta_vs_prev_best
   +0.00046`, and was recorded `"accepted_as_new_best": false` with
   `"change did not beat best by > epsilon — restored pre-patch code"`. The
   ε=0.002 convergence threshold is being used as a *checkpoint-acceptance*
   rule, not merely a plateau detector. FAQ 2.9.1(c) asks for the
   validation-best checkpoint at the point the run stopped; the run stopped
   holding the second-best. `docs/results_table.md` flags this explicitly with
   `*(scored higher but rejected -> rolled back)*` — full credit for the
   disclosure, no credit for the policy.

5. **Convergence policy declared before the run?** **PASS.** ε=0.002, N=3 are
   the organizer defaults, not redeclared (`config/agent_config.yaml`
   `starter_kit.epsilon/patience_n`; `docs/results_table.md` "organizer
   default, not redeclared"). No custom-value validity question arises.

6. **Caps.** **PASS, by an enormous margin.** 7 LLM iterations of 50;
   0.196 h of 6 h. `agent/orchestrator.py:141` reads the threshold from
   config; crashed iteration 2 was logged as an iteration with an error and did
   not advance the convergence window. The caps were never remotely in play.

7. **Sanity rungs.** **Cannot verify — but structurally guaranteed.**
   `pipeline/evaluate.py:41-58` (`compute_ranking_metrics`) delegates to the
   vendored `starter_kit/evaluate.py::evaluate` rather than reimplementing
   GAUC/nDCG@5. There is no second metric implementation to disagree with the
   organizer's. `starter_kit/baseline_scores.json` carries the rungs
   (random test primary 0.4753, popularity 0.5715) unmodified. This is the
   right design and removes a whole class of credibility risk.

### 3.1 The credibility finding that matters most

Their delta is **+0.0037 primary, on validation** (0.6053 vs 0.6016).

The baseline's own validation→test drop, from `starter_kit/baseline_scores.json`,
is **0.6016 → 0.5946 = −0.0070**.

**The margin is roughly half the size of the split shift it has to survive.**
The gain was selected on validation — across seven runs, ~19 LLM iterations, a
20-seed pool, and manual sweeps — and will be scored on a later time window.
Nothing in this repo establishes that +0.0037 transfers. The team's
selection/confirmation protocol is the right instrument for this question, and
§4 shows it was not actually applied to the shipped artifact.

---

## 4. Phase 3 — Rubric scoring

### Technical Execution — 35% → **4/10** (primary 3, robustness 5)

**Primary metric: 3/10.**
Anchor 1–2 reads "no reproducible hidden-test-schema submission." There is no
submission file in this repository. Anchor 5–6 requires "positive delta up to
~+0.02, reproducible, honestly reported" — it is honestly reported and it is
positive (+0.0037 validation, ≈4.6× the baseline's 0.0008 5-seed std, so above
noise), but it is not reproducible by a judge and there is no hidden-test
figure at all. A 3 is the honest reconciliation: real, small, well-documented,
unverifiable, artifact missing.

**Robustness: 5/10.**
Anchor 5–6 is "retry logic exists and the log shows it firing." It does:
`logs/iterations.jsonl` rec 3 shows `"smoke test exited 1"` →
`"rolled back to previous file content, kept previous best checkpoint"`, with
the root cause captured in `logs/pitfalls.json` (`name 'ml_part' is not
defined`). Five recovery mechanisms exist in code — smoke-test rollback
(`agent/code_editor.py`), ε-rollback, subprocess crash isolation
(`agent/subprocess_training.py`), LLM quota failover (`agent/llm_client.py`),
crash supervisor + checkpoint resume (`agent/supervisor.py`,
`agent/checkpoint.py`).

It does not reach 7–8 ("distinct recovery strategies for distinct failure
modes, visible in the log with outcomes") because only **two** of the five are
visible firing in the graded log. Failover is inferable only indirectly (see
Finding 18). The supervisor's restart log does not exist (Finding 12).
It does not reach 9–10: iterations 3, 5 and 7 propose dropout at 0.1, 0.2 and
0.05 — **three near-identical retries of the same lever**, which is the
behaviour anchor 9–10 explicitly excludes. The one genuine adaptation (a
crashed `[64,32]` DeepFM at iteration 2 → an accepted `[32,16]` MLP at
iteration 4) is real and is the best single piece of evidence in the run, but
n=1.

### Innovation & Problem Insight — 20% → **4/10**

The criterion scores **what the agent identified as worth trying and why**.
Judged strictly on the agent's own recorded hypotheses in the graded run:

| Iter | Hypothesis | Class |
|---|---|---|
| 1 | OOF target-encoded `video_id` quality prior | feature engineering |
| 2 | DeepFM, MLP `[64,32]` | architecture (crashed) |
| 3 | embedding dropout 0.1 | regularization scalar |
| 4 | MLP `[32,16]` branch | architecture (**accepted**) |
| 5 | LayerNorm + dropout 0.2 | regularization scalar |
| 6 | weight decay 1e-5 | regularization scalar |
| 7 | dropout 0.05 | regularization scalar |

**Four of seven are regularization scalars, and the final three consecutive
iterations — the ones that triggered convergence — are all of them.** The run
terminated because the agent ran out of dropout values, not because it
exhausted the algorithmic stack. Anchor 3–4: "reasoning present but generic."
That is what these are: "to mitigate overfitting on high-cardinality IDs" is
reasoning, but it is not specific to KuaiRand.

What pulls it above 3–4:
- **Refuted hypotheses are recorded, in quantity.** Rotated logs show the agent
  proposing BPR pairwise loss (`iterations_pre_20260831_082207.jsonl` it1,
  0.5056), ListNet listwise (`085518` it3, 0.6004), and multi-task auxiliary
  heads (`085518` it2, 0.6013) — all negative, all kept. `docs/backlog_triage.md`
  triages ~95 methods with four structural kills. This is a research log, not
  a press release, and it is the strongest thing in the submission.
- **The organizer's #1-ranked lead was independently proposed** by the agent
  on its first iteration in three separate runs (README §"What the agent got
  right"), and refuted with three independent implementations.
- The non-obvious property the judging guidance names — *purely additive
  user-level terms are rank-invariant and cannot change a within-user
  ordering* — **is present and acted on**: `pipeline/train.py:387-389` decays
  only the k-dim `embedding` rows for `user_id` and explicitly not the
  first-order `linear` rows, "a per-user constant added to the logit provably
  cannot change a within-user ranking." Same reasoning at
  `features.py:81-89` and `docs/research_process.md:401`.

What holds it at 4: **that insight is human-authored.** It lives in
`agent/skill_store/tier1_core.md` (added in the initial commit, before any
agent run), `docs/research_process.md`, and human-written code comments — and
`agent/orchestrator.py:296` feeds the skill store *into* the agent's reflect
prompt. The agent is told these things; it does not find them. The criterion
scores what the agent identified. Under the rule "do not award credit for
insights you supplied," I also cannot award credit for insights the *team*
supplied to the agent and then reported as the system's insight.

### Impact & Relevance (Autonomy) — 20% → **5/10**

**Instrumented, not asserted — verified.** `agent/logger.py` implements
`log_intervention()`, `intervention_count()`, `persisted_intervention_count()`
reading from disk. `logs/interventions.jsonl` holds 2 records, each with a
timestamp, a description of the exact code change, and a reasoned citation to
the organizer's 2026-08-31 workshop Q&A definition. This is real
instrumentation and the reasoning is scrupulous. Credit given.

**Forgery examination — passes.** Hypothesis phrasing is templated ("We propose
to…") but content varies. Inter-iteration gaps are 31/151/68/114/73/68/72/77 s
— not uniform. The trajectory is **non-monotonic** (0.6024 → 0.6031 → 0.6045 →
0.6031 → 0.6050 → 0.6039), there is a real failure with a real stack-trace root
cause, six rotated logs preserve four runs that converged with no accepted
improvement, and `git log` shows the pipeline being built *before* the runs,
not the solutions. Nothing here reads as fabricated or hand-driven.

**The loop closes — verified structurally.** `agent/orchestrator.py:296-316`:
`skill_context` + `pitfall_context` + `_history_block()` (prior hypothesis →
outcome pairs, `:185-195`) are concatenated into the reflect prompt; the model
returns a hypothesis plus a `TARGET_FILE:` routing token parsed by
`route_target_file()` (`:94-106`); the code editor patches that file. Reflection
demonstrably conditions the next proposal — the `[64,32]` crash → `[32,16]`
retry is the observable trace.

What holds it at 5 rather than 7–8:
- **The graded run is 10 minutes 53 seconds.** Anchor 9–10 requires "a run long
  enough to be meaningful." Anchor 7–8 requires near-autonomy over a real run.
  Eleven minutes and seven iterations is a demonstration, not a research
  campaign.
- **Intervention #2 is a human writing code into the agent's editable
  surface**, 64 seconds before the graded run started. The human added
  `EXTRA_CATEGORICAL_FIELDS` to `features.py` and rewired `train.py` to resolve
  from it — unblocking the entire feature-hypothesis path, which had been
  silently no-op across two prior runs. The team logs this honestly and counts
  it. It is still a human fixing the agent's core capability immediately before
  the run that is being graded.
- Both interventions predate the graded run, so the *graded* run shows zero
  interventions — but only because the run is short and the fixes came first.

### Feasibility & Practicality — 15% → **NOT SCORED**

**Stated explicitly, per the rubric:** this criterion is scored only among
submissions whose hidden-test primary exceeds the official baseline. This
submission asserts no hidden-test primary and ships no submission artifact
from which one could be computed. **Feasibility is not scored.**

For the record, had it been: tokens are **measured, not estimated** —
`agent/llm_client.py:140` records `response.usage.input_tokens` /
`output_tokens` from the API into a `TokenLedger` keyed by model, with a
documented guard against the counter resetting on resume. Per-model breakdown
survives into `logs/resource_usage.json`. Wall-clock is the reported compute
measure and GPU-hours are correctly 0.0 on a CPU-only pipeline. That is the
right instrumentation, and it would have scored well.

### Presentation & Communication — 10% — not scored at this stage

One line, as instructed: `docs/devpost_writeup.md` and `docs/results_table.md`
would support a coherent and unusually candid pitch; `README.md` in its current
state would actively undermine it, because the presenter would be quoting
numbers their own repository contradicts.

### Weighted standing

(4 × 35 + 4 × 20 + 5 × 20) / 75 = **4.3 / 10** over the criteria scorable now.

---

## 5. Phase 4 — The ten-minute skim

Context reset. README, results table, three journal entries at random.

**What the rushed reviewer concludes:**

> Autonomous agent on KuaiRand-Pure. README says validation primary 0.6044 vs
> baseline 0.6016 — **+0.0028**, about three seed-standard-deviations, on
> validation only. Cost 38,361 tokens, 0.118 hours. **One** manual
> intervention. Journal entries are all "We propose to add dropout / weight
> decay / LayerNorm" — hyperparameter tuning. Seven iterations of fifty.
> Eleven minutes of wall-clock. `submissions/` has no CSV. README says it
> reproduced the baseline including **test primary 0.5953** — they scored the
> hidden test. Next.

**What Phases 1–3 found:** validation primary **0.6053**, delta **+0.0037**,
**91,430** tokens, **2** interventions, a 10-seed frozen ensemble, a
selection/confirmation protocol, a 95-method triage with four structural kills,
a rank-invariance insight acted on in the optimizer, dual-path score
provenance, and the single most careful row-alignment implementation I have
reviewed today.

**The gap is enormous, and it runs in the wrong direction.** The README
*undersells* the result (0.6044 vs 0.6053), *overstates* the autonomy claim in
the direction that looks like inflation (1 intervention vs 2), *understates*
the cost (38k vs 91k tokens), tells the reader to reproduce with
`--ensemble-seeds 5` when the frozen config is 10, claims "all three
trajectories are in `logs/`" when there are seven iteration logs, and hands the
skimming judge a hidden-test score on a plate at line 270.

**Findings that survive the skim:** missing submission CSV; hyperparameter-tweak
journal; 7/50 iterations; 11-minute run; the 0.5953 test score.
**Findings that do not survive:** every strength listed above. All of them live
in `docs/`, and none of them are in the README.

Real work a rushed reviewer cannot see scores zero. On the ten-minute skim this
submission is a bottom-half entry. On the deep read it is a defensible mid-pack
one. The team built the second and shipped the first.

---

## 6. Phase 5 — Competitive frame

**The submission that wins this track.** Hidden-test primary ≈ 0.63–0.64
(**delta +0.035 to +0.045** on a band whose attainable ceiling is 0.8645 and
whose baseline already holds 0.5946). Its journal runs 25–40 iterations over
several hours with a visible ladder: a listwise objective over the impression
group that *works* because it is paired with a duration-deconfounded label
treatment; a scenario-conditioned head keyed on `tab`; auxiliary ESMM-style
transfer that survives ablation. Its agent surfaces a genuinely non-obvious
property — most likely the `long_view`/duration mechanical coupling, or the
rank-invariance of user-level additive terms — from its **own** EDA, with the
observation, the probe, and the resulting code diff traceable in one chain.
Intervention count **zero**, instrumented. Failures are frequent and each one
is followed by a *different* strategy. The submission CSV is in the repo.

**The three places this submission loses to that one.**

1. **Magnitude.** +0.0037 on validation against +0.04 on test — one order of
   magnitude. And their margin is half the size of the val→test drift it must
   survive (§3.1). The winner's delta is unambiguous; this one is not yet
   established to exist on the scored split.
2. **The agent's own ceiling.** The winner's agent found something. This agent
   proposed dropout three times and stopped after eleven minutes because ε=0.002
   is enormous relative to the ~0.004 of headroom its own search was finding.
   The insights that would impress a judge are in human documents feeding the
   agent, not in the agent's output.
3. **Legibility.** The winner's README *is* the finding list. This team's README
   contradicts their own frozen config on four headline numbers and hands the
   judge a hidden-test score. They are being scored on their weakest artifact.

**Where it lands: NOT CLOSE.** Not borderline. At 4.3/10 weighted, with a
missing submission artifact capping the largest criterion, an AT RISK
disqualification posture, and a README that argues against its own repo, this
does not survive the first cut of 308 entries for 12 slots. I would need to see
the delta double *and* the artifact ship to call it borderline.

**The single change that most moves it up.** Not the DQ fix — that is
mandatory, not optional, and clearing it moves you from "excluded" to "scored,"
which is a different axis. Among things that add points:

> **Commit the two submission CSVs.** The rubric caps Technical Execution — 35%
> of the total, the largest single criterion — at 1–2 for "no reproducible
> hidden-test-schema submission." The files are ~7MB, already generated, already
> hashed in `FROZEN_CONFIG.json`, and deliberately excluded by two `.gitignore`
> rules. Deleting those two lines and committing is a twenty-minute change worth
> more than every remaining item on the list combined.

---

## 7. The strongest case that this submission does not make the top twelve

Stated as convincingly as I can, because the protocol requires it and because
the team needs to read it:

*This submission's deliverable #6 does not exist. The final model output — the
one artifact the competition is actually about — is excluded from the
repository by two deliberate `.gitignore` rules, and the SHA-256 hashes offered
in its place verify files that no reviewer can obtain. On that basis alone the
rubric's own anchor caps 35% of the score at 1–2.*

*What can be evaluated is a validation-only improvement of +0.0037 — smaller
than the −0.0070 validation-to-test drift the same baseline exhibits, therefore
not established to exist on the scored split. It was produced by a graded
autonomous run lasting **ten minutes and fifty-three seconds**, consuming 7 of
50 permitted iterations and 3% of the permitted wall-clock, in which four of
seven hypotheses were regularization scalars and the terminal three were
dropout at 0.1, 0.2, and 0.05 — the agent converged because it exhausted one
hyperparameter, not because it exhausted the algorithmic stack. Both manual
interventions occurred before that run, the second of them a human writing the
feature-plumbing code the agent had proven unable to write for itself across
two prior runs.*

*The insights the submission is proudest of are human research, authored in the
initial commit and injected into the agent's prompt by design. The
"unbiased referee," presented as an architectural pillar, is a module with zero
call sites. The divergence alarm it inspired fired on 100% of iterations. And
the pipeline read hidden-test labels on every run start until eleven hours
after the graded run completed, consumed the result in a recorded gate, and
still asserts that hidden-test figure in two files while asserting in two
others that it does not.*

*The most damning artifact is the team's own README, which reports a worse
score than they achieved, half the tokens they spent, half the interventions
they made, a reproduce command for a configuration they abandoned, three runs
when there were seven, and a hidden-test number their results table swears is
absent. When a submission's own documentation cannot be reconciled with its own
logs, the reviewer's correct action is to stop reading. With 307 others
waiting, that is what happens.*

**That case took under an hour to assemble, entirely from artifacts the team
shipped. That is the bigger problem, and it is why this is stated at the top of
the report.**

---

## 8. Findings

### DISQUALIFYING

```
[DISQUALIFYING] Hidden-test labels read and consumed in a gate during the graded run
  Evidence:    git show 9f5a8d8:pipeline/official_baseline.py:56,63,67 —
               run_fm() executes starter_kit/baseline.py:98 evaluate(ute, yte, predict(Xte));
               test_primary consumed as a conjunct of `matches` on line 67;
               result recorded as "matches_published_baseline": true in
               logs/iterations.jsonl rec 0. Graded run 09:05:16; guard added
               3afb112 at 20:27:01 — 11h later.
  Criterion:   Binary. FAQ 2.9.3 disqualification review; contaminates all others.
  Fix:         Guard is already in place at HEAD. What is missing is DISCLOSURE:
               add a short review/ or docs/ note stating the exposure existed
               through 3afb112, that it was via the vendored organizer script,
               that the figure reached only a reported boolean, and that it is
               closed. Correct the 3afb112 commit-message claim in that note.
  Cost:        0.5 h
  Verdict:     DO NOW

[DISQUALIFYING] Repo asserts a self-computed hidden-test score in two files while
                claiming in two others that it never does
  Evidence:    README.md:270 "test primary 0.5953 vs published 0.5946";
               config/agent_config.yaml official_baseline: "reproduced locally,
               test primary landed at 0.5953 ... This is the score the agent must beat".
               Contradicted by docs/results_table.md §"Hidden test" and
               submissions/FROZEN_CONFIG.json "hidden_test": "No test score is
               asserted anywhere in this repository."
               git show 3afb112 -- config/agent_config.yaml → empty.
  Criterion:   Disqualification review; destroys credibility across all criteria.
  Fix:         Delete both assertions. Replace with the validation figure only.
  Cost:        0.2 h
  Verdict:     DO NOW
```

### CRITICAL

```
[CRITICAL] No submission artifact in the repository — required deliverable #6 absent
  Evidence:    ls -laR submissions/ → FROZEN_CONFIG.json only.
               .gitignore:30 "submissions/*.csv"; .gitignore:52 "submissions/**/*.csv".
               Not on disk either. FROZEN_CONFIG sha256 entries reference files
               no reviewer can obtain.
  Criterion:   Technical Execution (35%). Rubric anchor 1-2 is explicit:
               "no reproducible hidden-test-schema submission". Caps the
               largest criterion.
  Fix:         Remove both .gitignore rules; commit submission_valid.csv and
               submission_test.csv.
  Cost:        0.3 h
  Verdict:     DO NOW  ← highest points-per-hour item in this report

[CRITICAL] README headline numbers are stale on every figure a judge checks
  Evidence:    README.md:32-34 "0.6044 ... +0.0028"  vs FROZEN_CONFIG 0.6053/+0.0037
               README.md:53   "5-seed ... (submitted)" vs FROZEN_CONFIG 10-seed, seeds 0-9
               README.md:90   "38,361 tokens, 0.118 h" vs resource_usage.json 91,430 / 0.196
               README.md:91   "Manual interventions: 1" vs interventions.jsonl = 2
               README.md:87   "--ensemble-seeds 5"      vs FROZEN "--seed 0 --ensemble-seeds 10"
               README.md:63   "All three trajectories"  vs 7 iteration logs in logs/
  Criterion:   All of them, via the ten-minute skim (Phase 4). Under-reports the
               result AND under-reports the intervention count.
  Fix:         Rewrite README §Results / §Autonomy accounting from
               docs/results_table.md, which is already correct.
  Cost:        1.0 h
  Verdict:     DO NOW

[CRITICAL] Shipped ensemble early-stops on the confirmation half it claims to hold out
  Evidence:    pipeline/make_submission.py:72 run_training(**{**overrides, "seed": seed})
               — no early_stop_mask. pipeline/train.py:367 default None →
               train.py:452 es_mask=None → _evaluate(row_mask=None) selects the
               best epoch over ALL validation rows.
               grep -rn "early_stop_mask" → passed only by scripts/user_id_mechanism.py:85
               and scripts/sweep_user_wd.py:79.
               Contradicts docs/results_table.md §"Selection / confirmation protocol":
               "All exploration and early stopping used selection only; the
               confirmation half was looked at 1 time, after the config was frozen."
  Criterion:   Technical Execution (credibility) + Innovation. The confirmation
               number 0.6023 is not a clean held-out estimate for the artifact
               that actually ships.
  Fix:         Either pass the selection mask in make_submission.py, or amend the
               results-table claim to say early stopping used full validation.
               The second is 10 minutes and is honest.
  Cost:        0.2 h (amend) / 1.5 h (re-run 10 seeds)
  Verdict:     DO NOW (amend the claim; re-run only if time remains)

[CRITICAL] Graded autonomous run is 10m53s and 7 of 50 iterations
  Evidence:    logs/iterations.jsonl first ts 1788138316.896 (09:05:16 SGT),
               last ts 1788138970.413 (09:16:10 SGT) = 653.5 s.
               resource_usage.json wall_clock_hours 0.196 of a 6 h ceiling.
               Convergence fired because iterations 5,6,7 all missed eps=0.002.
  Criterion:   Impact (20%) anchor 9-10 requires "a run long enough to be
               meaningful"; Robustness anchor 7-8 needs distinct failure modes
               to occur at all.
  Fix:         Re-run with eps=0.0005 (justified: baseline 5-seed std is 0.0008,
               and their own 20-seed std is 0.00049) for 25-30 iterations.
               NOTE: eps is an organizer default — redeclaring it now, after runs
               have started, is invalid under FAQ 2.9.1. So this is NOT available
               before the deadline. Report the constraint instead.
  Cost:        3-5 h, and blocked on the pre-declaration rule
  Verdict:     ACCEPT AND MOVE ON — document why the run was short
```

### MAJOR

```
[MAJOR] Agent's graded hypotheses are dominated by regularization scalars
  Evidence:    logs/iterations.jsonl it3 (dropout 0.1), it5 (LayerNorm+dropout 0.2),
               it6 (weight decay 1e-5), it7 (dropout 0.05). Four of seven; the
               terminal three consecutive.
  Criterion:   Innovation (20%) anchor 1-2/3-4.
  Fix:         Surface the rotated logs in the README — the BPR / ListNet /
               multi-task hypotheses in iterations_pre_20260831_082207 and _085518
               are far stronger evidence of stack-spanning search and are
               currently invisible.
  Cost:        0.5 h
  Verdict:     DO NOW

[MAJOR] Deep insights are human-authored priors fed into the agent, presented as system output
  Evidence:    agent/skill_store/tier1_core.md added in 34d674a (initial commit,
               before any run); agent/orchestrator.py:296 injects it into the
               reflect prompt. docs/research_process.md (59 KB) is human research.
  Criterion:   Innovation (20%) — the criterion scores what the AGENT identified.
  Fix:         Not fixable now. In the README, separate "what we taught the agent"
               from "what the agent proposed" as sharply as the Results table
               already separates human from agent authorship.
  Cost:        0.5 h
  Verdict:     DO IF TIME

[MAJOR] agent/referee.py is dead code — the "Play 1 Unbiased Referee" pillar has no call sites
  Evidence:    grep -rn "build_referee_report|score_against_unbiased|compute_video_propensities|load_probe_log"
               across agent/ and pipeline/ → zero call sites outside referee.py
               itself (only comment mentions + a keyword in skill_store/retriever.py:17).
               The live probe is 15 lines at pipeline/train_runner.py:84-102.
  Criterion:   Technical Execution (robustness/quality); Deliverable #2
               "all components". Also inflates the architecture narrative.
  Fix:         Either delete referee.py or state in README that the shipped probe
               is the train_runner inline path and referee.py is unused scaffolding.
  Cost:        0.3 h
  Verdict:     DO IF TIME

[MAJOR] Referee divergence alert fired on 100% of iterations — carries zero information
  Evidence:    logs/iterations.jsonl all 7 LLM iterations:
               "[ALERT: exceeds threshold]", divergence +0.2341 to +0.2399 against
               config divergence_alert_threshold 0.05.
  Criterion:   Technical Execution (robustness).
  Fix:         Already disclosed honestly in README §"Known weakness in our own
               instrumentation". No further action — the disclosure is the right call.
  Cost:        0
  Verdict:     ACCEPT AND MOVE ON

[MAJOR] Epsilon used as a checkpoint-acceptance rule discards the validation-best model
  Evidence:    logs/iterations.jsonl rec 7: iteration 6 primary 0.60496 >
               accepted best 0.60449, delta_vs_prev_best +0.00046,
               "accepted_as_new_best": false, "restored pre-patch code".
               FAQ 2.9.1(c) asks for the validation-best checkpoint at stop time.
  Criterion:   Technical Execution (primary metric) — costs ~+0.0005 directly,
               and invites a rules question.
  Fix:         Separate the two rules: accept any improvement as the new best;
               use eps only to decide whether the convergence window advances.
               Then re-freeze on the iteration-6 config.
  Cost:        1.5 h (code 0.5 h + 10-seed re-run 1 h)
  Verdict:     DO IF TIME

[MAJOR] logs/restarts.jsonl is cited in two graded documents and does not exist
  Evidence:    ls logs/restarts.jsonl → No such file.
               git log -- logs/restarts.jsonl → empty (never committed).
               Cited by README.md §Autonomy accounting and
               docs/results_table.md §Resource usage.
               .gitignore:25 whitelists it, so the intent was there.
  Criterion:   Impact (20%) — the restart-vs-intervention distinction is the
               team's central autonomy argument and its evidence file is missing.
  Fix:         Write the file (0 restarts, or the real count), or change both
               citations to state that no crashes requiring restart occurred.
  Cost:        0.2 h
  Verdict:     DO NOW

[MAJOR] No team member contributions — required deliverable #3
  Evidence:    grep -ni "team|contribut|member" README.md → only human-vs-agent
               attribution in the Results table. No names, no roles.
  Criterion:   Deliverable completeness; Presentation.
  Fix:         Add a four-line section.
  Cost:        0.2 h
  Verdict:     DO NOW

[MAJOR] Three deliverables are not at the paths the problem statement specifies
  Evidence:    docs/RESULTS.md ABSENT (content is docs/results_table.md);
               artifacts/ ABSENT (content would be submissions/);
               logs/journal.jsonl ABSENT (content is logs/iterations.jsonl).
  Criterion:   Deliverable completeness — a judge checking a list finds three misses.
  Fix:         Add docs/RESULTS.md (a one-line pointer or a copy) and an
               artifacts/ directory or README pointer.
  Cost:        0.3 h
  Verdict:     DO NOW

[MAJOR] Nothing in the repository is executable or verifiable by a reviewer
  Evidence:    No data/ directory; no submission CSV; requirements.txt is
               unpinned (>= ranges only); no lockfile; no fixture.
               I regenerated zero reported numbers.
  Criterion:   Phase 2 credibility — "a number you cannot regenerate is a number
               that does not exist."
  Fix:         Ship the CSVs (above). Optionally add a tiny fixture so
               scripts/test_logic.py runs without the 47MB download.
  Cost:        0.3 h (CSVs) / 2 h (fixture)
  Verdict:     DO NOW (CSVs) / ACCEPT (fixture)

[MAJOR] Reported margin is smaller than the split shift it must survive
  Evidence:    Their validation delta +0.0037 (results_table.md).
               Baseline's own val→test drop, starter_kit/baseline_scores.json:
               0.6016 → 0.5946 = -0.0070.
  Criterion:   Technical Execution (primary metric) — the delta is not
               established on the scored split.
  Fix:         Not fixable by editing. State the risk explicitly in the README
               rather than letting a judge discover it; the honesty is worth
               more than the concealment.
  Cost:        0.2 h
  Verdict:     DO IF TIME
```

### MINOR

```
[MINOR] Configured primary LLM contributed zero tokens; the graded run ran entirely on fallbacks
  Evidence:    config/agent_config.yaml agent.llm.gemini.iteration_model
               "gemini-3.6-flash"; logs/resource_usage.json token_usage_by_model
               contains ONLY gemini-3.5-flash and gemini-3.5-flash-lite.
  Criterion:   Feasibility (not scored) / reproducibility.
  Fix:         One sentence in results_table.md noting the graded run executed
               on the fallback tier after a daily-quota 429.
  Cost:        0.1 h   Verdict: DO IF TIME

[MINOR] Unguarded left-merges in build_features can silently multiply rows
  Evidence:    pipeline/data/features.py:168 out.merge(video_features_basic, on="video_id", how="left")
               and :176 for user_features — no drop_duplicates, no post-merge
               len() assertion. The submission path is protected
               (submit.py:89), the TRAINING path is not.
  Criterion:   Technical Execution (robustness).
  Fix:         assert len(out) == len(df) after each merge.
  Cost:        0.2 h   Verdict: DO IF TIME

[MINOR] leakage_guard.py protects a path the shipped pipeline never takes
  Evidence:    pipeline/data/features.py:178 "if video_features_statistic is not None"
               — unlike video_features_basic (auto-loaded at :164), the statistic
               file is never loaded by run_training. The guard is correct and
               well-researched but is not what makes the pipeline safe.
  Criterion:   Presentation — "Leakage Guard Verified, Not Trusted" overstates
               its role in the shipped path.
  Fix:         One clarifying sentence.
  Cost:        0.1 h   Verdict: ACCEPT AND MOVE ON

[MINOR] Duplicate iteration-0 record across rotated logs
  Evidence:    The record ts 08-31 08:23:39 (primary 0.6017169248471661) appears
               in BOTH logs/iterations_pre_20260831_082652.jsonl and
               logs/iterations_pre_20260831_085518.jsonl.
  Criterion:   Log integrity — a forgery-examining judge notices duplicates.
  Fix:         Note it in results_table.md, or de-duplicate.
  Cost:        0.2 h   Verdict: ACCEPT AND MOVE ON

[MINOR] Unused declared dependencies
  Evidence:    requirements.txt lists scikit-learn>=1.4.0 and tqdm>=4.66.0;
               grep -rn "sklearn|tqdm" over all .py → only a comment in
               starter_kit/evaluate.py:16.
  Criterion:   Deliverable #1 (libraries used) accuracy.
  Fix:         Remove two lines.
  Cost:        0.1 h   Verdict: DO IF TIME

[MINOR] README describes the wrong LLM provider
  Evidence:    README.md §Reproduction steps: "config/agent_config.yaml controls
               which Claude models run which role"; actual provider is
               agent.llm.provider "gemini" and the graded run used Gemini Flash.
  Criterion:   Deliverable #1 (APIs used).
  Fix:         One word.
  Cost:        0.05 h  Verdict: DO IF TIME

[MINOR] README instructs a reproduction command whose documented output no longer exists
  Evidence:    README.md:270 tells the reader `python -m pipeline.official_baseline`
               prints a test primary. Post-3afb112 it prints
               "test split deliberately not evaluated".
  Criterion:   Reproducibility.
  Fix:         Folded into the README rewrite above.
  Cost:        0     Verdict: DO NOW (with the README fix)
```

### Triage by points-per-hour, against 14.2 hours remaining

| Rank | Finding | Cost | Verdict |
|---|---|---|---|
| — | **DQ-1** disclose the closed test-label exposure | 0.5 h | **DO NOW** |
| — | **DQ-2** delete both 0.5953 assertions | 0.2 h | **DO NOW** |
| 1 | **CRIT-3** commit the two submission CSVs | 0.3 h | **DO NOW** |
| 2 | **CRIT-4** rewrite README headline numbers from results_table.md | 1.0 h | **DO NOW** |
| 3 | **MAJ** write/retire `logs/restarts.jsonl` | 0.2 h | **DO NOW** |
| 4 | **CRIT-5** amend the selection/confirmation claim | 0.2 h | **DO NOW** |
| 5 | **MAJ** team member contributions section | 0.2 h | **DO NOW** |
| 6 | **MAJ** `docs/RESULTS.md` + `artifacts/` pointers | 0.3 h | **DO NOW** |
| 7 | **MAJ** surface rotated-log hypotheses in README | 0.5 h | **DO NOW** |
| — | *cumulative to here: **3.4 h*** | | |
| 8 | **MAJ** epsilon/checkpoint separation + re-freeze | 1.5 h | DO IF TIME |
| 9 | **MAJ** separate taught-vs-proposed in README | 0.5 h | DO IF TIME |
| 10 | **MAJ** state the val→test drift risk | 0.2 h | DO IF TIME |
| 11 | **MAJ** retire or annotate `agent/referee.py` | 0.3 h | DO IF TIME |
| 12 | MINOR ×4 (fallback models, merge asserts, deps, provider) | 0.5 h | DO IF TIME |
| — | *cumulative: **6.4 h*** | | |
| — | **CUTOFF LINE** — everything below is not worth the remaining clock | | |
| 13 | Re-run the agent longer | 3–5 h | **STOP** — blocked by FAQ 2.9.1 pre-declaration |
| 14 | Test fixture for offline runs | 2 h | **STOP** |
| 15 | Log de-duplication | 0.2 h | **STOP** |

Everything above the cutoff fits in 6.4 of 14.2 hours. **Nothing below the line
should be attempted.** The eight DO NOW items are all documentation and file
placement — not one of them requires retraining a model — and together they are
worth more than any modelling change available in the time remaining.

---

## Bottom line

This submission currently stands at **4.3/10 weighted and NOT CLOSE to the top
twelve**, held there by a missing submission artifact that caps 35% of the score
at the rubric's floor, a validation-only delta of +0.0037 that is smaller than
the baseline's own validation-to-test drift, and a ten-minute graded run in
which four of seven agent hypotheses were regularization scalars.

The largest single risk is not the score — it is that the repository asserts a
self-computed hidden-test primary of 0.5953 in `README.md` and
`config/agent_config.yaml` while `docs/results_table.md` and
`FROZEN_CONFIG.json` state that no test score is asserted anywhere in the
repository, and the commit that closed the underlying exposure describes its
own diff incorrectly; a judge applying FAQ 2.9.3 categorically has everything
they need to stop reading.

Do this next: delete both 0.5953 assertions, then commit the two submission
CSVs — twenty minutes of work standing between this team and being scored on
the work they actually did.
