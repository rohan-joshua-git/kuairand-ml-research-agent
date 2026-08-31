# Devpost Writeup

Track 2 — Autonomous Machine Learning Research Agent for Recommender Systems
(KuaiRand-Pure). Every number below was produced by running the code in this
repo; nothing is projected or rounded in our favour. Where the agent failed,
the failure is reported as a result, because the delta over baseline is
scored continuously and a false claim is worse than an honest negative.

## Inspiration

Machine learning engineers spend most of their time in one loop: read the
problem, inspect the data, engineer features, train and tune, evaluate,
reflect, and go again (see `README.md` Figure 1). It's structured and
repetitive — which is exactly what makes it a plausible target for an LLM
agent to run on its own, rather than a human grinding through it by hand.

We also found this exact idea has essentially never been run as a
competition. **AutoRecLab** (Beel et al., arXiv:2510.18104) explicitly
calls for benchmarks and competitions evaluating agents on producing
reproducible RecSys findings with minimal human input. **AgentX**
(Kuaishou, arXiv:2606.26859) — from the same company that published
KuaiRand — already runs a closed-loop agent internally on their production
ranking system. We built the open, offline, benchmarked version of that
idea on Kuaishou's own public dataset.

## What it does

An autonomous ML research agent that:
1. Trains a working reference pipeline on KuaiRand-Pure.
2. Iterates on it entirely on its own — proposing hypotheses, writing code,
   training, evaluating, and deciding what to try next — using only train
   and validation data.
3. Guards itself against the single biggest failure mode in this space:
   optimizing a biased validation proxy instead of genuinely improving.

Three things make this agent different from "fork AIDE and point it at a
dataset":

- **An unbiased referee.** KuaiRand-Pure ships a ~1.19M-row uniformly-
  random-exposure log outside the prescribed train/val/test split. We use
  it as a second, unbiased scoring channel, and drive course-correction off
  the *divergence* between biased-validation score and unbiased-probe
  score — not off validation score alone.
- **A compression gate.** Before any checkpoint is designated final, a
  fresh "reproducer" agent — with no memory of the search that found it —
  has to reproduce the approach from a short, honest summary. This
  directly targets a documented failure (SpecBench): an agent choosing a
  97%-validation/0%-held-out lookup-table artifact over a genuine
  53%/43% solution, because validation score was the only thing it
  searched against.
- **A warm-started, ablation-first search.** Instead of rewriting the whole
  pipeline every round, the agent ablates pipeline blocks each iteration to
  find which one is actually carrying the score, and targets only that
  block — informed by a pre-seeded, tiered domain-knowledge store rather
  than starting cold. Published ablations (HASTE) show this cuts
  iterations-to-best roughly in half versus flat/no knowledge loading.

We also found two undocumented quirks in the dataset itself: `is_click`
resolves to a genuine click in one UI and a duration-thresholded
"valid play" in another (keyed on the `tab` field), and the shipped
`video_features_statistic` columns are month-long averages that leak
across the train/val/test boundary. Both are handled explicitly (see
`pipeline/data/label.py` and `leakage_guard.py`) rather than silently.

## How we built it

- **Agent brain:** provider-agnostic behind one `LLMClient` protocol
  (`agent/llm_client.py`). Runs used Gemini Flash on the free tier; the
  Anthropic backend (Sonnet for code generation, Opus for reflection) is
  wired and selectable from `config/agent_config.yaml`. One verified gotcha
  is documented in code: Gemini 3.x thinks by default, and at the default
  thinking level internal thinking tokens can consume the entire output
  budget before any visible text is emitted — we pin `thinking_level="low"`
  and re-verified non-empty output at the exact budgets the loop uses.
- **Starting pipeline:** a PyTorch Factorization Machine over the official
  baseline's own five fields (user_id, video_id, author_id, tab,
  dur_bucket), early-stopped on validation *primary* rather than on loss.
  It reaches **valid GAUC 0.6676 / nDCG@5 0.5358 / primary 0.6017** against
  the published **0.6674 / 0.5357 / 0.6016** — parity within the 0.0008
  five-seed std. This was a deliberate correction: our first editable model
  scored 0.5201 and the agent burned iterations climbing toward parity
  instead of past it. Two measured causes — `duration_ms` fed raw into an
  MLP (mean ~9.7e4, std ~9.5e4, so it dominated small-initialised
  embeddings) and `author_id` never joined at all.
- **Scoring:** `pipeline/evaluate.py` imports and calls the organizer's
  vendored `starter_kit/evaluate.py` rather than reimplementing GAUC/nDCG@5,
  so a validation number cannot silently diverge from what actually grades
  the hidden test set.
- **Every scored run is a fresh subprocess.** This started as a bug fix and
  became a design rule. The orchestrator is long-lived and had imported the
  pipeline at startup, so `from x import y` bindings meant it was scoring
  the *pre-patch* code every iteration. A subprocess re-imports whatever is
  on disk, and additionally isolates the loop from crashes in
  agent-authored code.

Full run logs, intervention counts and token/wall-clock usage are in
`logs/` and `docs/results_table.md` (generated by `agent/report.py`).

## Challenges we ran into

**The agent proposed the right idea and could not implement it.** The first
live iteration produced the hypothesis "replace pointwise BCE with a
user-grouped BPR pairwise loss" — independently landing on the organizer's
own #1-ranked lead. But the editable allowlist covered only `features.py`
and `label.py`, and file routing was derived from the *ablation block name*,
computed before the hypothesis existed. So a loss-function hypothesis was
routed to the feature file, where a loss function cannot be written. The
agent did the only in-scope thing available: it added a `sort_values("user_id")`
step, preparing for a pairwise loss it had no way to write. We added
`train.py` to the allowlist and now route off what the hypothesis itself
says it needs to change (the reflection model emits a `TARGET_FILE` line,
with keyword routing as a fallback).

**A submission that passes every check and is still wrong.** `build_features`
sorts rows by `user_id`; the raw logs are not user-sorted. Scores were
therefore written against permuted rows while still satisfying the official
`--check`, because that check validates row *identity* columns, not whether
your score belongs to that row. Fixed by carrying an explicit original-position
column through the feature build and un-permuting before writing, verified
against the vendored writer plus a per-row spot check.

**Leakage that does not look like leakage.** `play_time_ms` was known to be
a near-direct proxy for `long_view` (corr 0.64). Auditing the same causal
class surfaced `comment_stay_time` (corr 0.17 — staying in the comments
implies you long-viewed), `profile_stay_time` and `is_profile_enter`: all
measured *during or after* the impression being predicted. Only
`duration_ms`, an item property known before exposure, survives as a numeric
input — which is exactly the official baseline's own conclusion.

**Free-tier outages killed runs mid-iteration.** A 503 took down one live run
and daily-quota 429s took down another. LLM calls now get bounded exponential
backoff plus failover across sibling Flash models (quota buckets are
per-model), and `agent/supervisor.py` relaunches a crashed orchestrator,
which resumes from `agent/checkpoint.py` rather than restarting the budget.

The running record of these is `logs/pitfalls.json`, which the agent itself
reads back into its prompt each iteration so it does not repeat a failure.

## Results

**Submitted: validation primary 0.6049 vs the official baseline 0.6016 —
+0.0033, both metrics up** (GAUC 0.6674 -> 0.6720, nDCG@5 0.5357 -> 0.5378),
scored by the organizer's own `submit.py --score`.

We claim **no hidden-test score**. It is scored once by the organizer, and a
validation number compared against a test number would overstate progress.

**The single largest improvement was found by the agent, autonomously.** It
proposed a DeepFM-lite architecture — a parallel [32, 16] MLP branch beside
the FM's linear and pairwise terms — which cleared ε=0.002 against its own
best and was checkpointed. It proposed that *after* its first, larger DeepFM
attempt crashed the smoke test and was rolled back automatically.

| Step | Valid primary | Author |
|---|---|---|
| Official FM baseline (published) | 0.6016 | organizer |
| Our editable pipeline (torch FM, official 5 fields) | 0.6017 | human |
| + session-position feature | 0.6024 | human |
| **+ agent's DeepFM-lite MLP branch (accepted)** | **0.6045** | **agent** |
| + 10-seed rank-average ensemble (submitted) | **0.6049** | human |

Three honest notes.

**The ensemble's mean gain is not established.** 5-seed against 1-seed is
+0.00027, while negative controls comparing the *same* model to itself reach
0.00077. It ships for **variance**, which is established: over 20 seeds the
single-seed std is 0.00049 against a 5-seed std of 0.00020, matching sqrt(n).
On a one-shot submission the floor is what matters — worst single seed 0.6036,
worst 5-seed ensemble 0.6046.

**We found our own champion figure was optimistic.** It had been recorded as
0.6047 with a 3-seed std of 0.0001. Seeds 0-2 were a lucky triple; over 20
seeds the mean is 0.60448 with std 0.00049 — an 8x larger spread. Every
3-seed standard deviation in our earlier notes should be read accordingly.

**Iteration 6 scored higher than the shipped checkpoint** (0.6050) but cleared
the accepted best by only +0.0005, below ε, so it was rejected and rolled back
rather than shipped.

Per-iteration trajectory, resource usage and the intervention count are in
`docs/results_table.md`, generated from `logs/iterations.jsonl`.

What the agent proposed, in order: pairwise BPR (0.5994), a multi-task
auxiliary `is_click` head (0.6013), listwise ListNet (0.6004). Each was
rolled back automatically for failing to clear ε=0.002, and the run converged
under the official N=3 rule.

### The negative results are the finding

We went looking for why the organizer's own priority list did not pay off, and
the answer is a property of the dataset rather than of the agent. Measuring
within-user GAUC of individual signals on a 4,000-user validation sample
(0.5 = no signal at all):

| Signal | within-user GAUC |
|---|---|
| Train-derived smoothed video quality prior | **0.6453** |
| `tab` | 0.5387 |
| Session position (impression order within a user-day) | 0.5148 |
| `duration_ms` / `hourmin` | ~0.486 |
| **user x author affinity** | **0.4981 — none** |
| **user x video affinity** | **0.4970 — none** |

The entire official FM reaches 0.6674. A single scalar item-quality prior gets
to 0.6453 of that alone. **This task is item-quality estimation with almost no
personalisation signal**, which explains three things at once: why adding
user-side features was already flat in the organizer's own ablation, why our
loss-alignment changes could not help (the objective was not the bottleneck),
and why behavioural-sequence modelling — the organizer's #2 unexplored lead —
is unlikely to pay off here. Repeat exposure is only 1.62% of validation rows,
and long_view rates on repeat pairs (0.3072) and new pairs (0.3134) are
effectively identical, so there is little history to model.

We also checked, so as not to misattribute the plateau: the model is not
undertrained (raising the epoch cap 12 -> 40 early-stops at 13 with an
identical score), and a LambdaMART ranker optimising nDCG directly scored
0.5901 — it spends its strongest splits on features that are constant within a
user and therefore cannot change that user's ordering.

### On the random-exposure log

KuaiRand-Pure ships a 1,186,059-row uniform-random-exposure log, and it is
tempting as extra training data. It is not usable: **897,721 of those rows
(75.7%) fall inside the hidden-test date window**, and its long_view rate is
0.0850 against 0.3133 in the standard log, because uniform exposure mostly
shows people videos they do not want. Training on it means test-period
contamination plus a train/serve distribution mismatch. We use it only as the
unbiased probe, which is its sanctioned use — and that distribution gap is
also why our referee's absolute divergence is always large, so only the change
in divergence across iterations is meaningful. We report that as an
instrumentation weakness rather than claiming the referee caught something.

### What we discovered by trying to falsify ourselves

After the score converged we kept going, but as research rather than a leaderboard
hunt: for each remaining unexplained signal, form a mechanism hypothesis and build
the control that would expose a false positive. Seven investigations, every one
with a control attached:

| Hypothesis | Test | Result | Decision |
|---|---|---|---|
| `user_id` helps via affinity or capacity | row-level shuffle, parameters held fixed | identity **108%**, capacity **-8%** | capacity refuted |
| The user embedding overfits, so shrink it | field-specific weight decay, 6 arms x 3 seeds | monotone **negative** | closed |
| Performance decays across the eval window | frozen model-free reference on the same days | gap **+0.0008** | closed |
| The gap to 0.8645 means signal is missing | simulate y ~ Bernoulli(q) from the calibrated model | irreducible **0.2556** > observed **0.2431** | closed |
| Users differ in feature *sensitivity* | 4-arm, out-of-fold, permuted + randomized controls | **-0.0012**, CI excludes zero | closed |
| Staleness explains that failure | early vs late estimation, sizes matched | **+0.00017** | refuted |
| DeepFM/GBDT disagreement is exploitable | per-group oracle + quality-matched control | **+0.0023** of an apparent +0.0314 | closed |

**Three of these are worth more than the score is.**

*`user_id` encodes identity, not capacity.* Deleting it costs -0.0091, yet
user-video affinity measures at chance three separate times. Permuting the
row-to-user link while holding parameter count, MLP input width and code
frequencies fixed loses the entire effect. The permutation has to be row-level:
a bijective user-to-row remap is a symmetry of the model and trains to an
identical result, which would have produced a convincing false null.

*The published ceiling is unreachable by any model.* 0.8645 is the organizer's
label-oracle ceiling and is correctly computed. But in a simulated world where
our model IS the true conditional probability — nothing left to learn by
construction — an oracle still beats it by 0.2556, while our real gap is 0.2431,
smaller. A long_view is a coin flip; an oracle that sees the realised label wins
by that margin regardless of model quality. The distance to 0.8645 is therefore
not evidence of remaining headroom.

*A per-group oracle is upward-biased, and here the bias was 12x the signal.* A
per-user oracle over DeepFM and GBDT showed +0.0314 of apparent headroom, which
would have justified days of gate-building. A quality-matched control — DeepFM
degraded with calibrated noise to the GBDT's exact score level, carrying no
information DeepFM lacks — reproduced +0.0291 of it. Real excess: +0.0023.

### The leak we introduced, found, and fixed

Our first sensitivity experiment produced a clean, significant result: real
sensitivities scored -0.0048 with a CI excluding zero, while permuted and
randomized versions of the identical fields were neutral.

Noise being free while real values did damage is not how a failed hypothesis
behaves — it is how a **leak** behaves. We had computed each user's statistic
from that user's own training labels, so a training row's feature contained that
row's label: an in-fold target encoding one level above where we had applied the
out-of-fold rule. Rebuilding it so each row's band comes from that user's *other*
folds recovered 75% of the drop.

With only a control and a treatment arm this would have been filed as
"sensitivities are harmful." The two null controls are what made it diagnosable.
We report it because the discipline is the point, not because it flatters us.

### What we measured versus what we believe

Kept separate deliberately. **Measured:** sensitivities are reliably estimable
(split-half 0.31-0.60), they decay 22-61% across a week, exposing them costs
-0.0012, both null controls are neutral, and estimating from 6x more data makes
the harm worse. **Interpretation, not established:** the model's own user
embedding already learns this modulation, and an explicit precomputed summary
adds a redundant, coarser pathway that generalises worse.

## How we built it: a human-relayed adversarial loop

Development used two AI assistants in opposing roles, with a human relaying
between them. **Claude Code** implemented, ran experiments and reported results.
**OpenAI** received those reports and attacked them — challenging conclusions,
demanding controls, proposing alternative explanations and setting stopping
rules. A human passed messages both ways and made the calls.

This was **not** an autonomous multi-agent system. The relay was manual. We
document it because it changed outcomes we can point at:

| The adversary's intervention | What it changed |
|---|---|
| "more seeds cannot hurt" is not true of a ranking metric | corrected an overstatement about variance before it reached the writeup |
| run the oracle diagnostic *before* building a gate | killed the conditional-blending branch in one GBDT fit rather than a full gate pipeline |
| add a *randomized* arm as a stronger negative control | produced the 4-arm design that made an in-fold leak diagnosable instead of reading as a failed hypothesis |
| test early-vs-late estimation directly | refuted staleness, which we had been treating as the likely mechanism |
| 0.8645 is the *label-oracle* ceiling, not "not a ceiling" | sharpened a claim that would not have survived review |
| freeze the submission before further research | produced the frozen, hash-verified artifact every later experiment was measured against |

The adversary rarely proposed better *models*. It proposed better *tests*, and
repeatedly stopped work that would have produced a confident wrong answer.

**This describes our development process, not the submitted system.** The agent
runs on Google Gemini; no OpenAI or Anthropic model is called by the pipeline at
scoring time.

## Verified reproducibility

The repository was cloned fresh from GitHub, data staged, and the pipeline
re-run end to end. Both artifacts regenerate **byte-identical** to their
recorded SHA-256s:

```
smoke test              0.4498908751631786   identical fingerprint
baseline reproduction   0.6015 vs published 0.6016   MATCHES
metric decomposition    1.599e-14 vs starter_kit/evaluate.py
submission_valid.csv    HASH MATCH
submission_test.csv     HASH MATCH
official checker        124,909 / 170,588 rows, both pass
official score (valid)  GAUC 0.6720 | nDCG@5 0.5378 | primary 0.6049
```

## Team

- **Rohan Joshua** — agent architecture and loop, evaluation protocol, research
  experiments and controls, submission pipeline and provenance.
- **Thaddus Lee** — referee integration, crash checkpointing and resume,
  autonomous ablation targeting, Starter Kit and dataset integration,
  metric/convergence alignment.
- **Waseem Akram** — audit of all technical files and documentation; research
  audit and submission audit.

## Built with

**Development tools:** VS Code, Claude Code (an AI coding assistant used during
development; it is not part of the submitted system).

**APIs:** Google Gemini — `gemini-3.5-flash` and `gemini-3.5-flash-lite`. The
scored run used 91,430 tokens in 0.1965 h across 9 iterations, with 2 logged
manual interventions. An Anthropic backend is implemented and selectable but was
not used for the scored run.

**Libraries:** PyTorch, pandas, numpy, scikit-learn, PyYAML, tqdm, google-genai,
anthropic. LightGBM is used only by a research-branch diagnostic and is not a
submission dependency.

**Data:** KuaiRand-Pure only, via the organizer's Starter Kit. No external
training data, no pretrained weights. The Zenodo caption and category
supplements were treated as out of scope, since neither the problem statement
nor the Starter Kit sanctions them.

## What's next

- **Automate the adversarial loop.** The critique loop described above was
  human-relayed. Wiring a second provider in as an automated adversary — one
  model proposes, a different model attacks the claim and demands controls —
  would make the critique a measurable part of the agent rather than a manual
  step. This is the extension we would build first.
- **Give the agent a signal it can actually exploit.** Our measurements say
  the ceiling here is item-quality estimation, which the FM already does well.
  The directions with any remaining room are content-side (video captions and
  category taxonomy exist as Zenodo supplements) rather than
  behavioural — but those are not referenced by the official problem statement,
  so we treated them as out of scope rather than quietly using them.
- **Make the compression gate re-train, not just reason.** As built, a fresh
  context judges a terse summary with no validation access. The stronger
  version re-runs training from that summary and compares scores.
- **Set the referee's alert on the change in divergence, not its level.** The
  two splits have structurally different label distributions, so the absolute
  gap is uninformative — see Results.
- **Persist "this hypothesis scored worse" across runs.** Pitfalls currently
  record crashes and gate rejections, so a fresh run would happily re-propose
  BPR. Carrying scored-worse outcomes into the pitfall store would stop that.
- **New-file creation in the editor**, so the agent can add an architecture
  variant under `pipeline/model/architectures/` rather than only rewriting
  files at fixed paths.
