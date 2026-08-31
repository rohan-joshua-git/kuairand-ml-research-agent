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

**Submitted: validation primary 0.6036 vs the official baseline 0.6016 —
+0.0020, both metrics up** (GAUC 0.6674 -> 0.6702, nDCG@5 0.5357 -> 0.5371),
scored by the organizer's own `submit.py --score`.

**The autonomous agent did not produce that gain, and we are not going to
imply otherwise.** The agent converged at parity, 0.6017. The +0.0020 comes
from two human changes made after analysing why it plateaued: a
session-position feature (0.6017 -> 0.6024) and a 5-seed rank-averaged
ensemble (-> 0.6036). Both are described below, and the results table
attributes every step.

| Step | Valid primary | Author |
|---|---|---|
| Official FM baseline (published) | 0.6016 | organizer |
| Our editable pipeline (torch FM, official 5 fields) | 0.6017 | human |
| Agent's 3 iterations | best 0.6013, all rejected | **agent** |
| + session-position feature | 0.6024 | human |
| + 5-seed ensemble (submitted) | **0.6036** | human |

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

## What's next

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
