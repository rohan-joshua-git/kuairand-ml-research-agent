# An agent that tries to prove itself wrong

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**

---

## The number we didn't report

At one point in this project our agent surfaced a candidate that scored **+0.0314**
over the official baseline. Ten times our final margin. It would have looked
extraordinary on a leaderboard.

We measured it properly and it was worth **+0.0023**.

The gap was winner's curse. We had picked the best of many noisy per-group
estimates, and the act of picking the maximum is itself biased upward — the
group that wins is disproportionately the one whose noise happened to be
positive. Once we estimated each group's effect out-of-fold instead of reading
it off the same data that selected it, **92% of the improvement evaporated.**

We report **+0.0033**. It is a small number. It is also a number we can defend,
and this writeup is mostly about the machinery that turned the first number into
the second.

---

## What we built

An autonomous research agent that proposes hypotheses about a recommender
system, writes the code to test them, runs controlled experiments, and — the
part we care about most — **tries to kill its own findings before believing
them.**

The task: rank videos within each user on KuaiRand-Pure. Scored by
**GAUC** (positive-count-weighted per-user AUC) and **nDCG@5**, with the
primary metric being their mean.

**Result: validation primary 0.6049 against the official FM baseline's 0.6016.**

| | baseline | ours | delta |
|---|---|---|---|
| GAUC | 0.6610\* | **0.6720** | +0.0046 |
| nDCG@5 | 0.5282\* | **0.5378** | +0.0021 |
| **primary** | **0.6016** | **0.6049** | **+0.0033** |

\* baseline validation figures; the organizer's published *hidden-test* primary
is 0.5946, a different split on a different scale. **We claim no hidden-test
score anywhere.** The hidden test is scored once, by the organizer. Every number
in this writeup is validation.

---

## The one structural insight everything rests on

**Only within-user order is scored.**

This sounds obvious and it is not. It has a brutal corollary: *anything constant
within a user cannot reorder that user's candidates.* We verified this
empirically — a feature that is constant per user produces a GAUC of exactly
0.500000.

That single fact killed more of our ideas than any experiment did, and it is why
several plausible-looking features are worthless here:

- **`tab` (the feed surface)** looks like the most powerful variable in the
  dataset: its `long_view` base rate runs from **0.004 to 0.489**. But 60% of
  users appear in only one tab, and for them it contributes **+0.0001**. Its
  entire apparent power is the level gap *between* tabs — which the metric never
  sees.
- **`author_id` and `music_id`** looked like strong pooling levels. Then we
  checked: **87% of authors and 98% of music IDs own exactly one video.**
  Target-encoded, they correlate 0.985 and 0.987 with the video prior. They are
  `video_id` wearing a hat. Only `tag` turned out to be a genuine pooling level.
  This corrected one of our *own* top-ranked recommendations.
- **17.5% of the metric is structurally unmovable.** 3,917 validation users have
  a single impression. Model, prior and oracle score them identically, and GAUC
  excludes them entirely.

---

## How we kept ourselves honest

Four pieces of machinery, all built before we needed them.

**1. A selection/confirmation split.** Validation users are partitioned by a
hashed user ID with a salt fixed *before the run started* — 11,270 selection
users, 11,107 confirmation. Every hypothesis, sweep and early-stopping decision
sees selection only. We budgeted a small number of confirmation looks and spent
one. You cannot tune against a number you refuse to look at.

**2. Negative controls on everything.** Permuted, randomized, quality-matched
and same-model-seed groups. A result that does not beat its own control is not a
result. This is what caught the winner's curse above: the "improvement" sat
comfortably inside the same-model control band.

**3. Knowing our own noise floor.** We had recorded a 3-seed standard deviation
of 0.0001 and were treating small gains as real. Running 20 seeds showed the
true single-seed std is **0.00049** — we had been under-estimating our own noise
by **8×**. Seeds 0–2 were an unusually tight triple. Every comparison after that
point is seed-matched over enough seeds to clear 2 standard errors.

**4. A rule against outcome features.** `play_time_ms` correlates **0.64** with
the label and predicts it at 84.7% accuracy from a threshold alone. It is not a
feature — it is the label in disguise, an *outcome* of the very impression being
predicted. Same for `profile_stay_time`, `comment_stay_time`, `is_profile_enter`.
They are available in the data and they are banned as inputs in our pipeline.
Anyone who feeds them in gets a spectacular validation score and a model that
cannot work.

---

## The leak we introduced, caught, and fixed

We are including this because we think it is the most instructive thing that
happened.

We added a per-user statistic as a feature. It scored **−0.0048** — actively
harmful. Meanwhile the permuted and randomized controls came back clean, costing
nothing.

**That is not how a failed hypothesis behaves. That is how a leak behaves.** A
useless feature costs you nothing; noise is free. Only a feature carrying
information the model can exploit at training time and not at inference time
does *damage*.

The statistic was computed from each user's own training labels. A training
row's feature therefore contained that row's own label — an in-fold target
encoding one level above where we had already applied the out-of-fold rule. We
knew to be out-of-fold at the *row* level and had not thought about the *user*
level.

Rebuilding it so each row's value comes from that user's other folds recovered
**75% of the drop.**

The lesson we took: the *sign and shape* of a result is diagnostic. We found
this by asking why real values were doing worse than noise, not by reading code.

---

## Six hypotheses, honestly tested, all null

| hypothesis | how we tested it | result |
|---|---|---|
| `user_id` helps via user×video affinity or via model capacity | row-level shuffle holding parameter count, MLP width and code-frequency fixed | identity **108%**, capacity **−8%** — capacity refuted |
| Field-specific weight decay on user embeddings | swept 5 orders of magnitude | monotone negative throughout |
| Temporal drift needs recency weighting | train-window sweep, matched sizes | no drift; gap **+0.0008**. The late-day decline is day *difficulty*, not staleness |
| User sensitivities (per-user response slopes) help | split-half reliability + two null controls | reliable (0.31–0.60) and real, but exposing them costs **−0.0012** |
| The gap to the 0.8645 oracle is missing signal | simulated a world where the model is exactly right | irreducible gap **0.2556** > observed **0.2431** |
| Model blending / gating exploits family disagreement | out-of-fold re-estimation | **+0.0023** of an apparent **+0.0314** |

One control deserves calling out. To test whether `user_id` contributes
*identity* or merely *capacity*, we permute the user→row assignment. Our first
instinct was a bijective user→user remap — and that would have been **wrong**: a
bijection is a *symmetry of the model*, so it trains to a numerically identical
result. It would have produced a clean, convincing, completely false null. The
permutation has to be at the row level.

We caught that before running it. It is the kind of mistake that does not
announce itself.

---

## The oracle ceiling is not a ceiling

The starter kit publishes a label-revealing oracle at **0.8645**, against a
baseline of 0.5946. Every team looking at that gap sees enormous headroom.

We tested whether it is reachable *by any probabilistic model*. Fit calibrated
probabilities `q` (mean predicted 0.3133 vs actual 0.3133), then simulate a
world where the model is exactly right by drawing `y ~ Bernoulli(q)`. In that
world, by construction, **there is nothing left to learn.** Score both the model
and a label-revealing oracle against those simulated labels:

| | primary |
|---|---|
| perfect-probability model vs simulated labels | 0.5915 |
| label oracle vs simulated labels | 0.8471 |
| **irreducible gap, pure coin-flip noise** | **0.2556** |
| our model vs real labels | 0.6053\* |
| oracle vs real labels | 0.8484 |
| **observed gap** | **0.2431** |

\* measured on an earlier artifact; see the note in `docs/research_process.md`.

**The observed gap is smaller than the gap a perfect model faces in a world with
nothing left to learn.** The oracle sees each impression's realised outcome. A
long-view is a coin flip, and no probability model can order a 0.31-chance item
that came up 1 above a 0.30-chance item that came up 0.

This does not prove no signal remains — if the model is missing structure then
`q` is wrong and the simulation flatters us. Nor does it impugn the organizer's
figure, which is exactly what it claims to be. What it establishes is narrower
and still useful: **the size of the gap to the oracle is not evidence of
remaining headroom.** Anyone arguing from "0.8645 minus 0.605 is huge" has to
make the case another way.

---

## Challenges

**Our own numbers were the hardest adversary.** Three separate times a promising
result turned out to be an artifact of how we measured it: the winner's curse,
the user-level leak, and the 8× noise under-estimate. None was found by a test
suite. Each was found by asking why a number had the shape it did.

**Choosing the smaller number.** Late in the project we found that our shipped
models were early-stopping over *all* validation rows, including the confirmation
half we had promised to hold out. Fixing it moved our headline from **0.6053 down
to 0.6049.** We shipped the lower number and kept the superseded file hashes in
the repo so the swap is auditable. The higher number was real; it just wasn't a
clean held-out estimate, and a number you can't defend is worth less than a
smaller one you can.

**Auditing our own audit.** One of our commit messages claimed a `grep` had
returned no consumers of a variable. The consumer was four lines below the
definition — the grep had excluded the file itself. Since history can't be
amended after pushing, we wrote the correction into `docs/COMPLIANCE_NOTE.md`.
An audit that misdescribes its own diff is worth less than no audit.

**Discipline about the hidden test.** `loader.py` defaults to `allow_test=False`;
exactly one file may flip it, and only on the final submission run. The test
submission is format-checked and **never locally scored.** We genuinely do not
know our hidden-test score. That is the point.

---

## How we built it: a human-relayed adversarial loop

Worth being precise, because it is unusual and it is *not* part of the submitted
system.

**The submitted agent** runs on Gemini (`gemini-3.6-flash`, with a documented
fallback to `gemini-3.5-flash` after a quota 429 during the graded run). It
proposes a hypothesis, writes a code diff, runs an ablation, evaluates against
the selection half, and keeps or rolls back.

**Our development process** ran a second loop, by hand. Claude acted as
researcher and engineer; its output was pasted into OpenAI's model, which was
prompted to attack the work — find the leak, name the confound, reject the
conclusion. The critique came back and was acted on. That relay is where the
bijection-symmetry trap, the winner's-curse re-estimation, and the
selection/confirmation contamination were all caught.

Two different loops, both adversarial by construction. The human in the middle
was the transport layer.

---

## Verified reproducibility

Not asserted — run, twice, from a fresh clone:

```
smoke fingerprint     0.4498908751631786          unchanged across every patch
Task Requirement #1   0.6015 vs published 0.6016  matched on validation
metric decomposition  max diff 1.599e-14          vs starter_kit/evaluate.py
official --score      GAUC 0.6720 | nDCG@5 0.5378 | primary 0.6049
submission CSVs       BYTE-IDENTICAL to the committed SHA-256s
```

The reported 0.6049 was produced twice by independent paths — a cached score
matrix through our own decomposition, and `make_submission` → CSV →
`starter_kit/evaluate.py`. They agree to four decimals on all three metrics,
which establishes that the number we report is the number the submitted artifact
actually scores.

---

## What we're proud of

- Reporting **+0.0033** when **+0.0314** was sitting there, and being able to
  show exactly why the larger number was false.
- Catching a leak we introduced ourselves, from the *shape* of the evidence.
- Nearly running a control that would have produced a convincing false null, and
  catching the symmetry first.
- Shipping the **lower** of two scores because the higher one wasn't clean.
- A repo where a judge can clone, run, and get our exact bytes back.

## The last night: we tested the two obvious next steps, and both lost

With the submission already frozen and byte-verified, we spent a final night on
the two levers we most expected to work. Fourteen arms, seed-matched, selection
half only. **Zero wins.**

**A ranking objective loses, monotonically.** GAUC is a within-user AUC, and AUC
is exactly the probability a positive outranks a negative from the same user, so
training pointwise BCE is a genuine objective mismatch. We implemented
within-user BPR and a hybrid. Every dose made it worse:

| α (weight on the pairwise term) | 0 (base) | 0.1 | 0.25 | 0.5 | 1.0 | pure BPR |
|---|---|---|---|---|---|---|
| selection primary | **0.60781** | 0.60682 | 0.60646 | 0.60612 | 0.60516 | 0.60488 |

The mechanism was in our own sampler's log: **382,579 pairable positives across
24,290 two-class users — 33.5% of training rows.** A pairwise loss can only use
rows from users who have *both* a positive and a negative. It discards the other
66.5% — and those rows still teach the model what a good *video* looks like,
which transfers to every user. The metric ignores single-class users. The
representation must not. Matching the loss to the metric meant throwing away two
thirds of the signal to do it.

**Fitted blend weights overfit, and we caught it in the act.** This was the one
technique we'd heard was working well for others. We ran it under a protocol
that can actually detect the failure: split the selection half *again* under a
different salt into `selA` (the only rows weights are fitted on) and `selB` (the
only rows results are reported on).

| | selA — fitted | selB — honest |
|---|---|---|
| base | 0.61174 | **0.60481** |
| greedy fitted blend | **0.61194** | **0.60393** |

**It wins where its weights were fitted and loses by −0.00089 where they were
not.** The +0.0002 isn't a small real gain; it's the search finding noise, and
the sign flips the moment it has to generalise. Same mechanism as the winner's
curse that opened this writeup, arriving through a different door.

We changed nothing. The submitted CSVs still hash to the frozen values.

## What we'd do next

- **A longer agent run.** The graded run converged in 7 hypotheses. The
  convergence rule also discarded a validation-best checkpoint worth ~+0.0005
  because it did not clear the acceptance threshold — disclosed in the README,
  and fixed in the code for future runs.
- **Attack the representation, not the objective or the ensemble.** Both of
  those are now measured dead ends. The evidence points at a local optimum that
  the available levers do not move, which is what our own oracle simulation
  predicted.

## Team

- **Rohan Joshua** — agent architecture and loop
- **Thaddus Lee** — referee integration
- **Waseem Akram** — auditor: technical files, documentation, research and
  submission audits

## Built with

`python` · `pytorch` · `pandas` · `numpy` · `gemini` · KuaiRand-Pure

---

*No hidden-test score is claimed in this writeup or anywhere in our repository.
Every figure above is validation, and the protocol that produced it is in
`pipeline/eval_protocol.py` with the salt fixed before the first run.*
