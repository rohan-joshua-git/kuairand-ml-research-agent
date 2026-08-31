# Compliance note — hidden-test exposure, disclosed

This note exists because an audit that asserts cleanliness without demonstrating
it is worth less than no audit. It records a hidden-test exposure that was
present in this pipeline during the graded run, how far it reached, when it was
closed, and one place where our own commit message described it incorrectly.

## What happened

`agent/orchestrator.py` calls `reproduce_official_baseline()` once at the start
of every run, to satisfy Task Requirement #1. Through commit `3afb112`, that
function was:

```python
# pipeline/official_baseline.py @ 9f5a8d8 (HEAD during the graded run)
splits = _sk_data.load(cfg["dataset"]["raw_dir"])
result = _sk_baseline.run_fm(splits, k=16, lr=0.001, epochs=40, ...)

valid_primary = float(result["valid"]["primary"])
test_primary  = float(result["test"]["primary"])

matches = (
    abs(valid_primary - ref["valid"]["primary"]) <= tolerance
    and abs(test_primary - ref["test"]["primary"]) <= tolerance   # <-- consumer
)
```

`run_fm` hard-codes `'test': evaluate(ute, yte, m.predict(Xte))`
(`starter_kit/baseline.py:98`), so it reads hidden-test **labels** (`yte`).
`test_primary` was then consumed as a conjunct of `matches`, and that boolean is
recorded in the graded run log as `matches_published_baseline: true`
(`logs/iterations.jsonl`, record 0).

This bypassed the `allow_test=False` guard in `pipeline/data/loader.py`, which
exists precisely to prevent it. The call went through the vendored starter-kit
loader instead.

## How far it reached

- **Not into model weights.** The vendored FM trained by `run_fm` is the
  organizer's reference baseline, discarded after the comparison. It is not the
  submitted model and shares no parameters with it.
- **Not into features.** Vocabularies, `dur_bucket` quantiles and any target
  encodings are fit on `split.train` only (`pipeline/train.py`).
- **Not into checkpoint selection.** Every acceptance decision uses
  `delta_vs_prev_best` computed from the **validation** primary against the
  config's `official_baseline.valid.primary` (0.6016). `test_primary` has no
  consumer outside the `matches` expression above.
- **Into the log, as a boolean.** `matches_published_baseline` is reported and
  never branched on.

## When it was closed

Commit `3afb112` (2026-08-31 20:27 SGT), **after the graded run completed at
09:16 SGT the same day**. `run_fm` hard-codes its test evaluation and
`starter_kit/` is vendored verbatim and must not be edited, so the fix starves
it of test rows rather than modifying vendored code:

```python
splits["test"] = splits["valid"]   # before encode(); no test label is ever encoded
```

The dead `test_*` fields were removed from `BaselineReproductionResult`, and
Task Requirement #1 is now satisfied on validation alone — which is what the
requirement asks for ("reaches the official baseline's reported validation
score").

## Correction to our own commit message

The message on `3afb112` states:

> "The figure was used nowhere (grep test_primary returned no consumers)"

**That is wrong.** `test_primary` had a consumer four lines below its own
definition, in the `matches` expression shown above. The grep behind that claim
excluded `official_baseline.py` itself and the result was reported as though it
covered the whole repository.

The substantive claim — that the figure never reached weights, features or
selection — is correct and is evidenced above. The claim that it had *no*
consumer is not. Git history cannot be amended after pushing, so the correction
is recorded here instead.

## Current state

- No file in this repository asserts a self-computed hidden-test score. The
  organizer's *published* figure (test primary 0.5946) appears in
  `config/agent_config.yaml` and once in `docs/results_table.md`, labelled as
  the organizer's own reference and never differenced against a validation
  number.
- `pipeline/data/loader.py` keeps `allow_test=False` by default;
  `pipeline/make_submission.py` is the only place it is set True, and the test
  split is written to the submission but never locally scored.
- `pipeline/official_baseline.py` carries the guard above with a comment
  explaining why it cannot simply be switched off by argument.

## Reader's own check

```bash
grep -rn "test_primary" --include="*.py" .          # no consumers remain
grep -rn "allow_test" --include="*.py" .            # only loader + make_submission
sed -n '/HIDDEN-TEST GUARD/,/splits\["test"\]/p' pipeline/official_baseline.py
```
