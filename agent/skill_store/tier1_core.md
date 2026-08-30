# Tier 1 — Core (always loaded)

This tier is small on purpose. HASTE's ablation found flat-loading a large
skill inventory performs *identically to loading no skills at all*, at
double the token cost — only tiered, targeted loading beats cold-start.
Tier 1 stays under ~1 page; deeper material lives in Tier 2/3 and is loaded
by `retriever.py` only when the current ablation target calls for it.

## Task framing

- Dataset: KuaiRand-Pure. Positive label: `is_click`. Metrics: GAUC,
  nDCG@5. Scored as `mean(GAUC_delta, nDCG@5_delta)` vs. the organizer
  baseline — **absolute** delta, not relative, on the hidden test set.
  Both metrics sit in a comparable range (perfect ranking: GAUC 1.0000,
  nDCG@5 0.7289; random: ~0.4753 combined) so neither dominates the mean —
  treat ranking quality (GAUC: does the model separate a user's clicked
  items from their non-clicked ones) and top-of-list precision (nDCG@5)
  as equally worth optimizing.
- Split is date-pinned: train = 4/08-4/21, val = first half of 4/22-5/08,
  test = second half (hidden — never load it, see `pipeline/data/loader.py`
  `allow_test` guard).
- You are scored once, on the validation-best checkpoint at convergence.
  A checkpoint that scores well on validation but doesn't generalize is
  worse than useless — see `agent/compression_gate.py` and use it before
  designating anything final.

## Dataset-specific traps (verified against the KuaiRand-Pure field spec)

1. **`is_click` is two different constructs.** In the two-column UI it's a
   genuine tap. In the single-column UI it's actually `valid_play`:
   `play_time_ms >= duration_ms` (videos under 7000ms) or
   `play_time_ms > 7000ms` (longer videos). See `pipeline/data/label.py`.
   Don't assume a uniform semantic — profile it first (`profile_label`).
2. **`video_features_statistic` columns leak.** They're running averages
   computed over the full month, which spans train/val/test. Don't use them
   raw — `pipeline/data/leakage_guard.py` drops them by default. If you
   reconstruct a point-in-time version, log that decision explicitly.
3. **KuaiRand-Pure is the debiasing/multi-task variant**, not the
   sequential-modeling variant (that's 27K/1K). Heavy sequential
   architectures (long user-history transformers) are likely to
   underperform here relative to debiasing and multi-task approaches.
   Don't burn iterations on sequence models as a first move.
4. **There's a fourth log file** (`log_random_...`) outside the prescribed
   split — uniformly-random exposure, ~1.19M interactions. It's in-dataset,
   not external data, but its use is gated by
   `config.referee.mode` pending organizer confirmation. See
   `agent/referee.py` and README "Open Questions" before relying on it for
   anything beyond diagnostics.

## Where to look next

- Need RecSys architecture/method background? -> Tier 2 (`tier2_domain.md`)
- Need a deep dive on a specific method you're about to implement? ->
  Tier 3 (`tier3_deep/`), loaded on demand by `retriever.py` keyed to the
  current ablation target.
