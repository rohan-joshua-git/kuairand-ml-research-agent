# Tier 3 deep dive — multi-task / scenario-conditioned heads

Loaded when the ablation target is the model architecture block and the
current bottleneck looks like label sparsity or label-semantics conflation
(see Tier 1, trap #1: `is_click` means two different things depending on
`tab`).

## Scenario-conditioned heads

This applies to the `is_click` AUXILIARY head only (the primary label is
`long_view`, unconditional — see Tier 1). Rather than a single shared
output head predicting one `is_click` auxiliary target, condition the
final layer(s) on `tab` (or on the coarser two-column/single-column split
derived from it):

- Shared bottom: embeddings + cross layers, as in the baseline.
- Per-scenario head: either (a) a small `tab`-indexed set of final linear
  layers, or (b) a single head with `tab` embedding concatenated in late
  (cheaper, less capacity for scenario-specific patterns).
- Rationale: a genuine "click" in the two-column UI and a duration-derived
  "valid_play" in the single-column UI are different behavioral signals
  with different base rates (verify with
  `pipeline/data/label.py::profile_label` before assuming this helps —
  it's a hypothesis to test, not a guaranteed win; on real data the
  valid_play-derivation assumption holds ~97.2% of the time, not exactly
  1.0). This only matters if the auxiliary `is_click` head is actually
  helping the primary `long_view` task — verify with an ablation before
  investing in scenario-conditioning.

## Auxiliary-task transfer (ESMM/PLE, see Tier 2)

Concretely, for KuaiRand:
- Primary task: `long_view` (the scored label — `pipeline/data/label.py`).
- Candidate auxiliary tasks (Starter Kit priority-3 lead): `is_click`
  (resolve via `resolve_auxiliary_click_label`, not the raw column — see
  the scenario-conditioning note above), `is_like` / `is_follow` (sparser
  but higher-precision positive signal), `is_comment` / `is_forward`.
  `play_time_ms` is excluded from any head — it's a near-direct proxy for
  `long_view` itself (see Tier 1 "Already tried" — 0.64 correlation),
  feeding it anywhere is leakage, not modeling.
- Start with the simplest viable setup: shared bottom + task-specific
  towers, weighted sum of losses (primary task weighted highest). Only move
  to PLE's explicit shared/specific expert split if the simple version
  shows auxiliary tasks are helping but a naive shared bottom seems to be
  causing negative transfer (auxiliary loss decreasing while primary task
  validation score stalls or drops).

## Caveat

More heads and more losses is more surface area for the compression gate
(`agent/compression_gate.py`) to catch overfitting on. Justify each
addition with an ablation result (`agent/ablation.py`), not by architecture
aesthetics.
