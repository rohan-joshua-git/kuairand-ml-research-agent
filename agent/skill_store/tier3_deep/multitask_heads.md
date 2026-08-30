# Tier 3 deep dive — multi-task / scenario-conditioned heads

Loaded when the ablation target is the model architecture block and the
current bottleneck looks like label sparsity or label-semantics conflation
(see Tier 1, trap #1: `is_click` means two different things depending on
`tab`).

## Scenario-conditioned heads (Play 3)

Rather than a single shared output head predicting one `is_click`, condition
the final layer(s) on `tab` (or on the coarser two-column/single-column
split derived from it):

- Shared bottom: embeddings + cross layers, as in the baseline.
- Per-scenario head: either (a) a small `tab`-indexed set of final linear
  layers, or (b) a single head with `tab` embedding concatenated in late
  (cheaper, less capacity for scenario-specific patterns).
- Rationale: a genuine "click" in the two-column UI and a duration-derived
  "valid_play" in the single-column UI are different behavioral signals
  with different base rates (verify with
  `pipeline/data/label.py::profile_label` before assuming this helps —
  it's a hypothesis to test, not a guaranteed win).

## Auxiliary-task transfer (ESMM/PLE, see Tier 2)

Concretely, for KuaiRand:
- Primary task: `is_click` (the scored label).
- Candidate auxiliary tasks: `long_view` (strong engagement signal, likely
  positively correlated with click but less sparse), `is_like` /
  `is_follow` (sparser but higher-precision positive signal).
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
