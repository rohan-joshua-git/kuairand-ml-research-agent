# Tier 3 deep dive — multi-task heads

Loaded when the ablation target is the model architecture block and the
current bottleneck looks like label sparsity (organizer headroom item #3 —
see Tier 1). The scored label is `long_view`; every other feedback signal
in the log (`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`,
`is_hate`) is fair game as an auxiliary task, none of it is pre-resolved
for you the way `long_view` is.

## `is_click`'s own quirk (still real, just no longer the scored label)

Per the KuaiRand field spec, `is_click` means different things depending on
`tab` (two-column UI: a genuine tap; single-column UI: actually
`valid_play`, a duration-derived proxy — see
`pipeline/data/label.py::profile_label` / `resolve_label`). If `is_click`
is used as an auxiliary task, feeding the raw conflated column in is still
a mistake — use `resolve_label(df, two_column_tabs, mode="click_only")` (or
a scenario-conditioned head keyed on `tab`) rather than the raw column, so
the auxiliary signal isn't itself noisy.

## Auxiliary-task transfer (ESMM/PLE, see Tier 2)

Concretely, for KuaiRand:
- Primary task: `long_view` (the scored label, resolved via
  `pipeline/data/label.py::resolve_primary_label`).
- Candidate auxiliary tasks: `is_click` (see quirk above — resolve it
  first), `is_like` / `is_follow` (sparser but higher-precision positive
  signal), `play_time_ms` (continuous, could support the watch-time/CWM
  direction in Tier 1 headroom item #4 instead of a binary auxiliary head).
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
