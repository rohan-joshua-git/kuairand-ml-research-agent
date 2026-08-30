"""
Agent-generated model variants land here, one module per iteration lineage
(e.g. `iter_003_multitask_esmm.py`). Nothing lives here at repo init time —
`agent/ablation.py` and `agent/orchestrator.py` populate this directory as
the agent proposes and keeps architecture changes across iterations.

Keeping variants as separate files (rather than overwriting one file) is
what lets `ablation.py` diff across the lineage and `pitfall_store.py`
attribute a regression to a specific iteration.
"""
