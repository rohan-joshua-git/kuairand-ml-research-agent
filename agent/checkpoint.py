"""
Checkpointing for the orchestrator's editable pipeline files + run state.

Two problems this solves:

1. Correctness within a single run. `code_editor.py` writes an
   LLM-authored patch straight to disk once it passes the smoke test, but
   passing the smoke test only means "doesn't crash" — it says nothing
   about whether the patch actually improved the score. Without this
   module, a patch that passed the smoke test but scored worse (or got
   rejected by `compression_gate.py`) stayed on disk anyway: every later
   iteration's ablation and training silently ran against accumulated
   not-actually-good code instead of the last known-best version, so
   `best_metrics` drifted out of sync with what was actually on disk. Every
   iteration now ends with the editable files matching `best_metrics`
   exactly — a new best snapshots them, anything else restores them.
2. Crash recovery across runs. The challenge brief specifically grades
   whether "long iterative runs neither crash, stall, nor diverge" before
   hitting the compute/wall-clock budget. A 6-hour unattended run can be
   killed by an API outage, an OOM, or any unhandled exception outside the
   smoke-tested block. `Orchestrator.run()` checks for an in-progress
   checkpoint on startup and resumes from it — including wall-clock and
   token-usage accounting — instead of losing the whole run and starting
   the iteration/wall-clock budget over from zero.

An interrupted mid-iteration patch (process died between "apply" and
"score") is handled by construction: resuming always restores files from
the last *saved* checkpoint, so an in-flight, never-fully-scored patch is
simply discarded, the same as any other rejected patch.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunState:
    iteration: int
    iterations_without_improvement: int
    best_score: float
    best_metrics: dict
    elapsed_hours_at_checkpoint: float
    token_usage_by_model: dict = field(default_factory=dict)
    saved_at: float = 0.0


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, editable_files: dict[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.files_dir = self.checkpoint_dir / "files"
        self.state_path = self.checkpoint_dir / "state.json"
        self.editable_files = editable_files

    def exists(self) -> bool:
        return self.state_path.exists()

    def save(self, state: RunState) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)
        for key, path in self.editable_files.items():
            if path.exists():
                shutil.copy2(path, self.files_dir / f"{key}.py")
        tmp_path = self.state_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)
        tmp_path.replace(self.state_path)  # atomic on POSIX — a crash mid-write never corrupts state.json

    def load_state(self) -> RunState:
        with open(self.state_path, "r", encoding="utf-8") as f:
            return RunState(**json.load(f))

    def restore_files(self) -> None:
        """Overwrites the live editable files with the checkpoint's saved
        copies. Used both to resume after a crash and, within a single run,
        to revert a patch that didn't end up being a new best."""
        for key, path in self.editable_files.items():
            snapshot = self.files_dir / f"{key}.py"
            if snapshot.exists():
                shutil.copy2(snapshot, path)

    def clear(self) -> None:
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
