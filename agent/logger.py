"""
Run logging — this is a graded deliverable, not paperwork. Per the
challenge rules, judges use these logs as their evidence for both the
Autonomy score (counting manual interventions) and the Robustness score
(recovery from failure, not failure count). If it isn't logged, it doesn't
count.

Writes append-only JSONL so a crashed run still leaves a readable trail up
to the crash point — never buffer-and-write-at-the-end.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class IterationRecord:
    iteration: int
    timestamp: float
    hypothesis: str
    code_diff_summary: str
    metrics: dict
    errors: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)


@dataclass
class InterventionRecord:
    timestamp: float
    iteration: int
    description: str
    reason: str


class RunLogger:
    def __init__(self, run_log_dir: str | Path, intervention_log_path: str | Path, resource_usage_path: str | Path):
        self.run_log_dir = Path(run_log_dir)
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_log_path = self.run_log_dir / "iterations.jsonl"
        self.intervention_log_path = Path(intervention_log_path)
        self.resource_usage_path = Path(resource_usage_path)
        # Recovered from any existing log rather than reset to 0, so a
        # crash-and-resume (see agent/checkpoint.py) doesn't undercount the
        # graded Autonomy metric — interventions.jsonl is append-only and
        # already survives a crash on disk, this just makes the in-memory
        # counter agree with it.
        self._intervention_count = 0
        if self.intervention_log_path.exists():
            with open(self.intervention_log_path, "r", encoding="utf-8") as f:
                self._intervention_count = sum(1 for line in f if line.strip())

    def log_iteration(self, record: IterationRecord) -> None:
        with open(self.iteration_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def log_intervention(self, iteration: int, description: str, reason: str) -> None:
        """Call this ONLY when a human actually steps in — this count feeds
        the Autonomy score directly, so it must be accurate, not padded down."""
        self._intervention_count += 1
        record = InterventionRecord(timestamp=time.time(), iteration=iteration, description=description, reason=reason)
        with open(self.intervention_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    @property
    def intervention_count(self) -> int:
        return self._intervention_count

    def write_resource_usage_report(self, token_usage_by_model: dict, wall_clock_hours: float) -> None:
        report = {
            "token_usage_by_model": token_usage_by_model,
            "total_tokens": sum(v["input_tokens"] + v["output_tokens"] for v in token_usage_by_model.values()),
            "wall_clock_hours": wall_clock_hours,
            "intervention_count": self._intervention_count,
            "generated_at": time.time(),
        }
        self.resource_usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.resource_usage_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def read_iterations(self) -> list[dict]:
        if not self.iteration_log_path.exists():
            return []
        with open(self.iteration_log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
