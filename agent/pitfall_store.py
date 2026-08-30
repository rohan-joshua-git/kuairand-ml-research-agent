"""
Structured failure+recovery record, in the spirit of RecSys Factory's
PitfallStore: a growing table of "we tried X, it failed because Y, we
recovered by Z" that the agent consults before repeating a mistake.

This is distinct from logger.py's per-iteration log: logger.py is a
chronological trail of everything that happened; pitfall_store.py is a
deduplicated, queryable lessons-learned table the orchestrator actively
reads from at the start of each iteration (fed into the Tier-1/Tier-2 skill
context via retriever.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Pitfall:
    id: str
    symptom: str        # what was observed (e.g. "val NDCG dropped after adding feature X")
    root_cause: str      # what actually caused it
    recovery: str          # what fixed it / how the agent routed around it
    stage: str               # which loop stage it happened in: inspect/engineer/train/evaluate
    first_seen_iteration: int
    times_seen: int = 1


class PitfallStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pitfalls: dict[str, Pitfall] = self._load()

    def _load(self) -> dict[str, Pitfall]:
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: Pitfall(**v) for k, v in data.items()}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self._pitfalls.items()}, f, indent=2)

    def record(self, id: str, symptom: str, root_cause: str, recovery: str, stage: str, iteration: int) -> None:
        if id in self._pitfalls:
            self._pitfalls[id].times_seen += 1
        else:
            self._pitfalls[id] = Pitfall(
                id=id,
                symptom=symptom,
                root_cause=root_cause,
                recovery=recovery,
                stage=stage,
                first_seen_iteration=iteration,
            )
        self._save()

    def relevant_to_stage(self, stage: str) -> list[Pitfall]:
        return [p for p in self._pitfalls.values() if p.stage == stage]

    def as_context_block(self, stage: str | None = None) -> str:
        """Renders known pitfalls as a compact block to prepend to the
        agent's prompt — this is what makes past failures actually change
        future behavior instead of just sitting in a log nobody reads."""
        pitfalls = self.relevant_to_stage(stage) if stage else list(self._pitfalls.values())
        if not pitfalls:
            return ""
        lines = ["Known pitfalls from prior iterations (avoid repeating these):"]
        for p in pitfalls:
            lines.append(f"- [{p.stage}] {p.symptom} -> caused by {p.root_cause} -> fix: {p.recovery}")
        return "\n".join(lines)
