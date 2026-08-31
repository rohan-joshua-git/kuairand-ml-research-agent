"""
Unattended run supervisor: relaunches `agent.orchestrator` if it dies, until
it converges, hits its budget, or exhausts the restart allowance.

Why this exists and why it is not "cheating autonomy": in the Track 2
workshop Q&A (2026-08-31) the organizer was asked directly whether
restarting a crashed process counts as a manual intervention. The answer
was no — "we only consider the manual intervention if you change the
agent's behaviour" — and the suggested remedy was literally to have a
second session restart the crashed one. This module is that second session,
automated, so the restart happens without a human touching anything.

What it does NOT do: it never edits code, never changes config, never
selects or discards a checkpoint, and never re-runs with different
hyperparameters. It re-executes the identical command. The orchestrator
resumes from `agent/checkpoint.py`, so a restart continues the same run
(same wall-clock budget, same best checkpoint) rather than starting over.

Every restart is recorded to logs/restarts.jsonl with the exit code and the
tail of the failure, so the report can state exactly how many crashes
happened and why — that record is evidence for the Robustness criterion
("recovery from failure, not failure count"), which is why it is written
even though restarts are not interventions.

Usage:
    python -m agent.supervisor                    # default 20 restarts
    python -m agent.supervisor --max-restarts 50
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESTART_LOG = REPO_ROOT / "logs" / "restarts.jsonl"

# Converged//budget-exhausted runs exit 0. Anything else is a crash worth
# retrying — but back off so a hard outage doesn't spin.
BACKOFF_S = [30, 60, 120, 300, 600]


def _record(attempt: int, returncode: int, tail: str) -> None:
    RESTART_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RESTART_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "attempt": attempt,
            "returncode": returncode,
            "failure_tail": tail[-1500:],
            "note": "process restart after crash; not a manual intervention "
                    "(organizer Q&A 2026-08-31) — no behaviour was changed",
        }) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--log", default="logs/orchestrator_run.log")
    args = parser.parse_args()

    log_path = REPO_ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(args.max_restarts + 1):
        print(f"[supervisor] Launching orchestrator (attempt {attempt + 1}/{args.max_restarts + 1})...", flush=True)
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n===== supervisor launch attempt {attempt + 1} @ {time.ctime()} =====\n")
            logf.flush()
            proc = subprocess.run(
                [sys.executable, "-u", "-m", "agent.orchestrator"],
                cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT,
            )

        if proc.returncode == 0:
            print("[supervisor] Orchestrator exited cleanly (converged or budget reached).", flush=True)
            return 0

        tail = ""
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        _record(attempt + 1, proc.returncode, tail)
        print(f"[supervisor] Orchestrator crashed (exit {proc.returncode}). Recorded to {RESTART_LOG.name}.", flush=True)

        if attempt == args.max_restarts:
            print("[supervisor] Restart allowance exhausted — giving up.", flush=True)
            return proc.returncode

        delay = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
        print(f"[supervisor] Restarting in {delay}s (orchestrator resumes from its checkpoint).", flush=True)
        time.sleep(delay)

    return 0


if __name__ == "__main__":
    sys.exit(main())
