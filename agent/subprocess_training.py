"""
Runs pipeline/train_runner.py as a subprocess and parses its metrics line.

Every scored training run the orchestrator makes goes through here (full
iteration runs AND cheap ablation probes) for the reasons documented in
pipeline/train_runner.py: fresh imports of whatever the agent's last patch
left on disk, and crash isolation from agent-written code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PREFIX = "TRAIN_RUNNER_METRICS: "


class TrainSubprocessError(RuntimeError):
    """Training subprocess failed (crash, timeout, or unparseable output).
    Carries the captured output for the iteration log / pitfall store."""

    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


@dataclass
class SubprocessTrainResult:
    gauc: float
    ndcg_at_5: float
    primary: float
    n_users: int
    epoch_losses: list[float]
    unbiased_primary: float | None
    unbiased_gauc: float | None
    unbiased_ndcg_at_5: float | None
    referee_error: str | None


def run_training_subprocess(
    epochs: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
    subsample_train: int | None = None,
    no_referee: bool = False,
    timeout_s: int = 2400,
) -> SubprocessTrainResult:
    cmd = [sys.executable, "-m", "pipeline.train_runner"]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if subsample_train is not None:
        cmd += ["--subsample-train", str(subsample_train)]
    if no_referee:
        cmd += ["--no-referee"]

    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise TrainSubprocessError(f"training subprocess timed out after {timeout_s}s", output=str(e)) from e

    if proc.returncode != 0:
        raise TrainSubprocessError(
            f"training subprocess exited {proc.returncode}",
            output=(proc.stdout + proc.stderr)[-2000:],
        )

    for line in proc.stdout.splitlines():
        if line.startswith(METRICS_PREFIX):
            payload = json.loads(line[len(METRICS_PREFIX):])
            return SubprocessTrainResult(
                gauc=payload["gauc"],
                ndcg_at_5=payload["ndcg_at_5"],
                primary=payload["primary"],
                n_users=payload["n_users"],
                epoch_losses=payload.get("epoch_losses", []),
                unbiased_primary=payload.get("unbiased_primary"),
                unbiased_gauc=payload.get("unbiased_gauc"),
                unbiased_ndcg_at_5=payload.get("unbiased_ndcg_at_5"),
                referee_error=payload.get("referee_error"),
            )

    raise TrainSubprocessError(
        "training subprocess exited 0 but printed no TRAIN_RUNNER_METRICS line",
        output=(proc.stdout + proc.stderr)[-2000:],
    )
