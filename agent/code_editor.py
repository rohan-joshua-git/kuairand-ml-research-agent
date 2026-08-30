"""
Applies LLM-proposed whole-file rewrites to the editable pipeline surface,
with backup/rollback and a subprocess smoke test before anything is kept.

Editable surface is deliberately restricted to a small allowlist
(`EDITABLE_FILES`) rather than "any file in the repo" — this keeps the
blast radius of a bad LLM-generated patch contained to files the
orchestrator knows how to sanity-check and roll back, which is what makes
the "recovery, not failure count" robustness story in the README credible
rather than aspirational.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EDITABLE_FILES = {
    "features": REPO_ROOT / "pipeline" / "data" / "features.py",
    "label": REPO_ROOT / "pipeline" / "data" / "label.py",
}


@dataclass
class PatchApplyResult:
    applied: bool
    smoke_test_passed: bool
    smoke_test_output: str
    error: str | None
    rolled_back: bool


def extract_code(llm_text: str) -> str:
    """LLM output is expected to be a full-file Python replacement, possibly
    wrapped in a ```python ... ``` fence. Strips the fence if present."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", llm_text, re.DOTALL)
    return match.group(1) if match else llm_text.strip()


def apply_and_smoke_test(target: str, new_code: str, smoke_test_module: str, timeout_s: int = 300) -> PatchApplyResult:
    """Backs up `target`'s current content, writes `new_code`, runs
    `python -m {smoke_test_module}` as a subprocess smoke test, and rolls
    back automatically on any failure (non-zero exit, exception, timeout).

    A subprocess (rather than in-process import) is used deliberately: a
    syntax error or crash in agent-generated code must not be able to take
    down the orchestrator process itself.
    """
    if target not in EDITABLE_FILES:
        raise ValueError(f"'{target}' is not in the editable file allowlist: {list(EDITABLE_FILES)}")

    path = EDITABLE_FILES[target]
    original = path.read_text(encoding="utf-8") if path.exists() else ""

    path.write_text(new_code, encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", smoke_test_module],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        path.write_text(original, encoding="utf-8")
        return PatchApplyResult(
            applied=False, smoke_test_passed=False,
            smoke_test_output=str(e), error=f"timeout after {timeout_s}s", rolled_back=True,
        )

    if proc.returncode != 0:
        path.write_text(original, encoding="utf-8")
        return PatchApplyResult(
            applied=False, smoke_test_passed=False,
            smoke_test_output=proc.stdout + proc.stderr,
            error=f"smoke test exited {proc.returncode}", rolled_back=True,
        )

    return PatchApplyResult(
        applied=True, smoke_test_passed=True,
        smoke_test_output=proc.stdout, error=None, rolled_back=False,
    )
