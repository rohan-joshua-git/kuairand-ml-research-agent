"""
Generates docs/results_table.md from the run's iteration log + resource
usage report — the "results table with deltas over baseline" and
"resource-usage report" deliverables. Run after `orchestrator.py` finishes
(or at any point mid-run, to check progress).
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.data.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_results_table(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    run_log_dir = Path(cfg["logging"]["run_log_dir"])
    iterations_path = run_log_dir / "iterations.jsonl"
    resource_path = Path(cfg["logging"]["resource_usage_report"])
    intervention_path = Path(cfg["logging"]["intervention_log"])

    if not iterations_path.exists():
        return "# Results\n\nNo run log found yet — run `python -m agent.orchestrator` first.\n"

    with open(iterations_path, "r", encoding="utf-8") as f:
        iterations = [json.loads(line) for line in f if line.strip()]

    resource = {}
    if resource_path.exists():
        with open(resource_path, "r", encoding="utf-8") as f:
            resource = json.load(f)

    intervention_count = 0
    if intervention_path.exists():
        with open(intervention_path, "r", encoding="utf-8") as f:
            intervention_count = sum(1 for line in f if line.strip())

    restart_path = run_log_dir / "restarts.jsonl"
    restart_count = 0
    if restart_path.exists():
        with open(restart_path, "r", encoding="utf-8") as f:
            restart_count = sum(1 for line in f if line.strip())

    lines = ["# Results\n"]
    lines.append(
        "**Note:** metrics are GAUC / nDCG@5 (primary = mean of the two), scored by the "
        "vendored organizer evaluate.py (`starter_kit/evaluate.py`). Official FM baseline: "
        "test GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 — see `starter_kit/baseline_scores.json`.\n"
    )
    # Identify the best scored iteration so the report states outright which
    # checkpoint is the submitted one — the organizer asked for the full
    # trajectory WITH the best run clearly marked (workshop Q&A 2026-08-31).
    # Match on ROW POSITION, not iteration number: iteration 0 legitimately has
    # two rows (official-baseline reproduction, then the agent's own starting
    # pipeline) and marking by number would tag both.
    scored = [(i, it) for i, it in enumerate(iterations)
              if isinstance(it.get("metrics", {}).get("primary"), (int, float))
              and "baseline reproduction" not in it.get("hypothesis", "")]
    best_row = max(scored, key=lambda pair: pair[1]["metrics"]["primary"])[0] if scored else None

    lines.append("| Iteration | Hypothesis | GAUC | nDCG@5 | Primary | Delta vs prev best | Delta vs baseline | Unbiased probe | Accepted | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row_index, it in enumerate(iterations):
        m = it.get("metrics", {})
        errors = "; ".join(it.get("errors", [])) or "-"
        marker = " **<- BEST (submitted)**" if row_index == best_row else ""
        unbiased = m.get("unbiased_primary")
        unbiased_s = f"{unbiased:.4f}" if isinstance(unbiased, (int, float)) else "-"
        d_base = m.get("delta_vs_official_baseline")
        d_base_s = f"{d_base:+.4f}" if isinstance(d_base, (int, float)) else "-"
        d_prev = m.get("delta_vs_prev_best")
        d_prev_s = f"{d_prev:+.4f}" if isinstance(d_prev, (int, float)) else "-"
        lines.append(
            f"| {it['iteration']}{marker} | {it['hypothesis'][:80]} | "
            f"{m.get('gauc', float('nan')):.4f} | {m.get('ndcg_at_5', float('nan')):.4f} | "
            f"{m.get('primary', float('nan')):.4f} | {d_prev_s} | {d_base_s} | {unbiased_s} | "
            f"{m.get('accepted_as_new_best', False)} | {errors} |"
        )

    lines.append("\n## Resource usage\n")
    if resource:
        total_tokens = resource.get("total_tokens", 0)
        wall_clock = resource.get("wall_clock_hours", resource.get("gpu_hours", 0.0))
        lines.append(f"- Total tokens: {total_tokens:,}")
        lines.append(f"- Wall-clock hours: {wall_clock:.3f}")
        lines.append(f"- GPU hours: {resource.get('gpu_hours', 0.0):.3f} (CPU-only pipeline)")
        lines.append(f"- Manual interventions: {intervention_count}")
        lines.append(
            f"- Automatic process restarts after a crash: {restart_count} "
            "(recorded in `logs/restarts.jsonl`). These are NOT manual interventions: "
            "the organizer confirmed in the Track 2 workshop Q&A (2026-08-31) that only "
            "changing the agent's behaviour counts, and `agent/supervisor.py` re-executes "
            "an identical command while the orchestrator resumes from its own checkpoint."
        )
        lines.append("\n### Token usage by model\n")
        for model, usage in resource.get("token_usage_by_model", {}).items():
            lines.append(f"- {model}: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
    else:
        lines.append("Resource usage report not yet generated.")

    return "\n".join(lines) + "\n"


def main() -> None:
    content = generate_results_table()
    out_path = REPO_ROOT / "docs" / "results_table.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
