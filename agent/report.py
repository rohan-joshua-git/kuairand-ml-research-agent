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

    lines = ["# Results\n"]
    lines.append(
        "**Note:** \"Delta vs official baseline\" is against the FM baseline's "
        "official validation score (`starter_kit/baseline_scores.json`, "
        "primary=0.6016) — the target Task Requirement #1 asks the agent to "
        "reproduce and then beat. \"Delta vs prev best\" is the self-referential "
        "improvement that drives the convergence rule.\n"
    )
    lines.append("| Iteration | Hypothesis | GAUC | nDCG@5 | Delta vs prev best | Delta vs official baseline | Accepted | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for it in iterations:
        m = it.get("metrics", {})
        errors = "; ".join(it.get("errors", [])) or "-"
        lines.append(
            f"| {it['iteration']} | {it['hypothesis'][:80]} | "
            f"{m.get('gauc', float('nan')):.4f} | {m.get('ndcg_at_5', float('nan')):.4f} | "
            f"{m.get('delta_vs_prev_best', float('nan')):+.4f} | {m.get('delta_vs_official_baseline', float('nan')):+.4f} | "
            f"{m.get('accepted_as_new_best', False)} | {errors} |"
        )

    lines.append("\n## Resource usage\n")
    if resource:
        lines.append(f"- Total tokens: {resource.get('total_tokens', 'n/a'):,}")
        lines.append(f"- Wall-clock hours: {resource.get('wall_clock_hours', 'n/a'):.3f}")
        lines.append(f"- Manual interventions: {intervention_count}")
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
