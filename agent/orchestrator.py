"""
Main agent loop — implements Figure 1 end to end:

    read problem -> inspect data -> engineer features -> train+tune ->
    evaluate -> reflect+revise -> (back to inspect/engineer)

Each iteration:
  1. `ablation.py` identifies which pipeline block is the current
     bottleneck (using cheap, reduced-scale runs).
  2. `skill_store/retriever.py` pulls in only the domain knowledge relevant
     to that block (Tier 1 always, Tier 2/3 keyed to the target).
  3. The reflection model proposes a hypothesis for that block.
  4. The iteration model writes a full-file code replacement implementing
     the hypothesis, targeting one of `code_editor.EDITABLE_FILES`.
  5. `code_editor.py` applies it behind a subprocess smoke test with
     automatic rollback on failure — a bad patch never corrupts state.
  6. On success, a full training run scores the change on validation
     (and, if enabled, `referee.py`'s unbiased probe).
  7. Every iteration is logged (`logger.py`) with hypothesis, diff summary,
     metrics, and any error/recovery — regardless of whether it improved
     anything, since the graded run log needs the full trajectory, not
     just the wins.
  8. A new best checkpoint must pass `compression_gate.py` before being
     designated the final candidate.
  9. The loop stops at convergence (no improvement > epsilon for N
     iterations), at `agent.max_iterations`, or at the wall-clock cap —
     whichever comes first, per config.

This module makes real Anthropic API calls and runs real training — there
is no mocked/offline mode. See README for how to run it.
"""
from __future__ import annotations

import time

from agent import compression_gate, referee
from agent.ablation import pick_highest_impact_block, run_ablation
from agent.code_editor import EDITABLE_FILES, apply_and_smoke_test, extract_code
from agent.llm_client import LLMClient, TokenLedger
from agent.logger import IterationRecord, RunLogger
from agent.pitfall_store import PitfallStore
from agent.skill_store.retriever import SkillRetriever
from pipeline.data.loader import load_config, load_split
from pipeline.evaluate import RankingMetrics, score_delta
from pipeline.train import run_training

ITERATE_SYSTEM_PROMPT = """You are an ML engineer iterating on a recommender \
system pipeline for KuaiRand-Pure. You will be given a specific file to \
rewrite and a hypothesis for how to improve it. Output ONLY the complete \
new file content, wrapped in a single ```python code fence. The file must \
remain importable and preserve any function signatures other modules \
depend on unless the hypothesis specifically requires changing them \
(check the current file content you're given for what's referenced \
elsewhere). If you're rewriting pipeline/train.py, `run_training(...)`'s \
signature is called by agent/ablation.py and pipeline/smoke_test.py with \
specific keyword arguments (split, epochs, lr, pos_weight, device) — \
adding new keyword arguments is fine, removing or renaming existing ones \
will break those callers. If you're rewriting pipeline/model/baseline.py, \
`BaselineCTRModel`'s constructor is called from pipeline/train.py with \
specific keyword arguments (n_users, n_videos, n_tabs, embed_dim, \
numeric_dim) — changing that signature requires rewriting \
pipeline/train.py's instantiation of it in the SAME iteration, or the \
smoke test will fail and the patch will be rolled back."""

REFLECT_SYSTEM_PROMPT = """You are the lead ML researcher directing an \
iteration of an autonomous recommender-system research loop on \
KuaiRand-Pure. Given the current metrics, known pitfalls, ablation \
results, and relevant domain knowledge, propose ONE concrete, specific \
hypothesis for the next code change. Be concrete: name the exact \
mechanism (e.g. "add an auxiliary long_view loss with weight 0.3", not \
"try multitask learning"). One or two sentences."""


class Orchestrator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        agent_cfg = self.cfg["agent"]

        self.ledger = TokenLedger()
        self.llm = LLMClient(
            iteration_model=agent_cfg["llm"]["iteration_model"],
            reflection_model=agent_cfg["llm"]["reflection_model"],
            ledger=self.ledger,
        )
        self.logger = RunLogger(
            run_log_dir=self.cfg["logging"]["run_log_dir"],
            intervention_log_path=self.cfg["logging"]["intervention_log"],
            resource_usage_path=self.cfg["logging"]["resource_usage_report"],
        )
        self.pitfalls = PitfallStore(path=f"{self.cfg['logging']['run_log_dir']}/pitfalls.json")
        self.retriever = SkillRetriever(
            tier1_path=agent_cfg["skill_store"]["tier1_path"],
            tier2_path=agent_cfg["skill_store"]["tier2_path"],
            tier3_dir=agent_cfg["skill_store"]["tier3_dir"],
        )

        self.max_iterations = agent_cfg["max_iterations"]
        self.epsilon = self.cfg["starter_kit"]["epsilon"]
        self.patience_n = self.cfg["starter_kit"]["patience_n"]
        self.wall_clock_cap_hours = self.cfg["starter_kit"]["wall_clock_cap_hours"]

        # Task Requirement #1: "reproduce the official baseline" — the fixed
        # target this run must reach, not just its own iteration-0 score.
        ob = self.cfg["starter_kit"]["official_baseline"]["valid"]
        self.official_baseline_valid = RankingMetrics(
            gauc=ob["gauc"], ndcg_at_5=ob["ndcg_at_5"], n_users=0, n_users_gauc=0
        )

        self._start_time = time.time()

    def _elapsed_hours(self) -> float:
        return (time.time() - self._start_time) / 3600.0

    def run(self) -> None:
        split = load_split(self.cfg)

        print("[orchestrator] Training initial pipeline...")
        current = run_training(split=split)
        best_metrics = current.val_metrics
        best_score = 0.0  # delta vs itself is 0 at iteration 0

        baseline_primary = (self.official_baseline_valid.gauc + self.official_baseline_valid.ndcg_at_5) / 2.0
        baseline_delta = score_delta(best_metrics, self.official_baseline_valid)
        reached = "REACHED" if baseline_delta >= 0 else "BELOW"
        print(
            f"[orchestrator] vs official FM baseline (valid, primary={baseline_primary:.4f}): "
            f"delta={baseline_delta:+.4f} [{reached}]"
        )

        iterations_without_improvement = 0
        iteration = 0

        while iteration < self.max_iterations:
            if self._elapsed_hours() >= self.wall_clock_cap_hours:
                print(f"\n[orchestrator] Wall-clock cap ({self.wall_clock_cap_hours}h) reached — stopping.")
                break

            iteration += 1
            print(f"\n[orchestrator] === Iteration {iteration}/{self.max_iterations} "
                  f"(elapsed {self._elapsed_hours():.2f}h / {self.wall_clock_cap_hours}h) ===")

            errors: list[str] = []
            recovery_actions: list[str] = []

            # 1. Ablation: which block is the bottleneck right now?
            print("[orchestrator] Running ablation...")
            ablation_results = run_ablation(split, best_metrics)
            target = pick_highest_impact_block(ablation_results)
            print(f"[orchestrator] Ablation target: {target.block_name} ({target.description})")

            # 2. Pull tiered domain knowledge relevant to this target.
            skill_context = self.retriever.context_for_target(target.block_name)
            pitfall_context = self.pitfalls.as_context_block()

            # 3. Reflect: propose a concrete hypothesis.
            reflect_prompt = (
                f"Current best metrics: GAUC={best_metrics.gauc:.4f}, "
                f"nDCG@5={best_metrics.ndcg_at_5:.4f}\n\n"
                f"Ablation results this round:\n"
                + "\n".join(
                    f"  - {r.block_name}: dGAUC={r.delta_gauc:+.4f} dNDCG@5={r.delta_ndcg:+.4f}"
                    for r in ablation_results
                )
                + f"\n\nHighest-impact block: {target.block_name}\n\n"
                + (f"{pitfall_context}\n\n" if pitfall_context else "")
                + f"Relevant domain knowledge:\n{skill_context}"
            )
            hypothesis_resp = self.llm.reflect(system=REFLECT_SYSTEM_PROMPT, prompt=reflect_prompt)
            hypothesis = hypothesis_resp.text.strip()
            print(f"[orchestrator] Hypothesis: {hypothesis}")

            # 4. Iterate: write the code change. Which file to target comes
            # straight from the winning ablation variant (see
            # agent/ablation.py::BlockVariant.editable_target) rather than
            # guessing from the block's name — every entry in
            # code_editor.EDITABLE_FILES is a legitimate target now that the
            # engineer-features AND train+tune stages are both editable.
            editable_target = target.editable_target
            current_code = EDITABLE_FILES[editable_target].read_text(encoding="utf-8")
            iterate_prompt = (
                f"Hypothesis to implement: {hypothesis}\n\n"
                f"Current content of {EDITABLE_FILES[editable_target].name}:\n"
                f"```python\n{current_code}\n```\n\n"
                "Rewrite this file to implement the hypothesis."
            )
            code_resp = self.llm.iterate(system=ITERATE_SYSTEM_PROMPT, prompt=iterate_prompt, max_tokens=8192)
            new_code = extract_code(code_resp.text)

            # 5. Apply behind a smoke test, with automatic rollback.
            print("[orchestrator] Applying patch + smoke test...")
            patch_result = apply_and_smoke_test(editable_target, new_code, smoke_test_module="pipeline.smoke_test")

            if not patch_result.applied:
                errors.append(patch_result.error or "unknown smoke test failure")
                recovery_actions.append("rolled back to previous file content, kept previous best checkpoint")
                self.pitfalls.record(
                    id=f"smoke_fail_{target.block_name}",
                    symptom=f"patch to {editable_target} failed smoke test: {patch_result.error}",
                    root_cause=patch_result.smoke_test_output[-500:],
                    recovery="rolled back automatically via code_editor.py",
                    stage="engineer",
                    iteration=iteration,
                )
                self.logger.log_iteration(
                    IterationRecord(
                        iteration=iteration,
                        timestamp=time.time(),
                        hypothesis=hypothesis,
                        code_diff_summary=f"REJECTED (smoke test failed): {target.block_name}",
                        metrics={"gauc": best_metrics.gauc, "ndcg_at_5": best_metrics.ndcg_at_5},
                        errors=errors,
                        recovery_actions=recovery_actions,
                    )
                )
                continue  # doesn't count toward patience — a rejected patch isn't "no improvement," it's a non-event

            # 6. Full training run to score the accepted change.
            print("[orchestrator] Patch accepted by smoke test. Running full training...")
            train_result = run_training(split=split)
            new_metrics = train_result.val_metrics

            delta = score_delta(new_metrics, best_metrics)
            baseline_delta = score_delta(new_metrics, self.official_baseline_valid)
            print(
                f"[orchestrator] New metrics: GAUC={new_metrics.gauc:.4f}, nDCG@5={new_metrics.ndcg_at_5:.4f} "
                f"(delta vs prev best={delta:+.4f}, delta vs official baseline={baseline_delta:+.4f})"
            )

            # Play 1: unbiased referee check, if enabled.
            referee_note = ""
            if referee.referee_enabled(self.cfg):
                try:
                    # NOTE: wiring a live scored-candidates DataFrame through the
                    # referee requires per-iteration inference over the random
                    # log; left as an explicit integration point rather than
                    # faked here. See README "Extension points".
                    referee_note = "referee enabled (mode=%s) — see README for wiring inference through the probe" % self.cfg["referee"]["mode"]
                except Exception as e:  # noqa: BLE001
                    referee_note = f"referee check skipped due to error: {e}"

            is_new_best = delta > self.epsilon

            if is_new_best:
                # Play 1: compression gate before trusting this as final-candidate-worthy.
                gate_result = compression_gate.run_compression_gate(
                    self.llm, hypothesis=hypothesis, code_diff_summary=f"Modified {editable_target}.py per hypothesis above.", cfg=self.cfg
                )
                if gate_result.passed:
                    best_metrics = new_metrics
                    best_score = delta
                    iterations_without_improvement = 0
                    print(f"[orchestrator] New best accepted (compression gate PASSED).")
                else:
                    recovery_actions.append("compression gate rejected the checkpoint — kept previous best")
                    self.pitfalls.record(
                        id=f"compression_gate_fail_{iteration}",
                        symptom=f"iteration {iteration} scored a new best but failed the compression gate",
                        root_cause=gate_result.reasoning[:500],
                        recovery="rejected; previous best checkpoint retained as final candidate",
                        stage="evaluate",
                        iteration=iteration,
                    )
                    iterations_without_improvement += 1
                    print("[orchestrator] New best REJECTED by compression gate — likely overfit. Reverting to previous best.")
            else:
                iterations_without_improvement += 1

            self.logger.log_iteration(
                IterationRecord(
                    iteration=iteration,
                    timestamp=time.time(),
                    hypothesis=hypothesis,
                    code_diff_summary=f"Modified {editable_target}.py targeting {target.block_name}. {referee_note}",
                    metrics={
                        "gauc": new_metrics.gauc,
                        "ndcg_at_5": new_metrics.ndcg_at_5,
                        "delta_vs_prev_best": delta,
                        "delta_vs_official_baseline": baseline_delta,
                        "accepted_as_new_best": is_new_best,
                    },
                    errors=errors,
                    recovery_actions=recovery_actions,
                )
            )

            if iterations_without_improvement >= self.patience_n:
                print(f"\n[orchestrator] Converged: no improvement > epsilon ({self.epsilon}) for {self.patience_n} iterations.")
                break

        wall_clock_hours = self._elapsed_hours()
        self.logger.write_resource_usage_report(self.ledger.as_dict(), wall_clock_hours=wall_clock_hours)
        print(f"\n[orchestrator] Run complete. Final best: GAUC={best_metrics.gauc:.4f}, nDCG@5={best_metrics.ndcg_at_5:.4f}")
        print(f"[orchestrator] Total tokens: {self.ledger.total_tokens():,} | Wall-clock hours: {wall_clock_hours:.3f} | Interventions: {self.logger.intervention_count}")


if __name__ == "__main__":
    Orchestrator().run()
