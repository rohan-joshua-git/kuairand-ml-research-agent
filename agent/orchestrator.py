"""
Main agent loop — implements Figure 1 end to end:

    read problem -> inspect data -> engineer features -> train+tune ->
    evaluate -> reflect+revise -> (back to inspect/engineer)

Each iteration:
  1. `ablation.py` identifies which pipeline block is the current
     bottleneck (cheap, subsampled subprocess probes).
  2. `skill_store/retriever.py` pulls in only the domain knowledge relevant
     to that block (Tier 1 always, Tier 2/3 keyed to the target).
  3. The reflection model proposes a hypothesis for the next change AND
     names which editable file the hypothesis requires (features / label /
     train) — file routing follows the hypothesis, not the ablation block
     name, because the first live run proved the two can disagree (the
     agent proposed a BPR loss, which lives in train.py, while block-name
     routing sent the edit to features.py).
  4. The iteration model writes a full-file code replacement implementing
     the hypothesis, targeting one of `code_editor.EDITABLE_FILES`.
  5. `code_editor.py` applies it behind a subprocess smoke test with
     automatic rollback on failure — a bad patch never corrupts state.
  6. On success, a full training run scores the change on validation, in a
     FRESH SUBPROCESS (`pipeline/train_runner.py`) so the code being scored
     is exactly what's on disk, and the run also scores the unbiased
     random-exposure probe (`referee.py`, Play 1).
  7. Every iteration is logged (`logger.py`) with hypothesis, diff summary,
     metrics, and any error/recovery — regardless of whether it improved
     anything, since the graded run log needs the full trajectory, not
     just the wins.
  8. A new best checkpoint must pass `compression_gate.py` before being
     designated the final candidate. A change that is NOT accepted as the
     new best is rolled back on disk, so the working tree always holds the
     best-known code state — which is what the final submission re-trains.
  9. The loop stops at convergence (no improvement > epsilon for N
     iterations) or at `agent.max_iterations`, per config.

This module makes real LLM API calls and runs real training — there is no
mocked/offline mode. See README for how to run it.
"""
from __future__ import annotations

import re
import time

from agent import compression_gate
from agent.ablation import pick_highest_impact_block, run_ablation
from agent.checkpoint import CheckpointManager, RunState
from agent.code_editor import EDITABLE_FILES, apply_and_smoke_test, extract_code
from agent.llm_client import TokenLedger, build_llm_client
from agent.logger import IterationRecord, RunLogger
from agent.pitfall_store import PitfallStore
from agent.skill_store.retriever import SkillRetriever
from agent.subprocess_training import TrainSubprocessError, run_training_subprocess
from pipeline.data.loader import load_config
from pipeline.evaluate import RankingMetrics
from pipeline.official_baseline import reproduce_official_baseline

ITERATE_SYSTEM_PROMPT = """You are an ML engineer iterating on a recommender \
system pipeline for KuaiRand-Pure. You will be given a specific file to \
rewrite and a hypothesis for how to improve it. Output ONLY the complete \
new file content, wrapped in a single ```python code fence. The file must \
remain importable and preserve any function signatures other modules \
depend on unless the hypothesis specifically requires changing them \
(check the current file content you're given for what's referenced \
elsewhere)."""

REFLECT_SYSTEM_PROMPT = """You are the lead ML researcher directing an \
iteration of an autonomous recommender-system research loop on \
KuaiRand-Pure. Given the current metrics, iteration history, known \
pitfalls, ablation results, and relevant domain knowledge, propose ONE \
concrete, specific hypothesis for the next code change. Be concrete: name \
the exact mechanism (e.g. "add an auxiliary long_view loss with weight \
0.3", not "try multitask learning"). Do not re-propose a hypothesis the \
history shows already failed to improve the score. One or two sentences, \
then end with a final line of exactly:
TARGET_FILE: <features|label|train>
where features = pipeline/data/features.py (feature engineering), \
label = pipeline/data/label.py (label resolution), \
train = pipeline/train.py (model definition, loss function, optimizer, \
training loop, hyperparameters)."""

# Fallback routing if the model doesn't emit a parseable TARGET_FILE line.
# Ordered most-specific first; checked against the hypothesis text.
_ROUTING_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("train", ("loss", "bpr", "pairwise", "listwise", "softmax", "rank",
               "optimizer", "learning rate", "epoch", "batch", "architecture",
               "embedding", "model", "head", "tower", "sequence", "attention",
               "dropout", "regulariz", "negative sampl", "margin")),
    ("label", ("label resolution", "relabel", "valid_play", "is_click",
               "auxiliary label", "primary label")),
]


def route_target_file(reflect_text: str) -> tuple[str, str]:
    """Returns (editable_target, hypothesis_text). Prefers the model's own
    TARGET_FILE declaration; falls back to keyword routing on the
    hypothesis text; defaults to 'features'."""
    match = re.search(r"TARGET_FILE:\s*[`*]*(features|label|train)", reflect_text, re.IGNORECASE)
    hypothesis = re.sub(r"TARGET_FILE:.*$", "", reflect_text, flags=re.IGNORECASE | re.MULTILINE).strip()
    if match:
        return match.group(1).lower(), hypothesis
    lowered = hypothesis.lower()
    for target, keywords in _ROUTING_RULES:
        if any(k in lowered for k in keywords):
            return target, hypothesis
    return "features", hypothesis


def _as_ranking_metrics(r) -> RankingMetrics:
    return RankingMetrics(gauc=r.gauc, ndcg_at_5=r.ndcg_at_5, primary=r.primary, n_users=r.n_users)


class Orchestrator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        agent_cfg = self.cfg["agent"]

        self.ledger = TokenLedger()
        self.llm = build_llm_client(self.cfg, ledger=self.ledger)
        self.checkpoint = CheckpointManager(
            checkpoint_dir=self.cfg["logging"]["checkpoint_dir"], editable_files=EDITABLE_FILES
        )
        # Resuming continues the SAME logical run, so its trajectory must keep
        # appending to the same log rather than rotating it away.
        self.logger = RunLogger(
            run_log_dir=self.cfg["logging"]["run_log_dir"],
            intervention_log_path=self.cfg["logging"]["intervention_log"],
            resource_usage_path=self.cfg["logging"]["resource_usage_report"],
            rotate_existing=not self.checkpoint.exists(),
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
        self.divergence_threshold = self.cfg["referee"]["divergence_alert_threshold"]

        self._start_time = time.time()
        self._history: list[dict] = []  # this run's (hypothesis, target, delta, accepted) trail for the reflect prompt

    def _save_checkpoint(self, iteration: int, iterations_without_improvement: int, best: RankingMetrics) -> None:
        """Snapshot the editable files + run state after every accepted best, so
        a crashed run resumes from the last known-good state instead of
        restarting the iteration and wall-clock budget from zero. The organizer
        confirmed (workshop Q&A 2026-08-31) that restarting a crashed process is
        not a manual intervention, so this is the sanctioned way to survive an
        API outage mid-run."""
        self.checkpoint.save(
            RunState(
                iteration=iteration,
                iterations_without_improvement=iterations_without_improvement,
                best_score=best.primary,
                best_metrics={"gauc": best.gauc, "ndcg_at_5": best.ndcg_at_5,
                              "primary": best.primary, "n_users": best.n_users},
                elapsed_hours_at_checkpoint=(time.time() - self._start_time) / 3600.0,
                token_usage_by_model=self.ledger.as_dict(),
                saved_at=time.time(),
            )
        )

    def _resume_if_checkpointed(self) -> tuple[RankingMetrics, int, int] | None:
        """Restores editable files + run state from a previous crashed run.
        Returns (best_metrics, next_iteration, iterations_without_improvement)
        or None when there is nothing to resume."""
        if not self.checkpoint.exists():
            return None
        state = self.checkpoint.load_state()
        self.checkpoint.restore_files()
        bm = state.best_metrics
        best = RankingMetrics(gauc=bm["gauc"], ndcg_at_5=bm["ndcg_at_5"],
                              primary=bm["primary"], n_users=bm.get("n_users", 0))
        # Charge already-spent wall-clock against the 6h ceiling, so a resumed
        # run cannot exceed the budget by restarting the clock.
        self._start_time = time.time() - state.elapsed_hours_at_checkpoint * 3600.0
        self._resumed_from_checkpoint = True
        print(f"[orchestrator] RESUMING from checkpoint: iteration {state.iteration}, "
              f"best primary={best.primary:.4f}, {state.elapsed_hours_at_checkpoint:.2f}h already spent.")
        return best, state.iteration, state.iterations_without_improvement

    def _history_block(self, limit: int = 8) -> str:
        if not self._history:
            return ""
        lines = ["Iteration history so far (hypothesis -> outcome):"]
        for h in self._history[-limit:]:
            lines.append(
                f"  - iter {h['iteration']} [{h['target']}]: {h['hypothesis'][:160]} -> "
                f"primary {h['primary']:.4f} (delta vs best {h['delta']:+.4f}, "
                f"{'ACCEPTED as new best' if h['accepted'] else h['outcome']})"
            )
        return "\n".join(lines)

    def _log_initial_training(self, initial, best_metrics: RankingMetrics, official_baseline_valid_primary: float) -> None:
        """Records the agent's starting editable pipeline as iteration 0 — the
        reference every later delta is measured against."""
        print(
            f"[orchestrator] Initial agent model: GAUC={best_metrics.gauc:.4f}, "
            f"nDCG@5={best_metrics.ndcg_at_5:.4f}, primary={best_metrics.primary:.4f}"
            + (f" | unbiased probe primary={initial.unbiased_primary:.4f}" if initial.unbiased_primary is not None else "")
        )
        self.logger.log_iteration(
            IterationRecord(
                iteration=0,
                timestamp=time.time(),
                hypothesis="(initial editable-pipeline training, not an LLM hypothesis)",
                code_diff_summary="Trained the agent's starting editable pipeline (pipeline/train.py as-is) to set the iteration-0 reference.",
                metrics={
                    "gauc": best_metrics.gauc,
                    "ndcg_at_5": best_metrics.ndcg_at_5,
                    "primary": best_metrics.primary,
                    "unbiased_primary": initial.unbiased_primary,
                    "delta_vs_official_baseline": best_metrics.primary - official_baseline_valid_primary,
                },
            )
        )

    def run(self) -> None:
        official_baseline_valid_primary = self.cfg["starter_kit"]["official_baseline"]["valid"]["primary"]

        # Task Requirement #1: reproduce the official baseline BEFORE any
        # LLM-driven iteration, and confirm it reaches the published
        # validation score. Trains the organizer's own vendored FM directly
        # (pipeline/official_baseline.py) — never a reimplementation. The
        # (also-computed) test-split number is logged for evidence only and
        # is never surfaced to the reflect/iterate prompts below — the
        # agent's decisions must never be informed by hidden-test scores.
        resumed = self._resume_if_checkpointed()

        print("[orchestrator] Reproducing official baseline (starter_kit FM, k=16, lr=0.001)...")
        baseline_repro = reproduce_official_baseline(self.cfg)
        print(
            f"[orchestrator] Official baseline reproduction: valid primary={baseline_repro.valid_primary:.4f} "
            f"(published {baseline_repro.published_valid_primary:.4f}) — "
            f"{'MATCHES' if baseline_repro.matches_published else 'DOES NOT MATCH'} published score."
        )
        self.logger.log_iteration(
            IterationRecord(
                iteration=0,
                timestamp=time.time(),
                hypothesis="(baseline reproduction, not an LLM hypothesis)",
                code_diff_summary="Reproduced the official FM baseline via starter_kit/baseline.py, per Task Requirement #1.",
                metrics={
                    "gauc": baseline_repro.valid_gauc,
                    "ndcg_at_5": baseline_repro.valid_ndcg_at_5,
                    "primary": baseline_repro.valid_primary,
                    "matches_published_baseline": baseline_repro.matches_published,
                },
                errors=[] if baseline_repro.matches_published else [
                    f"reproduced valid primary {baseline_repro.valid_primary:.4f} did not match published "
                    f"{baseline_repro.published_valid_primary:.4f} within tolerance"
                ],
            )
        )

        if resumed is not None:
            best_metrics, resumed_iteration, resumed_no_improve = resumed
            initial = None
        else:
            print("[orchestrator] Training initial agent pipeline (editable torch model, iteration 0)...")
            initial = run_training_subprocess()
            best_metrics = _as_ranking_metrics(initial)
        if initial is not None:
            self._log_initial_training(initial, best_metrics, official_baseline_valid_primary)


        if resumed is not None:
            iteration = resumed_iteration
            iterations_without_improvement = resumed_no_improve
        else:
            iterations_without_improvement = 0
            iteration = 0
            self._save_checkpoint(iteration, iterations_without_improvement, best_metrics)

        while iteration < self.max_iterations:
            iteration += 1
            elapsed_hours = (time.time() - self._start_time) / 3600.0
            if elapsed_hours >= self.cfg["starter_kit"]["wall_clock_ceiling_hours"]:
                print(f"\n[orchestrator] Wall-clock ceiling reached ({elapsed_hours:.2f}h) — stopping.")
                break
            print(f"\n[orchestrator] === Iteration {iteration}/{self.max_iterations} ===")

            errors: list[str] = []
            recovery_actions: list[str] = []

            # 1. Ablation: which block is the bottleneck right now?
            print("[orchestrator] Running ablation (cheap subsampled probes)...")
            ablation_results = run_ablation(best_metrics)
            ablation_target = pick_highest_impact_block(ablation_results)
            print(f"[orchestrator] Ablation signal: {ablation_target.block_name} ({ablation_target.description})")

            # 2. Pull tiered domain knowledge relevant to this signal.
            skill_context = self.retriever.context_for_target(ablation_target.block_name)
            pitfall_context = self.pitfalls.as_context_block()
            history_context = self._history_block()

            # 3. Reflect: propose a concrete hypothesis + the file it needs.
            reflect_prompt = (
                f"Current best validation metrics: GAUC={best_metrics.gauc:.4f}, "
                f"nDCG@5={best_metrics.ndcg_at_5:.4f}, primary={best_metrics.primary:.4f} "
                f"(official baseline primary: {official_baseline_valid_primary:.4f})\n\n"
                f"Ablation probe results this round (subsampled — compare variants to each other, not to zero):\n"
                + "\n".join(
                    f"  - {r.block_name}: probe primary={r.val_primary:.4f}" + (f" [FAILED: {r.error}]" if r.error else "")
                    for r in ablation_results
                )
                + "\n\n"
                + (f"{history_context}\n\n" if history_context else "")
                + (f"{pitfall_context}\n\n" if pitfall_context else "")
                + f"Relevant domain knowledge:\n{skill_context}"
            )
            hypothesis_resp = self.llm.reflect(system=REFLECT_SYSTEM_PROMPT, prompt=reflect_prompt)
            editable_target, hypothesis = route_target_file(hypothesis_resp.text.strip())
            print(f"[orchestrator] Hypothesis (-> {editable_target}.py): {hypothesis}")

            # 4. Iterate: write the code change.
            target_path = EDITABLE_FILES[editable_target]
            current_code = target_path.read_text(encoding="utf-8")
            iterate_prompt = (
                f"Hypothesis to implement: {hypothesis}\n\n"
                f"Current content of {target_path.name}:\n"
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
                    id=f"smoke_fail_{editable_target}",
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
                        code_diff_summary=f"REJECTED (smoke test failed): patch to {editable_target}.py",
                        metrics={"gauc": best_metrics.gauc, "ndcg_at_5": best_metrics.ndcg_at_5, "primary": best_metrics.primary},
                        errors=errors,
                        recovery_actions=recovery_actions,
                    )
                )
                self._history.append({
                    "iteration": iteration, "target": editable_target, "hypothesis": hypothesis,
                    "primary": best_metrics.primary, "delta": 0.0, "accepted": False,
                    "outcome": "patch failed smoke test, rolled back",
                })
                continue  # doesn't count toward patience — a rejected patch isn't "no improvement," it's a non-event

            # 6. Full training run (fresh subprocess) to score the accepted change.
            print("[orchestrator] Patch accepted by smoke test. Running full training in subprocess...")
            try:
                train_result = run_training_subprocess()
            except TrainSubprocessError as e:
                target_path.write_text(current_code, encoding="utf-8")
                errors.append(f"full training failed after smoke test passed: {e}")
                recovery_actions.append("restored pre-patch file content; kept previous best checkpoint")
                self.pitfalls.record(
                    id=f"full_train_fail_{editable_target}",
                    symptom=f"patch to {editable_target} passed the 1-epoch smoke test but failed the full run: {e}",
                    root_cause=e.output[-500:],
                    recovery="rolled back to pre-patch file content",
                    stage="train",
                    iteration=iteration,
                )
                self.logger.log_iteration(
                    IterationRecord(
                        iteration=iteration,
                        timestamp=time.time(),
                        hypothesis=hypothesis,
                        code_diff_summary=f"REJECTED (full training failed): patch to {editable_target}.py",
                        metrics={"gauc": best_metrics.gauc, "ndcg_at_5": best_metrics.ndcg_at_5, "primary": best_metrics.primary},
                        errors=errors,
                        recovery_actions=recovery_actions,
                    )
                )
                self._history.append({
                    "iteration": iteration, "target": editable_target, "hypothesis": hypothesis,
                    "primary": best_metrics.primary, "delta": 0.0, "accepted": False,
                    "outcome": "full training crashed, rolled back",
                })
                continue

            new_metrics = _as_ranking_metrics(train_result)
            delta = new_metrics.primary - best_metrics.primary
            delta_vs_official_baseline = new_metrics.primary - official_baseline_valid_primary
            divergence = (
                new_metrics.primary - train_result.unbiased_primary
                if train_result.unbiased_primary is not None else None
            )
            referee_note = (
                f"unbiased probe primary={train_result.unbiased_primary:.4f}, divergence={divergence:+.4f}"
                + (" [ALERT: exceeds threshold]" if divergence > self.divergence_threshold else "")
                if divergence is not None
                else (f"referee probe failed: {train_result.referee_error}" if train_result.referee_error else "referee disabled")
            )
            print(
                f"[orchestrator] New metrics: GAUC={new_metrics.gauc:.4f}, nDCG@5={new_metrics.ndcg_at_5:.4f}, "
                f"primary={new_metrics.primary:.4f} (delta vs prev best={delta:+.4f}, "
                f"delta vs official baseline={delta_vs_official_baseline:+.4f}) | {referee_note}"
            )

            is_new_best = delta > self.epsilon
            outcome = ""

            if is_new_best:
                # Play 1: compression gate before trusting this as final-candidate-worthy.
                gate_result = compression_gate.run_compression_gate(
                    self.llm, hypothesis=hypothesis, code_diff_summary=f"Modified {editable_target}.py per hypothesis above.", cfg=self.cfg
                )
                if gate_result.passed:
                    best_metrics = new_metrics
                    iterations_without_improvement = 0
                    self._save_checkpoint(iteration, iterations_without_improvement, best_metrics)
                    outcome = "accepted as new best (compression gate passed)"
                    print("[orchestrator] New best accepted (compression gate PASSED).")
                else:
                    is_new_best = False
                    target_path.write_text(current_code, encoding="utf-8")
                    recovery_actions.append("compression gate rejected the checkpoint — restored pre-patch code, kept previous best")
                    self.pitfalls.record(
                        id=f"compression_gate_fail_{iteration}",
                        symptom=f"iteration {iteration} scored a new best but failed the compression gate",
                        root_cause=gate_result.reasoning[:500],
                        recovery="rejected; pre-patch code restored; previous best checkpoint retained",
                        stage="evaluate",
                        iteration=iteration,
                    )
                    iterations_without_improvement += 1
                    outcome = "scored above best but REJECTED by compression gate, rolled back"
                    print("[orchestrator] New best REJECTED by compression gate — likely overfit. Reverting to previous best.")
            else:
                # Not an improvement above epsilon: restore the pre-patch code
                # so the working tree always equals the best-known state (and
                # so `best_metrics` stays reproducible from disk).
                target_path.write_text(current_code, encoding="utf-8")
                recovery_actions.append("change did not beat best by > epsilon — restored pre-patch code")
                iterations_without_improvement += 1
                outcome = f"no improvement > epsilon ({delta:+.4f}), rolled back"

            self.logger.log_iteration(
                IterationRecord(
                    iteration=iteration,
                    timestamp=time.time(),
                    hypothesis=hypothesis,
                    code_diff_summary=f"Modified {editable_target}.py (ablation signal: {ablation_target.block_name}). {referee_note}",
                    metrics={
                        "gauc": new_metrics.gauc,
                        "ndcg_at_5": new_metrics.ndcg_at_5,
                        "primary": new_metrics.primary,
                        "unbiased_primary": train_result.unbiased_primary,
                        "biased_unbiased_divergence": divergence,
                        "delta_vs_prev_best": delta,
                        "delta_vs_official_baseline": delta_vs_official_baseline,
                        "accepted_as_new_best": is_new_best,
                    },
                    errors=errors,
                    recovery_actions=recovery_actions,
                )
            )
            self._history.append({
                "iteration": iteration, "target": editable_target, "hypothesis": hypothesis,
                "primary": new_metrics.primary, "delta": delta, "accepted": is_new_best,
                "outcome": outcome,
            })

            if iterations_without_improvement >= self.patience_n:
                print(f"\n[orchestrator] Converged: no improvement > epsilon ({self.epsilon}) for {self.patience_n} iterations.")
                break

        wall_clock_hours = (time.time() - self._start_time) / 3600.0
        self.logger.write_resource_usage_report(self.ledger.as_dict(), wall_clock_hours=wall_clock_hours)
        final_delta = best_metrics.primary - official_baseline_valid_primary
        print(
            f"\n[orchestrator] Run complete. Final best: GAUC={best_metrics.gauc:.4f}, "
            f"nDCG@5={best_metrics.ndcg_at_5:.4f}, primary={best_metrics.primary:.4f} "
            f"(delta vs official baseline: {final_delta:+.4f})"
        )
        print(
            "[orchestrator] The working tree holds the best-known code state (non-improving patches were rolled back) — "
            "the submission step re-trains from it with a fixed seed."
        )
        print(f"[orchestrator] Total tokens: {self.ledger.total_tokens():,} | Wall-clock hours: {wall_clock_hours:.3f} | Interventions: {self.logger.intervention_count}")


if __name__ == "__main__":
    Orchestrator().run()
