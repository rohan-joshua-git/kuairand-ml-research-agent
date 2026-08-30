"""
Main agent loop — implements Figure 1 end to end:

    read problem -> inspect data -> engineer features -> train+tune ->
    evaluate -> reflect+revise -> (back to inspect/engineer)

Each iteration:
  1. `ablation.py` identifies which pipeline block is the current
     bottleneck (using cheap, reduced-scale runs). Every
     `ablation_grid_growth_interval` iterations, the agent first gets a
     chance to rewrite `ablation.py`'s own block-variant grid — see
     `_maybe_grow_ablation_grid` — so ablation *targeting* isn't stuck
     with a fixed seed grid a human has to keep extending by hand.
  2. `skill_store/retriever.py` pulls in only the domain knowledge relevant
     to that block (Tier 1 always, Tier 2/3 keyed to the target).
  3. The reflection model proposes a hypothesis for that block.
  4. The iteration model writes a full-file code replacement implementing
     the hypothesis, targeting one of `code_editor.EDITABLE_FILES`.
  5. `code_editor.py` applies it behind a subprocess smoke test with
     automatic rollback on failure — a bad patch never corrupts state.
  6. On success, in-process modules are reloaded (`_reload_editable_modules`)
     so the very training run that scores the patch actually runs the new
     code — `from x import y` only snapshots `y` at import time, so
     without this, an applied patch would silently keep being scored
     against its own pre-patch behavior. A full training run then scores
     the change on validation, plus `referee.py`'s unbiased probe if
     enabled.
  7. Every iteration is logged (`logger.py`) with hypothesis, diff summary,
     metrics, and any error/recovery — regardless of whether it improved
     anything, since the graded run log needs the full trajectory, not
     just the wins.
  8. A new best checkpoint must pass `compression_gate.py` before being
     designated the final candidate. Whether it passes or not, the
     iteration ends with the on-disk editable files matching
     `best_metrics` exactly (`checkpoint.py`) — a pass snapshots the new
     state as the checkpoint, anything else restores the files to the
     last snapshot, so the pipeline never silently drifts on a patch that
     didn't actually help.
  9. The loop stops at convergence (no improvement > epsilon for N
     iterations), at `agent.max_iterations`, or at the wall-clock cap —
     whichever comes first, per config.

`checkpoint.py` also makes a crashed run resumable: `run()` checks for an
existing checkpoint on startup and, if found, restores the editable files
and run state (iteration count, best metrics, elapsed wall-clock, token
usage) instead of starting over from iteration 0.

This module makes real Anthropic API calls and runs real training — there
is no mocked/offline mode. See README for how to run it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import asdict

from agent import compression_gate, referee
from agent.ablation import pick_highest_impact_block, run_ablation
from agent.checkpoint import CheckpointManager, RunState
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

GROW_ABLATION_SYSTEM_PROMPT = """You maintain the ablation grid \
(agent/ablation.py) for an autonomous ML research loop on KuaiRand-Pure. \
Given the current grid contents and pipeline/train.py's run_training(...) \
signature, add ONE new BlockVariant to default_block_variants() that \
probes something not yet covered — or, if you see a concrete flaw in \
pick_highest_impact_block's selection logic, fix that instead. You may \
ONLY use keyword arguments that already exist in run_training's current \
signature (shown below); inventing a new one is not an error (a bad \
variant is skipped gracefully, not a crash) but it won't produce a useful \
signal until pipeline/train.py separately learns to accept it. Preserve \
every existing BlockVariant unless you have a specific reason to remove \
one. Output ONLY the complete new file content for agent/ablation.py, \
wrapped in a single ```python code fence."""


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
        self.checkpoint = CheckpointManager(
            checkpoint_dir=self.cfg["logging"]["checkpoint_dir"],
            editable_files=EDITABLE_FILES,
        )

        self.max_iterations = agent_cfg["max_iterations"]
        self.grid_growth_interval = agent_cfg.get("ablation_grid_growth_interval") or 0
        self.epsilon = self.cfg["starter_kit"]["epsilon"]
        self.patience_n = self.cfg["starter_kit"]["patience_n"]
        self.wall_clock_cap_hours = self.cfg["starter_kit"]["wall_clock_cap_hours"]

        # Task Requirement #1: "reproduce the official baseline" — the fixed
        # target this run must reach, not just its own iteration-0 score.
        ob = self.cfg["starter_kit"]["official_baseline"]["valid"]
        self.official_baseline_valid = RankingMetrics(
            gauc=ob["gauc"], ndcg_at_5=ob["ndcg_at_5"], n_users=0, n_users_gauc=0
        )

        # Bound methods refreshed by _reload_editable_modules() after every
        # applied patch — see module docstring for why this is necessary.
        self.run_training = run_training
        self.run_ablation = run_ablation
        self.pick_highest_impact_block = pick_highest_impact_block

        self._referee_probe_df = None  # lazily loaded + cached, see _referee_check

        self._start_time = time.time()

    def _elapsed_hours(self) -> float:
        return (time.time() - self._start_time) / 3600.0

    _RELOAD_MODULE_ORDER = (
        "pipeline.data.features",
        "pipeline.data.label",
        "pipeline.model.baseline",
        "pipeline.train",
        "agent.ablation",
    )

    def _reload_editable_modules(self) -> None:
        """Drop and re-import every module `code_editor.EDITABLE_FILES` can
        point the agent at, in dependency order, and rebind this instance's
        references to them.

        `from x import y` only binds `y` to whatever object `x.y` was AT
        IMPORT TIME. Overwriting x.py on disk (what `code_editor.py` does
        on every applied patch) has no effect on an already-bound name
        without picking the new code up somehow. Without this method,
        every "full training run to score the change" after the very
        first patch would silently keep running the pre-patch code — the
        smoke test (a fresh subprocess) would correctly validate the new
        code, but the in-process scoring step that decides whether the
        patch becomes the new best would not, making the entire
        optimization loop not actually test what it just wrote.

        Uses `sys.modules.pop` + fresh `import_module`, not
        `importlib.reload` — reload() re-executes a module's source into
        its EXISTING namespace rather than a clean one, so a name defined
        by an older version and no longer assigned by the new version
        stays sitting in the module's `__dict__` as a stale leftover.
        Harmless for the common case (an unused dangling function/constant
        just wastes a little memory), but it can mask a real bug: if a
        patch renames a helper without updating every call site, the
        rename should be a clear NameError — reload() would instead
        silently resolve the old call to the still-present stale helper.
        Popping first and re-importing gives each module a genuinely
        fresh namespace, so a patch is only ever scored against what its
        own current source actually defines.
        """
        for mod_name in self._RELOAD_MODULE_ORDER:
            sys.modules.pop(mod_name, None)
        for mod_name in self._RELOAD_MODULE_ORDER:
            importlib.import_module(mod_name)

        from agent.ablation import pick_highest_impact_block as _pick, run_ablation as _ra
        from pipeline.train import run_training as _rt

        self.run_training = _rt
        self.run_ablation = _ra
        self.pick_highest_impact_block = _pick

    def _save_checkpoint(
        self, iteration: int, iterations_without_improvement: int, best_score: float, best_metrics: RankingMetrics
    ) -> None:
        self.checkpoint.save(
            RunState(
                iteration=iteration,
                iterations_without_improvement=iterations_without_improvement,
                best_score=best_score,
                best_metrics=asdict(best_metrics),
                elapsed_hours_at_checkpoint=self._elapsed_hours(),
                token_usage_by_model=self.ledger.as_dict(),
                saved_at=time.time(),
            )
        )

    def _revert_to_checkpoint(self) -> None:
        """Undoes an applied-but-not-accepted patch: restores every
        editable file to the last checkpointed (known-best) content and
        reloads in-process modules to match, so the next iteration starts
        clean rather than silently building on a patch that didn't help."""
        self.checkpoint.restore_files()
        self._reload_editable_modules()

    def _maybe_grow_ablation_grid(
        self,
        iteration: int,
        iterations_without_improvement: int,
        best_score: float,
        best_metrics: RankingMetrics,
    ) -> None:
        """Every `ablation_grid_growth_interval` iterations, let the agent
        rewrite its own ablation block-variant grid — see module docstring.
        Always kept on success (checkpointed immediately, using the
        CURRENT best_metrics/state passed in — growth doesn't change the
        score) regardless of whether the current iteration's main patch
        ends up accepted, since grid growth isn't a scored pipeline
        change, it's a search-strategy change with its own dedicated
        validity check.
        """
        if not self.grid_growth_interval or iteration % self.grid_growth_interval != 0:
            return

        print(f"[orchestrator] Iteration {iteration}: attempting to grow the ablation grid...")
        current_code = EDITABLE_FILES["ablation"].read_text(encoding="utf-8")
        train_code = EDITABLE_FILES["train"].read_text(encoding="utf-8")
        prompt = (
            f"Current agent/ablation.py:\n```python\n{current_code}\n```\n\n"
            f"Current pipeline/train.py (for its run_training(...) signature):\n"
            f"```python\n{train_code}\n```"
        )
        resp = self.llm.iterate(system=GROW_ABLATION_SYSTEM_PROMPT, prompt=prompt, max_tokens=8192)
        new_code = extract_code(resp.text)

        result = apply_and_smoke_test("ablation", new_code, smoke_test_module="agent.ablation_smoke_test")
        if result.applied:
            self._reload_editable_modules()
            self._save_checkpoint(iteration, iterations_without_improvement, best_score, best_metrics)
            print("[orchestrator] Ablation grid grown successfully — checkpointed.")
        else:
            print(f"[orchestrator] Ablation grid growth patch failed its smoke test, kept previous grid: {result.error}")
            self.pitfalls.record(
                id=f"ablation_grid_growth_fail_{iteration}",
                symptom=f"iteration {iteration}: ablation grid growth patch failed its smoke test: {result.error}",
                root_cause=result.smoke_test_output[-500:],
                recovery="rolled back automatically via code_editor.py",
                stage="engineer",
                iteration=iteration,
            )

    def _referee_check(self, iteration: int, biased_metrics: RankingMetrics, model, id_maps: dict) -> str:
        """Scores `model` against the unbiased random-exposure probe and
        returns a short note for the iteration log. Never raises — a
        referee failure is diagnostic, not fatal to the run."""
        if not referee.referee_enabled(self.cfg):
            return ""
        try:
            if self._referee_probe_df is None:
                self._referee_probe_df = referee.load_probe_sample(self.cfg)
            probe_scored = referee.score_probe_with_model(model, id_maps, self._referee_probe_df, self.cfg)
            report = referee.build_referee_report(biased_metrics, probe_scored, self.cfg)
            note = (
                f"referee: unbiased GAUC={report.unbiased_metrics.gauc:.4f} "
                f"nDCG@5={report.unbiased_metrics.ndcg_at_5:.4f} "
                f"(div_gauc={report.divergence_gauc:+.4f} div_ndcg={report.divergence_ndcg:+.4f})"
            )
            if report.alert:
                note += " ALERT: diverging from unbiased probe, possible overfit to biased validation"
                self.pitfalls.record(
                    id=f"referee_alert_{iteration}",
                    symptom=(
                        f"iteration {iteration}: validation score diverges from the unbiased random-log "
                        f"probe by more than {self.cfg['referee']['divergence_alert_threshold']}"
                    ),
                    root_cause=(
                        f"biased GAUC={biased_metrics.gauc:.4f} vs unbiased GAUC={report.unbiased_metrics.gauc:.4f}; "
                        f"biased nDCG@5={biased_metrics.ndcg_at_5:.4f} vs unbiased nDCG@5={report.unbiased_metrics.ndcg_at_5:.4f}"
                    ),
                    recovery="flagged for the next reflect+revise step, not auto-reverted — see agent/referee.py",
                    stage="evaluate",
                    iteration=iteration,
                )
            return note
        except Exception as e:  # noqa: BLE001 — diagnostic only, must never crash the loop
            return f"referee check skipped due to error: {e}"

    def run(self) -> None:
        split = load_split(self.cfg)

        if self.checkpoint.exists():
            print("[orchestrator] Found an existing checkpoint — resuming...")
            state = self.checkpoint.load_state()
            self.checkpoint.restore_files()
            self._reload_editable_modules()
            best_metrics = RankingMetrics(**state.best_metrics)
            best_score = state.best_score
            iteration = state.iteration
            iterations_without_improvement = state.iterations_without_improvement
            self._start_time = time.time() - state.elapsed_hours_at_checkpoint * 3600.0
            self.ledger.restore(state.token_usage_by_model)
            print(
                f"[orchestrator] Resumed at iteration {iteration} "
                f"(elapsed {state.elapsed_hours_at_checkpoint:.2f}h), "
                f"best GAUC={best_metrics.gauc:.4f} nDCG@5={best_metrics.ndcg_at_5:.4f}"
            )
        else:
            print("[orchestrator] Training initial pipeline...")
            current = self.run_training(split=split)
            best_metrics = current.val_metrics
            best_score = 0.0  # delta vs itself is 0 at iteration 0
            iteration = 0
            iterations_without_improvement = 0

            baseline_primary = (self.official_baseline_valid.gauc + self.official_baseline_valid.ndcg_at_5) / 2.0
            baseline_delta = score_delta(best_metrics, self.official_baseline_valid)
            reached = "REACHED" if baseline_delta >= 0 else "BELOW"
            print(
                f"[orchestrator] vs official FM baseline (valid, primary={baseline_primary:.4f}): "
                f"delta={baseline_delta:+.4f} [{reached}]"
            )
            self._save_checkpoint(iteration, iterations_without_improvement, best_score, best_metrics)

        while iteration < self.max_iterations:
            if self._elapsed_hours() >= self.wall_clock_cap_hours:
                print(f"\n[orchestrator] Wall-clock cap ({self.wall_clock_cap_hours}h) reached — stopping.")
                break

            iteration += 1
            print(f"\n[orchestrator] === Iteration {iteration}/{self.max_iterations} "
                  f"(elapsed {self._elapsed_hours():.2f}h / {self.wall_clock_cap_hours}h) ===")

            self._maybe_grow_ablation_grid(iteration, iterations_without_improvement, best_score, best_metrics)

            errors: list[str] = []
            recovery_actions: list[str] = []

            # 1. Ablation: which block is the bottleneck right now?
            print("[orchestrator] Running ablation...")
            ablation_results = self.run_ablation(split, best_metrics)
            target = self.pick_highest_impact_block(ablation_results)
            if target is None:
                print("[orchestrator] Every ablation variant failed this round — skipping to next iteration.")
                self.pitfalls.record(
                    id=f"ablation_all_failed_{iteration}",
                    symptom="every BlockVariant in the current ablation grid raised an exception this round",
                    root_cause="see agent/ablation.py::run_ablation's per-variant exception handling for details",
                    recovery="skipped this iteration's pipeline-file patch; a later grid-growth or train.py "
                    "edit should self-correct whatever kwarg mismatch caused this",
                    stage="engineer",
                    iteration=iteration,
                )
                continue  # doesn't count toward patience — mirrors smoke-test-failure treatment below
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

            # 6. Reload in-process modules so the training run below actually
            # exercises the patch just applied (see _reload_editable_modules),
            # then score it.
            print("[orchestrator] Patch accepted by smoke test. Running full training...")
            self._reload_editable_modules()
            train_result = self.run_training(split=split)
            new_metrics = train_result.val_metrics

            delta = score_delta(new_metrics, best_metrics)
            baseline_delta = score_delta(new_metrics, self.official_baseline_valid)
            print(
                f"[orchestrator] New metrics: GAUC={new_metrics.gauc:.4f}, nDCG@5={new_metrics.ndcg_at_5:.4f} "
                f"(delta vs prev best={delta:+.4f}, delta vs official baseline={baseline_delta:+.4f})"
            )

            # Play 1: unbiased referee check, if enabled.
            referee_note = self._referee_check(iteration, new_metrics, train_result.model, train_result.id_maps)

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
                    self._save_checkpoint(iteration, iterations_without_improvement, best_score, best_metrics)
                    print(f"[orchestrator] New best accepted (compression gate PASSED) — checkpointed.")
                else:
                    recovery_actions.append("compression gate rejected the checkpoint — reverted to previous best on disk")
                    self.pitfalls.record(
                        id=f"compression_gate_fail_{iteration}",
                        symptom=f"iteration {iteration} scored a new best but failed the compression gate",
                        root_cause=gate_result.reasoning[:500],
                        recovery="rejected; on-disk editable files reverted to previous best checkpoint",
                        stage="evaluate",
                        iteration=iteration,
                    )
                    iterations_without_improvement += 1
                    self._revert_to_checkpoint()
                    print("[orchestrator] New best REJECTED by compression gate — likely overfit. Reverted to previous best on disk.")
            else:
                iterations_without_improvement += 1
                recovery_actions.append("not a new best — reverted to previous best checkpoint on disk")
                self._revert_to_checkpoint()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Discard any existing checkpoint and start from iteration 0.")
    args = parser.parse_args()

    orchestrator = Orchestrator()
    if args.fresh:
        orchestrator.checkpoint.clear()
    orchestrator.run()
