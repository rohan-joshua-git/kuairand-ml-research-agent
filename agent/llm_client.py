"""
Thin wrapper around the Anthropic API. Every call the agent makes to Claude
goes through here, for two reasons: (1) one place to enforce the
iteration-model / reflection-model split from config, and (2) one place to
count tokens for the resource-usage report (Feasibility/Cost, 15% of the
score).

Requires ANTHROPIC_API_KEY in the environment. This module does not
fabricate responses if the key is missing — it raises, so a misconfigured
run fails loudly at the first LLM call rather than silently producing
garbage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class TokenLedger:
    """Running total of tokens spent this run, split by model — feeds
    logger.py's resource_usage_report."""

    def __init__(self):
        self._by_model: dict[str, dict[str, int]] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        bucket = self._by_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens

    def as_dict(self) -> dict:
        return dict(self._by_model)

    def total_tokens(self) -> int:
        return sum(v["input_tokens"] + v["output_tokens"] for v in self._by_model.values())

    def restore(self, snapshot: dict) -> None:
        """Replaces the running totals with a previously-saved snapshot
        (see agent/checkpoint.py) — used when resuming a crashed run so
        token accounting doesn't silently reset to zero and undercount
        the Feasibility/Cost metric."""
        self._by_model = {model: dict(counts) for model, counts in snapshot.items()}


class LLMClient:
    def __init__(self, iteration_model: str, reflection_model: str, ledger: TokenLedger | None = None):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the agent "
                "(the orchestrator makes real API calls — there is no offline/mock mode)."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.iteration_model = iteration_model
        self.reflection_model = reflection_model
        self.ledger = ledger or TokenLedger()

    def _call(self, model: str, system: str, prompt: str, max_tokens: int) -> LLMResponse:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        self.ledger.record(model, response.usage.input_tokens, response.usage.output_tokens)
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )

    def iterate(self, system: str, prompt: str, max_tokens: int = 4096) -> LLMResponse:
        """Cheap, high-volume calls: writing code diffs, running ablation
        analysis. Uses `iteration_model`."""
        return self._call(self.iteration_model, system, prompt, max_tokens)

    def reflect(self, system: str, prompt: str, max_tokens: int = 4096) -> LLMResponse:
        """Expensive, low-volume calls: ideation, reflect+revise decisions,
        final report synthesis. Uses `reflection_model`."""
        return self._call(self.reflection_model, system, prompt, max_tokens)
