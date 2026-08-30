"""
Tiered skill-store retriever. Deliberately dumb (keyword match, not
embeddings) — the point proven by HASTE's ablation is that *tiering*
matters (loading only what's relevant to the current target beats flat-
loading everything), not that retrieval needs to be sophisticated. Tier 1
is small enough to always include; Tier 2/3 are pulled in only when their
keywords match the current ablation target's description.
"""
from __future__ import annotations

from pathlib import Path

# ablation-target keyword -> Tier 3 filename(s). Extend as tier3_deep/ grows.
TIER3_KEYWORD_MAP = {
    "debias": ["autodebias.md"],
    "propensity": ["autodebias.md"],
    "referee": ["autodebias.md"],
    "multitask": ["multitask_heads.md"],
    "multi-task": ["multitask_heads.md"],
    "scenario": ["multitask_heads.md"],
    "auxiliary": ["multitask_heads.md"],
    "esmm": ["multitask_heads.md"],
    "ple": ["multitask_heads.md"],
}


class SkillRetriever:
    def __init__(self, tier1_path: str | Path, tier2_path: str | Path, tier3_dir: str | Path):
        self.tier1_path = Path(tier1_path)
        self.tier2_path = Path(tier2_path)
        self.tier3_dir = Path(tier3_dir)

    def _read(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def context_for_target(self, ablation_target: str, include_tier2: bool = True) -> str:
        """Builds the prompt context block for a given iteration.

        `ablation_target` is a free-text description of what this
        iteration is focused on (e.g. "debiasing block" or "multitask
        heads") — output of `agent/ablation.py`'s block-selection step.
        """
        parts = [self._read(self.tier1_path)]

        if include_tier2:
            parts.append(self._read(self.tier2_path))

        target_lower = ablation_target.lower()
        matched_files: set[str] = set()
        for keyword, filenames in TIER3_KEYWORD_MAP.items():
            if keyword in target_lower:
                matched_files.update(filenames)

        for filename in sorted(matched_files):
            content = self._read(self.tier3_dir / filename)
            if content:
                parts.append(content)

        return "\n\n---\n\n".join(p for p in parts if p)
