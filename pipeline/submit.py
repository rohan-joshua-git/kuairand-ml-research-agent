"""
Generates the final submission artifact.

The organizer-defined submission schema is a Day-1 blocker (see README /
config.starter_kit.submission_schema_path) — this module raises loudly
rather than guessing a schema and silently producing something that won't
parse on the organizer's side. Wire the real writer in once the Starter
Kit schema is confirmed; the TODO marks exactly where.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from pipeline.data.loader import load_config


class SubmissionSchemaUnknown(RuntimeError):
    pass


def write_submission(model: torch.nn.Module, id_maps: dict, out_path: str | Path) -> None:
    cfg = load_config()
    schema_path = cfg["starter_kit"].get("submission_schema_path")

    if not schema_path:
        raise SubmissionSchemaUnknown(
            "config.starter_kit.submission_schema_path is not set. Obtain the "
            "official submission schema from the organizer Starter Kit before "
            "generating a real submission — see README 'Open Questions'. "
            "(TODO: once the schema is known, replace this function body with "
            "the actual writer — e.g. per-user top-K ranked candidate lists in "
            "the organizer's specified format.)"
        )

    # Placeholder shape once a schema exists: swap this block for the real writer.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"note": "placeholder — schema not yet wired"}, f)
