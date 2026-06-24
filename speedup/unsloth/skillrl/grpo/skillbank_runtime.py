#!/usr/bin/env python3
"""Runtime SkillBank retrieval for GRPO.

This module deliberately reads from the SkillBank at runtime. It does not use
the distilled `rl_train_vlm_style_reasoning_skilltrigger_v2.jsonl` as training
data or as the source of skills.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SINGLE_ZOOM_DIR = THIS_DIR.parent / "single_zoom"
sys.path.insert(0, str(SINGLE_ZOOM_DIR))

from skillbank_single_zoom import (  # noqa: E402
    format_skill_block,
    load_skillbank,
    retrieve_skills,
    skill_ids,
    validate_no_rezoom,
)


DEFAULT_SKILLBANK = SINGLE_ZOOM_DIR / "data/geoskill_single_zoom_skillbank_v1.json"
FALLBACK_SKILLBANK = THIS_DIR.parent / "stageA/data/spatial_grounding_v0.json"


class RuntimeSkillBank:
    def __init__(self, path: str | Path | None = None, max_task: int = 2) -> None:
        requested = Path(path) if path else DEFAULT_SKILLBANK
        if not requested.exists():
            requested = FALLBACK_SKILLBANK
        self.path = requested
        self.max_task = max_task
        self.skillbank = load_skillbank(requested)
        validate_no_rezoom(self.skillbank)

    def build(self, sample: dict[str, Any]) -> dict[str, Any]:
        question = str(sample.get("question") or "")
        label = str(sample.get("label") or sample.get("question") or "target")
        skills = retrieve_skills(question, label, skillbank=self.skillbank, max_task=self.max_task)
        return {
            "skill_block": format_skill_block(skills),
            "retrieved_skill_ids": skill_ids(skills),
            "retrieved_skills": skills,
        }
