#!/usr/bin/env python3
"""Runtime SkillBank retrieval for the slime adapter."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SINGLE_ZOOM_DIR = THIS_DIR.parent / "single_zoom"
STAGEA_DIR = THIS_DIR.parent / "stageA"
sys.path.insert(0, str(SINGLE_ZOOM_DIR))
sys.path.insert(0, str(STAGEA_DIR))

from skillbank_single_zoom import (  # noqa: E402
    FORBIDDEN_IDS,
    format_skill_block,
    load_skillbank,
    retrieve_skills,
    skill_ids,
    validate_no_rezoom,
)
from skillbank_v1 import load_skillbank as load_stagea_skillbank  # noqa: E402


DEFAULT_SKILLBANK = SINGLE_ZOOM_DIR / "data/geoskill_single_zoom_skillbank_v1.json"
FALLBACK_SKILLBANK = STAGEA_DIR / "data/spatial_grounding_v0.json"


def _builtin_single_zoom_skillbank() -> dict[str, Any]:
    bank = deepcopy(load_stagea_skillbank(None))
    bank["version"] = "geoskill_single_zoom_builtin"
    bank["general_skills"] = [
        skill for skill in bank.get("general_skills", []) if skill.get("skill_id") not in FORBIDDEN_IDS
    ]
    for group, skills in list((bank.get("task_specific_skills") or {}).items()):
        bank["task_specific_skills"][group] = [
            skill for skill in skills if skill.get("skill_id") not in FORBIDDEN_IDS
        ]
    return bank


class RuntimeSkillBank:
    def __init__(self, path: str | Path | None = None, max_task: int = 2) -> None:
        requested = Path(path) if path else DEFAULT_SKILLBANK
        if requested.exists():
            self.skillbank = load_skillbank(requested)
            self.path = requested
        elif FALLBACK_SKILLBANK.exists():
            self.skillbank = load_skillbank(FALLBACK_SKILLBANK)
            self.path = FALLBACK_SKILLBANK
        else:
            self.skillbank = _builtin_single_zoom_skillbank()
            self.path = Path("<builtin_stageA_single_zoom>")
        self.max_task = max_task
        validate_no_rezoom(self.skillbank)

    def build(self, row: dict[str, Any]) -> dict[str, Any]:
        question = str(row.get("question") or "")
        label = str(row.get("label") or row.get("ground_truth") or row.get("question") or "target")
        skills = retrieve_skills(question, label, skillbank=self.skillbank, max_task=self.max_task)
        return {
            "skill_block": format_skill_block(skills),
            "retrieved_skill_ids": skill_ids(skills),
            "retrieved_skills": skills,
        }
