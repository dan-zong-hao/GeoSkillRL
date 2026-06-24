#!/usr/bin/env python3
"""GeoSkillRL reward adapter for ZoomEarth-style GRPO."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SKILLRL_DIR = THIS_DIR.parent
GRPO_DIR = SKILLRL_DIR / "grpo"
sys.path.insert(0, str(GRPO_DIR))

from reward_geo import compute_episode_reward  # noqa: E402


def compute_geo_episode_reward(
    *,
    sample: dict[str, Any],
    turn1_text: str,
    turn2_text: str,
) -> dict[str, Any]:
    """Compute the canonical GeoSkillRL reward for one two-stage rollout."""
    answer_text = turn2_text or ""
    if answer_text and not answer_text.startswith("<think>"):
        answer_text = "<think>" + answer_text
    return compute_episode_reward(
        question_id=str(sample["question_id"]),
        question=str(sample["question"]),
        zoom_text=turn1_text or "",
        answer_text=answer_text,
        gt_bbox_1024=[float(v) for v in sample["bbox_1024"]],
        ground_truth=sample["ground_truth"],
        image_size=sample["image_size"],
    )

