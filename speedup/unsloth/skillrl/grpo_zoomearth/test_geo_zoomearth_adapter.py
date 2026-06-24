#!/usr/bin/env python3
"""Lightweight checks for the GeoSkillRL ZoomEarth adapter."""
from __future__ import annotations

from pathlib import Path

from geo_data import DEFAULT_TRAIN_JSONL, GeoBBoxDataset
from geo_rewards import compute_geo_episode_reward
from geo_rollout import build_stage2_context, parse_zoom_bbox, prepare_sample

import sys

THIS_DIR = Path(__file__).resolve().parent
GRPO_DIR = THIS_DIR.parent / "grpo"
sys.path.insert(0, str(GRPO_DIR))

from skillbank_runtime import DEFAULT_SKILLBANK, RuntimeSkillBank  # noqa: E402


def test_dataset_filters_bbox_samples() -> None:
    dataset = GeoBBoxDataset(DEFAULT_TRAIN_JSONL)
    assert len(dataset) == 1472, len(dataset)
    assert all(row["type"] in {"object", "region"} for row in dataset.records)
    assert all(isinstance(row["bbox"], list) and len(row["bbox"]) == 4 for row in dataset.records)


def test_primitive_bbox_parser() -> None:
    text = "<zoom><|ref|>target<|/ref|><|box|>[[1,2,3,4]]<|/box|></zoom>"
    assert parse_zoom_bbox(text) == [1.0, 2.0, 3.0, 4.0]


def test_skill_prompt_and_no_forbidden_skills() -> None:
    dataset = GeoBBoxDataset(DEFAULT_TRAIN_JSONL, limit=1)
    skillbank = RuntimeSkillBank(DEFAULT_SKILLBANK)
    sample = prepare_sample(dataset[0], skillbank)
    assert "Retrieved grounding skills:" in sample["turn1_prompt"]
    assert "rezoom" not in sample["turn1_prompt"].lower()
    assert "strict_crop_verification" not in sample["turn1_prompt"]


def test_invalid_bbox_gets_parse_penalty_and_no_crop() -> None:
    dataset = GeoBBoxDataset(DEFAULT_TRAIN_JSONL, limit=1)
    skillbank = RuntimeSkillBank(DEFAULT_SKILLBANK)
    sample = prepare_sample(dataset[0], skillbank)
    invalid = "<zoom><|ref|>target<|/ref|><|box|>not a box<|/box|></zoom>"
    assert build_stage2_context(sample, invalid) is None
    reward = compute_geo_episode_reward(sample=sample, turn1_text=invalid, turn2_text="")
    assert reward["P_parse"] >= 1.0, reward
    assert reward["bbox_valid"] == 0.0, reward


def main() -> int:
    test_dataset_filters_bbox_samples()
    test_primitive_bbox_parser()
    test_skill_prompt_and_no_forbidden_skills()
    test_invalid_bbox_gets_parse_penalty_and_no_crop()
    print("geo_zoomearth adapter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

