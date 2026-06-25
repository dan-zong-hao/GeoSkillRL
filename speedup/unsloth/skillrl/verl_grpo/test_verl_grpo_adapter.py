#!/usr/bin/env python3
"""Lightweight checks for corrected ZoomEarth GRPO helpers."""
from __future__ import annotations

from geo_data import DEFAULT_TRAIN_JSONL, GeoBBoxDataset
from geo_rewards import answer_match, bbox_1024_to_orig, compute_geo_episode_reward
from geo_rollout import build_stage2_context, parse_zoom_bbox, prepare_sample
import skillbank_runtime as skillbank_module
from skillbank_runtime import DEFAULT_SKILLBANK, RuntimeSkillBank


def test_dataset_filters_bbox_samples() -> None:
    dataset = GeoBBoxDataset(DEFAULT_TRAIN_JSONL)
    assert len(dataset) == 1472, len(dataset)
    assert all(row["type"] in {"object", "region"} for row in dataset.records)
    assert all(isinstance(row["bbox"], list) and len(row["bbox"]) == 4 for row in dataset.records)


def test_answer_match_has_no_raw_substring_false_positive() -> None:
    assert answer_match("yes", "yeah")
    assert answer_match("light grey", "light gray")
    assert not answer_match("north", "no")
    assert not answer_match("no", "north")


def test_bbox_parser_and_xy_mapping() -> None:
    text = "<zoom><|ref|>target<|/ref|><|box|>[[1,2,3,4]]<|/box|></zoom>"
    assert parse_zoom_bbox(text) == [1.0, 2.0, 3.0, 4.0]
    mapped = bbox_1024_to_orig([0, 0, 1024, 1024], (2048, 1024), coord_mode="xy")
    assert mapped == [0.0, 0.0, 2048.0, 1024.0], mapped
    old = bbox_1024_to_orig([0, 0, 512, 512], (2048, 1024), coord_mode="max_side")
    assert old == [0.0, 0.0, 1024.0, 1024.0], old


def test_skill_prompt_uses_question_without_label_leakage() -> None:
    dataset = GeoBBoxDataset(DEFAULT_TRAIN_JSONL, limit=1)
    row = dataset[0]
    row["label"] = "UNAVAILABLE_GT_LABEL_SHOULD_NOT_APPEAR"
    skillbank = RuntimeSkillBank(DEFAULT_SKILLBANK)
    sample = prepare_sample(row, skillbank)
    assert "Retrieved grounding skills:" in sample["turn1_prompt"]
    assert "UNAVAILABLE_GT_LABEL_SHOULD_NOT_APPEAR" not in sample["turn1_prompt"]
    assert "rezoom" not in sample["turn1_prompt"].lower()
    assert "strict_crop_verification" not in sample["turn1_prompt"]


def test_skill_retrieval_receives_empty_label() -> None:
    skillbank = RuntimeSkillBank(DEFAULT_SKILLBANK)
    calls = []
    original = skillbank_module.retrieve_skills

    def fake_retrieve(question, label="", **kwargs):
        calls.append((question, label))
        return []

    try:
        skillbank_module.retrieve_skills = fake_retrieve
        skillbank.build({"question": "What color is the roof?", "label": "GT roof"})
    finally:
        skillbank_module.retrieve_skills = original
    assert calls and calls[-1][1] == "", calls


def test_invalid_bbox_gets_zoom_penalty_and_no_crop() -> None:
    dataset = GeoBBoxDataset(DEFAULT_TRAIN_JSONL, limit=1)
    skillbank = RuntimeSkillBank(DEFAULT_SKILLBANK)
    sample = prepare_sample(dataset[0], skillbank)
    invalid = "<zoom><|ref|>target<|/ref|><|box|>not a box<|/box|></zoom>"
    assert build_stage2_context(sample, invalid) is None
    reward = compute_geo_episode_reward(
        question_id=str(sample["question_id"]),
        question=sample["question"],
        zoom_text=invalid,
        answer_text="",
        gt_bbox_1024=sample["bbox_1024"],
        ground_truth=sample["ground_truth"],
        image_size=sample["image_size"],
    )
    assert reward["P_zoom_parse"] >= 1.0, reward
    assert reward["bbox_valid"] == 0.0, reward


def test_wrong_answer_good_box_does_not_dominate_correct_answer_bad_box() -> None:
    gt = [100.0, 100.0, 220.0, 220.0]
    good_zoom = "<zoom><|ref|>target<|/ref|><|box|>[[100,100,220,220]]<|/box|></zoom>"
    bad_zoom = "<zoom><|ref|>target<|/ref|><|box|>[[700,700,820,820]]<|/box|></zoom>"
    wrong = compute_geo_episode_reward(
        question_id="q",
        question="What color is the building?",
        zoom_text=good_zoom,
        answer_text="<answer>blue</answer>",
        gt_bbox_1024=gt,
        ground_truth="red",
        image_size=(1024, 1024),
    )
    correct = compute_geo_episode_reward(
        question_id="q",
        question="What color is the building?",
        zoom_text=bad_zoom,
        answer_text="<answer>red</answer>",
        gt_bbox_1024=gt,
        ground_truth="red",
        image_size=(1024, 1024),
    )
    assert wrong["answer_reward"] <= 0.0, wrong
    assert correct["answer_reward"] > 0.0, correct
    assert correct["total"] > wrong["total"], (wrong, correct)


def main() -> int:
    test_dataset_filters_bbox_samples()
    test_answer_match_has_no_raw_substring_false_positive()
    test_bbox_parser_and_xy_mapping()
    test_skill_prompt_uses_question_without_label_leakage()
    test_skill_retrieval_receives_empty_label()
    test_invalid_bbox_gets_zoom_penalty_and_no_crop()
    test_wrong_answer_good_box_does_not_dominate_correct_answer_bad_box()
    print("verl_grpo adapter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
