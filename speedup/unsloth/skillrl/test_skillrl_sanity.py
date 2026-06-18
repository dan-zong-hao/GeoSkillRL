#!/usr/bin/env python3
"""Small sanity checks for SkillRL offline utilities."""
from __future__ import annotations

from mine_failures import classify_failure
from skillbank import load_skillbank, retrieve_skills
from spatial import parse_locator, spatial_reward


def test_locator_parser() -> None:
    assert parse_locator("the upper-left building")["family"] == "corner"
    assert "top" in parse_locator("top-most warehouse")["axes"]
    assert "bottom" in parse_locator("southern field")["axes"]
    assert "left" in parse_locator("left-most breakwater")["axes"]
    assert "right" in parse_locator("east side tank")["axes"]
    assert "largest" in parse_locator("largest airplane")["backlog_families"]
    assert "smallest" in parse_locator("smallest vehicle")["backlog_families"]
    assert "nearest" in parse_locator("nearest storage tank")["backlog_families"]
    assert "farthest" in parse_locator("farthest ship")["backlog_families"]


def test_skill_retrieval() -> None:
    bank = load_skillbank(None)
    ids = [s["skill_id"] for s in retrieve_skills("What color is the largest building?", bank)]
    assert "rank_largest_instance" in ids
    subpart_ids = [s["skill_id"] for s in retrieve_skills("What is at the right-side stadium entrance?", bank, label="right-side stadium entrance")]
    assert "subpart_side_locator" in subpart_ids
    assert "dir_right_extremum_instance" not in subpart_ids
    rel_ids = [s["skill_id"] for s in retrieve_skills("What is located to the left of the warehouse?", bank)]
    assert "rel_left_of_anchor" in rel_ids


def test_spatial_reward() -> None:
    top_ok = spatial_reward([100, 50, 200, 100], [100, 300, 200, 360], "top-most object")
    assert top_ok["spatial_reward"] == 1.0 and top_ok["spatial_penalty"] == 0.0
    top_bad = spatial_reward([100, 500, 200, 560], [100, 100, 200, 160], "top-most object")
    assert top_bad["spatial_reward"] == 0.0 and top_bad["spatial_penalty"] == 1.0


def test_false_grounding_tag() -> None:
    row = {
        "question_id": "x",
        "type": "object",
        "question": "What is the color of the top-most tank?",
        "bbox_pred_1024": [500, 500, 600, 600],
        "bbox_gt_1024": [100, 100, 200, 200],
        "iou": 0.0,
        "correct_final": True,
        "trajectory": "[Crop Verification] The crop correctly shows the target.\n<answer>white</answer>",
    }
    failure = classify_failure(row, 0.3, load_skillbank(None))
    assert failure is not None
    assert failure["bbox_wrong"] is True
    assert failure["crop_claim_correct"] is True
    assert failure["deer_horse"] is True
    assert failure["false_grounded_correct"] is True


def main() -> int:
    test_locator_parser()
    test_skill_retrieval()
    test_spatial_reward()
    test_false_grounding_tag()
    print("[ok] SkillRL sanity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
