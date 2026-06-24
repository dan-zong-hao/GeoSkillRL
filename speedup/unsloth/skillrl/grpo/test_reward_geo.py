#!/usr/bin/env python3
"""Lightweight reward sanity checks."""
from __future__ import annotations

from reward_geo import compute_episode_reward


def zoom(box):
    return f"<zoom><|ref|>target<|/ref|><|box|>[[{box[0]},{box[1]},{box[2]},{box[3]}]]<|/box|></zoom>"


def answer(text):
    return f"<answer>{text}</answer>"


def reward(box, ans, gt=(100, 100, 200, 200), truth="yes"):
    return compute_episode_reward(
        question_id="q",
        question="Is the left object visible?",
        zoom_text=zoom(box) if box is not None else "not a zoom",
        answer_text=answer(ans) if ans is not None else "",
        gt_bbox_1024=list(gt),
        ground_truth=truth,
        image_size=(1024, 1024),
    )


def main() -> int:
    perfect = reward((100, 100, 200, 200), "yes")
    good_wrong = reward((100, 100, 200, 200), "no")
    bad_correct = reward((500, 500, 600, 600), "yes")
    parse_fail = reward(None, None)
    full_image = reward((0, 0, 1024, 1024), "yes")

    assert perfect["total"] > good_wrong["total"], (perfect, good_wrong)
    assert good_wrong["total"] > bad_correct["total"], (good_wrong, bad_correct)
    assert parse_fail["total"] < 0, parse_fail
    assert full_image["P_area"] > 0.5, full_image
    assert bad_correct["P_false_grounded"] > 0, bad_correct
    print("reward_geo checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
