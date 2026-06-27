#!/usr/bin/env python3
"""Lightweight reward sanity checks for the slime adapter."""
from __future__ import annotations

from reward_geo import compute_reward


def primitive_zoom(box):
    return f"<zoom><|ref|>target<|/ref|><|box|>[[{box[0]},{box[1]},{box[2]},{box[3]}]]<|/box|></zoom>"


def compact_zoom(box):
    return f"<zoom>[[{box[0]},{box[1]},{box[2]},{box[3]}]]</zoom>"


def answer(text):
    return f"<answer>{text}</answer>"


def reward(zoom_text, ans, *, version="legacy", gt=(100, 100, 200, 200), truth="yes"):
    return compute_reward(
        version=version,
        question_id="q",
        question="Is the left object visible?",
        zoom_text=zoom_text,
        answer_text=answer(ans) if ans is not None else "",
        gt_bbox_1024=list(gt),
        ground_truth=truth,
        image_size=(1024, 1024),
        referent_phrase="left object",
        stage1_locator_axes=["left"],
    )


def main() -> int:
    perfect = reward(primitive_zoom((100, 100, 200, 200)), "yes")
    good_wrong = reward(primitive_zoom((100, 100, 200, 200)), "no")
    bad_correct = reward(primitive_zoom((500, 500, 600, 600)), "yes")
    parse_fail = reward("not a zoom", None)
    full_image = reward(primitive_zoom((0, 0, 1024, 1024)), "yes")
    assert perfect["total"] > good_wrong["total"], (perfect, good_wrong)
    assert good_wrong["total"] > bad_correct["total"], (good_wrong, bad_correct)
    assert parse_fail["total"] < 0, parse_fail
    assert full_image["P_area"] > 0.5, full_image
    assert bad_correct["P_false_grounded"] > 0, bad_correct

    strict = reward(primitive_zoom((100, 100, 200, 200)), "yes", version="strict_v3")
    compact = reward(compact_zoom((100, 100, 200, 200)), "yes", version="strict_v3")
    assert strict["zoom_primitive_format_ok"]
    assert not compact["zoom_primitive_format_ok"]
    assert strict["total"] > compact["total"], (strict, compact)
    print("reward_geo checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

