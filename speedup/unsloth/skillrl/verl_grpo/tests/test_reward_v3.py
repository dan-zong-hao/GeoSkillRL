from __future__ import annotations

from reward.components_v3 import compute_reward_v3


def _reward(
    zoom_text: str,
    *,
    question: str = "What color is the left-most field?",
    gt_bbox_1024: list[float] | None = None,
    answer_text: str = "<answer>yes</answer>",
    ground_truth: str = "yes",
    category: str = "",
):
    return compute_reward_v3(
        question_id="q1",
        question=question,
        zoom_text=zoom_text,
        answer_text=answer_text,
        gt_bbox_1024=gt_bbox_1024 or [10, 10, 110, 110],
        ground_truth=ground_truth,
        image_size=(1024, 1024),
        referent_phrase="left-most field",
        stage1_locator_axes=["left"],
        category=category,
    )


def test_answer_correct_with_bad_iou_is_not_grounded_reward():
    reward = _reward("<zoom><|ref|>field<|/ref|><|box|>[[700,700,800,800]]<|/box|></zoom>")
    assert reward["answer_correct"]
    assert reward["iou"] < 0.3
    assert reward["R_grounded_answer"] == 0.0
    assert reward["R_task_success"] == 0.0
    assert reward["P_false_grounded"] > 0.0


def test_strict_primitive_format_scores_above_compact_bbox():
    strict = _reward("<zoom><|ref|>field<|/ref|><|box|>[[10,10,110,110]]<|/box|></zoom>")
    compact = _reward("<zoom>[[10,10,110,110]]</zoom>")
    assert strict["zoom_primitive_format_ok"]
    assert not compact["zoom_primitive_format_ok"]
    assert not compact["zoom_parse_ok"]
    assert compact["R_format"] == 0.0
    assert compact["P_protocol"] > 0.0
    assert strict["total"] > compact["total"]


def test_wrong_answer_high_iou_task_reward_is_small():
    reward = _reward(
        "<zoom><|ref|>field<|/ref|><|box|>[[10,10,110,110]]<|/box|></zoom>",
        answer_text="<answer>no</answer>",
    )
    assert reward["iou"] >= 0.5
    assert not reward["answer_correct"]
    assert reward["R_task_success"] == 0.3


def test_color_answer_partial_similarity_scores_grounded_answer():
    reward = _reward(
        "<zoom><|ref|>field<|/ref|><|box|>[[10,10,110,110]]<|/box|></zoom>",
        answer_text="<answer>reddish-brown</answer>",
        ground_truth="light brown",
        category="Object color / pattern",
    )
    assert not reward["answer_correct"]
    assert 0.0 < reward["answer_score"] < 1.0
    assert reward["R_grounded_answer"] > 0.0


def test_spatial_violation_reduces_reward():
    good = _reward("<zoom><|ref|>left-most field<|/ref|><|box|>[[10,10,110,110]]<|/box|></zoom>")
    bad = _reward("<zoom><|ref|>left-most field<|/ref|><|box|>[[300,10,400,110]]<|/box|></zoom>")
    assert bad["spatial_penalty"] == 1.0
    assert bad["P_spatial"] > 0.0
    assert good["total"] > bad["total"]
