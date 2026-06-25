#!/usr/bin/env python3
"""Optional simplified reward for post-parity experiments."""
from __future__ import annotations

from typing import Any

from agent.protocol import extract_answer, extract_zoom, strict_answer_match
from reward.components_legacy import apo_iou_fixed512_from_1024, area_penalty, center_reward, compute_episode_reward


def compute_reward_v2(
    *,
    question_id: str,
    question: str,
    zoom_text: str,
    answer_text: str,
    gt_bbox_1024: list[float],
    ground_truth: Any,
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    legacy = compute_episode_reward(
        question_id=question_id,
        question=question,
        zoom_text=zoom_text,
        answer_text=answer_text,
        gt_bbox_1024=gt_bbox_1024,
        ground_truth=ground_truth,
        image_size=image_size,
    )
    parsed_zoom = extract_zoom(zoom_text)
    parsed_answer = extract_answer(answer_text)
    iou = apo_iou_fixed512_from_1024(parsed_zoom.bbox_1024, gt_bbox_1024, image_size)
    answer_ok = strict_answer_match(parsed_answer.answer_pred, ground_truth)
    region_guided = center_reward(float(legacy["center_distance"]))
    fmt = 1.0 if parsed_zoom.parse_ok and parsed_answer.parse_ok else 0.0
    parse_penalty = 1.0 if not parsed_zoom.parse_ok else 0.0
    oversized = area_penalty(parsed_zoom.bbox_1024, gt_bbox_1024)
    score = iou + region_guided + (1.0 if answer_ok else 0.0) + 0.05 * fmt - parse_penalty - oversized
    legacy.update(
        {
            "total": float(score),
            "score": float(score),
            "answer_correct": bool(answer_ok),
            "reward_version": "v2",
        }
    )
    return legacy

