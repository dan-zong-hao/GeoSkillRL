#!/usr/bin/env python3
"""Stage-1 grounding-focused reward for ZoomEarth GRPO."""
from __future__ import annotations

import math
from typing import Any

from agent.protocol import canonical_bbox_1024, extract_answer, extract_zoom, legacy_answer_match, normalize_answer
from agent.skill_retriever import extract_referent_phrase, parse_locator
from reward.components_legacy import (
    apo_iou_fixed512_from_1024,
    area_penalty,
    center_distance_1024,
    center_reward,
    coverage,
)


def _coerce_axes(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    axes: list[str] = []
    for axis in value:
        name = str(axis).strip().lower()
        if name in {"top", "bottom", "left", "right"} and name not in axes:
            axes.append(name)
    return axes


COLOR_FAMILIES: dict[str, set[str]] = {
    "black": {"black"},
    "white": {"white"},
    "gray": {"gray", "grey", "silver"},
    "brown": {"brown", "tan", "beige"},
    "red": {"red", "reddish"},
    "orange": {"orange"},
    "yellow": {"yellow", "gold", "golden"},
    "green": {"green"},
    "blue": {"blue"},
    "purple": {"purple", "violet"},
    "pink": {"pink"},
}
COLOR_MODIFIERS = {
    "light",
    "dark",
    "pale",
    "bright",
    "deep",
    "dull",
    "faded",
    "reddish",
    "yellowish",
    "greenish",
    "bluish",
    "grayish",
    "greyish",
}


def _canonical_color_tokens(value: Any) -> set[str]:
    normalized = normalize_answer(value).replace("grey", "gray")
    return {token for token in normalized.split() if token}


def _color_families(tokens: set[str]) -> set[str]:
    families: set[str] = set()
    for family, aliases in COLOR_FAMILIES.items():
        if tokens & {alias.replace("grey", "gray") for alias in aliases}:
            families.add(family)
    return families


def color_similarity(pred: Any, gt: Any) -> float:
    if legacy_answer_match(pred, gt):
        return 1.0
    pred_tokens = _canonical_color_tokens(pred)
    gt_tokens = _canonical_color_tokens(gt)
    if not pred_tokens or not gt_tokens:
        return 0.0
    if pred_tokens == gt_tokens:
        return 1.0
    shared_families = _color_families(pred_tokens) & _color_families(gt_tokens)
    if not shared_families:
        return 0.0
    pred_core = pred_tokens - COLOR_MODIFIERS
    gt_core = gt_tokens - COLOR_MODIFIERS
    if pred_core and pred_core == gt_core:
        return 0.8
    return 0.6


def answer_similarity(pred: Any, gt: Any, *, category: str, question: str) -> float:
    if "color" in (category or "").lower() or "color" in (question or "").lower():
        return color_similarity(pred, gt)
    return 1.0 if legacy_answer_match(pred, gt) else 0.0


def _centers(pred_bbox: list[float] | None, gt_bbox: list[float] | None) -> tuple[float, float, float, float] | None:
    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    if pred is None or gt is None:
        return None
    px, py = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    gx, gy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
    return px, py, gx, gy


def axis_reward(pred_bbox: list[float] | None, gt_bbox: list[float] | None, axes: list[str], tau: float = 64.0) -> float:
    centers = _centers(pred_bbox, gt_bbox)
    if not axes or centers is None:
        return 0.0
    px, py, gx, gy = centers

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    scores: list[float] = []
    if "left" in axes:
        scores.append(sigmoid((gx - px) / tau))
    if "right" in axes:
        scores.append(sigmoid((px - gx) / tau))
    if "top" in axes:
        scores.append(sigmoid((gy - py) / tau))
    if "bottom" in axes:
        scores.append(sigmoid((py - gy) / tau))
    return float(sum(scores) / len(scores)) if scores else 0.0


def spatial_reward_from_axes(
    pred_bbox: list[float] | None,
    gt_bbox: list[float] | None,
    axes: list[str],
    margin: float = 32.0,
) -> dict[str, Any]:
    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    violation = "none"
    if axes and pred is None:
        violation = "missing_bbox"
    elif axes and gt is None:
        violation = "missing_gt"
    elif axes and pred is not None and gt is not None:
        px, py, gx, gy = _centers(pred, gt) or (0.0, 0.0, 0.0, 0.0)
        violations: list[str] = []
        if "top" in axes and py > gy + margin:
            violations.append("top_violated")
        if "bottom" in axes and py < gy - margin:
            violations.append("bottom_violated")
        if "left" in axes and px > gx + margin:
            violations.append("left_violated")
        if "right" in axes and px < gx - margin:
            violations.append("right_violated")
        violation = "+".join(violations) if violations else "none"
    applicable = bool(axes)
    ok = applicable and violation == "none" and pred is not None and gt is not None
    return {
        "spatial_applicable": 1.0 if applicable else 0.0,
        "spatial_ok": 1.0 if ok else 0.0,
        "spatial_penalty": 1.0 if applicable and violation != "none" else 0.0,
        "spatial_violation": violation,
    }


def compute_reward_v3(
    *,
    question_id: str,
    question: str,
    zoom_text: str,
    answer_text: str,
    gt_bbox_1024: list[float],
    ground_truth: Any,
    image_size: tuple[int, int] | None,
    referent_phrase: str | None = None,
    stage1_locator_axes: list[str] | str | None = None,
    category: str = "",
) -> dict[str, Any]:
    parsed_zoom = extract_zoom(zoom_text)
    parsed_answer = extract_answer(answer_text)
    pred_bbox = canonical_bbox_1024(parsed_zoom.bbox_1024)
    gt_bbox = canonical_bbox_1024(gt_bbox_1024) or [0.0, 0.0, 0.0, 0.0]
    answer_pred = parsed_answer.answer_pred
    answer_score = answer_similarity(answer_pred, ground_truth, category=category, question=question)
    ans_ok = answer_score >= 1.0

    referent = str(referent_phrase or extract_referent_phrase(question) or question or "")
    axes = _coerce_axes(stage1_locator_axes)
    if not axes:
        axes = list(parse_locator(referent).get("axes") or [])

    iou = float(apo_iou_fixed512_from_1024(pred_bbox, gt_bbox, image_size))
    hit03 = 1.0 if iou >= 0.3 else 0.0
    hit05 = 1.0 if iou >= 0.5 else 0.0
    cdist = center_distance_1024(pred_bbox, gt_bbox)
    creward = center_reward(cdist)
    cov = coverage(pred_bbox, gt_bbox)
    r_axis = axis_reward(pred_bbox, gt_bbox, axes)
    spatial = spatial_reward_from_axes(pred_bbox, gt_bbox, axes)

    if ans_ok and iou >= 0.5:
        r_task = 2.0
    elif ans_ok and iou >= 0.3:
        r_task = 1.0
    elif answer_pred is not None and (not ans_ok) and iou >= 0.5:
        r_task = 0.3
    elif answer_pred is not None and (not ans_ok) and iou >= 0.3:
        r_task = 0.1
    else:
        r_task = 0.0

    r_bbox_region = iou + 0.50 * creward + 0.25 * hit03 + 0.30 * hit05 + 0.10 * cov + 0.20 * r_axis
    r_grounded_answer = 0.6 * answer_score * min(iou / 0.5, 1.0)
    r_format = 0.05 if parsed_zoom.primitive_format_ok and parsed_answer.parse_ok else 0.0
    r_spatial = 0.3 * float(spatial["spatial_ok"]) if float(spatial["spatial_applicable"]) else 0.0

    p_parse = 0.0
    if pred_bbox is None:
        p_parse += 1.0
    if answer_pred is None:
        p_parse += 0.5
    p_area = area_penalty(pred_bbox, gt_bbox)
    p_false_grounded = 0.4 if ans_ok and iou < 0.3 else 0.0
    p_spatial = 0.4 * float(spatial["spatial_penalty"])
    p_protocol = 0.0
    if pred_bbox is not None and not parsed_zoom.primitive_format_ok:
        if parsed_zoom.bbox_format in {"compact", "json"}:
            p_protocol += 0.3
        if not parsed_zoom.has_ref:
            p_protocol += 0.1
        if not parsed_zoom.has_box:
            p_protocol += 0.1
        if parsed_zoom.has_ref and parsed_zoom.has_box:
            p_protocol += 0.05

    total = (
        r_task
        + r_bbox_region
        + r_grounded_answer
        + r_format
        + r_spatial
        - p_parse
        - p_area
        - p_false_grounded
        - p_spatial
        - p_protocol
    )

    return {
        "total": float(total),
        "score": float(total),
        "question_id": str(question_id),
        "pred_bbox_1024": pred_bbox,
        "gt_bbox_1024": [float(v) for v in gt_bbox],
        "iou": float(iou),
        "hit03": hit03,
        "hit05": hit05,
        "center_distance": float(cdist),
        "center_reward": float(creward),
        "coverage": float(cov),
        "answer_pred": answer_pred,
        "answer_gt": str(ground_truth),
        "answer_correct": bool(ans_ok),
        "answer_score": float(answer_score),
        "bbox_valid": 1.0 if pred_bbox is not None else 0.0,
        "zoom_parse_ok": bool(parsed_zoom.parse_ok),
        "answer_parse_ok": bool(parsed_answer.parse_ok),
        "zoom_has_ref": bool(parsed_zoom.has_ref),
        "zoom_has_box": bool(parsed_zoom.has_box),
        "zoom_primitive_format_ok": bool(parsed_zoom.primitive_format_ok),
        "zoom_ref_text": parsed_zoom.ref_text,
        "zoom_bbox_format": parsed_zoom.bbox_format,
        "referent_phrase": referent,
        "stage1_locator_axes": axes,
        "category": str(category or ""),
        "R_task_success": float(r_task),
        "R_bbox_region": float(r_bbox_region),
        "R_grounded_answer": float(r_grounded_answer),
        "R_format": float(r_format),
        "R_spatial": float(r_spatial),
        "R_axis": float(r_axis),
        "P_parse": float(p_parse),
        "P_area": float(p_area),
        "P_false_grounded": float(p_false_grounded),
        "P_spatial": float(p_spatial),
        "P_protocol": float(p_protocol),
        "spatial_applicable": float(spatial["spatial_applicable"]),
        "spatial_ok": float(spatial["spatial_ok"]),
        "spatial_penalty": float(spatial["spatial_penalty"]),
        "spatial_violation": str(spatial["spatial_violation"]),
        "reward_version": "v3",
    }
