#!/usr/bin/env python3
"""Legacy GeoSkillRL reward ported for verl custom reward managers."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from agent.protocol import canonical_bbox_1024, extract_answer, extract_zoom, legacy_answer_match
from agent.skill_retriever import parse_locator


def bbox_area(bbox: list[float] | None) -> float:
    box = canonical_bbox_1024(bbox)
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_intersection(a: list[float] | None, b: list[float] | None) -> float:
    a = canonical_bbox_1024(a)
    b = canonical_bbox_1024(b)
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def _scale_1024_to_orig(bbox: list[float], image_size: tuple[int, int]) -> list[float]:
    scale = max(float(image_size[0]), float(image_size[1])) / 1024.0
    return [float(v) * scale for v in bbox]


def _shift_inside(x1: float, x2: float, low: float, high: float) -> tuple[float, float]:
    span = x2 - x1
    if high - low <= span:
        return low, high
    if x1 < low:
        x2 += low - x1
        x1 = low
    if x2 > high:
        x1 -= x2 - high
        x2 = high
    return max(low, x1), min(high, x2)


def _expand_fixed512_orig(bbox_orig: list[float], image_size: tuple[int, int], size: int = 512) -> list[float]:
    width, height = image_size
    side_x = min(float(size), float(width))
    side_y = min(float(size), float(height))
    cx = (bbox_orig[0] + bbox_orig[2]) / 2.0
    cy = (bbox_orig[1] + bbox_orig[3]) / 2.0
    x1, x2 = cx - side_x / 2.0, cx + side_x / 2.0
    y1, y2 = cy - side_y / 2.0, cy + side_y / 2.0
    x1, x2 = _shift_inside(x1, x2, 0.0, float(width))
    y1, y2 = _shift_inside(y1, y2, 0.0, float(height))
    return [x1, y1, x2, y2]


def apo_iou_fixed512_from_1024(
    pred_1024: list[float] | None,
    gt_1024: list[float] | None,
    image_size: tuple[int, int] | None,
) -> float:
    pred = canonical_bbox_1024(pred_1024)
    gt = canonical_bbox_1024(gt_1024)
    if pred is None or gt is None or image_size is None:
        return 0.0
    pred_512 = _expand_fixed512_orig(_scale_1024_to_orig(pred, image_size), image_size, 512)
    gt_512 = _expand_fixed512_orig(_scale_1024_to_orig(gt, image_size), image_size, 512)
    return _iou(pred_512, gt_512)


def coverage(pred_bbox: list[float] | None, gt_bbox: list[float] | None) -> float:
    gt_area = max(1.0, bbox_area(gt_bbox))
    return min(1.0, bbox_intersection(pred_bbox, gt_bbox) / gt_area)


def center_distance_1024(pred_bbox: list[float] | None, gt_bbox: list[float] | None) -> float:
    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    if pred is None or gt is None:
        return 1024.0 * math.sqrt(2.0)
    pcx, pcy = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    gcx, gcy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
    return math.hypot(pcx - gcx, pcy - gcy)


def center_reward(distance_1024: float) -> float:
    return math.tanh(200.0 / (float(distance_1024) + 1.0))


def area_penalty(pred_bbox: list[float] | None, gt_bbox: list[float] | None) -> float:
    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    if pred is None or gt is None:
        return 0.0
    pred_area = max(1.0, bbox_area(pred))
    gt_area = max(1.0, bbox_area(gt))
    penalty = min(0.6, 0.2 * abs(math.log(pred_area / gt_area)))
    if pred_area > 0.65 * 1024.0 * 1024.0:
        penalty += 0.5
    return float(penalty)


def format_reward(zoom_text: str, answer_text: str, pred_bbox: list[float] | None) -> tuple[float, float]:
    p_parse = 0.0
    r_format = 0.0
    pred = canonical_bbox_1024(pred_bbox)
    if pred is None:
        p_parse += 1.0
    else:
        r_format += 0.2
        if all(0.0 <= float(v) <= 1024.0 for v in pred) and pred[0] < pred[2] and pred[1] < pred[3]:
            r_format += 0.1
    if (zoom_text or "").count("<zoom>") == 1 and (zoom_text or "").count("</zoom>") == 1:
        r_format += 0.1
    if (answer_text or "").count("<answer>") == 1 and (answer_text or "").count("</answer>") == 1:
        r_format += 0.1
    else:
        p_parse += 0.5
    return r_format, p_parse


def spatial_reward(pred_bbox: list[float] | None, gt_bbox: list[float] | None, question: str, margin: float = 32.0) -> dict[str, Any]:
    locator = parse_locator(question or "")
    axes = list(locator.get("axes") or [])
    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    violation = "none"
    if axes and pred is None:
        violation = "missing_bbox"
    elif axes and gt is None:
        violation = "missing_gt"
    elif axes and pred is not None and gt is not None:
        px, py = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
        gx, gy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
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


@dataclass
class RewardBreakdown:
    total: float
    question_id: str
    pred_bbox_1024: list[float] | None
    gt_bbox_1024: list[float]
    iou: float
    hit03: float
    hit05: float
    center_distance: float
    center_reward: float
    coverage: float
    answer_pred: str | None
    answer_gt: str
    answer_correct: bool
    bbox_valid: float
    zoom_parse_ok: bool
    answer_parse_ok: bool
    R_task_success: float
    R_bbox_region: float
    R_grounded_answer: float
    R_format: float
    R_spatial: float
    P_parse: float
    P_area: float
    P_false_grounded: float
    spatial_applicable: float
    spatial_ok: float
    spatial_penalty: float
    spatial_violation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_episode_reward(
    *,
    question_id: str,
    question: str,
    zoom_text: str,
    answer_text: str,
    gt_bbox_1024: list[float],
    ground_truth: Any,
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    parsed_zoom = extract_zoom(zoom_text)
    pred_bbox = canonical_bbox_1024(parsed_zoom.bbox_1024)
    gt_bbox = canonical_bbox_1024(gt_bbox_1024) or [0.0, 0.0, 0.0, 0.0]
    parsed_answer = extract_answer(answer_text)
    answer_pred = parsed_answer.answer_pred
    ans_ok = legacy_answer_match(answer_pred, ground_truth)

    iou = float(apo_iou_fixed512_from_1024(pred_bbox, gt_bbox, image_size))
    hit03 = 1.0 if iou >= 0.3 else 0.0
    hit05 = 1.0 if iou >= 0.5 else 0.0
    cdist = center_distance_1024(pred_bbox, gt_bbox)
    creward = center_reward(cdist)
    cov = coverage(pred_bbox, gt_bbox)

    if ans_ok and iou >= 0.5:
        r_task = 2.0
    elif ans_ok and iou >= 0.3:
        r_task = 1.0
    elif (not ans_ok) and iou >= 0.5:
        r_task = 0.8
    elif (not ans_ok) and iou >= 0.3:
        r_task = 0.4
    elif ans_ok and iou < 0.3:
        r_task = 0.1
    else:
        r_task = 0.0

    r_bbox_region = iou + 0.25 * creward + 0.25 * hit03 + 0.35 * hit05 + 0.10 * cov
    r_grounded_answer = 0.6 * (1.0 if ans_ok else 0.0) * min(iou / 0.5, 1.0)
    r_format, p_parse = format_reward(zoom_text, answer_text, pred_bbox)
    spatial = spatial_reward(pred_bbox, gt_bbox, question)
    r_spatial = 0.3 * float(spatial["spatial_ok"]) if float(spatial["spatial_applicable"]) else 0.0
    p_area = area_penalty(pred_bbox, gt_bbox)
    p_false_grounded = 0.4 if ans_ok and iou < 0.3 else 0.0
    total = r_task + r_bbox_region + r_grounded_answer + r_format + r_spatial - p_parse - p_area - p_false_grounded

    return RewardBreakdown(
        total=float(total),
        question_id=str(question_id),
        pred_bbox_1024=pred_bbox,
        gt_bbox_1024=[float(v) for v in gt_bbox],
        iou=float(iou),
        hit03=hit03,
        hit05=hit05,
        center_distance=float(cdist),
        center_reward=float(creward),
        coverage=float(cov),
        answer_pred=answer_pred,
        answer_gt=str(ground_truth),
        answer_correct=bool(ans_ok),
        bbox_valid=1.0 if pred_bbox is not None else 0.0,
        zoom_parse_ok=pred_bbox is not None,
        answer_parse_ok=answer_pred is not None,
        R_task_success=float(r_task),
        R_bbox_region=float(r_bbox_region),
        R_grounded_answer=float(r_grounded_answer),
        R_format=float(r_format),
        R_spatial=float(r_spatial),
        P_parse=float(p_parse),
        P_area=float(p_area),
        P_false_grounded=float(p_false_grounded),
        spatial_applicable=float(spatial["spatial_applicable"]),
        spatial_ok=float(spatial["spatial_ok"]),
        spatial_penalty=float(spatial["spatial_penalty"]),
        spatial_violation=str(spatial["spatial_violation"]),
    ).to_dict()
