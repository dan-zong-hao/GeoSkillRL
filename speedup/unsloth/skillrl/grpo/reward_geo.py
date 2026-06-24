#!/usr/bin/env python3
"""GeoSkillRL GRPO reward utilities.

The scalar environment reward follows:

    R = R_task_success + R_bbox_region + R_grounded_answer + R_format
        + R_spatial - P_parse - P_area - P_false_grounded

KL is intentionally not mixed into this reward. The trainer adds KL in the
loss so logs remain interpretable as task/outcome signals.
"""
from __future__ import annotations

import math
import re
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SINGLE_ZOOM_DIR = THIS_DIR.parent / "single_zoom"
OLDTEST_DIR = THIS_DIR.parent / "oldtest"
sys.path.insert(0, str(SINGLE_ZOOM_DIR))
sys.path.insert(0, str(OLDTEST_DIR))

from bbox_rewards import apo_iou_fixed512_from_1024, canonical_bbox_1024  # noqa: E402
from bbox_rewards_single_zoom import extract_first_bbox  # noqa: E402
from spatial import parse_locator, spatial_reward as _spatial_reward  # noqa: E402


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = " ".join(value.split())
    aliases = {"yeah": "yes", "yep": "yes", "true": "yes", "nope": "no", "false": "no"}
    value = aliases.get(value, value)
    try:
        number = float(value)
        if number.is_integer():
            value = str(int(number))
    except Exception:
        pass
    return value


def answer_match(pred: Any, gt: Any) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if not pred_norm or not gt_norm:
        return False
    return pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm


def extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else None


def bbox_area(bbox: list[float] | None) -> float:
    bbox = canonical_bbox_1024(bbox)
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_intersection(a: list[float] | None, b: list[float] | None) -> float:
    a = canonical_bbox_1024(a)
    b = canonical_bbox_1024(b)
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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
    zoom_text = zoom_text or ""
    answer_text = answer_text or ""
    p_parse = 0.0
    r_format = 0.0
    pred = canonical_bbox_1024(pred_bbox)
    if pred is None:
        p_parse += 1.0
    else:
        r_format += 0.2
        if all(0.0 <= float(v) <= 1024.0 for v in pred) and pred[0] < pred[2] and pred[1] < pred[3]:
            r_format += 0.1
    if zoom_text.count("<zoom>") == 1 and zoom_text.count("</zoom>") == 1:
        r_format += 0.1
    if answer_text.count("<answer>") == 1 and answer_text.count("</answer>") == 1:
        r_format += 0.1
    else:
        p_parse += 0.5
    return r_format, p_parse


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
    normalized_zoom_text = (zoom_text or "").replace("</|box|>", "<|/box|>").replace("</|ref|>", "<|/ref|>")
    pred_bbox = canonical_bbox_1024(extract_first_bbox(normalized_zoom_text))
    gt_bbox = canonical_bbox_1024(gt_bbox_1024) or [0.0, 0.0, 0.0, 0.0]
    answer_pred = extract_answer(answer_text)
    ans_ok = answer_match(answer_pred, ground_truth)

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
    r_format, p_parse = format_reward(normalized_zoom_text, answer_text, pred_bbox)
    spatial = _spatial_reward(pred_bbox, gt_bbox, parse_locator(question or ""))
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
