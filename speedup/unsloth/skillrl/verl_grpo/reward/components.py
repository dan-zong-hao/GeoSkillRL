"""Pure reward components for GeoSkillRL ZoomEarth GRPO."""
from __future__ import annotations

import math
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import parse_zoom_text, validate_bbox_1024
from speedup.unsloth.skillrl.verl_grpo.agent.crop_environment import bbox_1024_to_original, expand_fixed512_orig


def canonical_bbox_1024(bbox: Sequence[Any] | None, *, clamp: bool = True) -> list[float] | None:
    valid = validate_bbox_1024(bbox, require_in_range=False)
    if valid is None:
        return None
    if not clamp:
        return valid if all(0.0 <= v <= 1024.0 for v in valid) else None
    x1, y1, x2, y2 = valid
    x1 = max(0.0, min(1024.0, x1))
    y1 = max(0.0, min(1024.0, y1))
    x2 = max(0.0, min(1024.0, x2))
    y2 = max(0.0, min(1024.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(bbox: Sequence[float] | None) -> float:
    box = canonical_bbox_1024(bbox)
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou_1024(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    box_a = canonical_bbox_1024(a)
    box_b = canonical_bbox_1024(b)
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return float(inter / (union + 1e-9))


def _iou_orig(a: Sequence[float], b: Sequence[float]) -> float:
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


def apo_iou_fixed512(
    pred_bbox_1024: Sequence[float] | None,
    gt_bbox_1024: Sequence[float] | None,
    image_size: tuple[int, int] | list[int] | None,
    *,
    coord_mode: str = "max_side",
    crop_size: int = 512,
) -> float:
    pred = canonical_bbox_1024(pred_bbox_1024)
    gt = canonical_bbox_1024(gt_bbox_1024)
    if pred is None or gt is None or image_size is None:
        return 0.0
    size = (int(image_size[0]), int(image_size[1]))
    pred_orig = bbox_1024_to_original(pred, size, coord_mode=coord_mode)
    gt_orig = bbox_1024_to_original(gt, size, coord_mode=coord_mode)
    pred_crop = expand_fixed512_orig(pred_orig, size, size=crop_size)
    gt_crop = expand_fixed512_orig(gt_orig, size, size=crop_size)
    return _iou_orig(pred_crop, gt_crop)


def center_distance(pred_bbox_1024: Sequence[float] | None, gt_bbox_1024: Sequence[float] | None) -> float:
    pred = canonical_bbox_1024(pred_bbox_1024)
    gt = canonical_bbox_1024(gt_bbox_1024)
    if pred is None or gt is None:
        return 1024.0 * math.sqrt(2.0)
    pcx, pcy = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    gcx, gcy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
    return float(math.hypot(pcx - gcx, pcy - gcy))


def region_guided_reward(
    pred_bbox_1024: Sequence[float] | None,
    gt_bbox_1024: Sequence[float] | None,
    *,
    alpha: float = 200.0,
    eps: float = 0.2,
) -> float:
    dist = center_distance(pred_bbox_1024, gt_bbox_1024)
    if dist >= 1024.0 * math.sqrt(2.0):
        return 0.0
    x = alpha / (dist + eps)
    return float(max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-x)))))


def area_penalty(
    pred_bbox_1024: Sequence[float] | None,
    gt_bbox_1024: Sequence[float] | None,
    *,
    max_penalty: float = 0.6,
    full_image_penalty: float = 0.5,
) -> float:
    pred = canonical_bbox_1024(pred_bbox_1024)
    gt = canonical_bbox_1024(gt_bbox_1024)
    if pred is None or gt is None:
        return 0.0
    pred_area = max(1.0, bbox_area(pred))
    gt_area = max(1.0, bbox_area(gt))
    penalty = min(max_penalty, 0.2 * abs(math.log(pred_area / gt_area)))
    if pred_area > 0.65 * 1024.0 * 1024.0:
        penalty += full_image_penalty
    return float(penalty)


def pattern_reward(text: str, pred_bbox_1024: Sequence[float] | None = None) -> float:
    parsed = parse_zoom_text(text)
    if not parsed.zoom_parse_ok:
        return 0.0
    reward = 0.6
    zoom = parsed.zoom_text
    if zoom.count("<|ref|>") == 1 and zoom.count("<|/ref|>") == 1:
        reward += 0.2
    if pred_bbox_1024 is not None or parsed.pred_bbox_1024 is not None:
        reward += 0.2
    return float(min(1.0, reward))


def parse_reward(text: str) -> tuple[float, float, bool]:
    parsed = parse_zoom_text(text)
    if parsed.zoom_parse_ok:
        return 1.0, 0.0, True
    penalty = 1.0
    if parsed.error in {"multiple_zoom", "zoom_unclosed"}:
        penalty = 0.8
    if parsed.error == "zoom_missing":
        penalty = 1.2
    return 0.0, penalty, False


def _normalize_answer(text: Any) -> str:
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


def _contains_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def answer_match(pred: Any, gt: Any) -> bool:
    pred_norm = _normalize_answer(pred)
    gt_norm = _normalize_answer(gt)
    if not pred_norm or not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True
    return _contains_phrase(gt_norm, pred_norm) or _contains_phrase(pred_norm, gt_norm)


@dataclass
class ZoomReward:
    score: float
    apo_iou: float
    hit03: float
    hit05: float
    region_guided: float
    format: float
    parse_reward: float
    parse_penalty: float
    area_penalty: float
    zoom_parse_ok: bool
    bbox_area_ratio: float
    iou_1024: float
    center_distance: float
    pred_bbox_1024: list[float] | None
    gt_bbox_1024: list[float] | None
    answer_correct: bool = False
    answer_parse_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_zoom_reward(
    *,
    zoom_text: str,
    gt_bbox_1024: Sequence[float] | None,
    image_size: tuple[int, int] | list[int] | None,
    coord_mode: str = "max_side",
    pred_bbox_1024: Sequence[float] | None = None,
    answer_text: str | None = None,
    ground_truth: Any = None,
    area_penalty_weight: float = 1.0,
) -> dict[str, Any]:
    parsed = parse_zoom_text(zoom_text)
    pred = canonical_bbox_1024(pred_bbox_1024 if pred_bbox_1024 is not None else parsed.pred_bbox_1024)
    gt = canonical_bbox_1024(gt_bbox_1024)
    pr, pp, ok = parse_reward(zoom_text)
    fmt = pattern_reward(zoom_text, pred)
    apo = apo_iou_fixed512(pred, gt, image_size, coord_mode=coord_mode)
    hit03 = 1.0 if apo >= 0.3 else 0.0
    hit05 = 1.0 if apo >= 0.5 else 0.0
    rg = region_guided_reward(pred, gt)
    ap = area_penalty(pred, gt)
    score = apo + rg + 0.05 * fmt - pp - area_penalty_weight * ap
    pred_area = bbox_area(pred)
    gt_area = max(1.0, bbox_area(gt))
    answer_parse_ok = bool(answer_text and re.search(r"<answer>\s*.*?\s*</answer>", answer_text, re.DOTALL))
    answer_correct = answer_match(answer_text, ground_truth) if ground_truth is not None else False
    return ZoomReward(
        score=float(score),
        apo_iou=float(apo),
        hit03=hit03,
        hit05=hit05,
        region_guided=float(rg),
        format=float(fmt),
        parse_reward=float(pr),
        parse_penalty=float(pp),
        area_penalty=float(ap),
        zoom_parse_ok=bool(ok and pred is not None),
        bbox_area_ratio=float(pred_area / gt_area) if pred is not None else 0.0,
        iou_1024=float(iou_1024(pred, gt)),
        center_distance=float(center_distance(pred, gt)),
        pred_bbox_1024=pred,
        gt_bbox_1024=gt,
        answer_correct=bool(answer_correct),
        answer_parse_ok=bool(answer_parse_ok),
    ).to_dict()
