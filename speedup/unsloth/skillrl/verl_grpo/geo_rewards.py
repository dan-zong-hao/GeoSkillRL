#!/usr/bin/env python3
"""Corrected reward split for ZoomEarth-style GeoSkillRL GRPO."""
from __future__ import annotations

import math
import re
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
SKILLRL_DIR = THIS_DIR.parent
SINGLE_ZOOM_DIR = SKILLRL_DIR / "single_zoom"
OLDTEST_DIR = SKILLRL_DIR / "oldtest"
sys.path.insert(0, str(SINGLE_ZOOM_DIR))
sys.path.insert(0, str(OLDTEST_DIR))

from bbox_rewards import canonical_bbox_1024  # noqa: E402
from bbox_rewards_single_zoom import extract_first_bbox  # noqa: E402
from spatial import parse_locator, spatial_reward as _spatial_reward  # noqa: E402

sys.path.insert(0, str(THIS_DIR))


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
YES_ALIASES = {"yes", "yeah", "yep", "true"}
NO_ALIASES = {"no", "nope", "false"}
TOKEN_ALIASES = {
    "grey": "gray",
    "centre": "center",
}


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())


def normalize_answer(text: Any) -> str:
    value = _normalize_text(text)
    if value in YES_ALIASES:
        return "yes"
    if value in NO_ALIASES:
        return "no"
    value = " ".join(TOKEN_ALIASES.get(token, token) for token in value.split())
    number = _parse_number(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return value


def _parse_number(text: str) -> float | None:
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _gt_candidates(gt: Any) -> Iterable[str]:
    if isinstance(gt, (list, tuple, set)):
        for item in gt:
            yield normalize_answer(item)
        return
    text = str(gt or "")
    if "|" in text or ";" in text:
        for part in re.split(r"[|;]", text):
            yield normalize_answer(part)
    else:
        yield normalize_answer(text)


def _boundary_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def answer_match(pred: Any, gt: Any) -> bool:
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return False
    for gt_norm in _gt_candidates(gt):
        if not gt_norm:
            continue
        if gt_norm in {"yes", "no"} or pred_norm in {"yes", "no"}:
            if pred_norm == gt_norm:
                return True
            continue
        pred_num = _parse_number(pred_norm)
        gt_num = _parse_number(gt_norm)
        if pred_num is not None or gt_num is not None:
            if pred_num is not None and gt_num is not None and abs(pred_num - gt_num) <= 1e-6:
                return True
            continue
        if pred_norm == gt_norm:
            return True
        # Allow phrase answers embedded as independent tokens, but never raw
        # substring matches such as "no" in "north".
        if len(gt_norm) >= 3 and _boundary_contains(pred_norm, gt_norm):
            return True
        if len(pred_norm) >= 3 and _boundary_contains(gt_norm, pred_norm):
            return True
    return False


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


def bbox_1024_to_orig(
    bbox: list[float] | None,
    image_size: tuple[int, int] | None,
    *,
    coord_mode: str = "xy",
) -> list[float] | None:
    box = canonical_bbox_1024(bbox)
    if box is None or image_size is None:
        return None
    width, height = float(image_size[0]), float(image_size[1])
    if coord_mode == "max_side":
        scale = max(width, height) / 1024.0
        return [float(v) * scale for v in box]
    if coord_mode != "xy":
        raise ValueError(f"unknown bbox coord_mode={coord_mode!r}; expected 'xy' or 'max_side'")
    sx, sy = width / 1024.0, height / 1024.0
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def _iou_orig(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


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
    cx = (float(bbox_orig[0]) + float(bbox_orig[2])) / 2.0
    cy = (float(bbox_orig[1]) + float(bbox_orig[3])) / 2.0
    x1, x2 = cx - side_x / 2.0, cx + side_x / 2.0
    y1, y2 = cy - side_y / 2.0, cy + side_y / 2.0
    x1, x2 = _shift_inside(x1, x2, 0.0, float(width))
    y1, y2 = _shift_inside(y1, y2, 0.0, float(height))
    return [x1, y1, x2, y2]


def apo_iou_fixed512_from_1024(
    pred_1024: list[float] | None,
    gt_1024: list[float] | None,
    image_size: tuple[int, int] | None,
    *,
    coord_mode: str = "xy",
) -> float:
    pred_orig = bbox_1024_to_orig(pred_1024, image_size, coord_mode=coord_mode)
    gt_orig = bbox_1024_to_orig(gt_1024, image_size, coord_mode=coord_mode)
    if pred_orig is None or gt_orig is None or image_size is None:
        return 0.0
    pred_512 = _expand_fixed512_orig(pred_orig, image_size, 512)
    gt_512 = _expand_fixed512_orig(gt_orig, image_size, 512)
    return _iou_orig(pred_512, gt_512)


def format_reward_split(zoom_text: str, answer_text: str, pred_bbox: list[float] | None) -> tuple[float, float, float, float]:
    zoom_text = zoom_text or ""
    answer_text = answer_text or ""
    zoom_parse_penalty = 0.0
    answer_parse_penalty = 0.0
    zoom_format = 0.0
    answer_format = 0.0
    pred = canonical_bbox_1024(pred_bbox)
    if pred is None:
        zoom_parse_penalty += 1.0
    else:
        zoom_format += 0.2
        if all(0.0 <= float(v) <= 1024.0 for v in pred) and pred[0] < pred[2] and pred[1] < pred[3]:
            zoom_format += 0.1
    if zoom_text.count("<zoom>") == 1 and zoom_text.count("</zoom>") == 1:
        zoom_format += 0.1
    if answer_text.count("<answer>") == 1 and answer_text.count("</answer>") == 1:
        answer_format += 0.2
    else:
        answer_parse_penalty += 0.5
    return zoom_format, answer_format, zoom_parse_penalty, answer_parse_penalty


@dataclass
class RewardBreakdown:
    total: float
    zoom_reward: float
    answer_reward: float
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
    R_zoom_bbox: float
    R_zoom_format: float
    R_answer_correct: float
    R_answer_grounded: float
    R_answer_format: float
    R_spatial: float
    P_zoom_parse: float
    P_answer_parse: float
    P_area: float
    spatial_applicable: float
    spatial_ok: float
    spatial_penalty: float
    spatial_violation: str
    bbox_coord_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_geo_episode_reward(
    *,
    question_id: str,
    question: str,
    zoom_text: str,
    answer_text: str,
    gt_bbox_1024: list[float],
    ground_truth: Any,
    image_size: tuple[int, int] | None,
    bbox_coord_mode: str = "xy",
) -> dict[str, Any]:
    normalized_zoom_text = (zoom_text or "").replace("</|box|>", "<|/box|>").replace("</|ref|>", "<|/ref|>")
    pred_bbox = canonical_bbox_1024(extract_first_bbox(normalized_zoom_text))
    gt_bbox = canonical_bbox_1024(gt_bbox_1024) or [0.0, 0.0, 0.0, 0.0]
    answer_pred = extract_answer(answer_text)
    ans_ok = answer_match(answer_pred, ground_truth)

    iou = float(apo_iou_fixed512_from_1024(pred_bbox, gt_bbox, image_size, coord_mode=bbox_coord_mode))
    hit03 = 1.0 if iou >= 0.3 else 0.0
    hit05 = 1.0 if iou >= 0.5 else 0.0
    cdist = center_distance_1024(pred_bbox, gt_bbox)
    creward = center_reward(cdist)
    cov = coverage(pred_bbox, gt_bbox)
    zoom_format, answer_format, p_zoom_parse, p_answer_parse = format_reward_split(
        normalized_zoom_text,
        answer_text,
        pred_bbox,
    )
    spatial = _spatial_reward(pred_bbox, gt_bbox, parse_locator(question or ""))
    r_spatial = 0.3 * float(spatial["spatial_ok"]) if float(spatial["spatial_applicable"]) else 0.0
    p_area = area_penalty(pred_bbox, gt_bbox)

    r_zoom_bbox = iou + 0.25 * creward + 0.25 * hit03 + 0.35 * hit05 + 0.10 * cov
    zoom_reward = r_zoom_bbox + zoom_format + r_spatial - p_zoom_parse - p_area

    r_answer_correct = 1.0 if ans_ok else 0.0
    r_answer_grounded = 0.25 * r_answer_correct * min(iou / 0.5, 1.0)
    answer_reward = r_answer_correct + r_answer_grounded + answer_format - p_answer_parse
    if not ans_ok:
        answer_reward = min(answer_reward, 0.0)

    total = zoom_reward + 2.0 * answer_reward
    return RewardBreakdown(
        total=float(total),
        zoom_reward=float(zoom_reward),
        answer_reward=float(answer_reward),
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
        R_zoom_bbox=float(r_zoom_bbox),
        R_zoom_format=float(zoom_format),
        R_answer_correct=float(r_answer_correct),
        R_answer_grounded=float(r_answer_grounded),
        R_answer_format=float(answer_format),
        R_spatial=float(r_spatial),
        P_zoom_parse=float(p_zoom_parse),
        P_answer_parse=float(p_answer_parse),
        P_area=float(p_area),
        spatial_applicable=float(spatial["spatial_applicable"]),
        spatial_ok=float(spatial["spatial_ok"]),
        spatial_penalty=float(spatial["spatial_penalty"]),
        spatial_violation=str(spatial["spatial_violation"]),
        bbox_coord_mode=str(bbox_coord_mode),
    ).to_dict()
