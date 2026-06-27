#!/usr/bin/env python3
"""GeoSkillRL reward functions for slime.

The default `legacy` reward keeps parity with the current non-verl `grpo`
reward and the default `verl_grpo/reward_manager.py` path.  The optional
`strict_v3` mode ports the stricter experimental reward from `verl_grpo`
without depending on veRL.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .protocol import canonical_bbox_1024, extract_answer, extract_zoom, legacy_answer_match
except ImportError:  # pragma: no cover - supports slime loading by file path/module path.
    from protocol import canonical_bbox_1024, extract_answer, extract_zoom, legacy_answer_match


LOCATOR_PATTERNS = {
    "top": [r"\btop[- ]?most\b", r"\bupper[- ]?most\b", r"\bnorth(?:ern)?[- ]?most\b", r"\btop\b", r"\bupper\b"],
    "bottom": [r"\bbottom[- ]?most\b", r"\blower[- ]?most\b", r"\bsouth(?:ern)?[- ]?most\b", r"\bbottom\b", r"\blower\b"],
    "left": [r"\bleft[- ]?most\b", r"\bwest(?:ern)?[- ]?most\b", r"\bleft\b", r"\bwest(?:ern)?\b"],
    "right": [r"\bright[- ]?most\b", r"\beast(?:ern)?[- ]?most\b", r"\bright\b", r"\beast(?:ern)?\b"],
}


def parse_locator(text: str) -> dict[str, Any]:
    q = (text or "").lower()
    axes = [name for name, patterns in LOCATOR_PATTERNS.items() if any(re.search(p, q) for p in patterns)]
    vertical = [axis for axis in axes if axis in {"top", "bottom"}]
    horizontal = [axis for axis in axes if axis in {"left", "right"}]
    family = "corner" if vertical and horizontal else (axes[0] if axes else "none")
    return {"has_locator": bool(axes), "family": family, "axes": axes}


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
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
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


def spatial_reward(pred_bbox: list[float] | None, gt_bbox: list[float] | None, question: str) -> dict[str, Any]:
    return spatial_reward_from_axes(pred_bbox, gt_bbox, list(parse_locator(question).get("axes") or []))


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


def compute_legacy_reward(
    *,
    question_id: str,
    question: str,
    zoom_text: str,
    answer_text: str,
    gt_bbox_1024: list[float],
    ground_truth: Any,
    image_size: tuple[int, int] | None,
    **_: Any,
) -> dict[str, Any]:
    parsed_zoom = extract_zoom((zoom_text or "").replace("</|box|>", "<|/box|>").replace("</|ref|>", "<|/ref|>"))
    parsed_answer = extract_answer(answer_text)
    pred_bbox = canonical_bbox_1024(parsed_zoom.bbox_1024)
    gt_bbox = canonical_bbox_1024(gt_bbox_1024) or [0.0, 0.0, 0.0, 0.0]
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
    r_format, p_parse = format_reward(parsed_zoom.zoom_text or zoom_text, answer_text, pred_bbox)
    spatial = spatial_reward(pred_bbox, gt_bbox, question)
    r_spatial = 0.3 * float(spatial["spatial_ok"]) if float(spatial["spatial_applicable"]) else 0.0
    p_area = area_penalty(pred_bbox, gt_bbox)
    p_false_grounded = 0.4 if ans_ok and iou < 0.3 else 0.0
    total = r_task + r_bbox_region + r_grounded_answer + r_format + r_spatial - p_parse - p_area - p_false_grounded

    result = RewardBreakdown(
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
    result["score"] = float(total)
    result["reward_version"] = "legacy"
    return result


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


def compute_strict_v3_reward(
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
) -> dict[str, Any]:
    parsed_zoom = extract_zoom(zoom_text, require_primitive=True)
    parsed_answer = extract_answer(answer_text)
    pred_bbox = canonical_bbox_1024(parsed_zoom.bbox_1024)
    gt_bbox = canonical_bbox_1024(gt_bbox_1024) or [0.0, 0.0, 0.0, 0.0]
    answer_pred = parsed_answer.answer_pred
    ans_ok = legacy_answer_match(answer_pred, ground_truth)

    referent = str(referent_phrase or question or "")
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
    elif (not ans_ok) and iou >= 0.5:
        r_task = 0.8
    elif (not ans_ok) and iou >= 0.3:
        r_task = 0.4
    else:
        r_task = 0.0

    r_bbox_region = iou + 0.50 * creward + 0.25 * hit03 + 0.30 * hit05 + 0.10 * cov + 0.20 * r_axis
    r_grounded_answer = 0.6 * (1.0 if ans_ok and iou >= 0.3 else 0.0) * min(iou / 0.5, 1.0)
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
        "bbox_valid": 1.0 if pred_bbox is not None else 0.0,
        "zoom_parse_ok": pred_bbox is not None,
        "answer_parse_ok": answer_pred is not None,
        "zoom_has_ref": bool(parsed_zoom.has_ref),
        "zoom_has_box": bool(parsed_zoom.has_box),
        "zoom_primitive_format_ok": bool(parsed_zoom.primitive_format_ok),
        "referent_phrase": referent,
        "stage1_locator_axes": axes,
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
        "reward_version": "strict_v3",
    }


def compute_reward(
    *,
    version: str = "legacy",
    **kwargs: Any,
) -> dict[str, Any]:
    if version in {"legacy", "grpo_v1"}:
        return compute_legacy_reward(**kwargs)
    if version in {"strict_v3", "v3"}:
        return compute_strict_v3_reward(**kwargs)
    raise ValueError(f"unsupported reward version: {version}")


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def build_reward_kwargs(sample: Any) -> dict[str, Any]:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    label = _maybe_json(getattr(sample, "label", None))
    if isinstance(label, dict):
        ground_truth = label.get("ground_truth")
        gt_bbox = label.get("gt_bbox_1024") or label.get("bbox")
    else:
        ground_truth = metadata.get("ground_truth", label)
        gt_bbox = metadata.get("gt_bbox_1024") or metadata.get("bbox")
    image_size_value = metadata.get("image_size")
    image_size = None
    if isinstance(image_size_value, (list, tuple)) and len(image_size_value) == 2:
        image_size = (int(image_size_value[0]), int(image_size_value[1]))
    return {
        "question_id": str(metadata.get("question_id") or ""),
        "question": str(metadata.get("question") or ""),
        "zoom_text": str(metadata.get("zoom_text") or ""),
        "answer_text": str(metadata.get("answer_text") or ""),
        "gt_bbox_1024": [float(v) for v in (gt_bbox or [0.0, 0.0, 0.0, 0.0])],
        "ground_truth": ground_truth,
        "image_size": image_size,
        "referent_phrase": metadata.get("referent_phrase"),
        "stage1_locator_axes": metadata.get("stage1_locator_axes"),
    }


async def reward_func(args: Any, sample_or_samples: Any, **_: Any) -> float | list[float]:
    """slime --custom-rm-path entrypoint.

    It accepts either a single Sample or a list[Sample], matching slime's
    single and batched custom reward contracts.
    """

    async def _one(sample: Any) -> float:
        if getattr(sample, "reward", None) is not None and not isinstance(sample.reward, dict):
            return float(sample.reward)
        metadata = dict(getattr(sample, "metadata", {}) or {})
        version = str(
            os.environ.get("ZOOMEARTH_REWARD_VERSION")
            or metadata.get("reward_version")
            or getattr(args, "zoomearth_reward_version", "legacy")
        )
        reward = compute_reward(version=version, **build_reward_kwargs(sample))
        metadata["reward_breakdown"] = reward
        sample.metadata = metadata
        sample.reward = float(reward["total"])
        return float(reward["total"])

    if isinstance(sample_or_samples, list):
        return [await _one(sample) for sample in sample_or_samples]
    return await _one(sample_or_samples)
