#!/usr/bin/env python3
"""Protocol parsing helpers for GeoSkillRL slime rollouts."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


ZOOM_RE = re.compile(r"<zoom>\s*(.*?)\s*</zoom>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
REF_RE = re.compile(r"<\|ref\|>\s*(.*?)\s*<\|/ref\|>", re.DOTALL | re.IGNORECASE)
PRIMITIVE_BBOX_RE = re.compile(r"<\|box\|>\s*\[\[(.*?)\]\]\s*<\|/box\|>", re.DOTALL | re.IGNORECASE)
JSON_BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE)
COMPACT_BBOX_RE = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)


@dataclass(frozen=True)
class ParsedZoom:
    zoom_text: str
    bbox_1024: list[float] | None
    parse_ok: bool
    ref_text: str = ""
    has_ref: bool = False
    has_box: bool = False
    primitive_format_ok: bool = False


@dataclass(frozen=True)
class ParsedAnswer:
    answer_text: str
    answer_pred: str | None
    parse_ok: bool


def _parse_float_list(raw: str) -> list[float] | None:
    try:
        values = [float(x.strip()) for x in raw.split(",")]
    except Exception:
        return None
    return values[:4] if len(values) >= 4 else None


def canonical_bbox_1024(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    if not all(v == v and abs(v) != float("inf") for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(1024.0, x1))
    y1 = max(0.0, min(1024.0, y1))
    x2 = max(0.0, min(1024.0, x2))
    y2 = max(0.0, min(1024.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def extract_zoom(text: str, *, require_primitive: bool = False) -> ParsedZoom:
    raw = text or ""
    match = ZOOM_RE.search(raw)
    if not match:
        return ParsedZoom("", None, False)
    zoom_text = raw[match.start() : match.end()]
    body = match.group(1)
    ref_match = REF_RE.search(body)
    ref_text = ref_match.group(1).strip() if ref_match else ""
    has_ref = ref_match is not None and bool(ref_text)

    bbox = None
    primitive_match = PRIMITIVE_BBOX_RE.search(body)
    has_box = primitive_match is not None
    if primitive_match:
        bbox = canonical_bbox_1024(_parse_float_list(primitive_match.group(1)))
    primitive_format_ok = bool(has_ref and has_box and bbox is not None)

    if bbox is None and not require_primitive:
        regexes = (JSON_BBOX_RE,)
        if "<|box|>" not in body and "<|/box|>" not in body:
            regexes = regexes + (COMPACT_BBOX_RE,)
        for regex in regexes:
            bbox_match = regex.search(body)
            if bbox_match:
                bbox = canonical_bbox_1024(_parse_float_list(bbox_match.group(1)))
                break

    parse_ok = primitive_format_ok if require_primitive else bbox is not None
    return ParsedZoom(
        zoom_text=zoom_text,
        bbox_1024=bbox if parse_ok else None,
        parse_ok=parse_ok,
        ref_text=ref_text,
        has_ref=has_ref,
        has_box=has_box,
        primitive_format_ok=primitive_format_ok,
    )


def extract_answer(text: str) -> ParsedAnswer:
    raw = text or ""
    match = ANSWER_RE.search(raw)
    if not match:
        return ParsedAnswer("", None, False)
    answer_text = raw[match.start() : match.end()]
    pred = match.group(1).strip()
    return ParsedAnswer(answer_text, pred, bool(pred))


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


def legacy_answer_match(pred: Any, gt: Any) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if not pred_norm or not gt_norm:
        return False
    return pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm


def strict_answer_match(pred: Any, gt: Any) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if not pred_norm or not gt_norm:
        return False
    if gt_norm in {"yes", "no"} or pred_norm in {"yes", "no"}:
        return pred_norm == gt_norm
    try:
        return abs(float(pred_norm) - float(gt_norm)) <= 1e-6
    except Exception:
        return pred_norm == gt_norm

