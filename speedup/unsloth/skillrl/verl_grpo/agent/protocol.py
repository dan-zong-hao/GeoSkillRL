#!/usr/bin/env python3
"""Protocol parsing and masking helpers for bbox-only ZoomEarth GRPO."""
from __future__ import annotations

import re
import os
import string
from dataclasses import dataclass
from typing import Any


ZOOM_RE = re.compile(r"<zoom>\s*(.*?)\s*</zoom>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
JSON_BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE)
REF_RE = re.compile(r"<\|ref\|>\s*(.*?)\s*<\|/ref\|>", re.DOTALL | re.IGNORECASE)
PRIMITIVE_BBOX_RE = re.compile(r"<\|box\|>\s*\[\[(.*?)\]\]\s*<\|/box\|>", re.DOTALL | re.IGNORECASE)
PRIMITIVE_ZOOM_BODY_RE = re.compile(
    r"^\s*<\|ref\|>\s*(?P<ref>.*?)\s*<\|/ref\|>\s*"
    r"<\|box\|>\s*\[\[(?P<bbox>.*?)\]\]\s*<\|/box\|>\s*$",
    re.DOTALL | re.IGNORECASE,
)
COMPACT_BBOX_RE = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)

EXTRA_FIELD_DEFAULTS: dict[str, Any] = {
    "zoom_text": "",
    "answer_text": "",
    "answer_pred": None,
    "stage1_raw_text": "",
    "stage2_raw_text": "",
    "zoom_parse_ok": False,
    "zoom_has_ref": False,
    "zoom_has_box": False,
    "zoom_primitive_format_ok": False,
    "zoom_ref_text": "",
    "zoom_bbox_format": "",
    "answer_parse_ok": False,
    "tool_error": "",
    "pred_bbox_1024": None,
    "crop_created": False,
    "stage1_tokens": 0,
    "stage2_tokens": 0,
    "tool_observation_tokens": 0,
    "retrieved_skill_ids": [],
}


@dataclass(frozen=True)
class ParsedZoom:
    zoom_text: str
    bbox_1024: list[float] | None
    parse_ok: bool
    has_ref: bool = False
    has_box: bool = False
    primitive_format_ok: bool = False
    ref_text: str = ""
    bbox_format: str = ""


@dataclass(frozen=True)
class ParsedAnswer:
    answer_text: str
    answer_pred: str | None
    parse_ok: bool


def stable_extra_fields(**updates: Any) -> dict[str, Any]:
    fields = dict(EXTRA_FIELD_DEFAULTS)
    fields.update(updates)
    return fields


def _parse_float_list(raw: str) -> list[float] | None:
    try:
        nums = [float(x.strip()) for x in raw.split(",")]
    except Exception:
        return None
    if len(nums) < 4:
        return None
    return nums[:4]


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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def extract_zoom(text: str, *, allow_compact: bool | None = None, allow_json: bool | None = None) -> ParsedZoom:
    raw = text or ""
    match = ZOOM_RE.search(raw)
    if not match:
        return ParsedZoom("", None, False)
    if allow_compact is None:
        allow_compact = _env_flag("ALLOW_COMPACT_ZOOM", False)
    if allow_json is None:
        allow_json = bool(allow_compact)
    zoom_text = raw[match.start() : match.end()]
    body = match.group(1)
    ref_match = REF_RE.search(body)
    ref_text = (ref_match.group(1).strip() if ref_match else "")
    has_ref = bool(ref_text)

    primitive_match = PRIMITIVE_ZOOM_BODY_RE.match(body)
    if primitive_match:
        bbox = canonical_bbox_1024(_parse_float_list(primitive_match.group("bbox")))
        primitive_ref = primitive_match.group("ref").strip()
        primitive_ok = bbox is not None and bool(primitive_ref)
        return ParsedZoom(
            zoom_text,
            bbox,
            primitive_ok,
            has_ref=bool(primitive_ref),
            has_box=bbox is not None,
            primitive_format_ok=primitive_ok,
            ref_text=primitive_ref,
            bbox_format="primitive" if bbox is not None else "",
        )

    box_match = PRIMITIVE_BBOX_RE.search(body)
    if box_match:
        bbox = canonical_bbox_1024(_parse_float_list(box_match.group(1)))
        return ParsedZoom(
            zoom_text,
            bbox,
            False,
            has_ref=has_ref,
            has_box=bbox is not None,
            primitive_format_ok=False,
            ref_text=ref_text,
            bbox_format="primitive" if bbox is not None else "",
        )

    json_match = JSON_BBOX_RE.search(body)
    if json_match:
        bbox = canonical_bbox_1024(_parse_float_list(json_match.group(1)))
        return ParsedZoom(
            zoom_text,
            bbox,
            bool(bbox is not None and allow_json),
            has_ref=has_ref,
            has_box=False,
            primitive_format_ok=False,
            ref_text=ref_text,
            bbox_format="json" if bbox is not None else "",
        )

    if "<|box|>" not in body and "<|/box|>" not in body:
        compact_match = COMPACT_BBOX_RE.search(body)
        if compact_match:
            bbox = canonical_bbox_1024(_parse_float_list(compact_match.group(1)))
            return ParsedZoom(
                zoom_text,
                bbox,
                bool(bbox is not None and allow_compact),
                has_ref=has_ref,
                has_box=False,
                primitive_format_ok=False,
                ref_text=ref_text,
                bbox_format="compact" if bbox is not None else "",
            )

    return ParsedZoom(zoom_text, None, False, has_ref=has_ref, ref_text=ref_text)


def extract_answer(text: str) -> ParsedAnswer:
    raw = text or ""
    match = ANSWER_RE.search(raw)
    if not match:
        return ParsedAnswer("", None, False)
    answer_text = raw[match.start() : match.end()]
    pred = match.group(1).strip()
    return ParsedAnswer(answer_text, pred, bool(pred))


def truncate_after(text: str, tag: str) -> str:
    idx = (text or "").lower().find(tag.lower())
    return text[: idx + len(tag)] if idx >= 0 else text


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


def mask_from_char_span(tokenizer: Any, token_ids: list[int], start: int, end: int) -> list[int]:
    """Mark generated tokens whose decoded text overlaps [start, end)."""
    if not token_ids or start < 0 or end <= start:
        return [0] * len(token_ids)
    mask: list[int] = []
    prev_len = 0
    for i in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:i], skip_special_tokens=False)
        cur_len = len(decoded)
        overlaps = cur_len > start and prev_len < end
        mask.append(1 if overlaps else 0)
        prev_len = cur_len
    return mask


def tag_mask_or_all(tokenizer: Any, token_ids: list[int], raw_text: str, tag: str) -> tuple[list[int], bool]:
    """Mask a tag span; if missing or empty, fall back to all generated tokens."""
    if not token_ids:
        return [], False
    regex = ZOOM_RE if tag == "zoom" else ANSWER_RE
    match = regex.search(raw_text or "")
    if not match:
        return [1] * len(token_ids), False
    mask = mask_from_char_span(tokenizer, token_ids, match.start(), match.end())
    if sum(mask) == 0:
        return [1] * len(token_ids), False
    return mask, True
