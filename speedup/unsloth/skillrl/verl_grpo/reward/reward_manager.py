#!/usr/bin/env python3
"""verl custom reward function entrypoint."""
from __future__ import annotations

import json
import os
from typing import Any

from agent.protocol import extract_answer, extract_zoom
from reward.components_legacy import compute_episode_reward
from reward.components_v2 import compute_reward_v2


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _image_size(extra_info: dict[str, Any]) -> tuple[int, int] | None:
    value = extra_info.get("image_size")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _structured_texts(solution_str: str, extra_info: dict[str, Any]) -> tuple[str, str]:
    # AgentLoop fields may be surfaced as extra_info or nested under reward_extra_info/tool_extra_fields
    for container_key in ("extra_fields", "tool_extra_fields", "reward_extra_info"):
        nested = extra_info.get(container_key)
        if isinstance(nested, dict):
            extra_info = {**extra_info, **nested}
    zoom_text = str(extra_info.get("zoom_text") or "")
    answer_text = str(extra_info.get("answer_text") or "")
    if not zoom_text:
        parsed = extract_zoom(solution_str or "")
        zoom_text = parsed.zoom_text
    if not answer_text:
        parsed = extract_answer(solution_str or "")
        answer_text = parsed.answer_text
    return zoom_text, answer_text


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    extra_info = dict(extra_info or {})
    gt_payload = _maybe_json(ground_truth)
    if isinstance(gt_payload, dict):
        ground_truth_value = gt_payload.get("ground_truth")
        gt_bbox = gt_payload.get("gt_bbox_1024") or gt_payload.get("bbox")
    else:
        ground_truth_value = gt_payload
        gt_bbox = extra_info.get("gt_bbox_1024") or extra_info.get("bbox")
    zoom_text, answer_text = _structured_texts(solution_str, extra_info)
    version = os.environ.get("ZOOMEARTH_REWARD_VERSION", extra_info.get("reward_version", "legacy"))
    kwargs = dict(
        question_id=str(extra_info.get("question_id") or ""),
        question=str(extra_info.get("question") or ""),
        zoom_text=zoom_text,
        answer_text=answer_text,
        gt_bbox_1024=[float(v) for v in (gt_bbox or [0.0, 0.0, 0.0, 0.0])],
        ground_truth=ground_truth_value,
        image_size=_image_size(extra_info),
    )
    result = compute_reward_v2(**kwargs) if version == "v2" else compute_episode_reward(**kwargs)
    result["score"] = float(result["total"])
    result["data_source"] = data_source
    result["reward_version"] = version
    return result

