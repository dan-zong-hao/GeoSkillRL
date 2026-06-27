#!/usr/bin/env python3
"""slime custom-generate implementation for two-turn GeoSkillRL rollouts."""
from __future__ import annotations

from typing import Any

import torch

try:
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.processing_utils import encode_image_for_rollout_engine
    from slime.utils.types import Sample
except Exception as exc:  # pragma: no cover - imported only inside a slime runtime.
    GenerateState = None  # type: ignore[assignment]
    post = None  # type: ignore[assignment]
    encode_image_for_rollout_engine = None  # type: ignore[assignment]

    class Sample:  # type: ignore[no-redef]
        class Status:
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            FAILED = "failed"

    _SLIME_IMPORT_ERROR = exc
else:
    _SLIME_IMPORT_ERROR = None

try:
    from .crop_environment import make_legacy_crop
    from .protocol import extract_answer, extract_zoom
    from .reward_geo import build_reward_kwargs, compute_reward
except ImportError:  # pragma: no cover
    from crop_environment import make_legacy_crop
    from protocol import extract_answer, extract_zoom
    from reward_geo import build_reward_kwargs, compute_reward


DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]

OBSERVATION_TEXT = (
    "Zoomed crop image to verify. Use the crop to answer the question. "
    "If the crop does not contain the requested target, answer from the available visual evidence."
)


def _ensure_slime() -> None:
    if _SLIME_IMPORT_ERROR is not None:
        raise RuntimeError(f"slime runtime imports failed: {_SLIME_IMPORT_ERROR}") from _SLIME_IMPORT_ERROR


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    sample.metadata = metadata
    return metadata


def _get_int(args: Any, name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default))
    except Exception:
        return default


def _get_bool(args: Any, name: str, default: bool) -> bool:
    value = getattr(args, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _sampling_params(base: dict[str, Any], *, max_new_tokens: int, stop: str) -> dict[str, Any]:
    params = dict(base)
    params["max_new_tokens"] = max(1, int(max_new_tokens))
    existing = params.get("stop")
    if existing is None:
        stops: list[str] = []
    elif isinstance(existing, str):
        stops = [existing]
    else:
        stops = list(existing)
    params["stop"] = list(dict.fromkeys(stops + [stop]))
    params["no_stop_trim"] = True
    params["spaces_between_special_tokens"] = False
    return params


def _initial_inputs(sample: Any, state: Any) -> tuple[list[int], list[Any], list[dict | None]]:
    processor = state.processor
    tokenizer = state.tokenizer
    multimodal_train_inputs_buffer: list[dict | None] = []
    if processor is not None:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = [int(token) for token in processor_output["input_ids"][0]]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in {"input_ids", "attention_mask"}
        } or None
        if sample.multimodal_train_inputs:
            multimodal_train_inputs_buffer.append(sample.multimodal_train_inputs)
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]]
    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    return prompt_ids, image_data, multimodal_train_inputs_buffer


async def _generate_once(
    *,
    args: Any,
    state: Any,
    tokens: list[int],
    text: str | None,
    image_data: list[Any],
    sampling_params: dict[str, Any],
) -> tuple[str, list[int], list[float], dict[str, Any]]:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        if text is None:
            raise ValueError("multimodal SGLang generation requires text so image placeholders stay aligned")
        payload["text"] = text
        payload["image_data"] = image_data
    else:
        payload["input_ids"] = tokens
    output = await post(url, payload)
    meta_info = output.get("meta_info", {}) or {}
    if "output_token_logprobs" in meta_info:
        new_tokens = [int(item[1]) for item in meta_info["output_token_logprobs"]]
        new_log_probs = [float(item[0]) for item in meta_info["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    return str(output.get("text") or ""), new_tokens, new_log_probs, meta_info


def _encode_observation_for_generation(
    tokenizer: Any,
    processor: Any,
    crop: Any,
    *,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
) -> tuple[list[int], str | None, list[Any], dict | None, dict | None]:
    message = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OBSERVATION_TEXT}]}
    apply_kwargs = apply_chat_template_kwargs or {}
    trim_length = 0
    obs_text = None
    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        if isinstance(formatted_prompt, str) and isinstance(dummy_prompt, str) and formatted_prompt.startswith(dummy_prompt):
            obs_text = formatted_prompt[len(dummy_prompt) :]
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]
        obs_text = f"<image>\n{OBSERVATION_TEXT}"

    multimodal_inputs = {"images": [crop], "videos": None}
    multimodal_train_inputs = None
    if processor is not None:
        processor_output = processor(text=formatted_prompt, images=[crop])
        obs_ids = [int(token) for token in processor_output["input_ids"][0]]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in {"input_ids", "attention_mask"}
        } or None
    else:
        obs_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    if trim_length:
        obs_ids = obs_ids[trim_length:]
    if obs_ids and tokenizer.bos_token_id is not None and obs_ids[0] == tokenizer.bos_token_id:
        obs_ids = obs_ids[1:]
    if obs_text is None:
        obs_text = tokenizer.decode(obs_ids, skip_special_tokens=False)
    image_data = [encode_image_for_rollout_engine(crop)]
    return obs_ids, obs_text, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    values_by_key: dict[str, list[Any]] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, value in chunk.items():
            if value is not None:
                values_by_key.setdefault(key, []).append(value)
    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(value, torch.Tensor) for value in values):
            merged[key] = torch.cat(values, dim=0)
    return merged or None


def _update_multimodal_inputs(sample: Any, obs_multimodal_inputs: dict | None) -> None:
    if not obs_multimodal_inputs:
        return
    if not sample.multimodal_inputs:
        sample.multimodal_inputs = obs_multimodal_inputs
        return
    for key, value in obs_multimodal_inputs.items():
        if value is None:
            continue
        if isinstance(value, list) and isinstance(sample.multimodal_inputs.get(key), list):
            sample.multimodal_inputs[key].extend(value)
        else:
            sample.multimodal_inputs[key] = value


def _compute_and_store_reward(args: Any, sample: Any, *, version: str) -> float:
    reward = compute_reward(version=version, **build_reward_kwargs(sample))
    metadata = _metadata(sample)
    metadata["reward_breakdown"] = reward
    sample.reward = float(reward["total"])
    return float(reward["total"])


def _set_terminal_response(sample: Any, tokenizer: Any, response_tokens: list[int], status: Any) -> Any:
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(
        getattr(sample, "_geoskill_mm_train_buffer", [])
    )
    if hasattr(sample, "_geoskill_mm_train_buffer"):
        delattr(sample, "_geoskill_mm_train_buffer")
    sample.status = status
    return sample


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False) -> Any:
    """slime custom-generate entrypoint."""
    _ensure_slime()
    assert not getattr(args, "partial_rollout", False), "GeoSkillRL two-turn rollout does not support partial_rollout"

    state = GenerateState(args)
    tokenizer = state.tokenizer
    metadata = _metadata(sample)
    reward_version = str(metadata.get("reward_version") or getattr(args, "zoomearth_reward_version", "legacy"))
    require_primitive = _get_bool(args, "zoomearth_require_primitive_zoom", False)
    crop_max_size = _get_int(args, "zoomearth_crop_max_size", 512)
    stage1_max_tokens = _get_int(args, "zoomearth_stage1_max_tokens", 512)
    stage2_max_tokens = _get_int(args, "zoomearth_stage2_max_tokens", 512)

    prompt_ids, current_image_data, mm_train_buffer = _initial_inputs(sample, state)
    sample._geoskill_mm_train_buffer = mm_train_buffer
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []

    stage1_text, stage1_tokens, stage1_log_probs, stage1_meta = await _generate_once(
        args=args,
        state=state,
        tokens=sample.tokens,
        text=sample.prompt if isinstance(sample.prompt, str) else None,
        image_data=current_image_data,
        sampling_params=_sampling_params(sampling_params, max_new_tokens=stage1_max_tokens, stop="</zoom>"),
    )
    sample.append_response_tokens(
        args,
        tokens=stage1_tokens,
        log_probs=stage1_log_probs,
        trainable=True,
        meta_info=stage1_meta,
        text=stage1_text,
    )
    response_tokens.extend(stage1_tokens)

    zoom = extract_zoom(stage1_text, require_primitive=require_primitive)
    metadata.update(
        {
            "stage1_raw_text": stage1_text,
            "zoom_text": zoom.zoom_text,
            "zoom_parse_ok": bool(zoom.parse_ok),
            "zoom_has_ref": bool(zoom.has_ref),
            "zoom_has_box": bool(zoom.has_box),
            "zoom_primitive_format_ok": bool(zoom.primitive_format_ok),
            "zoom_ref_text": zoom.ref_text,
            "pred_bbox_1024": zoom.bbox_1024,
            "stage1_tokens": len(stage1_tokens),
            "stage2_tokens": 0,
            "tool_observation_tokens": 0,
            "crop_created": False,
            "tool_error": "",
        }
    )

    if not zoom.parse_ok:
        metadata["answer_text"] = ""
        metadata["answer_pred"] = None
        metadata["answer_parse_ok"] = False
        metadata["trajectory_text"] = f"[TURN1]\n{stage1_text}\n"
        _compute_and_store_reward(args, sample, version=reward_version)
        return _set_terminal_response(sample, tokenizer, response_tokens, Sample.Status.COMPLETED)

    try:
        crop, crop_meta = make_legacy_crop(
            metadata.get("original_image_path") or metadata.get("global_image_path"),
            zoom.bbox_1024 or [0, 0, 0, 0],
            max_size=crop_max_size,
        )
    except Exception as exc:
        metadata["tool_error"] = f"{type(exc).__name__}: {exc}"
        metadata["answer_text"] = ""
        metadata["answer_pred"] = None
        metadata["answer_parse_ok"] = False
        _compute_and_store_reward(args, sample, version=reward_version)
        return _set_terminal_response(sample, tokenizer, response_tokens, Sample.Status.FAILED)

    obs_ids, obs_text, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = _encode_observation_for_generation(
        tokenizer,
        state.processor,
        crop,
        apply_chat_template=bool(getattr(args, "apply_chat_template", True)),
        apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
    )
    sample.append_response_tokens(args, tokens=obs_ids, trainable=False)
    response_tokens.extend(obs_ids)
    current_image_data = list(current_image_data or []) + list(obs_image_data or [])
    _update_multimodal_inputs(sample, obs_multimodal_inputs)
    if obs_multimodal_train_inputs:
        mm_train_buffer.append(obs_multimodal_train_inputs)
    metadata.update(
        {
            "crop_created": True,
            "crop_meta": crop_meta,
            "tool_observation_tokens": len(obs_ids),
        }
    )

    stage2_prompt_text = None
    if isinstance(sample.prompt, str) and obs_text is not None:
        stage2_prompt_text = sample.prompt + stage1_text + obs_text

    stage2_text, stage2_tokens, stage2_log_probs, stage2_meta = await _generate_once(
        args=args,
        state=state,
        tokens=sample.tokens,
        text=stage2_prompt_text,
        image_data=current_image_data,
        sampling_params=_sampling_params(sampling_params, max_new_tokens=stage2_max_tokens, stop="</answer>"),
    )
    sample.append_response_tokens(
        args,
        tokens=stage2_tokens,
        log_probs=stage2_log_probs,
        trainable=True,
        meta_info=stage2_meta,
        text=stage2_text,
    )
    response_tokens.extend(stage2_tokens)
    answer = extract_answer(stage2_text)
    metadata.update(
        {
            "stage2_raw_text": stage2_text,
            "answer_text": answer.answer_text,
            "answer_pred": answer.answer_pred,
            "answer_parse_ok": bool(answer.parse_ok),
            "stage2_tokens": len(stage2_tokens),
            "trajectory_text": f"[TURN1]\n{stage1_text}\n[TURN2]\n{stage2_text}\n",
        }
    )
    _compute_and_store_reward(args, sample, version=reward_version)
    return _set_terminal_response(sample, tokenizer, response_tokens, Sample.Status.COMPLETED)
