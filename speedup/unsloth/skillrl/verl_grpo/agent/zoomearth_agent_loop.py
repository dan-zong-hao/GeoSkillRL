#!/usr/bin/env python3
"""verl AgentLoop for bbox-only two-stage ZoomEarth GRPO."""
from __future__ import annotations

import copy
import logging
import os
from typing import Any
from uuid import uuid4

from agent.crop_environment import make_legacy_crop
from agent.protocol import extract_answer, extract_zoom, stable_extra_fields, tag_mask_or_all

try:  # pragma: no cover - exercised in the verl runtime environment.
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    from verl.utils.profiler import simple_timer
    from verl.utils.rollout_trace import rollout_trace_op
    from verl.utils.tokenizer import get_processor_token_id
except Exception:  # pragma: no cover - lets local unit tests import helpers without verl installed.
    class AgentLoopBase:  # type: ignore[no-redef]
        pass

    class AgentLoopOutput:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def register(_name):  # type: ignore[no-redef]
        def deco(cls):
            return cls

        return deco

    def get_processor_token_id(processor, token_name: str):  # type: ignore[no-redef]
        token_id = getattr(processor, f"{token_name}_token_id", None)
        return int(token_id) if token_id is not None else None

    def rollout_trace_op(func):  # type: ignore[no-redef]
        return func

    class simple_timer:  # type: ignore[no-redef]
        def __init__(self, name, metrics):
            self.name = name
            self.metrics = metrics

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def build_stage2_messages(role: str) -> list[dict[str, Any]]:
    text = (
        "Zoomed crop image to verify. Use the crop to answer the question. "
        "If the crop does not contain the requested target, answer from the available visual evidence."
    )
    if role == "tool":
        return [{"role": "tool", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    if role == "user":
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    if role == "assistant+user":
        return [
            {"role": "assistant", "content": "I will inspect the zoomed crop before answering."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
    raise ValueError(f"unsupported stage2_observation_role={role!r}")


@register("zoomearth_full_agent")
class ZoomEarthAgentLoop(AgentLoopBase):
    """Two-stage bbox-only ZoomEarth loop.

    The first generation emits a zoom action, then a processor-encoded crop
    observation is appended before the second generation emits an answer.
    """

    def __init__(
        self,
        *args,
        stage2_observation_role: str = "user",
        crop_max_size: int = 512,
        stage1_max_tokens: int = 512,
        stage2_max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.stage2_observation_role = stage2_observation_role
        self.crop_max_size = crop_max_size
        self.stage1_max_tokens = stage1_max_tokens
        self.stage2_max_tokens = stage2_max_tokens

    async def _generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: list[Any] | None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        metrics: dict[str, Any],
    ):
        with simple_timer("generate_sequences", metrics):
            return await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _sampling_params_with_max_tokens(
        self,
        sampling_params: dict[str, Any],
        max_tokens: int,
        *,
        stop: list[str] | None = None,
        bad_words: list[str] | None = None,
    ) -> dict[str, Any]:
        params = dict(sampling_params)
        params["max_tokens"] = max(1, int(max_tokens))
        if stop:
            existing_stop = params.get("stop")
            if existing_stop is None:
                stops: list[str] = []
            elif isinstance(existing_stop, str):
                stops = [existing_stop]
            else:
                stops = list(existing_stop)
            params["stop"] = list(dict.fromkeys(stops + stop))
            params["include_stop_str_in_output"] = True
        if bad_words:
            existing_bad_words = params.get("bad_words") or []
            params["bad_words"] = list(dict.fromkeys(list(existing_bad_words) + bad_words))
        return params

    def _generated_bad_words(self, *, forbid_zoom: bool = False, forbid_answer: bool = False) -> list[str]:
        words = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
        if forbid_zoom:
            words.extend(["<zoom>", "</zoom>"])
        if forbid_answer:
            words.extend(["<answer>", "</answer>"])
        return words

    def _generated_mm_token_ids(self) -> set[int]:
        processor = getattr(self, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None) or getattr(self, "tokenizer", None)
        token_ids: set[int] = set()
        for token_name in ("image", "video"):
            token_id = get_processor_token_id(processor, token_name)
            if token_id is not None:
                token_ids.add(int(token_id))
        special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
        for key in ("vision_bos_token", "vision_eos_token", "image_token", "video_token"):
            token = special_tokens_map.get(key)
            if not token:
                continue
            try:
                ids = tokenizer.encode(token, add_special_tokens=False)
            except Exception:
                ids = []
            if len(ids) == 1:
                token_ids.add(int(ids[0]))
        return token_ids

    def _strip_generated_mm_tokens(
        self,
        token_ids: list[int],
        log_probs: list[float] | None = None,
    ) -> tuple[list[int], list[float] | None, int]:
        blocked = self._generated_mm_token_ids()
        if not blocked:
            return token_ids, log_probs, 0
        kept_ids: list[int] = []
        kept_log_probs: list[float] | None = [] if log_probs is not None else None
        removed = 0
        for idx, token_id in enumerate(token_ids):
            if token_id in blocked:
                removed += 1
                continue
            kept_ids.append(token_id)
            if kept_log_probs is not None:
                kept_log_probs.append(log_probs[idx] if idx < len(log_probs) else 0.0)
        return kept_ids, kept_log_probs, removed

    def _truncate_after_marker(
        self,
        token_ids: list[int],
        log_probs: list[float] | None,
        raw_text: str,
        marker: str,
    ) -> tuple[list[int], list[float] | None, str, int]:
        marker_end = (raw_text or "").lower().find(marker.lower())
        if marker_end < 0:
            return token_ids, log_probs, raw_text, 0
        marker_end += len(marker)
        for end_idx in range(1, len(token_ids) + 1):
            decoded = self._decode(token_ids[:end_idx])
            if len(decoded) >= marker_end:
                trimmed_log_probs = log_probs[:end_idx] if log_probs is not None else None
                return token_ids[:end_idx], trimmed_log_probs, decoded, len(token_ids) - end_idx
        return token_ids, log_probs, raw_text, 0

    def _assistant_end_ids(self, generated_ids: list[int]) -> list[int]:
        tokenizer = getattr(self, "tokenizer", None)
        token_ids: list[int] = []
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            try:
                token_ids = [int(eos_token_id)]
            except Exception:
                token_ids = []
        if not token_ids:
            eos_token = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
            try:
                token_ids = [int(v) for v in tokenizer.encode(eos_token, add_special_tokens=False)]
            except Exception:
                token_ids = []
        if token_ids and generated_ids[-len(token_ids) :] == token_ids:
            return []
        return token_ids

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        extra_info = dict(kwargs.get("extra_info", {}) or {})
        retrieved_skill_ids = list(extra_info.get("retrieved_skill_ids") or [])
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        output1 = await self._generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=self._sampling_params_with_max_tokens(
                sampling_params,
                self.stage1_max_tokens,
                stop=["</zoom>"],
                bad_words=self._generated_bad_words(forbid_answer=True),
            ),
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
        )
        stage1_ids, stage1_logprobs, stage1_mm_removed = self._strip_generated_mm_tokens(
            list(output1.token_ids),
            list(output1.log_probs) if output1.log_probs else None,
        )
        stage1_raw_text = self._decode(stage1_ids)
        zoom = extract_zoom(stage1_raw_text)
        stage1_trailing_removed = 0
        if zoom.parse_ok:
            stage1_ids, stage1_logprobs, stage1_raw_text, stage1_trailing_removed = self._truncate_after_marker(
                stage1_ids,
                stage1_logprobs,
                stage1_raw_text,
                "</zoom>",
            )
            zoom = extract_zoom(stage1_raw_text)
        extra_fields = stable_extra_fields(
            zoom_text=zoom.zoom_text,
            stage1_raw_text=stage1_raw_text,
            zoom_parse_ok=zoom.parse_ok,
            pred_bbox_1024=zoom.bbox_1024,
            stage1_tokens=len(stage1_ids),
            retrieved_skill_ids=retrieved_skill_ids,
        )
        if stage1_mm_removed:
            extra_fields["stage1_mm_tokens_stripped"] = stage1_mm_removed
        if stage1_trailing_removed:
            extra_fields["stage1_trailing_tokens_stripped"] = stage1_trailing_removed
        stage1_mask, _ = tag_mask_or_all(self.tokenizer, stage1_ids, stage1_raw_text, "zoom")
        if not zoom.parse_ok:
            stage1_mask = [1] * len(stage1_ids)
            return self._final_output(
                prompt_ids=prompt_ids,
                response_ids=stage1_ids,
                response_mask=stage1_mask,
                response_logprobs=stage1_logprobs,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                metrics=metrics,
                extra_fields=extra_fields,
                routed_experts=getattr(output1, "routed_experts", None),
                num_turns=2,
            )

        try:
            crop, crop_meta = make_legacy_crop(
                extra_info.get("original_image_path") or extra_info.get("global_image_path"),
                zoom.bbox_1024 or [0, 0, 0, 0],
                max_size=self.crop_max_size,
            )
        except Exception as exc:
            extra_fields["tool_error"] = f"{type(exc).__name__}: {exc}"
            raise

        observation_messages = build_stage2_messages(self.stage2_observation_role)
        observation_ids = await self.apply_chat_template(
            observation_messages,
            images=[crop],
            videos=None,
            remove_system_prompt=True,
        )
        assistant_end_ids = self._assistant_end_ids(stage1_ids)
        stage2_token_budget = self.response_length - len(stage1_ids) - len(assistant_end_ids) - len(observation_ids)
        stage2_prompt_ids = prompt_ids + stage1_ids + assistant_end_ids + observation_ids
        stage2_images = list(images or []) + [crop]
        multi_modal_data_out = dict(multi_modal_data)
        multi_modal_data_out["images"] = stage2_images
        extra_fields.update(
            {
                "crop_created": True,
                "assistant_end_tokens": len(assistant_end_ids),
                "tool_observation_tokens": len(observation_ids),
                "stage2_token_budget": stage2_token_budget,
                "crop_meta": crop_meta,
            }
        )
        if stage2_token_budget < 1:
            extra_fields["tool_error"] = "stage1 response left no room for crop answer"
            return self._final_output(
                prompt_ids=prompt_ids,
                response_ids=stage1_ids,
                response_mask=stage1_mask,
                response_logprobs=stage1_logprobs,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                metrics=metrics,
                extra_fields=extra_fields,
                routed_experts=getattr(output1, "routed_experts", None),
                num_turns=2,
            )

        output2 = await self._generate(
            request_id=request_id,
            prompt_ids=stage2_prompt_ids,
            sampling_params=self._sampling_params_with_max_tokens(
                sampling_params,
                min(self.stage2_max_tokens, stage2_token_budget),
                stop=["</answer>"],
                bad_words=self._generated_bad_words(forbid_zoom=True),
            ),
            images=stage2_images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
        )
        stage2_ids, stage2_logprobs, stage2_mm_removed = self._strip_generated_mm_tokens(
            list(output2.token_ids),
            list(output2.log_probs) if output2.log_probs else None,
        )
        stage2_raw_text = self._decode(stage2_ids)
        answer = extract_answer(stage2_raw_text)
        stage2_trailing_removed = 0
        if answer.parse_ok:
            stage2_ids, stage2_logprobs, stage2_raw_text, stage2_trailing_removed = self._truncate_after_marker(
                stage2_ids,
                stage2_logprobs,
                stage2_raw_text,
                "</answer>",
            )
            answer = extract_answer(stage2_raw_text)
        stage2_mask, _ = tag_mask_or_all(self.tokenizer, stage2_ids, stage2_raw_text, "answer")
        if not answer.parse_ok:
            stage2_mask = [1] * len(stage2_ids)
        extra_fields.update(
            {
                "answer_text": answer.answer_text,
                "answer_pred": answer.answer_pred,
                "stage2_raw_text": stage2_raw_text,
                "answer_parse_ok": answer.parse_ok,
                "stage2_tokens": len(stage2_ids),
                "trajectory_text": stage1_raw_text + "\n" + stage2_raw_text,
            }
        )
        if stage2_mm_removed:
            extra_fields["stage2_mm_tokens_stripped"] = stage2_mm_removed
        if stage2_trailing_removed:
            extra_fields["stage2_trailing_tokens_stripped"] = stage2_trailing_removed

        response_ids = stage1_ids + assistant_end_ids + observation_ids + stage2_ids
        response_mask = stage1_mask + [0] * len(assistant_end_ids) + [0] * len(observation_ids) + stage2_mask
        response_logprobs = None
        if stage1_logprobs or stage2_logprobs:
            response_logprobs = (
                list(stage1_logprobs or [0.0] * len(stage1_ids))
                + [0.0] * len(assistant_end_ids)
                + [0.0] * len(observation_ids)
                + list(stage2_logprobs or [0.0] * len(stage2_ids))
            )
        routed_experts = getattr(output2, "routed_experts", None)
        if routed_experts is None:
            routed_experts = getattr(output1, "routed_experts", None)
        return self._final_output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data_out,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            extra_fields=extra_fields,
            routed_experts=routed_experts,
            num_turns=4,
        )

    def _final_output(
        self,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any],
        metrics: dict[str, Any],
        extra_fields: dict[str, Any],
        routed_experts: Any,
        num_turns: int,
    ) -> AgentLoopOutput:
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs is not None:
            response_logprobs = response_logprobs[: self.response_length]
        if routed_experts is not None:
            routed_experts = routed_experts[: len(prompt_ids) + self.response_length]
        metrics.setdefault("num_preempted", -1)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            reward_score=None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=stable_extra_fields(**extra_fields),
        )
