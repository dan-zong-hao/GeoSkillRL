"""Official verl AgentLoop for GeoSkillRL ZoomEarth rollouts."""
from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from PIL import Image

from speedup.unsloth.skillrl.verl_grpo.agent.crop_environment import crop_from_original
from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import (
    build_zoom_response_mask,
    count_masked,
    decode_token_ids,
    ensure_aligned_response,
    extract_answer,
)

try:  # Imported lazily in unit tests that do not have official verl on sys.path.
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
    from verl.utils.profiler import simple_timer
    from verl.utils.rollout_trace import rollout_trace_op
    from verl.workers.rollout.replica import TokenOutput
except Exception:  # pragma: no cover
    AgentLoopBase = object  # type: ignore
    AgentLoopOutput = None  # type: ignore
    TokenOutput = Any  # type: ignore

    def simple_timer(_name, _metrics):
        class _Timer:
            def __enter__(self):
                return None

            def __exit__(self, *_exc):
                return False

        return _Timer()

    def rollout_trace_op(fn):
        return fn


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    out = tokenizer(text, add_special_tokens=False)
    return list(out["input_ids"] if isinstance(out, dict) else out)


def build_crop_observation_text() -> str:
    return (
        "\nZoomed crop image to verify:\n"
        f"{VISION_TOKEN}\n"
        "Use the crop to answer the question. If the crop does not contain the requested target, "
        "answer from the available visual evidence without claiming the crop is correct.\n"
    )


class ZoomEarthAgentLoop(AgentLoopBase):
    """Two-stage ZoomEarth loop with token-in-token-out Stage 1 optimization.

    Stage 1 is the only optimized segment by default. Stage 2 can be enabled for
    validation so answer accuracy is measured without assigning answer signal
    onto zoom tokens.
    """

    def __init__(
        self,
        *args,
        generate_stage2: bool = False,
        optimize_answer: bool = False,
        stage1_max_tokens: int = 256,
        stage2_max_tokens: int = 128,
        coord_mode: str = "max_side",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generate_stage2 = bool(generate_stage2)
        self.optimize_answer = bool(optimize_answer)
        self.stage1_max_tokens = int(stage1_max_tokens)
        self.stage2_max_tokens = int(stage2_max_tokens)
        self.coord_mode = coord_mode
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = list(multi_modal_data.get("images") or [])
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images or None,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics: dict[str, Any] = {}
        request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
        stage1_sampling = {**sampling_params, "max_tokens": self.stage1_max_tokens}
        with simple_timer("generate_stage1", metrics):
            stage1: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=stage1_sampling,
                image_data=images or None,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        metrics["num_preempted"] = stage1.num_preempted if getattr(stage1, "num_preempted", None) is not None else -1

        response_ids = list(stage1.token_ids)
        response_mask, parsed = build_zoom_response_mask(response_ids, self.tokenizer)
        response_logprobs = (
            list(stage1.log_probs)
            if getattr(stage1, "log_probs", None) is not None
            else [0.0] * len(response_ids)
        )
        stage1_text = decode_token_ids(self.tokenizer, response_ids)
        answer_text = ""
        stage2_tokens = 0
        crop_created = False
        crop_info: dict[str, Any] | None = None

        if self.generate_stage2:
            crop_image = None
            if parsed.zoom_parse_ok and parsed.pred_bbox_1024 is not None:
                original_image_path = extra_info.get("original_image_path")
                if original_image_path:
                    try:
                        crop_image, crop_result = crop_from_original(
                            original_image_path,
                            parsed.pred_bbox_1024,
                            coord_mode=extra_info.get("bbox_coord_mode", self.coord_mode),
                        )
                        crop_info = crop_result.to_dict()
                        crop_created = True
                    except Exception as exc:  # keep malformed environment from crashing validation rollout
                        logger.warning("crop creation failed for %s: %s", extra_info.get("question_id"), exc)

            if crop_image is not None:
                obs_ids = _encode_no_special(self.tokenizer, build_crop_observation_text())
                stage2_prompt_ids = prompt_ids + response_ids + obs_ids
                stage2_images = images + [crop_image]
                stage2_sampling = {**sampling_params, "max_tokens": self.stage2_max_tokens}
                with simple_timer("generate_stage2", metrics):
                    stage2: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=stage2_prompt_ids,
                        sampling_params=stage2_sampling,
                        image_data=stage2_images,
                        video_data=videos,
                        audio_data=audios,
                        mm_processor_kwargs=mm_processor_kwargs,
                        priority=priority,
                    )
                stage2_ids = list(stage2.token_ids)
                stage2_tokens = len(stage2_ids)
                response_ids = response_ids + obs_ids + stage2_ids
                response_mask = response_mask + [0] * len(obs_ids) + (
                    [1] * len(stage2_ids) if self.optimize_answer else [0] * len(stage2_ids)
                )
                stage2_logprobs = (
                    list(stage2.log_probs)
                    if getattr(stage2, "log_probs", None) is not None
                    else [0.0] * len(stage2_ids)
                )
                response_logprobs = response_logprobs + [0.0] * len(obs_ids) + stage2_logprobs
                answer_text = decode_token_ids(self.tokenizer, stage2_ids)
                multi_modal_data["images"] = stage2_images

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs = response_logprobs[: self.response_length]
        ensure_aligned_response(response_ids, response_mask, response_logprobs)

        extra_fields = dict(getattr(stage1, "extra_fields", {}) or {})
        extra_fields.update(
            {
                "zoom_text": parsed.zoom_text,
                "answer_text": answer_text,
                "pred_bbox_1024": parsed.pred_bbox_1024,
                "zoom_parse_ok": parsed.zoom_parse_ok,
                "answer_parse_ok": extract_answer(answer_text) is not None,
                "zoom_mask_tokens": count_masked(response_mask),
                "stage1_tokens": len(stage1.token_ids),
                "stage2_tokens": stage2_tokens,
                "crop_created": crop_created,
                "crop_info": crop_info,
                "stage1_raw_text": stage1_text,
                "zoom_parse_error": parsed.error,
            }
        )
        output_multi_modal = {}
        if multi_modal_data.get("images") is not None:
            output_multi_modal["images"] = multi_modal_data["images"]
        if videos is not None:
            output_multi_modal["videos"] = videos
        if audios is not None:
            output_multi_modal["audios"] = audios

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=(
                stage1.routed_experts[: len(prompt_ids) + self.response_length]
                if getattr(stage1, "routed_experts", None) is not None
                else None
            ),
            multi_modal_data=output_multi_modal,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=3 if self.generate_stage2 and crop_created else 2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
