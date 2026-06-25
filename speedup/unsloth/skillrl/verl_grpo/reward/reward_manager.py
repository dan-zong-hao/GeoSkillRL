"""GeoSkillRL reward manager for official verl."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import decode_token_ids
from speedup.unsloth.skillrl.verl_grpo.reward.components import compute_zoom_reward

try:
    from verl import DataProto
    from verl.workers.reward_manager.abstract import AbstractRewardManager
except Exception:  # pragma: no cover
    DataProto = Any  # type: ignore

    class AbstractRewardManager:  # type: ignore
        pass


class GeoRewardManager(AbstractRewardManager):
    """Reward manager that reads raw AgentLoop fields instead of cleaned decode."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int = 0,
        compute_score: Any | None = None,
        reward_fn_key: str = "data_source",
        config: Any | None = None,
        coord_mode: str = "max_side",
        area_penalty_weight: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine or 0)
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.config = config
        self.coord_mode = coord_mode
        self.area_penalty_weight = float(area_penalty_weight)

    @staticmethod
    def _as_python(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        return value

    @staticmethod
    def _metric_extra_info(score: dict[str, Any]) -> dict[str, float]:
        metric_info: dict[str, float] = {}
        for key, value in score.items():
            if isinstance(value, bool):
                metric_info[key] = float(value)
            elif isinstance(value, int | float):
                metric_info[key] = float(value)
        return metric_info

    def score_item(
        self,
        *,
        response_ids: list[int],
        non_tensor: dict[str, Any],
    ) -> dict[str, Any]:
        extra_info = dict(non_tensor.get("extra_info", {}) or {})
        extra_fields = dict(non_tensor.get("extra_fields", {}) or {})
        for key in (
            "zoom_text",
            "answer_text",
            "pred_bbox_1024",
            "zoom_parse_ok",
            "answer_parse_ok",
            "zoom_parse_error",
        ):
            if key in non_tensor and key not in extra_fields:
                value = non_tensor[key]
                if hasattr(value, "item"):
                    value = value.item()
                extra_fields[key] = value
        reward_model = dict(non_tensor.get("reward_model", {}) or {})

        zoom_text = extra_fields.get("zoom_text")
        if zoom_text is None:
            zoom_text = decode_token_ids(self.tokenizer, response_ids)
        pred_bbox = extra_fields.get("pred_bbox_1024")
        answer_text = extra_fields.get("answer_text", "")
        gt_bbox = extra_info.get("gt_bbox_1024")
        image_size = extra_info.get("image_size")
        coord_mode = extra_info.get("bbox_coord_mode", self.coord_mode)
        ground_truth = reward_model.get("ground_truth")

        score = compute_zoom_reward(
            zoom_text=str(zoom_text or ""),
            pred_bbox_1024=pred_bbox,
            gt_bbox_1024=gt_bbox,
            image_size=image_size,
            coord_mode=coord_mode,
            answer_text=str(answer_text or ""),
            ground_truth=ground_truth,
            area_penalty_weight=self.area_penalty_weight,
        )
        # Preserve AgentLoop parse flags if available; they were computed from
        # raw Stage 1 tokens before any optional validation continuation.
        if "zoom_parse_ok" in extra_fields:
            score["zoom_parse_ok"] = bool(extra_fields["zoom_parse_ok"])
        if "answer_parse_ok" in extra_fields:
            score["answer_parse_ok"] = bool(extra_fields["answer_parse_ok"])
        score["question_id"] = extra_info.get("question_id")
        return score

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = None
        if hasattr(self, "_extract_reward_from_rm_scores"):
            reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)  # type: ignore[attr-defined]
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list[Any]] = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids_tensor = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids_tensor[:valid_response_length].tolist()

            non_tensor = dict(data_item.non_tensor_batch)
            score = self.score_item(response_ids=valid_response_ids, non_tensor=non_tensor)
            reward = float(score["score"])
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward
            for key, value in score.items():
                reward_extra_info[key].append(value)

            if i < self.num_examine:
                print("[question_id]", score.get("question_id"))
                print("[zoom_text]", non_tensor.get("extra_fields", {}).get("zoom_text", ""))
                print("[score]", reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Official verl async reward-loop API for one agent-loop trajectory."""
        data = data[-1:]
        data_item = data[0]

        response_ids_tensor = data_item.batch["responses"]
        response_length = response_ids_tensor.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids_tensor[:valid_response_length].tolist()

        non_tensor = {key: self._as_python(value) for key, value in dict(data_item.non_tensor_batch).items()}
        extra_fields = dict(non_tensor.get("extra_fields", {}) or {})
        tool_extra_fields = self._as_python(non_tensor.get("tool_extra_fields"))
        if isinstance(tool_extra_fields, dict):
            extra_fields.update(tool_extra_fields)
        non_tensor["extra_fields"] = extra_fields

        score = self.score_item(response_ids=valid_response_ids, non_tensor=non_tensor)
        reward = float(score["score"])
        reward_extra_info = self._metric_extra_info(score)
        reward_extra_info.setdefault("acc", reward)
        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
