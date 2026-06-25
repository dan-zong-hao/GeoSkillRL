#!/usr/bin/env python3
"""Corrected two-stage GRPO trainer for GeoSkillRL ZoomEarth."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sized

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Sampler
from transformers import Trainer, TrainingArguments

try:
    from trl.models import create_reference_model
except Exception:  # pragma: no cover - import shape differs across TRL releases.
    create_reference_model = None

from geo_rewards import compute_geo_episode_reward
from geo_rollout import (
    build_stage2_context,
    clean_generation_text,
    ensure_closed_zoom,
    parse_zoom_bbox,
    prepare_sample,
)
from skillbank_runtime import RuntimeSkillBank


THIS_DIR = Path(__file__).resolve().parent


@dataclass
class GeoGRPOConfig(TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    dataloader_drop_last: bool = field(default=True)
    learning_rate: float = field(default=1e-7)
    num_generations: int = field(default=4)
    beta: float = field(default=0.04)
    epsilon: float = field(default=0.2)
    epsilon_high: Optional[float] = field(default=None)
    stage1_max_completion_length: int = field(default=512)
    stage2_max_completion_length: int = field(default=256)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.8)
    top_k: Optional[int] = field(default=20)
    max_pixels: int = field(default=64 * 64 * 28 * 28)
    min_pixels: int = field(default=32 * 32)
    rollout_log_path: str = field(default=str(THIS_DIR / "logs/geoskill_verl_grpo_rollouts.jsonl"))
    save_rollouts: bool = field(default=True)
    save_rollout_tokens: bool = field(default=False)
    objective: str = field(default="zoom_only")
    answer_loss_weight: float = field(default=1.0)
    bbox_coord_mode: str = field(default="xy")
    advantage_eps: float = field(default=1e-6)
    malformed_zoom_mask_weight: float = field(default=0.25)
    malformed_answer_mask_weight: float = field(default=1.0)
    block_positive_wrong_answer_advantage: bool = field(default=True)


class RepeatRandomSampler(Sampler[int]):
    """Repeat each sampled index contiguously so GRPO groups stay adjacent."""

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.mini_repeat_count = int(mini_repeat_count)
        self.batch_size = max(1, int(batch_size))
        self.repeat_count = max(1, int(repeat_count))
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(len(self.data_source), generator=self.generator).tolist()
        chunks = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        chunks = [chunk for chunk in chunks if len(chunk) == self.batch_size]
        for chunk in chunks:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        full_chunks = len(self.data_source) // self.batch_size
        return full_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count


class GeoVLMGRPOTrainer(Trainer):
    def __init__(
        self,
        *,
        skillbank_path: str | Path,
        processing_class,
        **kwargs,
    ) -> None:
        args = kwargs["args"]
        model = kwargs["model"]
        if args.objective not in {"zoom_only", "split"}:
            raise ValueError("--objective must be either 'zoom_only' or 'split'")
        if args.bbox_coord_mode not in {"xy", "max_side"}:
            raise ValueError("--bbox_coord_mode must be either 'xy' or 'max_side'")
        self.num_generations = int(args.num_generations)
        self.beta = float(args.beta)
        self.epsilon_low = float(args.epsilon)
        self.epsilon_high = float(args.epsilon_high if args.epsilon_high is not None else args.epsilon)
        self.processing_class = processing_class
        self.skillbank = RuntimeSkillBank(skillbank_path)
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._step = 0
        self.ref_model = None
        if self.beta > 0:
            if create_reference_model is None:
                raise ImportError("trl.models.create_reference_model is required when beta > 0")
            self.ref_model = create_reference_model(model)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)

        super().__init__(processing_class=processing_class, **kwargs)
        self.model_accepts_loss_kwargs = False
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self) -> None:
        if self._signature_columns is None:
            self._signature_columns = ["question", "image_name", "bbox", "ground_truth", "type"]

    def _get_train_sampler(self, train_dataset=None) -> Sampler[int] | None:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        global_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes
        if global_batch_size % self.num_generations != 0:
            raise ValueError(
                f"global train batch size ({self.args.per_device_train_batch_size} x "
                f"{self.accelerator.num_processes}) must be divisible by num_generations={self.num_generations}"
            )
        effective_batch_size = global_batch_size * self.args.gradient_accumulation_steps
        return RepeatRandomSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=max(1, effective_batch_size // self.num_generations),
            repeat_count=1,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset=None) -> Sampler[int] | None:
        if eval_dataset is None:
            return None
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=1,
            repeat_count=1,
            seed=self.args.seed,
        )

    def _prepare_inputs(self, inputs):
        return inputs

    @contextmanager
    def _model_for_generation(self, model):
        unwrapped = self.accelerator.unwrap_model(model)
        was_training = unwrapped.training
        was_grad_ckpt = bool(getattr(unwrapped, "is_gradient_checkpointing", False))
        prev_use_cache = getattr(unwrapped.config, "use_cache", False)
        unwrapped.eval()
        if was_grad_ckpt and hasattr(unwrapped, "gradient_checkpointing_disable"):
            unwrapped.gradient_checkpointing_disable()
        for module in unwrapped.modules():
            if getattr(module, "gradient_checkpointing", False):
                module.gradient_checkpointing = False
        unwrapped.config.use_cache = True
        try:
            yield unwrapped
        finally:
            unwrapped.config.use_cache = prev_use_cache
            if was_grad_ckpt and hasattr(unwrapped, "gradient_checkpointing_enable"):
                unwrapped.gradient_checkpointing_enable()
            unwrapped.train(was_training)

    def _processor_inputs(self, prompt: str, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self.processing_class(
            text=[prompt],
            images=[images],
            return_tensors="pt",
            padding="longest",
        ).to(self.accelerator.device)
        if "labels" in inputs:
            inputs.pop("labels")
        return inputs

    def _generate_part(
        self,
        model,
        prompt: str,
        images: list[Image.Image],
        *,
        focus_tag: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
    ) -> dict[str, Any]:
        prompt_inputs = self._processor_inputs(prompt, images)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        tok = self.processing_class.tokenizer
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "num_beams": 1,
            "temperature": float(self.args.temperature),
            "top_p": float(self.args.top_p),
            "use_cache": True,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
        }
        if self.args.top_k is not None:
            kwargs["top_k"] = int(self.args.top_k)
        if stop_strings:
            kwargs["stop_strings"] = stop_strings
            kwargs["tokenizer"] = tok
        used_stop_strings = bool(stop_strings)
        with torch.inference_mode(), self._model_for_generation(model) as gen_model:
            try:
                generated = gen_model.generate(**prompt_inputs, **kwargs)
            except ValueError as exc:
                if "stop_strings" not in str(exc):
                    raise
                kwargs.pop("stop_strings", None)
                kwargs.pop("tokenizer", None)
                used_stop_strings = False
                generated = gen_model.generate(**prompt_inputs, **kwargs)

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        response_ids = sequences[:, prompt_len:].detach()
        raw_text = tok.decode(response_ids[0], skip_special_tokens=False)
        response_text = clean_generation_text(raw_text, stop_strings=stop_strings)
        retokenized_after_clean = response_text != raw_text
        if retokenized_after_clean:
            response_ids = tok(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
                self.accelerator.device
            )

        with torch.no_grad():
            old_logps = self._score_response_logps(model, prompt_inputs, response_ids).detach()
        loss_mask = self._response_focus_mask(
            response_ids,
            focus_tag,
            fallback_weight=(
                float(self.args.malformed_zoom_mask_weight)
                if focus_tag == "zoom"
                else float(self.args.malformed_answer_mask_weight)
            ),
        )
        return {
            "prompt": prompt,
            "images": images,
            "prompt_inputs": prompt_inputs,
            "response_ids": response_ids.detach(),
            "response_text": response_text,
            "raw_response_text": raw_text,
            "old_logps": old_logps,
            "loss_mask": loss_mask.detach(),
            "focus_tag": focus_tag,
            "prompt_len": prompt_len,
            "used_stop_strings": used_stop_strings,
            "retokenized_after_clean": retokenized_after_clean,
        }

    def _build_full_inputs(
        self,
        prompt_inputs: dict[str, torch.Tensor],
        response_ids: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], int]:
        prompt_ids = prompt_inputs["input_ids"]
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        response_ids = response_ids.to(prompt_ids.device)
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        prompt_attention = prompt_inputs.get("attention_mask")
        if prompt_attention is None:
            prompt_attention = torch.ones_like(prompt_ids)
        response_attention = torch.ones(
            response_ids.shape,
            dtype=prompt_attention.dtype,
            device=prompt_attention.device,
        )
        full_attention = torch.cat([prompt_attention, response_attention], dim=1)
        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": full_ids,
            "attention_mask": full_attention,
        }
        for key, value in prompt_inputs.items():
            if key in {"input_ids", "attention_mask", "labels"}:
                continue
            if torch.is_tensor(value):
                if value.dim() == 2 and value.shape == prompt_ids.shape:
                    pad_value = 0
                    if key == "position_ids":
                        last = value[:, -1:].to(response_ids.device)
                        extension = last + torch.arange(
                            1,
                            response_ids.shape[1] + 1,
                            dtype=value.dtype,
                            device=response_ids.device,
                        ).unsqueeze(0)
                    else:
                        extension = torch.full(
                            response_ids.shape,
                            pad_value,
                            dtype=value.dtype,
                            device=response_ids.device,
                        )
                    model_inputs[key] = torch.cat([value, extension], dim=1)
                    continue
                model_inputs[key] = value
        return model_inputs, int(prompt_ids.shape[1])

    def _score_response_logps(
        self,
        model,
        prompt_inputs: dict[str, torch.Tensor],
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        if response_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.accelerator.device)
        model_inputs, prompt_len = self._build_full_inputs(prompt_inputs, response_ids)
        out = model(**model_inputs, use_cache=False)
        logits = out.logits[:, :-1, :]
        labels = model_inputs["input_ids"][:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        response_len = int(response_ids.shape[-1])
        start = max(0, prompt_len - 1)
        end = start + response_len
        return token_logps[start:end]

    def _special_token_ids(self) -> set[int]:
        tok = self.processing_class.tokenizer
        tokens = [
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        ids: set[int] = set()
        for token in tokens:
            token_id = tok.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tok.unk_token_id:
                ids.add(int(token_id))
        for token_id in [tok.pad_token_id, tok.eos_token_id]:
            if token_id is not None:
                ids.add(int(token_id))
        return ids

    @staticmethod
    def _find_subsequence(values: list[int], pattern: list[int], *, start: int = 0) -> int:
        if not pattern:
            return -1
        limit = len(values) - len(pattern) + 1
        for idx in range(max(0, start), max(0, limit)):
            if values[idx : idx + len(pattern)] == pattern:
                return idx
        return -1

    def _response_focus_mask(
        self,
        response_ids: torch.Tensor,
        focus_tag: str,
        *,
        fallback_weight: float,
    ) -> torch.Tensor:
        if response_ids.dim() == 2:
            ids = response_ids.squeeze(0).tolist()
        else:
            ids = response_ids.tolist()
        mask = torch.zeros(len(ids), dtype=torch.float32, device=self.accelerator.device)
        if not ids:
            return mask
        tok = self.processing_class.tokenizer
        open_ids = tok(f"<{focus_tag}>", add_special_tokens=False)["input_ids"]
        close_ids = tok(f"</{focus_tag}>", add_special_tokens=False)["input_ids"]
        start = self._find_subsequence(ids, open_ids)
        end = self._find_subsequence(ids, close_ids, start=start + len(open_ids) if start >= 0 else 0)
        if start >= 0 and end >= start:
            mask[start : end + len(close_ids)] = 1.0
        elif fallback_weight > 0:
            mask[:] = float(fallback_weight)

        special_ids = self._special_token_ids()
        if special_ids:
            labels = torch.tensor(ids, dtype=torch.long, device=self.accelerator.device)
            special = torch.tensor(sorted(special_ids), dtype=torch.long, device=self.accelerator.device)
            mask = mask * (~torch.isin(labels, special)).float()
        return mask

    def _group_normalized_advantages(
        self,
        local_values: list[float],
        *,
        local_positive_mask: list[float] | None = None,
        block_positive_for_mask_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = torch.tensor(local_values, dtype=torch.float32, device=self.accelerator.device)
        gathered = self.accelerator.gather(values)
        if gathered.numel() % self.num_generations != 0:
            raise ValueError(
                f"gathered reward count {gathered.numel()} is not divisible by num_generations={self.num_generations}"
            )
        grouped = gathered.view(-1, self.num_generations)
        means = grouped.mean(dim=1)
        stds = grouped.std(dim=1, unbiased=False)
        mean_all = means.repeat_interleave(self.num_generations, dim=0)
        std_all = stds.repeat_interleave(self.num_generations, dim=0)
        zero_var_all = (std_all < float(self.args.advantage_eps)).float()
        advantages_all = (gathered - mean_all) / std_all.clamp_min(float(self.args.advantage_eps))
        advantages_all = torch.where(zero_var_all.bool(), torch.zeros_like(advantages_all), advantages_all)

        if local_positive_mask is not None and block_positive_for_mask_zero:
            positive = torch.tensor(local_positive_mask, dtype=torch.float32, device=self.accelerator.device)
            gathered_positive = self.accelerator.gather(positive)
            advantages_all = torch.where(
                gathered_positive > 0.5,
                advantages_all,
                torch.minimum(advantages_all, torch.zeros_like(advantages_all)),
            )

        local_count = len(local_values)
        process_slice = slice(
            self.accelerator.process_index * local_count,
            (self.accelerator.process_index + 1) * local_count,
        )
        return (
            advantages_all[process_slice],
            mean_all[process_slice],
            std_all[process_slice],
            zero_var_all[process_slice],
        )

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]], model) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        rollouts: list[dict[str, Any]] = []
        parts: list[list[dict[str, Any]]] = []
        zoom_rewards: list[float] = []
        answer_rewards: list[float] = []
        answer_correct: list[float] = []

        for row in inputs:
            sample = prepare_sample(row, self.skillbank, bbox_coord_mode=str(self.args.bbox_coord_mode))
            turn1_part = self._generate_part(
                model,
                sample["turn1_prompt"],
                [sample["global_small"]],
                focus_tag="zoom",
                max_new_tokens=self.args.stage1_max_completion_length,
                stop_strings=["</zoom>", "</answer>"],
            )
            turn1_text = ensure_closed_zoom(str(turn1_part["response_text"] or ""))
            stage2 = build_stage2_context(sample, turn1_text)
            turn2_text = ""
            row_parts = [turn1_part]

            if stage2 is not None:
                turn2_part = self._generate_part(
                    model,
                    stage2["prompt"],
                    stage2["images"],
                    focus_tag="answer",
                    max_new_tokens=self.args.stage2_max_completion_length,
                    stop_strings=["</answer>"],
                )
                turn2_text = str(turn2_part["response_text"] or "")
                row_parts.append(turn2_part)

            reward = compute_geo_episode_reward(
                question_id=str(sample["question_id"]),
                question=sample["question"],
                zoom_text=turn1_text,
                answer_text=turn2_text,
                gt_bbox_1024=sample["bbox_1024"],
                ground_truth=sample["ground_truth"],
                image_size=sample["image_size"],
                bbox_coord_mode=str(self.args.bbox_coord_mode),
            )
            zoom_rewards.append(float(reward["zoom_reward"]))
            answer_rewards.append(float(reward["answer_reward"]))
            answer_correct.append(1.0 if reward.get("answer_correct") else 0.0)
            pred_bbox = parse_zoom_bbox(turn1_text)

            rollouts.append(
                {
                    "question_id": sample["question_id"],
                    "image_name": sample["image_name"],
                    "type": sample["type"],
                    "category": sample["category"],
                    "higher_level": sample["higher_level"],
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "skill_block": sample["skill_block"],
                    "retrieved_skill_ids": sample["retrieved_skill_ids"],
                    "turn1_completion": turn1_text,
                    "turn2_completion": turn2_text,
                    "pred_bbox_1024": pred_bbox,
                    "gt_bbox_1024": sample["bbox_1024"],
                    "answer_pred": reward.get("answer_pred"),
                    "reward": reward,
                    "turn1_tokens": int(turn1_part["response_ids"].shape[-1]),
                    "turn2_tokens": int(row_parts[1]["response_ids"].shape[-1]) if len(row_parts) > 1 else 0,
                    "turn1_old_logprob_sum": float(turn1_part["old_logps"].sum().item()),
                    "turn2_old_logprob_sum": (
                        float(row_parts[1]["old_logps"].sum().item()) if len(row_parts) > 1 else 0.0
                    ),
                    "turn1_mask_tokens": float(turn1_part["loss_mask"].sum().item()),
                    "turn2_mask_tokens": float(row_parts[1]["loss_mask"].sum().item()) if len(row_parts) > 1 else 0.0,
                    "trajectory": f"[TURN1]\n{turn1_text}\n[TURN2]\n{turn2_text}\n",
                }
            )
            samples.append(sample)
            parts.append(row_parts)

        zoom_adv, zoom_mean, zoom_std, zoom_zero = self._group_normalized_advantages(zoom_rewards)
        answer_adv, answer_mean, answer_std, answer_zero = self._group_normalized_advantages(
            answer_rewards,
            local_positive_mask=answer_correct,
            block_positive_for_mask_zero=bool(self.args.block_positive_wrong_answer_advantage),
        )

        for idx, rollout in enumerate(rollouts):
            rollout["zoom_advantage"] = float(zoom_adv[idx].detach().item())
            rollout["answer_advantage"] = float(answer_adv[idx].detach().item())
            rollout["zoom_reward_mean"] = float(zoom_mean[idx].detach().item())
            rollout["answer_reward_mean"] = float(answer_mean[idx].detach().item())
            rollout["zoom_reward_std"] = float(zoom_std[idx].detach().item())
            rollout["answer_reward_std"] = float(answer_std[idx].detach().item())
            rollout["zoom_zero_variance_group"] = float(zoom_zero[idx].detach().item())
            rollout["answer_zero_variance_group"] = float(answer_zero[idx].detach().item())
            if self.args.save_rollout_tokens:
                rollout["turn1_response_ids"] = parts[idx][0]["response_ids"].squeeze(0).detach().cpu().tolist()
                if len(parts[idx]) > 1:
                    rollout["turn2_response_ids"] = parts[idx][1]["response_ids"].squeeze(0).detach().cpu().tolist()

        self._append_reward_metrics(rollouts)
        return {
            "parts": parts,
            "zoom_advantages": zoom_adv,
            "answer_advantages": answer_adv,
            "rollouts": rollouts,
            "samples": samples,
        }

    def _append_reward_metrics(self, local_rollouts: list[dict[str, Any]]) -> None:
        for reward_key, metric_name in [
            ("total", "reward_total"),
            ("zoom_reward", "reward_zoom"),
            ("answer_reward", "reward_answer_split"),
            ("iou", "reward_iou"),
            ("hit03", "reward_hit03"),
            ("hit05", "reward_hit05"),
            ("answer_correct", "answer_correct"),
            ("bbox_valid", "bbox_valid_rate"),
            ("P_zoom_parse", "zoom_parse_penalty"),
            ("P_answer_parse", "answer_parse_penalty"),
        ]:
            values = [float(row["reward"].get(reward_key, 0.0)) for row in local_rollouts]
            if values:
                local = torch.tensor(values, dtype=torch.float32, device=self.accelerator.device)
                gathered = self.accelerator.gather_for_metrics(local)
                self._metrics[metric_name].append(float(gathered.mean().item()))

        for key in ["zoom_zero_variance_group", "answer_zero_variance_group", "turn1_mask_tokens", "turn2_mask_tokens"]:
            values = [float(row.get(key, 0.0)) for row in local_rollouts]
            if values:
                local = torch.tensor(values, dtype=torch.float32, device=self.accelerator.device)
                gathered = self.accelerator.gather_for_metrics(local)
                self._metrics[key].append(float(gathered.mean().item()))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GeoVLMGRPOTrainer does not return model outputs")
        generated = self._generate_and_score_completions(inputs, model)
        losses: list[torch.Tensor] = []
        kl_values: list[torch.Tensor] = []
        clip_values: list[torch.Tensor] = []
        turn_loss_values: dict[str, list[torch.Tensor]] = {"zoom": [], "answer": []}

        for sample_idx, row_parts in enumerate(generated["parts"]):
            for part in row_parts:
                focus_tag = str(part["focus_tag"])
                if focus_tag == "zoom":
                    adv = generated["zoom_advantages"][sample_idx].to(self.accelerator.device)
                    loss_weight = 1.0
                    zero_mask = False
                else:
                    adv = generated["answer_advantages"][sample_idx].to(self.accelerator.device)
                    loss_weight = float(self.args.answer_loss_weight)
                    zero_mask = self.args.objective == "zoom_only"

                logps = self._score_response_logps(model, part["prompt_inputs"], part["response_ids"])
                old_logps = part["old_logps"].to(logps.device)
                mask = part["loss_mask"].to(logps.device)
                usable = min(logps.numel(), old_logps.numel(), mask.numel())
                if usable == 0:
                    continue
                logps = logps[:usable]
                old_logps = old_logps[:usable]
                mask = mask[:usable]
                if zero_mask:
                    mask = torch.zeros_like(mask)
                denom = mask.sum().clamp_min(1.0)

                ratio = torch.exp(logps - old_logps.detach())
                clipped = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)
                policy = -torch.minimum(ratio * adv, clipped * adv)
                clip_values.append((((ratio - 1.0).abs() > self.epsilon_high).float() * mask).sum() / denom)

                if self.beta > 0 and self.ref_model is not None and mask.sum().item() > 0:
                    with torch.no_grad():
                        ref_logps = self._score_response_logps(
                            self.ref_model,
                            part["prompt_inputs"],
                            part["response_ids"],
                        )
                    ref_logps = ref_logps[:usable].detach()
                    delta = ref_logps - logps
                    kl = torch.exp(delta) - delta - 1.0
                    policy = policy + self.beta * kl
                    kl_values.append(((kl * mask).sum() / denom).detach())

                part_loss = loss_weight * (policy * mask).sum() / denom
                if mask.sum().item() > 0:
                    losses.append(part_loss)
                    turn_loss_values.setdefault(focus_tag, []).append(part_loss.detach())

        if not losses:
            loss = torch.zeros((), dtype=torch.float32, device=self.accelerator.device, requires_grad=True)
        else:
            loss = torch.stack(losses).mean()

        kl_mean = torch.stack(kl_values).mean() if kl_values else torch.tensor(0.0, device=loss.device)
        clip_mean = torch.stack(clip_values).mean() if clip_values else torch.tensor(0.0, device=loss.device)
        gathered_kl = self.accelerator.gather_for_metrics(kl_mean.detach().view(1))
        gathered_loss = self.accelerator.gather_for_metrics(loss.detach().view(1))
        gathered_clip = self.accelerator.gather_for_metrics(clip_mean.detach().view(1))
        self._metrics["kl"].append(float(gathered_kl.mean().item()))
        self._metrics["loss_geo"].append(float(gathered_loss.mean().item()))
        self._metrics["clip_fraction"].append(float(gathered_clip.mean().item()))
        for tag, values in turn_loss_values.items():
            if values:
                local = torch.stack(values).mean().view(1)
                gathered = self.accelerator.gather_for_metrics(local)
                self._metrics[f"loss_{tag}"].append(float(gathered.mean().item()))

        self._write_rollout_log(generated["rollouts"], loss=loss.detach(), kl=kl_mean.detach())
        self._step += 1
        return loss

    def _write_rollout_log(self, rollouts: list[dict[str, Any]], *, loss: torch.Tensor, kl: torch.Tensor) -> None:
        if not self.args.save_rollouts:
            return
        path = Path(self.args.rollout_log_path)
        if self.accelerator.num_processes > 1:
            path = path.with_name(f"{path.stem}_rank{self.accelerator.process_index}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        loss_value = float(loss.item()) if torch.isfinite(loss).item() else math.nan
        kl_value = float(kl.item()) if torch.isfinite(kl).item() else math.nan
        with path.open("a", encoding="utf-8") as f:
            for row in rollouts:
                clean = dict(row)
                clean.update(
                    {
                        "global_step": int(self.state.global_step),
                        "trainer_step": int(self._step),
                        "rank": int(self.accelerator.process_index),
                        "loss": loss_value,
                        "kl": kl_value,
                        "objective": str(self.args.objective),
                        "bbox_coord_mode": str(self.args.bbox_coord_mode),
                    }
                )
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(values) / len(values) for key, values in self._metrics.items() if values}
        logs = {**logs, **metrics}
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)
        self._metrics.clear()
