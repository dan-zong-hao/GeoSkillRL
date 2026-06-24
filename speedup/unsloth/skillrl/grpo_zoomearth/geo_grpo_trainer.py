#!/usr/bin/env python3
"""ZoomEarth-style GRPO trainer specialized for GeoSkillRL."""
from __future__ import annotations

import json
import math
import sys
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


THIS_DIR = Path(__file__).resolve().parent
SKILLRL_DIR = THIS_DIR.parent
GRPO_DIR = SKILLRL_DIR / "grpo"
sys.path.insert(0, str(GRPO_DIR))

from geo_rewards import compute_geo_episode_reward  # noqa: E402
from geo_rollout import (  # noqa: E402
    build_stage2_context,
    clean_generation_text,
    ensure_closed_zoom,
    parse_zoom_bbox,
    prepare_sample,
)
from skillbank_runtime import RuntimeSkillBank  # noqa: E402


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
    rollout_log_path: str = field(default=str(THIS_DIR / "logs/geoskill_grpo_zoomearth_v1_rollouts.jsonl"))
    save_rollouts: bool = field(default=True)


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
        self.seed = seed
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

    def _generate_text(
        self,
        model,
        prompt: str,
        images: list[Image.Image],
        *,
        max_new_tokens: int,
        stop_strings: list[str] | None,
    ) -> str:
        inputs = self.processing_class(
            text=[prompt],
            images=[images],
            return_tensors="pt",
            padding="longest",
        ).to(self.accelerator.device)
        if "labels" in inputs:
            inputs.pop("labels")
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
        with torch.inference_mode(), self._model_for_generation(model) as gen_model:
            try:
                generated = gen_model.generate(**inputs, **kwargs)
            except ValueError as exc:
                if "stop_strings" not in str(exc):
                    raise
                kwargs.pop("stop_strings", None)
                kwargs.pop("tokenizer", None)
                generated = gen_model.generate(**inputs, **kwargs)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        text = tok.decode(generated[0], skip_special_tokens=False)
        return clean_generation_text(text, stop_strings=stop_strings)

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]], model) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        rollouts: list[dict[str, Any]] = []
        parts: list[list[dict[str, Any]]] = []
        rewards: list[float] = []
        for row in inputs:
            sample = prepare_sample(row, self.skillbank)
            turn1_text = self._generate_text(
                model,
                sample["turn1_prompt"],
                [sample["global_small"]],
                max_new_tokens=self.args.stage1_max_completion_length,
                stop_strings=["</zoom>", "</answer>"],
            )
            turn1_text = ensure_closed_zoom(turn1_text)
            stage2 = build_stage2_context(sample, turn1_text)
            turn2_text = ""
            row_parts = [
                {
                    "prompt": sample["turn1_prompt"],
                    "completion": turn1_text + "<|im_end|>",
                    "images": [sample["global_small"]],
                    "focus_tag": "zoom",
                    "zero_mask": False,
                }
            ]
            if stage2 is not None:
                turn2_text = self._generate_text(
                    model,
                    stage2["prompt"],
                    stage2["images"],
                    max_new_tokens=self.args.stage2_max_completion_length,
                    stop_strings=["</answer>"],
                )
                if turn2_text:
                    row_parts.append(
                        {
                            "prompt": stage2["prompt"],
                            "completion": turn2_text + "<|im_end|>",
                            "images": stage2["images"],
                            "focus_tag": "answer",
                            "zero_mask": False,
                        }
                    )
            reward = compute_geo_episode_reward(sample=sample, turn1_text=turn1_text, turn2_text=turn2_text)
            rewards.append(float(reward["total"]))
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
                }
            )
            samples.append(sample)
            parts.append(row_parts)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.accelerator.device)
        gathered_rewards = self.accelerator.gather(reward_tensor)
        if gathered_rewards.numel() % self.num_generations != 0:
            raise ValueError(
                f"gathered reward count {gathered_rewards.numel()} is not divisible by "
                f"num_generations={self.num_generations}"
            )
        grouped = gathered_rewards.view(-1, self.num_generations)
        mean_grouped = grouped.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped = grouped.std(dim=1, unbiased=False).repeat_interleave(self.num_generations, dim=0)
        advantages_all = gathered_rewards - mean_grouped
        process_slice = slice(
            self.accelerator.process_index * len(rewards),
            (self.accelerator.process_index + 1) * len(rewards),
        )
        advantages = advantages_all[process_slice]
        local_group_means = mean_grouped[process_slice]
        local_group_stds = std_grouped[process_slice]
        for rollout, advantage, group_mean, group_std in zip(
            rollouts,
            advantages.detach().tolist(),
            local_group_means.detach().tolist(),
            local_group_stds.detach().tolist(),
        ):
            turn1_text = str(rollout.get("turn1_completion") or "")
            turn2_text = str(rollout.get("turn2_completion") or "")
            rollout["advantage"] = float(advantage)
            rollout["group_reward_mean"] = float(group_mean)
            rollout["group_reward_std"] = float(group_std)
            rollout["turn1_chars"] = len(turn1_text)
            rollout["turn2_chars"] = len(turn2_text)
            rollout["turn1_tokens"] = len(self.processing_class.tokenizer(turn1_text, add_special_tokens=False)["input_ids"])
            rollout["turn2_tokens"] = len(self.processing_class.tokenizer(turn2_text, add_special_tokens=False)["input_ids"])
            rollout["trajectory"] = f"[TURN1]\n{turn1_text}\n[TURN2]\n{turn2_text}\n"

        self._append_reward_metrics(gathered_rewards, std_grouped, rollouts)
        return {
            "parts": parts,
            "advantages": advantages,
            "rollouts": rollouts,
            "samples": samples,
            "reward_std": std_grouped[process_slice],
        }

    def _append_reward_metrics(
        self,
        gathered_rewards: torch.Tensor,
        std_grouped: torch.Tensor,
        local_rollouts: list[dict[str, Any]],
    ) -> None:
        self._metrics["reward"].append(float(gathered_rewards.mean().item()))
        self._metrics["reward_std"].append(float(std_grouped.mean().item()))
        for key, metric_name in [
            ("iou", "reward_iou"),
            ("hit03", "reward_hit03"),
            ("hit05", "reward_hit05"),
            ("answer_correct", "reward_answer"),
            ("bbox_valid", "bbox_valid_rate"),
            ("R_format", "reward_format"),
        ]:
            values = [float(row["reward"].get(key, 0.0)) for row in local_rollouts]
            if values:
                local = torch.tensor(values, dtype=torch.float32, device=self.accelerator.device)
                gathered = self.accelerator.gather_for_metrics(local)
                self._metrics[metric_name].append(float(gathered.mean().item()))

    def _special_token_ids(self) -> list[int]:
        tok = self.processing_class.tokenizer
        ids: list[int] = []
        for token in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_end|>"]:
            token_id = tok.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tok.unk_token_id:
                ids.append(int(token_id))
        return ids

    def _focus_mask(self, prompt: str, completion: str, token_count: int, focus_tag: str) -> torch.Tensor:
        tok = self.processing_class.tokenizer
        prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        mask = torch.zeros(token_count, dtype=torch.float32, device=self.accelerator.device)
        start = max(0, prompt_len - 1)
        if start < token_count:
            mask[start:] = 1.0

        tag_start = completion.find(f"<{focus_tag}>")
        tag_end = completion.find(f"</{focus_tag}>")
        if tag_start >= 0 and tag_end >= tag_start:
            tag_end += len(f"</{focus_tag}>")
            start_len = len(tok(prompt + completion[:tag_start], add_special_tokens=False)["input_ids"])
            end_len = len(tok(prompt + completion[:tag_end], add_special_tokens=False)["input_ids"])
            focus = torch.zeros_like(mask)
            focus_start = max(0, start_len - 1)
            focus_end = min(token_count, max(focus_start + 1, end_len - 1))
            focus[focus_start:focus_end] = 1.0
            if focus.sum().item() > 0:
                mask = mask * focus
        elif focus_tag == "zoom":
            # Stage 1 should optimize the primitive zoom. A malformed generation
            # still gets a small policy signal over generated tokens.
            mask = mask * 0.25
        return mask

    def _compute_logps_and_mask(
        self,
        model,
        prompt: str,
        completion: str,
        images: list[Image.Image],
        *,
        focus_tag: str,
        zero_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not completion:
            completion = f"<{focus_tag}></{focus_tag}><|im_end|>"
        full_text = prompt + completion
        enc = self.processing_class(
            text=[full_text],
            images=[images],
            return_tensors="pt",
            padding="longest",
        ).to(self.accelerator.device)
        if "labels" in enc:
            enc.pop("labels")
        input_ids = enc["input_ids"]
        out = model(**enc, use_cache=False)
        logits = out.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        mask = self._focus_mask(prompt, completion, token_logps.numel(), focus_tag)
        special_ids = self._special_token_ids()
        if special_ids:
            special = torch.tensor(special_ids, device=shift_labels.device)
            labels = shift_labels.squeeze(0)
            mask = mask * (~torch.isin(labels, special)).float()
        if zero_mask:
            mask = torch.zeros_like(mask)
        return token_logps, mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GeoVLMGRPOTrainer does not return model outputs")
        generated = self._generate_and_score_completions(inputs, model)
        advantages = generated["advantages"]
        losses: list[torch.Tensor] = []
        kl_values: list[torch.Tensor] = []
        for row_parts, advantage in zip(generated["parts"], advantages):
            adv = advantage.to(self.accelerator.device)
            part_losses: list[torch.Tensor] = []
            for part in row_parts:
                logps, mask = self._compute_logps_and_mask(
                    model,
                    part["prompt"],
                    part["completion"],
                    part["images"],
                    focus_tag=part["focus_tag"],
                    zero_mask=part.get("zero_mask", False),
                )
                denom = mask.sum().clamp_min(1.0)
                old_logps = logps.detach()
                ratio = torch.exp(logps - old_logps)
                clipped = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)
                policy = -torch.minimum(ratio * adv, clipped * adv)
                if self.beta > 0 and self.ref_model is not None:
                    with torch.no_grad():
                        ref_logps, _ = self._compute_logps_and_mask(
                            self.ref_model,
                            part["prompt"],
                            part["completion"],
                            part["images"],
                            focus_tag=part["focus_tag"],
                            zero_mask=part.get("zero_mask", False),
                        )
                    delta = ref_logps.detach() - logps
                    kl = torch.exp(delta) - delta - 1.0
                    policy = policy + self.beta * kl
                    kl_values.append(((kl * mask).sum() / denom).detach())
                part_losses.append((policy * mask).sum() / denom)
            if part_losses:
                losses.append(torch.stack(part_losses).mean())
        if not losses:
            loss = torch.zeros((), dtype=torch.float32, device=self.accelerator.device, requires_grad=True)
        else:
            loss = torch.stack(losses).mean()
        kl_mean = torch.stack(kl_values).mean() if kl_values else torch.tensor(0.0, device=loss.device)
        gathered_kl = self.accelerator.gather_for_metrics(kl_mean.detach().view(1))
        gathered_loss = self.accelerator.gather_for_metrics(loss.detach().view(1))
        self._metrics["kl"].append(float(gathered_kl.mean().item()))
        self._metrics["loss_geo"].append(float(gathered_loss.mean().item()))
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
