#!/usr/bin/env python3
"""Two-turn single-zoom rollout environment for GeoSkillRL GRPO."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parents[1] / "geoskillrl_apo_experiments"
SINGLE_ZOOM_DIR = THIS_DIR.parent / "single_zoom"
OLDTEST_DIR = THIS_DIR.parent / "oldtest"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SINGLE_ZOOM_DIR))
sys.path.insert(0, str(OLDTEST_DIR))

import eval_zoom_only_primitive as base_eval  # noqa: E402
from bbox_rewards import canonical_bbox_1024  # noqa: E402
from bbox_rewards_single_zoom import extract_first_bbox  # noqa: E402
from reward_geo import compute_episode_reward  # noqa: E402
from skillbank_runtime import RuntimeSkillBank  # noqa: E402


VISION_TOKEN = base_eval.VISION_TOKEN

SYSTEM_PROMPT_GRPO = (
    "You are an intelligent remote sensing analyst. Given a satellite image, "
    "a question, and retrieved grounding skills, you may use exactly one zoom "
    "tool before answering:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 global image frame. The bbox "
    "must describe the target object or region, not an expanded full-image "
    "crop. First output one <zoom>; after the crop is shown, output exactly "
    "one <answer>...</answer>. Do not output a second zoom."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_image(name: str) -> Path | None:
    return base_eval.find_image(name)


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    return base_eval.resize_image(image, max_size=max_size)


def cut_image(image: Image.Image, bbox_orig: list[float]) -> Image.Image:
    return base_eval.cut_image(image, bbox_orig)


def valid_bbox_sample(row: dict[str, Any]) -> bool:
    bbox = row.get("bbox")
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and str(row.get("type") or "").lower() in {"object", "region"}
        and find_image(str(row.get("image_name") or row.get("image") or "")) is not None
    )


def load_bbox_samples(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(path) if valid_bbox_sample(r)]
    return rows[:limit] if limit else rows


def build_turn1_prompt(question: str, skill_block: str) -> str:
    user = VISION_TOKEN + str(question or "")
    if skill_block:
        user += "\n\n" + skill_block
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT_GRPO}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_turn2_prompt(turn1_prompt: str, turn1_text: str) -> str:
    turn1_trunc = base_eval.truncate_after(turn1_text, "</zoom>")
    return (
        turn1_prompt
        + turn1_trunc
        + "\nZoomed crop image to verify:\n"
        + VISION_TOKEN
        + "\nUse the crop to answer the question. If the crop does not contain the requested target, "
          "answer from the available visual evidence without claiming the crop is correct.\n"
        + "<think>"
    )


def prepare_sample(row: dict[str, Any], skillbank: RuntimeSkillBank) -> dict[str, Any] | None:
    image_name = str(row.get("image_name") or row.get("image") or "")
    image_path = find_image(image_name)
    if image_path is None:
        return None
    image = Image.open(image_path).convert("RGB")
    global_small = resize_image(image)
    skill_info = skillbank.build(row)
    return {
        "row": row,
        "question_id": row.get("question_id"),
        "image_name": image_name,
        "image": image,
        "global_small": global_small,
        "image_size": image.size,
        "scale": max(image.size) / 1024.0,
        "question": str(row.get("question") or ""),
        "ground_truth": row.get("ground_truth"),
        "bbox_1024": [float(v) for v in row["bbox"][:4]],
        "skill_block": skill_info["skill_block"],
        "retrieved_skill_ids": skill_info["retrieved_skill_ids"],
        "turn1_prompt": build_turn1_prompt(str(row.get("question") or ""), skill_info["skill_block"]),
    }


def decode_generated(processor: Any, generated_ids: torch.Tensor, stop_strings: list[str] | None = None) -> str:
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    return base_eval.clean_generation_text(text, stop_strings=stop_strings)


def generate_batch(
    model: Any,
    processor: Any,
    prompts: list[str],
    images: list[list[Image.Image]],
    *,
    max_new_tokens: int,
    do_sample: bool,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    stop_strings: list[str] | None = None,
) -> list[str]:
    if not prompts:
        return []
    device = next(model.parameters()).device
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding="longest").to(device)
    tok = processor.tokenizer
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
        num_return_sequences=num_return_sequences,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if stop_strings:
        kwargs["stop_strings"] = stop_strings
        kwargs["tokenizer"] = tok
    with torch.inference_mode():
        gen = model.generate(**inputs, **kwargs)
    gen = gen[:, inputs["input_ids"].shape[1]:]
    return [decode_generated(processor, gen[i], stop_strings=stop_strings) for i in range(gen.shape[0])]


def rollout_one_sample(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    *,
    num_generations: int,
    turn1_max_new_tokens: int,
    turn2_max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool = True,
) -> list[dict[str, Any]]:
    was_training = model.training
    was_grad_ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            module.gradient_checkpointing = False
    prev_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    try:
        turn1_texts = generate_batch(
            model,
            processor,
            [sample["turn1_prompt"]],
            [[sample["global_small"]]],
            max_new_tokens=turn1_max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_generations,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["</zoom>", "</answer>"],
        )
        rollouts: list[dict[str, Any]] = []
        for turn1_text in turn1_texts:
            if "</zoom>" not in turn1_text and "<zoom>" in turn1_text:
                turn1_text = turn1_text + "</zoom>"
            pred_bbox = canonical_bbox_1024(extract_first_bbox(turn1_text))
            turn2_prompt = build_turn2_prompt(sample["turn1_prompt"], turn1_text)
            # The turn-2 prompt always contains the original global image token
            # plus a crop image token. If zoom parsing fails, use the global
            # image as a dummy crop so logprob/KL can still run with a zero
            # mask under DDP.
            turn2_images = [sample["global_small"], sample["global_small"]]
            turn2_text = ""
            if pred_bbox is not None:
                bbox_orig = [v * sample["scale"] for v in pred_bbox]
                crop_small = resize_image(cut_image(sample["image"], bbox_orig))
                turn2_images = [sample["global_small"], crop_small]
                try:
                    turn2_text = generate_batch(
                        model,
                        processor,
                        [turn2_prompt],
                        [turn2_images],
                        max_new_tokens=turn2_max_new_tokens,
                        do_sample=do_sample,
                        num_return_sequences=1,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_strings=["</answer>"],
                    )[0]
                except Exception:
                    turn2_text = ""
            reward_turn2_text = turn2_text
            if reward_turn2_text and not reward_turn2_text.startswith("<think>"):
                reward_turn2_text = "<think>" + reward_turn2_text
            reward = compute_episode_reward(
                question_id=str(sample["question_id"]),
                question=sample["question"],
                zoom_text=turn1_text,
                answer_text=reward_turn2_text,
                gt_bbox_1024=sample["bbox_1024"],
                ground_truth=sample["ground_truth"],
                image_size=sample["image_size"],
            )
            rollouts.append({
                "question_id": sample["question_id"],
                "image_name": sample["image_name"],
                "type": sample["row"].get("type"),
                "category": sample["row"].get("category"),
                "higher_level": sample["row"].get("higher_level"),
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "turn1_prompt": sample["turn1_prompt"],
                "turn1_completion": turn1_text + "<|im_end|>",
                "turn1_images": [sample["global_small"]],
                "turn2_prompt": turn2_prompt,
                "turn2_completion": turn2_text + "<|im_end|>" if turn2_text else "",
                "turn2_images": turn2_images,
                "trajectory": f"[TURN1]\n{turn1_text}\n[TURN2]\n{reward_turn2_text}\n",
                "skill_block": sample["skill_block"],
                "retrieved_skill_ids": sample["retrieved_skill_ids"],
                "pred_bbox_1024": reward["pred_bbox_1024"],
                "gt_bbox_1024": sample["bbox_1024"],
                "answer_pred": reward["answer_pred"],
                "reward": reward,
            })
        totals = torch.tensor([float(r["reward"]["total"]) for r in rollouts], dtype=torch.float32)
        advantages = totals - totals.mean()
        for row, adv in zip(rollouts, advantages):
            row["advantage"] = float(adv.item())
        return rollouts
    finally:
        model.config.use_cache = prev_use_cache
        if was_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train(was_training)
