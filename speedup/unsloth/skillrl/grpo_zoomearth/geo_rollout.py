#!/usr/bin/env python3
"""Prompt, image, and rollout helpers for GeoSkillRL GRPO."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
SKILLRL_DIR = THIS_DIR.parent
GRPO_DIR = SKILLRL_DIR / "grpo"
SINGLE_ZOOM_DIR = SKILLRL_DIR / "single_zoom"
EVAL_DIR = THIS_DIR.parents[1] / "geoskillrl_apo_experiments"
sys.path.insert(0, str(GRPO_DIR))
sys.path.insert(0, str(SINGLE_ZOOM_DIR))
sys.path.insert(0, str(EVAL_DIR))

import eval_zoom_only_primitive as base_eval  # noqa: E402
from bbox_rewards import canonical_bbox_1024  # noqa: E402
from bbox_rewards_single_zoom import extract_first_bbox  # noqa: E402
from skillbank_runtime import RuntimeSkillBank  # noqa: E402


VISION_TOKEN = base_eval.VISION_TOKEN

SYSTEM_PROMPT_GRPO = (
    "You are an intelligent remote sensing analyst. Given a satellite image, "
    "a question, and retrieved grounding skills, you must use exactly one zoom "
    "tool before answering:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 global image frame. The bbox "
    "must describe the target object or region, not an expanded full-image "
    "crop. First output one <zoom>; after the crop is shown, output exactly "
    "one <answer>...</answer>. Do not output a second zoom."
)


def find_image(name: str) -> Path | None:
    return base_eval.find_image(name)


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    return base_eval.resize_image(image, max_size=max_size)


def cut_image(image: Image.Image, bbox_orig: list[float]) -> Image.Image:
    return base_eval.cut_image(image, bbox_orig)


def clean_generation_text(text: str, stop_strings: list[str] | None = None) -> str:
    return base_eval.clean_generation_text(text, stop_strings=stop_strings)


def truncate_after(text: str, tag: str) -> str:
    return base_eval.truncate_after(text, tag)


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
    turn1_trunc = truncate_after(turn1_text, "</zoom>")
    return (
        turn1_prompt
        + turn1_trunc
        + "\nZoomed crop image to verify:\n"
        + VISION_TOKEN
        + "\nUse the crop to answer the question. If the crop does not contain the requested target, "
        "answer from the available visual evidence without claiming the crop is correct.\n"
        + "<think>"
    )


def prepare_sample(row: dict[str, Any], skillbank: RuntimeSkillBank) -> dict[str, Any]:
    image_name = str(row.get("image_name") or row.get("image") or "")
    image_path = find_image(image_name)
    if image_path is None:
        raise FileNotFoundError(f"image not found for {image_name}")
    image = Image.open(image_path).convert("RGB")
    skill_info = skillbank.build(row)
    question = str(row.get("question") or "")
    return {
        "row": row,
        "question_id": row.get("question_id"),
        "image_name": image_name,
        "image_path": str(image_path),
        "image": image,
        "global_small": resize_image(image),
        "image_size": image.size,
        "scale": max(image.size) / 1024.0,
        "question": question,
        "ground_truth": row.get("ground_truth"),
        "bbox_1024": [float(v) for v in row["bbox"][:4]],
        "type": row.get("type"),
        "category": row.get("category"),
        "higher_level": row.get("higher_level"),
        "skill_block": skill_info["skill_block"],
        "retrieved_skill_ids": skill_info["retrieved_skill_ids"],
        "turn1_prompt": build_turn1_prompt(question, skill_info["skill_block"]),
    }


def parse_zoom_bbox(turn1_text: str) -> list[float] | None:
    normalized = (turn1_text or "").replace("</|box|>", "<|/box|>").replace("</|ref|>", "<|/ref|>")
    return canonical_bbox_1024(extract_first_bbox(normalized))


def build_stage2_context(sample: dict[str, Any], turn1_text: str) -> dict[str, Any] | None:
    pred_bbox = parse_zoom_bbox(turn1_text)
    if pred_bbox is None:
        return None
    bbox_orig = [float(v) * float(sample["scale"]) for v in pred_bbox]
    crop_small = resize_image(cut_image(sample["image"], bbox_orig))
    return {
        "prompt": build_turn2_prompt(sample["turn1_prompt"], turn1_text),
        "images": [sample["global_small"], crop_small],
        "pred_bbox_1024": pred_bbox,
    }


def ensure_closed_zoom(turn1_text: str) -> str:
    if "</zoom>" not in (turn1_text or "") and "<zoom>" in (turn1_text or ""):
        return turn1_text + "</zoom>"
    return turn1_text
