#!/usr/bin/env python3
"""Cold-start SFT for train-safe SkillBank-triggered VLM-style trajectories.

This trainer is intentionally separate from ``single_zoom``:
  * the input rows include both bbox and global QA samples;
  * bbox rows supervise a single fixed GT zoom, then crop-based reasoning,
    then the fixed GT answer;
  * global rows skip zoom and supervise direct image-level reasoning + answer;
  * skill blocks are injected into the user prompt, but assistant outputs do
    not mention skill ids.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


VQA_ROOT = Path("/root/autodl-tmp/VQA")
THIS_DIR = Path(__file__).resolve().parent
STAGEA_DIR = THIS_DIR.parent / "stageA"
sys.path.insert(0, str(STAGEA_DIR))

import train_stageA_sft as stagea  # noqa: E402
import train_bridge_sft_zoom as bridge  # noqa: E402


DEFAULT_MODEL = (
    VQA_ROOT
    / "speedup/unsloth/geoskillrl_apo_experiments/output"
    / "no_seg_vlm_rewrite_primitive/final_hf"
)
DEFAULT_TRAIN = (
    VQA_ROOT
    / "json_data/zoom_json/rl/rl_train_vlm_style_reasoning_skilltrigger_v2.jsonl"
)
DEFAULT_OUT = THIS_DIR / "output/geoskill_coldstart_skilltrigger_v2"

SYSTEM_PROMPT_COLDSTART = (
    "You are an intelligent remote sensing analyst. Given a satellite image, "
    "a question, and optional retrieved grounding skills, answer using the "
    "same protocol as the training trajectory. For object or region questions, "
    "first reason over the global image, then choose exactly one precise "
    "grounding zoom for the target:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 global image frame and must "
    "describe the target referent, not a broad context crop. After the zoom, "
    "inspect the provided crop image and answer from crop evidence. For global "
    "questions where the whole image is sufficient, skip zoom. Always wrap "
    "reasoning in <think>...</think> and end with exactly one "
    "<answer>...</answer>. Do not mention skill ids in the final answer, do "
    "not call <seg>, and do not output more than one zoom."
)


class ColdstartVLMStyleDataset(bridge.Dataset):
    """Trajectory-level dataset: global image in prompt, crop after zoom."""

    def __init__(self, records: list[dict], img_dirs: list[str]):
        self.records = records
        self.dirs = img_dirs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        path = bridge._find_image(row["image_name"], self.dirs)
        if path is None:
            raise FileNotFoundError(f"image not found: {row['image_name']}")

        img_full = Image.open(path).convert("RGB")
        assistant_text = row["assistant_text"]
        images = [bridge.resize_image(img_full)]

        if "</zoom>" in assistant_text:
            scale = row.get("scale", 1.0)
            tool_bboxes = row.get("tool_bboxes_1024") or row.get("tool_bboxes")
            if not tool_bboxes and row.get("bbox"):
                tool_bboxes = [row["bbox"]]
            for bbox in tool_bboxes or []:
                if bbox and len(bbox) == 4:
                    bbox_orig = [scale * float(v) for v in bbox]
                    crop = bridge.resize_image(bridge.cut_image(img_full, bbox_orig))
                    images.append(crop)

        return {
            "question": row.get("question", ""),
            "assistant_text": assistant_text,
            "images": images,
            "question_id": row.get("question_id", ""),
            "skill_block": row.get("skill_block", ""),
            # Crop is an observation after the model emits <zoom>, not part
            # of the initial user prompt.
            "prompt_crop_count": 0,
            "is_step_level": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--train_jsonl", default=str(DEFAULT_TRAIN))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--img_dir", action="append", default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--bbox_loss_alpha", type=float, default=2.0)
    parser.add_argument("--save_steps", type=int, default=9999)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--dry_run_batch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stagea.SYSTEM_PROMPT_STAGEA_PRIMITIVE = SYSTEM_PROMPT_COLDSTART
    stagea.patch_bridge()
    bridge.ZoomOnlyDataset = ColdstartVLMStyleDataset
    bridge.SYSTEM_PROMPT_PRIMITIVE = SYSTEM_PROMPT_COLDSTART

    cfg = bridge.TrainConfig(
        base_model=args.model,
        pretrained_adapter="",
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        img_folders=(args.img_dir or []) + list(bridge.IMAGE_DIRS),
        num_train_epochs=args.num_train_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        max_length=args.max_length,
        freeze_vision=args.freeze_vision,
        primitive_grammar=True,
        bbox_loss_alpha=args.bbox_loss_alpha,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
    )
    if args.dry_run_batch:
        bridge.dry_run_batch(cfg)
    else:
        bridge.train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
