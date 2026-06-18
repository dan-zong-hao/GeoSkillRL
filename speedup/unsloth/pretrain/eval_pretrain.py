"""
Quick evaluation of pretrained model: bbox format + IoU on val set.
Batch inference for speed (~10x faster than sample-by-sample).

Usage:
  python eval_pretrain.py \
    --model_path output/final_hf \
    --val_jsonl data/val.jsonl \
    --max_samples 200
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

VQA_ROOT = Path("/root/autodl-tmp/VQA")

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGES_DIR_CANDIDATES = ["/root/autodl-tmp/dataset/DIOR-RSVG/JPEGImages"]
SPECIAL_TOKENS = ["<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>"]

SYSTEM_PROMPT = (
    "You are a helpful assistant that can understand images and reason "
    "with visual primitives."
)

BOX_RE = re.compile(r"<\|box\|>\[\[(\d+),(\d+),(\d+),(\d+)\]\]")
REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>")


def load_image(name: str) -> Image.Image:
    for d in IMAGES_DIR_CANDIDATES:
        p = Path(d) / name
        if p.exists():
            return Image.open(p).convert("RGB")
        p = Path(d) / Path(name).name
        if p.exists():
            return Image.open(p).convert("RGB")
    return Image.new("RGB", (800, 800), (128, 128, 128))


def parse_box(text: str) -> Optional[List[int]]:
    m = BOX_RE.search(text)
    return [int(m.group(i)) for i in range(1, 5)] if m else None


def parse_ref(text: str) -> Optional[str]:
    m = REF_RE.search(text)
    return m.group(1).strip() if m else None


def compute_iou(b1: List[int], b2: List[int]) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default=str(VQA_ROOT / "models" / "Qwen3.5-4B"))
    parser.add_argument("--val_jsonl", default="data/val.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Load model ----
    print(f"Loading base model: {args.base_model}")
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model, load_in_4bit=False,
        use_gradient_checkpointing="unsloth", local_files_only=True,
    )
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.model_path, is_trainable=False)
    model = FastVisionModel.for_inference(model)
    model.eval()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.base_model, max_pixels=800 * 800 * 4)
    processor.tokenizer = tokenizer

    # ---- Load val data (positive samples only) ----
    with open(args.val_jsonl, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    pos_samples = [s for s in samples if s.get("boxes")]
    if len(pos_samples) > args.max_samples:
        pos_samples = random.sample(pos_samples, args.max_samples)
    print(f"Evaluating {len(pos_samples)} positive samples (batch_size={args.batch_size})...")

    # ---- Batch inference ----
    ious = []
    format_ok = 0
    ref_ok = 0

    for i in tqdm(range(0, len(pos_samples), args.batch_size)):
        batch = pos_samples[i:i + args.batch_size]

        # Build prompts and load images
        prompts, images = [], []
        for s in batch:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{VISION_TOKEN}{s['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompts.append(prompt)
            img = load_image(s["image"])
            images.append([img])

        tok = processor(
            text=prompts, images=images, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in tok.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, s in enumerate(batch):
            new_tokens = output_ids[j, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=False)

            pred_box = parse_box(response)
            pred_label = parse_ref(response)
            gt_box = s["boxes"][0]

            if pred_box is not None:
                format_ok += 1
                ious.append(compute_iou(pred_box, gt_box))
            else:
                ious.append(0.0)

            if pred_label and pred_label.lower() == s.get("label", "").lower():
                ref_ok += 1

    # ---- Results ----
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    nonzero = [x for x in ious if x > 0.01]
    high_iou = [x for x in ious if x > 0.5]

    print(f"\n{'='*55}")
    print(f"Samples:          {len(pos_samples)}")
    print(f"Format OK:        {format_ok}/{len(pos_samples)}  ({100*format_ok/len(pos_samples):.1f}%)")
    print(f"Label match:      {ref_ok}/{len(pos_samples)}  ({100*ref_ok/len(pos_samples):.1f}%)")
    print(f"Mean IoU:         {100*mean_iou:.1f}%")
    if nonzero:
        print(f"Non-zero IoU:     {len(nonzero)} ({100*len(nonzero)/len(ious):.1f}%)  mean={100*sum(nonzero)/len(nonzero):.1f}%")
    print(f"IoU > 50%:        {len(high_iou)} ({100*len(high_iou)/len(ious):.1f}%)")

    # IoU distribution
    buckets = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    print(f"\nIoU distribution:")
    for lo, hi in buckets:
        c = sum(1 for x in ious if lo <= x < hi)
        print(f"  [{lo*100:3.0f}% - {hi*100:3.0f}%): {c:4d}  ({100*c/len(ious):5.1f}%)")

    # Show a few examples
    print(f"\n--- Sample outputs ---")
    for s in random.sample(pos_samples, min(5, len(pos_samples))):
        img = load_image(s["image"])
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{VISION_TOKEN}{s['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        tok = processor(text=[prompt], images=[[img]], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in tok.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        resp = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        pb = parse_box(resp)
        gt = s["boxes"][0]
        iou = compute_iou(pb, gt) if pb else 0.0
        print(f"  Q: {s['question'][:80]}")
        print(f"  GT: {s['label']} {gt}")
        print(f"  Pred: {parse_ref(resp)} {pb}  IoU={100*iou:.1f}%")
        print(f"  Full: {resp.strip()[:120]}")
        print()


if __name__ == "__main__":
    main()
