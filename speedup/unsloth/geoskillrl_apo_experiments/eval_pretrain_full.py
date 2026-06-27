#!/usr/bin/env python3
"""Full DIOR-RSVG primitive grounding evaluation for Stage 1.

Outputs JSON and a one-line table with format OK, mean IoU, Recall@0.3,
Recall@0.5, and IoU buckets.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

VQA_ROOT = Path('/root/autodl-tmp/VQA')
VISION_TOKEN = '<|vision_start|><|image_pad|><|vision_end|>'
SPECIAL_TOKENS = ['<|ref|>', '<|/ref|>', '<|box|>', '<|/box|>']
SYSTEM_PROMPT = 'You are a helpful assistant that can understand images and reason with visual primitives.'
BOX_RE = re.compile(r'<\|box\|>\s*\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]\s*<\|/box\|>')
REF_RE = re.compile(r'<\|ref\|>(.*?)<\|/ref\|>', re.DOTALL)
IMAGE_DIRS = [Path('/root/autodl-tmp/dataset/DIOR-RSVG/JPEGImages')]
Image.MAX_IMAGE_PIXELS = None


def load_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return [json.loads(line) for line in f if line.strip()]


def find_image(name: str) -> Optional[Path]:
    for d in IMAGE_DIRS:
        p = d / name
        if p.exists():
            return p
        p = d / Path(name).name
        if p.exists():
            return p
    return None


def parse_box(text: str) -> Optional[list[int]]:
    m = BOX_RE.search(text or '')
    if not m:
        return None
    return [int(m.group(i)) for i in range(1, 5)]


def parse_ref(text: str) -> Optional[str]:
    m = REF_RE.search(text or '')
    return m.group(1).strip() if m else None


def iou(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (aa + bb - inter + 1e-9))


def build_prompt(question: str) -> str:
    return (
        f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        f'<|im_start|>user\n{VISION_TOKEN}{question}<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )


def buckets(values: list[float]) -> dict[str, int]:
    ranges = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    return {f'{lo:.1f}-{hi:.1f}': sum(1 for v in values if lo <= v < hi) for lo, hi in ranges}


def write_outputs(summary: dict, rows: list[dict], output_dir: Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{tag}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output_dir / f'{tag}_predictions.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    line = (
        f"stage1: samples={summary['samples']} format={summary['format_ok_rate']:.3f} "
        f"label={summary['label_match_rate']:.3f} mean_iou={summary['mean_iou']:.4f} "
        f"r03={summary['recall_0_3']:.3f} r05={summary['recall_0_5']:.3f}"
    )
    (output_dir / f'{tag}_table.txt').write_text(line + '\n')
    print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=Path, default=VQA_ROOT / 'speedup/unsloth/pretrain/output/final_hf')
    p.add_argument('--base_model', type=Path, default=VQA_ROOT / 'models/Qwen3.5-4B')
    p.add_argument('--val_jsonl', type=Path, default=VQA_ROOT / 'speedup/unsloth/pretrain/data/val.jsonl')
    p.add_argument('--output_dir', type=Path, default=VQA_ROOT / 'speedup/unsloth/geoskillrl_apo_experiments/results/stage1_pretrain')
    p.add_argument('--tag', default='stage1_pretrain_full')
    p.add_argument('--limit', type=int, default=0, help='0 means all positive val samples')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    samples = [s for s in load_jsonl(args.val_jsonl) if s.get('boxes')]
    if args.limit:
        random.Random(args.seed).shuffle(samples)
        samples = samples[:args.limit]

    from unsloth import FastVisionModel
    from transformers import AutoProcessor
    from peft import PeftModel

    t0 = time.perf_counter()
    model, tokenizer = FastVisionModel.from_pretrained(
        str(args.base_model), load_in_4bit=False,
        use_gradient_checkpointing='unsloth', local_files_only=True,
    )
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, str(args.model_path), is_trainable=False)
    model = FastVisionModel.for_inference(model)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(args.base_model), max_pixels=800 * 800 * 4)
    processor.tokenizer = tokenizer
    print(f'[model] ready in {time.perf_counter() - t0:.1f}s, samples={len(samples)}')

    rows: list[dict] = []
    ious: list[float] = []
    format_ok = 0
    label_ok = 0

    for i in tqdm(range(0, len(samples), args.batch_size), desc='Stage1 eval'):
        batch = samples[i:i + args.batch_size]
        prompts: list[str] = []
        images: list[list[Image.Image]] = []
        keep: list[dict] = []
        for sample in batch:
            img_path = find_image(sample.get('image', ''))
            if not img_path:
                continue
            prompts.append(build_prompt(sample.get('question', '')))
            images.append([Image.open(img_path).convert('RGB')])
            keep.append(sample)
        if not keep:
            continue
        tok = processor(text=prompts, images=images, return_tensors='pt', padding=True)
        inputs = {k: v.to(model.device) for k, v in tok.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, sample in enumerate(keep):
            resp = tokenizer.decode(out[j, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
            pred_box = parse_box(resp)
            pred_ref = parse_ref(resp)
            gt_box = sample.get('boxes', [[]])[0]
            score = iou(pred_box, gt_box) if pred_box else 0.0
            ious.append(score)
            format_ok += int(pred_box is not None)
            label_ok += int(bool(pred_ref) and pred_ref.lower() == str(sample.get('label', '')).lower())
            rows.append({
                'image': sample.get('image'),
                'question': sample.get('question'),
                'label': sample.get('label'),
                'gt_box': gt_box,
                'pred_box': pred_box,
                'pred_ref': pred_ref,
                'iou': round(score, 6),
                'response': resp,
            })

    n = max(len(rows), 1)
    summary = {
        'model_path': str(args.model_path),
        'val_jsonl': str(args.val_jsonl),
        'samples': len(rows),
        'format_ok_rate': format_ok / n,
        'label_match_rate': label_ok / n,
        'mean_iou': sum(ious) / n,
        'recall_0_3': sum(1 for x in ious if x >= 0.3) / n,
        'recall_0_5': sum(1 for x in ious if x >= 0.5) / n,
        'iou_buckets': buckets(ious),
    }
    write_outputs(summary, rows, args.output_dir, args.tag)


if __name__ == '__main__':
    main()
