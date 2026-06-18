"""
Unsloth-accelerated full-parameter SFT for Qwen3.5-4B: zoom+seg+think+answer chain.

Loads the pretrained LoRA adapter, merges into base weights, then runs full
fine-tuning (no LoRA) on the LRS-GRO sft_causal_reasoning dataset.

VRAM: ~60GB (bf16 weights 8GB + grads 8GB + Adam fp32 32GB + activations ~12GB)

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_sft_unsloth.py
  accelerate launch --num_processes 2 train_sft_unsloth.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from unsloth import FastVisionModel
from peft import PeftModel

Image.MAX_IMAGE_PIXELS = None

VQA_ROOT = Path("/root/autodl-tmp/VQA")

# Special tokens from pretraining (must add before loading pretrained adapter)
SPECIAL_TOKENS = ["<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>"]

SYSTEM_PROMPT = (
    "You are an intelligent remote sensing analyst. Given a question about a "
    "satellite image, you MAY use two tools to focus before answering:\n"
    '  1. <zoom>[{"bbox_2d":[x1,y1,x2,y2],"label":"<short>"}]</zoom>\n'
    '  2. <seg>{"prompt":"<text>"}</seg>\n'
    "Protocol: wrap reasoning in <think>...</think>. At most ONE <zoom> per "
    "trajectory; <seg> may only appear AFTER <zoom>. End with exactly one "
    "<answer>...</answer> (single word or short phrase). If the whole image "
    "is enough, skip zoom/seg. Never say 'uncertain'."
)

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

IMAGES_DIR_CANDIDATES = [
    "/root/autodl-tmp/dataset/lrs_gro/image",
    "/root/autodl-tmp/dataset/lrs_gro/images",
    "/root/autodl-tmp/dataset/lrs_gro",
]


# ---------------------------------------------------------------------------
# Image utils
# ---------------------------------------------------------------------------
def cut_image(image: Image.Image, bbox, min_size: int = 512) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1, ny1 = cx - min_size // 2, cy - min_size // 2
        nx2, ny2 = nx1 + min_size, ny1 + min_size
        if nx1 < 0: nx2 += -nx1; nx1 = 0
        if ny1 < 0: ny2 += -ny1; ny1 = 0
        if nx2 > image.width: nx1 -= nx2 - image.width; nx2 = image.width
        if ny2 > image.height: ny1 -= ny2 - image.height; ny2 = image.height
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(image.width, nx1 + min_size), min(image.height, ny1 + min_size)
        return image.crop((int(nx1), int(ny1), int(nx2), int(ny2)))
    return image.crop((x1, y1, x2, y2))


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return image


def _find_image(image_name: str, cands: List[str]) -> Optional[Path]:
    for d in cands:
        p = Path(d) / image_name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Dataset (identical to original SFT)
# ---------------------------------------------------------------------------
class ZoomSegDataset(Dataset):
    def __init__(self, records: list, img_dir_candidates: List[str]):
        self.records = records
        self.cands = img_dir_candidates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        path = _find_image(r["image_name"], self.cands)
        if path is None:
            raise FileNotFoundError(
                f"image not found: {r['image_name']} (searched: {self.cands})"
            )
        img_full = Image.open(path).convert("RGB")
        assistant_text = r["assistant_text"]

        images: List[Image.Image] = [resize_image(img_full)]

        has_zoom = "</zoom>" in assistant_text
        has_seg = "</seg>" in assistant_text

        if has_zoom and r.get("bbox"):
            scale = r.get("scale", 1.0)
            bbox_orig = [scale * v for v in r["bbox"]]
            crop = resize_image(cut_image(img_full, bbox_orig))
            images.append(crop)

        if has_seg:
            overlay_path = r.get("overlay_path")
            if overlay_path and Path(overlay_path).exists():
                images.append(Image.open(overlay_path).convert("RGB"))
            else:
                assistant_text = _strip_seg_block(assistant_text)

        return {
            "question": r["question"],
            "assistant_text": assistant_text,
            "images": images,
            "question_id": r["question_id"],
        }


def _strip_seg_block(text: str) -> str:
    text = re.sub(
        r"<seg>.*?</seg>\s*<think>.*?</think>\s*",
        "", text, count=1, flags=re.DOTALL,
    )
    return text


# ---------------------------------------------------------------------------
# Collator (identical to original SFT)
# ---------------------------------------------------------------------------
def build_collator(processor, max_length: int):
    pad_id = processor.tokenizer.pad_token_id
    ignore_index = -100

    def collate(examples):
        texts_prompt: List[str] = []
        texts_full: List[str] = []
        images_flat: List[List[Image.Image]] = []

        for ex in examples:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{VISION_TOKEN}{ex['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            assistant = ex["assistant_text"]
            if "</zoom>" in assistant:
                assistant = assistant.replace(
                    "</zoom>\n<think>",
                    f"</zoom>\n{VISION_TOKEN}\n<think>", 1,
                )
            if "</seg>" in assistant:
                assistant = assistant.replace(
                    "</seg>\n<think>",
                    f"</seg>\n{VISION_TOKEN}\n<think>", 1,
                )

            full = prompt + assistant + "<|im_end|>"
            texts_prompt.append(prompt)
            texts_full.append(full)
            images_flat.append(ex["images"])

        tok = processor(
            text=texts_full, images=images_flat, return_tensors="pt",
            padding="longest", max_length=max_length, truncation=True,
        )

        labels = tok["input_ids"].clone()
        for i, p in enumerate(texts_prompt):
            plen = len(processor.tokenizer(p, add_special_tokens=False)["input_ids"])
            labels[i, :plen] = ignore_index
        labels[labels == pad_id] = ignore_index
        tok["labels"] = labels
        return tok

    return collate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    base_model: str = str(VQA_ROOT / "models" / "Qwen3.5-4B")
    pretrained_adapter: str = str(VQA_ROOT / "speedup" / "unsloth" / "pretrain" / "output" / "final_hf")
    train_jsonl: str = str(
        VQA_ROOT / "json_data" / "zoom_seg_json" / "sft_causal_reasoning"
        / "sft-00000-of-00001.zoom_seg.think.jsonl"
    )
    output_dir: str = str(VQA_ROOT / "speedup" / "unsloth" / "sft" / "output")
    img_folders: List[str] = field(default_factory=lambda: list(IMAGES_DIR_CANDIDATES))

    # Training
    num_train_epochs: int = 3
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 4
    lr: float = 3e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 4096
    max_pixels: int = 64 * 64 * 28 * 28  # ~3.2M pixels

    seed: int = 42
    dtype: str = "bfloat16"
    save_steps: int = 200
    log_steps: int = 10
    freeze_vision: bool = False  # full-FT: train vision + language


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------
def prepare_model(config: TrainConfig):
    """Load base model, merge pretrained LoRA, apply new LoRA for SFT."""

    # 1. Load base model
    model, tokenizer = FastVisionModel.from_pretrained(
        config.base_model,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        local_files_only=True,
    )
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer
    print(f"[model] Base loaded: {config.base_model}")

    # 2. Add special tokens (must match pretraining exactly)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    print(f"[model] Special tokens added: {SPECIAL_TOKENS}")

    # 3. Load pretrained LoRA adapter and merge into base weights
    print(f"[model] Loading pretrained adapter: {config.pretrained_adapter}")
    model = PeftModel.from_pretrained(model, config.pretrained_adapter, is_trainable=True)
    model = model.merge_and_unload()
    print("[model] Pretrained adapter merged into base weights.")

    # 4. Ensure all parameters are trainable (full-FT)
    for p in model.parameters():
        p.requires_grad = True

    # 5. Freeze vision tower if requested
    if config.freeze_vision:
        visual = getattr(model, "visual", None)
        if visual is not None:
            for p in visual.parameters():
                p.requires_grad = False
            print("[model] Vision tower frozen.")

    # 6. Full-parameter training (no new LoRA) — use Unsloth optimizations
    model = FastVisionModel.for_training(model)

    # Processor (need image processing)
    processor = AutoProcessor.from_pretrained(
        config.base_model, max_pixels=config.max_pixels,
    )
    processor.tokenizer = tokenizer

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, processor


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def setup_logger(output_dir: str) -> str:
    return os.path.join(output_dir, "train.log")


def log_metric(log_path: str, msg: str):
    from datetime import datetime
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def train(config: TrainConfig):
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    log_path = setup_logger(config.output_dir)
    log_metric(log_path, f"Starting SFT (Unsloth): {vars(config)}")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.dtype == "bfloat16" else "fp16",
        log_with=["tensorboard"],
        project_dir=config.output_dir,
    )

    model, processor = prepare_model(config)
    collate = build_collator(processor, config.max_length)

    # Dataset
    with open(config.train_jsonl) as f:
        records = [json.loads(line) for line in f]
    before = len(records)
    records = [r for r in records if _find_image(r["image_name"], config.img_folders) is not None]
    print(f"[data] {len(records)}/{before} records have resolvable images")

    dataset = ZoomSegDataset(records, config.img_folders)
    loader = DataLoader(
        dataset, batch_size=config.batch_size_per_gpu,
        collate_fn=collate, shuffle=True, num_workers=2, pin_memory=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay,
    )

    loader, model, optimizer = accelerator.prepare(loader, model, optimizer)

    micro_per_epoch = len(loader)
    updates_per_epoch = math.ceil(micro_per_epoch / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.num_train_epochs
    warmup_steps = max(1, int(total_updates * config.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates,
    )
    scheduler = accelerator.prepare(scheduler)

    accelerator.print(
        f"[sched] dataset={len(dataset)} micro/epoch={micro_per_epoch} "
        f"updates/epoch={updates_per_epoch} total_updates={total_updates} "
        f"warmup={warmup_steps} gpus={accelerator.num_processes}"
    )

    accelerator.init_trackers(
        project_name="sft-unsloth",
        config={k: v for k, v in vars(config).items()
                if isinstance(v, (int, float, str, bool))},
    )

    progress = tqdm(total=total_updates, disable=not accelerator.is_local_main_process)
    global_step = 0
    total_loss = torch.tensor(0.0, device=accelerator.device)
    loss_count = 0

    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                total_loss += loss.detach()
                loss_count += 1
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % config.log_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).sum()
                        / accelerator.num_processes
                        / max(1, loss_count)
                    ).item()
                    lr = scheduler.get_last_lr()[0]
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": global_step / total_updates * config.num_train_epochs,
                    }, step=global_step)
                    if accelerator.is_main_process:
                        log_metric(log_path,
                            f"step={global_step}/{total_updates} "
                            f"loss={avg_loss:.4f} lr={lr:.2e} "
                            f"epoch={global_step/total_updates*config.num_train_epochs:.2f}"
                        )
                    total_loss.zero_()
                    loss_count = 0

                if global_step % config.save_steps == 0:
                    save_checkpoint(accelerator, model, processor, epoch, global_step, config)

        save_checkpoint(accelerator, model, processor, epoch, global_step, config)

    save_final(accelerator, model, processor, config)
    accelerator.end_training()


def save_checkpoint(accelerator, model, processor, epoch, step, config):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    ckpt_dir = Path(config.output_dir) / f"checkpoint-{step}"
    hf_dir = ckpt_dir / "final_hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped = FastVisionModel.for_inference(unwrapped)
    unwrapped.save_pretrained(str(hf_dir), safe_serialization=True)
    processor.save_pretrained(str(hf_dir))

    info = {"epoch": epoch, "step": step, "latest_checkpoint": str(hf_dir)}
    with open(Path(config.output_dir) / "training_info.json", "w") as f:
        json.dump(info, f)
    accelerator.print(f"[save] checkpoint-{step} -> {hf_dir}")

    unwrapped = FastVisionModel.for_training(unwrapped)


def save_final(accelerator, model, processor, config):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    out = Path(config.output_dir) / "final_hf"
    out.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped = FastVisionModel.for_inference(unwrapped)
    unwrapped.save_pretrained(str(out), safe_serialization=True)
    processor.save_pretrained(str(out))
    print(f"[done] Final model -> {out}")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unsloth LoRA SFT for zoom+seg reasoning")
    p.add_argument("--base_model", default=str(VQA_ROOT / "models" / "Qwen3.5-4B"))
    p.add_argument("--pretrained_adapter", default=str(VQA_ROOT / "speedup" / "unsloth" / "pretrain" / "output" / "final_hf"))
    p.add_argument("--train_jsonl", default=str(
        VQA_ROOT / "json_data" / "zoom_seg_json" / "sft_causal_reasoning"
        / "sft-00000-of-00001.zoom_seg.think.jsonl"
    ))
    p.add_argument("--output_dir", default=str(VQA_ROOT / "speedup" / "unsloth" / "sft" / "output"))
    p.add_argument("--img_dir", action="append", default=None)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size_per_gpu", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_vision", action="store_true",
                   help="freeze vision tower (default: train all)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        base_model=args.base_model,
        pretrained_adapter=args.pretrained_adapter,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        img_folders=(args.img_dir or []) + list(IMAGES_DIR_CANDIDATES),
        num_train_epochs=args.num_train_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        seed=args.seed,
        freeze_vision=args.freeze_vision,
    )
    train(cfg)


if __name__ == "__main__":
    main()
