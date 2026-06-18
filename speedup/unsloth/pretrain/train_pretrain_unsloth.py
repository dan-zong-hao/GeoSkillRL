"""
Unsloth-accelerated LoRA pretraining for Qwen3.5-4B: visual primitive format (bbox).

Uses unsloth.FastVisionModel for 2x faster training + lower VRAM.
Keep the same collator and accelerate training loop from train_pretrain.py,
but swap model loading to Unsloth.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_pretrain_unsloth.py
  accelerate launch --num_processes 2 train_pretrain_unsloth.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from unsloth import FastVisionModel

Image.MAX_IMAGE_PIXELS = None

VQA_ROOT = Path("/root/autodl-tmp/VQA")

# Visual primitive special tokens (TVP format)
REF_START = "<|ref|>"
REF_END = "<|/ref|>"
BOX_START = "<|box|>"
BOX_END = "<|/box|>"
SPECIAL_TOKENS = [REF_START, REF_END, BOX_START, BOX_END]

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGES_DIR_CANDIDATES = [
    "/root/autodl-tmp/dataset/DIOR-RSVG/JPEGImages",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PretrainDataset(Dataset):
    def __init__(self, jsonl_path: str, img_dirs: List[str]):
        self.img_dirs = img_dirs
        with open(jsonl_path, encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self._load_image(s["image"])
        return {
            "image": img,
            "image_name": s["image"],
            "question": s["question"],
            "assistant_text": s["assistant_text"],
        }

    def _load_image(self, name: str) -> Image.Image:
        for d in self.img_dirs:
            p = Path(d) / name
            if p.exists():
                return Image.open(p).convert("RGB")
            p = Path(d) / Path(name).name
            if p.exists():
                return Image.open(p).convert("RGB")
        return Image.new("RGB", (800, 800), (128, 128, 128))


# ---------------------------------------------------------------------------
# Collator (same as train_pretrain.py)
# ---------------------------------------------------------------------------
def build_collator(processor, max_length: int):
    pad_id = processor.tokenizer.pad_token_id
    ignore_index = -100

    SYSTEM_PROMPT = (
        "You are a helpful assistant that can understand images and reason "
        "with visual primitives."
    )

    def collate(examples):
        texts_prompt: List[str] = []
        texts_full: List[str] = []
        images: List[List[Image.Image]] = []

        for ex in examples:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{VISION_TOKEN}{ex['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            full = prompt + ex["assistant_text"] + "<|im_end|>"
            texts_prompt.append(prompt)
            texts_full.append(full)
            images.append([ex["image"]])

        tok = processor(
            text=texts_full,
            images=images,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
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
    model_name: str = str(VQA_ROOT / "models" / "Qwen3.5-4B")
    train_jsonl: str = str(VQA_ROOT / "speedup" / "pretrain" / "data" / "train.jsonl")
    val_jsonl: str = str(VQA_ROOT / "speedup" / "pretrain" / "data" / "val.jsonl")
    output_dir: str = str(VQA_ROOT / "speedup" / "pretrain" / "output")
    img_dirs: List[str] = field(default_factory=lambda: list(IMAGES_DIR_CANDIDATES))

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Training
    num_train_epochs: int = 3
    batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 8
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 2048
    max_pixels: int = 800 * 800 * 4

    seed: int = 42
    dtype: str = "bfloat16"
    save_steps: int = 5000  # > total steps so only saves at epoch end + final
    log_steps: int = 10
    freeze_vision: bool = True


# ---------------------------------------------------------------------------
# Model preparation (Unsloth path)
# ---------------------------------------------------------------------------
def prepare_model(config: TrainConfig):
    """Load Qwen3.5-4B via Unsloth, add special tokens, apply LoRA."""

    # 1. Load base model with Unsloth (local model, no HF download needed)
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model_name,
        load_in_4bit=False,                # bf16 LoRA recommended for Qwen3.5 MoE
        use_gradient_checkpointing="unsloth",
        local_files_only=True,             # Skip HF connectivity check
    )
    print(f"[model] Loaded: {config.model_name}")
    print(f"[model] Type: {type(model).__name__}")

    # Unsloth returns a processor; unwrap to get the underlying tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    # 2. Add visual primitive special tokens
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    print(f"[tokenizer] Added {n_added} special tokens: {SPECIAL_TOKENS}")

    # Resize embeddings BEFORE applying LoRA
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load processor (need image processing; Unsloth tokenizer = text only)
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        max_pixels=config.max_pixels,
    )
    processor.tokenizer = tokenizer  # Use Unsloth-patched tokenizer

    # 4. Apply LoRA via Unsloth
    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        finetune_vision_layers=not config.freeze_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # 5. Prepare for training
    model = FastVisionModel.for_training(model)
    print("[model] LoRA applied, ready for training.")

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, processor


# ---------------------------------------------------------------------------
# Training loop (same structure as train_pretrain.py)
# ---------------------------------------------------------------------------
def setup_logger(output_dir: str) -> str:
    """Create a text log file and return its path."""
    log_path = os.path.join(output_dir, "train.log")
    return log_path


def log_metric(log_path: str, msg: str):
    """Append a timestamped line to the training log."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def train(config: TrainConfig):
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    log_path = setup_logger(config.output_dir)
    log_metric(log_path, f"Starting training: {vars(config)}")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.dtype == "bfloat16" else "fp16",
        log_with=["tensorboard"],
        project_dir=config.output_dir,
    )

    model, processor = prepare_model(config)
    collate = build_collator(processor, config.max_length)

    train_ds = PretrainDataset(config.train_jsonl, config.img_dirs)
    val_path = Path(config.val_jsonl)
    val_ds = PretrainDataset(str(val_path), config.img_dirs) if val_path.exists() else None

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size_per_gpu,
        collate_fn=collate, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size_per_gpu,
            collate_fn=collate, shuffle=False, num_workers=2, pin_memory=True,
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    micro_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(micro_per_epoch / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.num_train_epochs
    warmup_steps = max(1, int(total_updates * config.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates,
    )
    scheduler = accelerator.prepare(scheduler)

    accelerator.print(
        f"[train] samples={len(train_ds)} micro/epoch={micro_per_epoch} "
        f"updates/epoch={updates_per_epoch} total_updates={total_updates} "
        f"warmup={warmup_steps} gpus={accelerator.num_processes}"
    )

    accelerator.init_trackers(
        project_name="pretrain-vprim-unsloth",
        config={k: v for k, v in vars(config).items()
                if isinstance(v, (int, float, str, bool))},
    )
    progress = tqdm(total=total_updates, disable=not accelerator.is_local_main_process)

    global_step = 0
    total_loss = torch.tensor(0.0, device=accelerator.device)
    loss_count = 0

    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        for batch in train_loader:
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
                    epoch_progress = global_step / total_updates * config.num_train_epochs
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch_progress,
                    }, step=global_step)
                    if accelerator.is_main_process:
                        log_metric(log_path,
                            f"step={global_step}/{total_updates} "
                            f"loss={avg_loss:.4f} lr={lr:.2e} "
                            f"epoch={epoch_progress:.2f}"
                        )
                    total_loss.zero_()
                    loss_count = 0

                if global_step % config.save_steps == 0:
                    save_checkpoint(accelerator, model, processor, epoch, global_step, config)

        if val_loader is not None:
            val_loss = run_validation(model, val_loader, accelerator)
            accelerator.print(f"[epoch {epoch}] val_loss={val_loss:.4f}")
            accelerator.log({"val/loss": val_loss}, step=global_step)

        save_checkpoint(accelerator, model, processor, epoch, global_step, config)

    save_final(accelerator, model, processor, config)
    accelerator.end_training()


def run_validation(model, val_loader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            losses.append(outputs.loss.detach())
    model.train()
    gathered = accelerator.gather(torch.stack(losses))
    return gathered.mean().item()


def save_checkpoint(accelerator, model, processor, epoch, step, config):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    ckpt_dir = Path(config.output_dir) / f"checkpoint-{step}"
    hf_dir = ckpt_dir / "final_hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    # Unsloth: use for_inference before saving
    unwrapped = accelerator.unwrap_model(model)
    unwrapped = FastVisionModel.for_inference(unwrapped)
    unwrapped.save_pretrained(str(hf_dir), safe_serialization=True)
    processor.save_pretrained(str(hf_dir))

    info = {"epoch": epoch, "step": step, "latest_checkpoint": str(hf_dir)}
    with open(Path(config.output_dir) / "training_info.json", "w") as f:
        json.dump(info, f)
    accelerator.print(f"[save] checkpoint-{step} -> {hf_dir}")

    # Switch back to training mode
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
    accelerator.print(f"[done] Final model -> {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unsloth LoRA pretraining for Qwen3.5-4B")
    p.add_argument("--model_name", default=str(VQA_ROOT / "models" / "Qwen3.5-4B"))
    p.add_argument("--train_jsonl", default=str(VQA_ROOT / "speedup" / "pretrain" / "data" / "train.jsonl"))
    p.add_argument("--val_jsonl", default=str(VQA_ROOT / "speedup" / "pretrain" / "data" / "val.jsonl"))
    p.add_argument("--output_dir", default=str(VQA_ROOT / "speedup" / "pretrain" / "output"))
    p.add_argument("--img_dir", action="append", default=None)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size_per_gpu", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_freeze_vision", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        img_dirs=(args.img_dir or []) + list(IMAGES_DIR_CANDIDATES),
        num_train_epochs=args.num_train_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        seed=args.seed,
        freeze_vision=not args.no_freeze_vision,
    )
    train(cfg)


if __name__ == "__main__":
    main()
