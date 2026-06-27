"""
Unsloth full-parameter SFT for Qwen3.5-4B: zoom-only verified dataset.

Loads pretrained LoRA adapter, merges into base, then full-FT on the
verified zoom data (with crop-verification positive+negative examples).

No seg tool - only think → zoom → think(verify) → answer.

VRAM: ~60GB

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_sft_zoom.py
  accelerate launch --num_processes 2 train_sft_zoom.py
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
import torch.nn.functional as F
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from unsloth import FastVisionModel
from peft import PeftModel

Image.MAX_IMAGE_PIXELS = None

VQA_ROOT = Path("/root/autodl-tmp/VQA")

SPECIAL_TOKENS = ["<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>"]

SYSTEM_PROMPT_JSON = (
    "You are an intelligent remote sensing analyst. Given a question about a "
    "satellite image, you MAY use a zoom tool to focus before answering:\n"
    '  <zoom>[{"bbox_2d":[x1,y1,x2,y2],"label":"<short>"}]</zoom>\n'
    "Protocol: wrap reasoning in <think>...</think>. At most ONE <zoom> per "
    "trajectory. End with exactly one <answer>...</answer> (single word or "
    "short phrase). If the whole image is enough, skip zoom. After zooming, "
    "ALWAYS verify the crop actually contains the target before analyzing. "
    "If the crop doesn't match the target, honestly report the error rather "
    "than hallucinating. Never say 'uncertain'."
)

SYSTEM_PROMPT_PRIMITIVE = (
    "You are an intelligent remote sensing analyst. Given a question about a "
    "satellite image, you MAY use a zoom tool to focus before answering:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 image frame and must describe "
    "the target object or region, not an expanded crop window. Protocol: wrap "
    "reasoning in <think>...</think>. At most TWO <zoom> calls are allowed "
    "only when the first crop is explicitly verified as wrong or partial. End "
    "with exactly one <answer>...</answer> (single word or short phrase). If "
    "the whole image is enough, skip zoom. After zooming, ALWAYS verify the "
    "crop actually contains the target before analyzing. If the crop doesn't "
    "match the target, honestly report the error rather than hallucinating. "
    "Never say 'uncertain'."
)
VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

IMAGE_DIRS = [
    "/root/autodl-tmp/dataset/lrs_gro/image",
    "/root/autodl-tmp/dataset/lrs_gro/images",
    "/root/autodl-tmp/dataset/lrs_gro",
    "/root/autodl-tmp/dataset/DIOR-RSVG/JPEGImages",
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
        return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return image


def _find_image(name: str, cands: List[str]) -> Optional[Path]:
    for d in cands:
        p = Path(d) / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Dataset — zoom-only (global + optional crop, no overlay)
# ---------------------------------------------------------------------------
class ZoomOnlyDataset(Dataset):
    def __init__(self, records: list, img_dirs: List[str]):
        self.records = records
        self.dirs = img_dirs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        path = _find_image(r["image_name"], self.dirs)
        if path is None:
            raise FileNotFoundError(f"image not found: {r['image_name']}")

        img_full = Image.open(path).convert("RGB")
        assistant_text = r["assistant_text"]
        images = [resize_image(img_full)]

        has_zoom = "</zoom>" in assistant_text
        if has_zoom:
            scale = r.get("scale", 1.0)
            tool_bboxes = r.get("tool_bboxes_1024") or r.get("tool_bboxes")
            if not tool_bboxes and r.get("bbox"):
                tool_bboxes = [r["bbox"]]
            for bbox in tool_bboxes or []:
                if bbox and len(bbox) == 4:
                    bbox_orig = [scale * v for v in bbox]
                    crop = resize_image(cut_image(img_full, bbox_orig))
                    images.append(crop)

        return {
            "question": r.get("question", ""),
            "assistant_text": assistant_text,
            "images": images,
            "question_id": r.get("question_id", ""),
        }


# ---------------------------------------------------------------------------
# Collator — zoom-only (no seg)
# ---------------------------------------------------------------------------
def build_collator(processor, max_length: int, system_prompt: str = SYSTEM_PROMPT_JSON, bbox_loss_alpha: float = 0.0):
    pad_id = processor.tokenizer.pad_token_id
    ignore_index = -100

    def collate(examples):
        texts_prompt: List[str] = []
        texts_full: List[str] = []
        images_flat: List[List[Image.Image]] = []

        for ex in examples:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{VISION_TOKEN}{ex['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            assistant = ex["assistant_text"]
            # Inject one crop vision token after each zoom observation.
            if "</zoom>" in assistant:
                assistant = assistant.replace(
                    "</zoom>\n<think>",
                    f"</zoom>\n{VISION_TOKEN}\n<think>",
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

        if bbox_loss_alpha > 0:
            weights = torch.ones_like(labels, dtype=torch.float)
            span_res = [
                re.compile(r"<\|box\|>.*?<\|/box\|>", re.DOTALL),
                re.compile(r'"bbox_2d"\s*:\s*\[[^\]]+\]', re.DOTALL),
            ]
            for i, full in enumerate(texts_full):
                for span_re in span_res:
                    for match in span_re.finditer(full):
                        start = len(processor.tokenizer(
                            full[:match.start()], add_special_tokens=False,
                        )["input_ids"])
                        end = len(processor.tokenizer(
                            full[:match.end()], add_special_tokens=False,
                        )["input_ids"])
                        if end > start:
                            weights[i, start:end] = 1.0 + float(bbox_loss_alpha)
            weights[labels == ignore_index] = 0.0
            tok["loss_weights"] = weights
        return tok

    return collate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    base_model: str = str(VQA_ROOT / "models" / "Qwen3.5-4B")
    pretrained_adapter: str = str(VQA_ROOT / "speedup" / "unsloth" / "pretrain" / "output" / "final_hf")
    train_jsonl: str = str(VQA_ROOT / "json_data" / "zoom_json" / "geoskillrl_apo" / "bridge_sft_primitive.jsonl")
    output_dir: str = str(VQA_ROOT / "speedup" / "unsloth" / "sft" / "only_zoom" / "output")
    img_folders: List[str] = field(default_factory=lambda: list(IMAGE_DIRS))

    num_train_epochs: int = 3
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 4
    lr: float = 3e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 4096
    max_pixels: int = 64 * 64 * 28 * 28

    seed: int = 42
    dtype: str = "bfloat16"
    save_steps: int = 9999
    log_steps: int = 10
    freeze_vision: bool = False
    primitive_grammar: bool = False
    bbox_loss_alpha: float = 0.0
    max_steps: int = 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def prepare_model(config: TrainConfig):
    model, tokenizer = FastVisionModel.from_pretrained(
        config.base_model, load_in_4bit=False,
        use_gradient_checkpointing="unsloth", local_files_only=True,
    )
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    print(f"[model] Loading pretrained adapter: {config.pretrained_adapter}")
    model = PeftModel.from_pretrained(model, config.pretrained_adapter, is_trainable=True)
    model = model.merge_and_unload()
    print("[model] Pretrained adapter merged.")

    for p in model.parameters():
        p.requires_grad = True

    if config.freeze_vision:
        visual = getattr(model, "visual", None)
        if visual is not None:
            for p in visual.parameters():
                p.requires_grad = False
            print("[model] Vision tower frozen.")

    model = FastVisionModel.for_training(model)

    processor = AutoProcessor.from_pretrained(config.base_model, max_pixels=config.max_pixels)
    processor.tokenizer = tokenizer

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, processor


# ---------------------------------------------------------------------------
# Training
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
    log_metric(log_path, f"Starting zoom-only SFT: {vars(config)}")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=["tensorboard"],
        project_dir=config.output_dir,
    )

    model, processor = prepare_model(config)
    system_prompt = SYSTEM_PROMPT_PRIMITIVE if config.primitive_grammar else SYSTEM_PROMPT_JSON
    collate = build_collator(
        processor, config.max_length,
        system_prompt=system_prompt,
        bbox_loss_alpha=config.bbox_loss_alpha,
    )

    with open(config.train_jsonl) as f:
        records = [json.loads(line) for line in f]
    before = len(records)
    records = [r for r in records if _find_image(r["image_name"], config.img_folders) is not None]
    print(f"[data] {len(records)}/{before} images found")

    dataset = ZoomOnlyDataset(records, config.img_folders)
    loader = DataLoader(
        dataset, batch_size=config.batch_size_per_gpu,
        collate_fn=collate, shuffle=True, num_workers=2, pin_memory=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, betas=(0.9, 0.999),
                                   weight_decay=config.weight_decay)

    loader, model, optimizer = accelerator.prepare(loader, model, optimizer)

    micro = len(loader)
    updates_per_epoch = math.ceil(micro / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.num_train_epochs
    if config.max_steps > 0:
        total_updates = min(total_updates, config.max_steps)
    warmup = max(1, int(total_updates * config.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_updates,
    )
    scheduler = accelerator.prepare(scheduler)

    accelerator.print(
        f"[sched] dataset={len(dataset)} micro/epoch={micro} "
        f"updates/epoch={updates_per_epoch} total={total_updates} warmup={warmup}"
    )

    accelerator.init_trackers(
        project_name="sft-zoom-only",
        config={k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool))},
    )

    progress = tqdm(total=total_updates, disable=not accelerator.is_local_main_process)
    global_step = 0
    total_loss = torch.tensor(0.0, device=accelerator.device)
    loss_count = 0

    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                loss_weights = batch.pop("loss_weights", None)
                if loss_weights is None:
                    outputs = model(**batch)
                    loss = outputs.loss
                else:
                    labels = batch.pop("labels")
                    outputs = model(**batch)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_weights = loss_weights[..., 1:].contiguous()
                    per_token = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view_as(shift_labels)
                    mask = (shift_labels != -100).float()
                    weighted = per_token * shift_weights * mask
                    denom = (shift_weights * mask).sum().clamp_min(1.0)
                    loss = weighted.sum() / denom
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
                    avg_loss = (accelerator.gather(total_loss).sum()
                                / accelerator.num_processes / max(1, loss_count)).item()
                    lr = scheduler.get_last_lr()[0]
                    accelerator.log({
                        "train/loss": avg_loss, "train/lr": lr,
                        "train/epoch": global_step / total_updates * config.num_train_epochs,
                    }, step=global_step)
                    if accelerator.is_main_process:
                        log_metric(log_path,
                            f"step={global_step}/{total_updates} loss={avg_loss:.4f} lr={lr:.2e}"
                        )
                    total_loss.zero_()
                    loss_count = 0

                if global_step % config.save_steps == 0:
                    save_checkpoint(accelerator, model, processor, epoch, global_step, config)

                if config.max_steps > 0 and global_step >= config.max_steps:
                    break

        save_checkpoint(accelerator, model, processor, epoch, global_step, config)
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

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
    accelerator.print(f"[save] checkpoint-{step}")
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
    print(f"[done] {out}")


def dry_run_batch(config: TrainConfig, take: int = 4):
    processor = AutoProcessor.from_pretrained(config.base_model, max_pixels=config.max_pixels)
    tokenizer = processor.tokenizer
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    processor.tokenizer = tokenizer

    with open(config.train_jsonl) as f:
        records = [json.loads(line) for line in f]
    records = [r for r in records if _find_image(r["image_name"], config.img_folders) is not None]
    zoom_records = [r for r in records if "</zoom>" in r.get("assistant_text", "")]
    plain_records = [r for r in records if "</zoom>" not in r.get("assistant_text", "")]
    sample = (zoom_records[: max(1, take - 1)] + plain_records[:1])[:take]
    if not sample:
        raise RuntimeError("no records with available images for dry run")

    system_prompt = SYSTEM_PROMPT_PRIMITIVE if config.primitive_grammar else SYSTEM_PROMPT_JSON
    collate = build_collator(
        processor, config.max_length,
        system_prompt=system_prompt,
        bbox_loss_alpha=config.bbox_loss_alpha,
    )
    loader = DataLoader(
        ZoomOnlyDataset(sample, config.img_folders),
        batch_size=min(config.batch_size_per_gpu, len(sample)),
        collate_fn=collate, shuffle=False, num_workers=0, pin_memory=False,
    )
    batch = next(iter(loader))
    labels = batch.get("labels")
    loss_weights = batch.get("loss_weights")
    supervised = int((labels != -100).sum().item()) if labels is not None else 0
    boosted = int((loss_weights > 1.0).sum().item()) if loss_weights is not None else 0
    print(
        json.dumps({
            "dry_run_records": len(sample),
            "zoom_records": sum("</zoom>" in r.get("assistant_text", "") for r in sample),
            "keys": sorted(batch.keys()),
            "input_shape": list(batch["input_ids"].shape),
            "supervised_tokens": supervised,
            "bbox_boosted_tokens": boosted,
        }, indent=2)
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=str(VQA_ROOT / "models" / "Qwen3.5-4B"))
    p.add_argument("--pretrained_adapter", default=str(VQA_ROOT / "speedup" / "unsloth" / "pretrain" / "output" / "final_hf"))
    p.add_argument("--train_jsonl", default=str(VQA_ROOT / "json_data" / "zoom_json" / "geoskillrl_apo" / "bridge_sft_primitive.jsonl"))
    p.add_argument("--output_dir", default=str(VQA_ROOT / "speedup" / "unsloth" / "sft" / "only_zoom" / "output"))
    p.add_argument("--img_dir", action="append", default=None)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size_per_gpu", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--freeze_vision", action="store_true")
    p.add_argument("--primitive_grammar", action="store_true")
    p.add_argument("--bbox_loss_alpha", type=float, default=0.0)
    p.add_argument("--save_steps", type=int, default=9999)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--dry_run_batch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        base_model=args.base_model,
        pretrained_adapter=args.pretrained_adapter,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        img_folders=(args.img_dir or []) + list(IMAGE_DIRS),
        num_train_epochs=args.num_train_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        max_length=args.max_length,
        freeze_vision=args.freeze_vision,
        primitive_grammar=args.primitive_grammar,
        bbox_loss_alpha=args.bbox_loss_alpha,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
    )
    if args.dry_run_batch:
        dry_run_batch(cfg)
        return
    train(cfg)


if __name__ == "__main__":
    main()
