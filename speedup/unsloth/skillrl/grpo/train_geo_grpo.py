#!/usr/bin/env python3
"""GeoSkillRL GRPO V1: SkillBank-runtime zoom+answer episodes."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, get_linear_schedule_with_warmup


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from rollout_env import load_bbox_samples, prepare_sample, rollout_one_sample  # noqa: E402
from skillbank_runtime import DEFAULT_SKILLBANK, RuntimeSkillBank  # noqa: E402


DEFAULT_MODEL = Path(
    "/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf"
)
DEFAULT_TRAIN = THIS_DIR.parent / "stageA/data/splits/rl_train.jsonl"
DEFAULT_OUT = THIS_DIR / "output/geoskill_grpo_skillbank_v1"
DEFAULT_ROLLOUT_LOG = THIS_DIR / "logs/geoskill_grpo_skillbank_v1_rollouts.jsonl"
SPECIAL_TOKENS = ["<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>"]


@dataclass
class Config:
    model_path: str = str(DEFAULT_MODEL)
    train_jsonl: str = str(DEFAULT_TRAIN)
    skillbank_path: str = str(DEFAULT_SKILLBANK)
    output_dir: str = str(DEFAULT_OUT)
    rollout_log_path: str = str(DEFAULT_ROLLOUT_LOG)
    data_limit: int = 0
    num_train_epochs: int = 1
    batch_size_per_device: int = 1
    num_generations: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-7
    warmup_steps: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    beta: float = 0.08
    clip_eps: float = 0.2
    seed: int = 42
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.8
    rollout_top_k: int = 20
    turn1_max_new_tokens: int = 512
    turn2_max_new_tokens: int = 256
    max_pixels: int = 64 * 64 * 28 * 28
    save_steps: int = 100000
    log_steps: int = 1
    max_steps: int = -1
    save_rollouts: bool = True
    skip_final_save: bool = False


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=os.environ.get("DDP_BACKEND", "nccl"), init_method="env://")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    return distributed, rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def rank_print(rank: int, *args, **kwargs) -> None:
    if is_main(rank):
        print(*args, **kwargs)


def barrier(distributed: bool, local_rank: int) -> None:
    if distributed:
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()


def reduce_mean(x: float, device: str, distributed: bool) -> float:
    t = torch.tensor(float(x), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t.item()


class BBoxDataset(Dataset):
    def __init__(self, path: str, limit: int = 0):
        self.records = load_bbox_samples(path, limit=limit)
        print(f"[data] grpo bbox prompts={len(self.records)} from {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate(batch):
    return batch


def special_token_ids(tokenizer) -> list[int]:
    ids = []
    for tok in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.append(int(tid))
    return ids


def compute_logps_and_mask(
    model,
    processor,
    prompt: str,
    completion: str,
    images: list[Image.Image],
    *,
    focus_tag: str,
    zero_mask: bool = False,
):
    if not completion:
        completion = f"<{focus_tag}></{focus_tag}><|im_end|>"
    full = prompt + completion
    device = next(model.parameters()).device
    enc = processor(text=[full], images=[images], return_tensors="pt", padding="longest").to(device)
    input_ids = enc["input_ids"]
    prompt_len = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    out = model(**enc, use_cache=False)
    logits = out.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    shift_labels = input_ids[:, 1:]
    token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    completion_logps = token_logps[:, prompt_len - 1:].squeeze(0)
    completion_labels = shift_labels[:, prompt_len - 1:].squeeze(0)
    mask = torch.ones_like(completion_logps, dtype=torch.float32)
    ids = special_token_ids(processor.tokenizer)
    if ids:
        special = torch.tensor(ids, device=completion_labels.device)
        mask = mask * (~torch.isin(completion_labels, special)).float()
    start_tag = f"<{focus_tag}>"
    end_tag = f"</{focus_tag}>"
    start = completion.find(start_tag)
    end = completion.find(end_tag)
    if start >= 0 and end >= start:
        end += len(end_tag)
        start_len = len(processor.tokenizer(prompt + completion[:start], add_special_tokens=False)["input_ids"])
        end_len = len(processor.tokenizer(prompt + completion[:end], add_special_tokens=False)["input_ids"])
        s = max(0, start_len - prompt_len)
        e = min(mask.numel(), max(s + 1, end_len - prompt_len))
        focus = torch.zeros_like(mask)
        focus[s:e] = 1.0
        if (focus * mask).sum().item() > 0:
            mask = mask * focus
    if zero_mask:
        mask = torch.zeros_like(mask)
    return completion_logps, mask


def grpo_loss(model, ref_model, processor, rollouts: list[dict[str, Any]], cfg: Config):
    losses = []
    kl_values = []
    for row in rollouts:
        parts = [
            (row["turn1_prompt"], row["turn1_completion"], row["turn1_images"], "zoom", False),
        ]
        # DDP requires all ranks to execute the same number of model forwards.
        # If zoom parsing failed and no answer was generated, run a zero-masked
        # dummy turn-2 forward. It contributes no gradient or KL but keeps ranks
        # synchronized.
        parts.append((
            row["turn2_prompt"],
            row.get("turn2_completion") or "",
            row["turn2_images"],
            "answer",
            not bool(row.get("turn2_completion")),
        ))
        adv = torch.tensor(float(row["advantage"]), dtype=torch.float32, device=next(model.parameters()).device)
        part_losses = []
        for prompt, completion, images, focus_tag, zero_mask in parts:
            logps, mask = compute_logps_and_mask(
                model, processor, prompt, completion, images, focus_tag=focus_tag, zero_mask=zero_mask
            )
            denom = mask.sum().clamp_min(1.0)
            old_logps = logps.detach()
            ratio = torch.exp(logps - old_logps)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            policy = -torch.minimum(ratio * adv, clipped * adv)
            if ref_model is not None and cfg.beta > 0:
                with torch.no_grad():
                    ref_logps, _ = compute_logps_and_mask(
                        ref_model, processor, prompt, completion, images, focus_tag=focus_tag, zero_mask=zero_mask
                    )
                delta = ref_logps.detach() - logps
                kl = torch.exp(delta) - delta - 1.0
                policy = policy + cfg.beta * kl
                kl_values.append(((kl * mask).sum() / denom).detach())
            part_losses.append((policy * mask).sum() / denom)
        if part_losses:
            losses.append(torch.stack(part_losses).mean())
    if not losses:
        return None, None
    loss = torch.stack(losses).mean()
    kl_mean = torch.stack(kl_values).mean() if kl_values else torch.tensor(0.0, device=loss.device)
    return loss, kl_mean


def metric_mean(rollouts: list[dict[str, Any]], key: str) -> float:
    return sum(float(r["reward"].get(key, 0.0)) for r in rollouts) / max(len(rollouts), 1)


def reward_std(rollouts: list[dict[str, Any]]) -> float:
    vals = torch.tensor([float(r["reward"]["total"]) for r in rollouts], dtype=torch.float32)
    return vals.std(unbiased=False).item() if len(vals) else 0.0


def write_rollouts(path: Path, rollouts: list[dict[str, Any]], *, epoch: int, iteration: int, rank: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rollouts:
            clean = {k: v for k, v in row.items() if k not in {"turn1_images", "turn2_images", "image", "global_small"}}
            clean.update({"epoch": epoch, "iteration": iteration, "rank": rank})
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def train(args) -> None:
    distributed, rank, local_rank, world_size, device = init_distributed()
    cfg = Config(**vars(args))
    random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_print(rank, f"[ddp] enabled={distributed} rank={rank}/{world_size} local_rank={local_rank} device={device}")
    rank_print(rank, f"[load] policy={cfg.model_path}")
    processor = AutoProcessor.from_pretrained(cfg.model_path, max_pixels=cfg.max_pixels)
    for tok in SPECIAL_TOKENS:
        if tok not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
            break
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False
    model.train()
    policy_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if distributed else model
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ref_model = None
    if cfg.beta > 0:
        rank_print(rank, f"[load] ref_model beta={cfg.beta}")
        ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        ref_model.resize_token_embeddings(len(processor.tokenizer))
        ref_model.eval()
        ref_model.requires_grad_(False)
        ref_model.config.use_cache = False

    skillbank = RuntimeSkillBank(cfg.skillbank_path)
    dataset = BBoxDataset(cfg.train_jsonl, limit=cfg.data_limit)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed, drop_last=False) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.batch_size_per_device, shuffle=(sampler is None), sampler=sampler, num_workers=0, collate_fn=collate)
    total_updates = max(1, (len(loader) * cfg.num_train_epochs) // max(1, cfg.gradient_accumulation_steps))
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_updates)
    rank_print(rank, f"[sched] updates={total_updates} num_generations={cfg.num_generations} skillbank={skillbank.path}")

    log_f = None
    if is_main(rank):
        log_f = (out_dir / "train.log").open("a", encoding="utf-8")
        log_f.write(f"\n=== geoskill grpo start {datetime.now().isoformat()} world_size={world_size} ===\n")

    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    global_step = 0
    stop_training = False
    rollout_log = Path(cfg.rollout_log_path)
    if distributed:
        rollout_log = rollout_log.with_name(f"{rollout_log.stem}_rank{rank}{rollout_log.suffix}")
    if cfg.save_rollouts and rollout_log.exists():
        rollout_log.unlink()

    for epoch in range(cfg.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for it, batch in enumerate(loader):
            if cfg.max_steps >= 0 and it >= cfg.max_steps:
                stop_training = True
                break
            t0 = time.time()
            rollouts: list[dict[str, Any]] = []
            for row in batch:
                sample = prepare_sample(row, skillbank)
                if sample is None:
                    continue
                rollouts.extend(rollout_one_sample(
                    model,
                    processor,
                    sample,
                    num_generations=cfg.num_generations,
                    turn1_max_new_tokens=cfg.turn1_max_new_tokens,
                    turn2_max_new_tokens=cfg.turn2_max_new_tokens,
                    temperature=cfg.rollout_temperature,
                    top_p=cfg.rollout_top_p,
                    top_k=cfg.rollout_top_k,
                ))
            if cfg.save_rollouts:
                write_rollouts(rollout_log, rollouts, epoch=epoch, iteration=it, rank=rank)
            local_can_update = 1 if (rollouts and reward_std(rollouts) >= 1e-8) else 0
            can_update = torch.tensor(local_can_update, dtype=torch.int32, device=device)
            if distributed:
                dist.all_reduce(can_update, op=dist.ReduceOp.MIN)
            if can_update.item() == 0:
                rank_print(rank, f"[ep{epoch} it{it}] skip no rollouts or no reward variance on at least one rank")
                continue

            loss, kl_mean = grpo_loss(policy_model, ref_model, processor, rollouts, cfg)
            has_loss = torch.tensor(0 if loss is None else 1, dtype=torch.int32, device=device)
            if distributed:
                dist.all_reduce(has_loss, op=dist.ReduceOp.MIN)
            if has_loss.item() == 0:
                rank_print(rank, f"[ep{epoch} it{it}] skip no loss")
                continue

            (loss / cfg.gradient_accumulation_steps).backward()
            accumulated += 1
            if accumulated % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            local = {
                "reward_total": metric_mean(rollouts, "total"),
                "reward_std": reward_std(rollouts),
                "reward_iou": metric_mean(rollouts, "iou"),
                "reward_hit03": metric_mean(rollouts, "hit03"),
                "reward_hit05": metric_mean(rollouts, "hit05"),
                "reward_answer": metric_mean(rollouts, "answer_correct"),
                "reward_format": metric_mean(rollouts, "R_format"),
                "bbox_valid_rate": metric_mean(rollouts, "bbox_valid"),
                "loss": float(loss.detach().item()),
                "kl": float(kl_mean.detach().item()) if kl_mean is not None else 0.0,
                "dt": time.time() - t0,
            }
            reduced = {k: reduce_mean(v, device, distributed) for k, v in local.items()}
            if is_main(rank) and it % cfg.log_steps == 0:
                qids = ",".join(str(x.get("question_id")) for x in batch)
                msg = (
                    f"[ep{epoch} it{it} step{global_step}] qids={qids} "
                    f"total={reduced['reward_total']:.3f} std={reduced['reward_std']:.3f} "
                    f"iou={reduced['reward_iou']:.3f} h03={reduced['reward_hit03']:.2f} "
                    f"h05={reduced['reward_hit05']:.2f} ans={reduced['reward_answer']:.2f} "
                    f"fmt={reduced['reward_format']:.2f} bbox={reduced['bbox_valid_rate']:.2f} "
                    f"loss={reduced['loss']:.4f} kl={reduced['kl']:.5f} dt={reduced['dt']:.1f}s"
                )
                print(msg)
                if log_f:
                    log_f.write(msg + "\n")
                    for gi, row in enumerate(rollouts[: min(4, len(rollouts))]):
                        log_f.write(
                            f"  r{gi} qid={row['question_id']} total={row['reward']['total']:.3f} "
                            f"iou={row['reward']['iou']:.3f} ans={row['reward']['answer_correct']} "
                            f"pred={row['pred_bbox_1024']} gt={row['gt_bbox_1024']} "
                            f"answer={row.get('answer_pred')}\n"
                        )
                    log_f.flush()

            if is_main(rank) and global_step > 0 and global_step % cfg.save_steps == 0 and accumulated % cfg.gradient_accumulation_steps == 0:
                ckpt = out_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt, safe_serialization=True)
                processor.save_pretrained(ckpt)
                print(f"[save] {ckpt}")
        if stop_training:
            break

    barrier(distributed, local_rank)
    if is_main(rank):
        if cfg.skip_final_save:
            print("[done] skip_final_save enabled")
        else:
            final = out_dir / "final_hf"
            final.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final, safe_serialization=True)
            processor.save_pretrained(final)
            print(f"[done] final ckpt -> {final}")
        if log_f:
            log_f.close()
    barrier(distributed, local_rank)
    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name
        default = field.default
        if isinstance(default, bool):
            if default is False:
                parser.add_argument(name, action="store_true", default=default)
            else:
                parser.add_argument("--no_" + field.name, action="store_false", dest=field.name, default=default)
        else:
            parser.add_argument(name, type=type(default), default=default)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
