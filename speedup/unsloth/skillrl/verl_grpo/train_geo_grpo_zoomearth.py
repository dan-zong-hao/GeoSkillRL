#!/usr/bin/env python3
"""Entry point for corrected GeoSkillRL ZoomEarth GRPO training."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoProcessor, HfArgumentParser, Qwen3_5ForConditionalGeneration, set_seed

from geo_data import DEFAULT_TRAIN_JSONL, GeoBBoxDataset, identity_collator
from geo_grpo_trainer import GeoGRPOConfig, GeoVLMGRPOTrainer


THIS_DIR = Path(__file__).resolve().parent
SKILLRL_DIR = THIS_DIR.parent
DEFAULT_MODEL = Path(
    "/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf"
)
DEFAULT_SKILLBANK = SKILLRL_DIR / "single_zoom/data/geoskill_single_zoom_skillbank_v1.json"
DEFAULT_OUTPUT = THIS_DIR / "output/geoskill_verl_grpo_zoom_only_v1"
SPECIAL_TOKENS = ["<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>"]


@dataclass
class GeoScriptArguments:
    model_path: str = field(default=str(DEFAULT_MODEL))
    train_jsonl: str = field(default=str(DEFAULT_TRAIN_JSONL))
    skillbank_path: str = field(default=str(DEFAULT_SKILLBANK))
    data_limit: int = field(default=0)
    attn_implementation: str = field(default="sdpa")
    skip_final_save: bool = field(default=False)


def ensure_special_tokens(processor) -> None:
    tok = processor.tokenizer
    missing = [token for token in SPECIAL_TOKENS if token not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    processor.pad_token_id = tok.pad_token_id
    processor.eos_token_id = tok.eos_token_id


def main() -> None:
    parser = HfArgumentParser((GeoScriptArguments, GeoGRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if not training_args.output_dir:
        training_args.output_dir = str(DEFAULT_OUTPUT)
    set_seed(training_args.seed)

    processor = AutoProcessor.from_pretrained(
        script_args.model_path,
        max_pixels=training_args.max_pixels,
        min_pixels=training_args.min_pixels,
    )
    ensure_special_tokens(processor)
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        script_args.model_path,
        torch_dtype=dtype,
        attn_implementation=script_args.attn_implementation,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    dataset = GeoBBoxDataset(script_args.train_jsonl, limit=script_args.data_limit)
    if training_args.process_index == 0:
        print(f"[data] bbox samples={len(dataset)} from {script_args.train_jsonl}")
        print(f"[skillbank] {script_args.skillbank_path}")
        print(f"[output] {training_args.output_dir}")
        print(f"[objective] {training_args.objective}, bbox_coord_mode={training_args.bbox_coord_mode}")

    trainer = GeoVLMGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=identity_collator,
        processing_class=processor,
        skillbank_path=script_args.skillbank_path,
    )
    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    if resume_from_checkpoint:
        resume: str | bool = resume_from_checkpoint
        if str(resume).lower() in {"1", "true", "yes"}:
            resume = True
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    if not script_args.skip_final_save:
        final_dir = Path(training_args.output_dir) / "final_hf"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        processor.save_pretrained(final_dir)
        if trainer.is_world_process_zero():
            print(f"[done] final checkpoint -> {final_dir}")
    elif trainer.is_world_process_zero():
        print("[done] skip_final_save enabled")


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "offline")
    main()

