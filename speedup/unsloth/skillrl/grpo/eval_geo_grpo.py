#!/usr/bin/env python3
"""Evaluate GeoSkillRL GRPO checkpoints on bbox RL-dev with live SkillBank."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from rollout_env import load_bbox_samples, prepare_sample, rollout_one_sample  # noqa: E402
from skillbank_runtime import DEFAULT_SKILLBANK, RuntimeSkillBank  # noqa: E402


DEFAULT_MODEL = Path(
    "/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf"
)
DEFAULT_DEV = THIS_DIR.parent / "stageA/data/splits/rl_dev.jsonl"


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def summarize(rows: list[dict[str, Any]], *, tag: str, model: str, eval_jsonl: str) -> dict[str, Any]:
    n = len(rows)
    reward_rows = [r["reward"] for r in rows]
    return {
        "tag": tag,
        "model": model,
        "eval_jsonl": eval_jsonl,
        "n_samples": n,
        "table": {
            "Answer Acc": pct(sum(1 for r in reward_rows if r.get("answer_correct")), n),
            "APO IoU": pct(sum(float(r.get("iou") or 0.0) for r in reward_rows), n),
            "Hit@0.3": pct(sum(1 for r in reward_rows if float(r.get("iou") or 0.0) >= 0.3), n),
            "Hit@0.5": pct(sum(1 for r in reward_rows if float(r.get("iou") or 0.0) >= 0.5), n),
            "BBox Pred Rate": pct(sum(1 for r in reward_rows if r.get("pred_bbox_1024")), n),
            "Zoom Parse Fail": pct(sum(1 for r in reward_rows if not r.get("zoom_parse_ok")), n),
            "Answer Parse Fail": pct(sum(1 for r in reward_rows if not r.get("answer_parse_ok")), n),
            "Mean Reward": sum(float(r.get("total") or 0.0) for r in reward_rows) / max(n, 1),
        },
        "reward_means": {
            key: sum(float(r.get(key) or 0.0) for r in reward_rows) / max(n, 1)
            for key in [
                "R_task_success",
                "R_bbox_region",
                "R_grounded_answer",
                "R_format",
                "R_spatial",
                "P_parse",
                "P_area",
                "P_false_grounded",
                "total",
            ]
        },
        "counts": {
            "by_type": dict(Counter(str(r.get("type") or "unknown").lower() for r in rows)),
            "retrieved_skill_ids": dict(Counter(
                sid for r in rows for sid in (r.get("retrieved_skill_ids") or [])
            ).most_common(50)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--eval_jsonl", default=str(DEFAULT_DEV))
    parser.add_argument("--skillbank_path", default=str(DEFAULT_SKILLBANK))
    parser.add_argument("--output_dir", default=str(THIS_DIR / "results/rl_dev_grpo_skillbank_v1"))
    parser.add_argument("--tag", default="grpo_skillbank_v1_rl_dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--turn1_max_new_tokens", type=int, default=512)
    parser.add_argument("--turn2_max_new_tokens", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=64 * 64 * 28 * 28)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{args.tag}_predictions.jsonl"
    summary_path = out_dir / f"{args.tag}_summary.json"

    rows = load_bbox_samples(args.eval_jsonl, limit=args.limit)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_id]
    print(json.dumps({
        "model": args.model,
        "eval_jsonl": args.eval_jsonl,
        "bbox_samples": len(rows),
        "skillbank_path": args.skillbank_path,
        "output": str(pred_path),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
    }, ensure_ascii=False, indent=2))
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    skillbank = RuntimeSkillBank(args.skillbank_path)

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with pred_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="grpo eval"):
            sample = prepare_sample(row, skillbank)
            if sample is None:
                continue
            try:
                rollout = rollout_one_sample(
                    model,
                    processor,
                    sample,
                    num_generations=1,
                    turn1_max_new_tokens=args.turn1_max_new_tokens,
                    turn2_max_new_tokens=args.turn2_max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    do_sample=False,
                )[0]
            except Exception as exc:  # noqa: BLE001
                rollout = {
                    "question_id": row.get("question_id"),
                    "image_name": row.get("image_name") or row.get("image"),
                    "type": row.get("type"),
                    "category": row.get("category"),
                    "higher_level": row.get("higher_level"),
                    "question": row.get("question"),
                    "ground_truth": row.get("ground_truth"),
                    "trajectory": "",
                    "retrieved_skill_ids": [],
                    "reward": {
                        "total": -1.5,
                        "iou": 0.0,
                        "answer_correct": False,
                        "zoom_parse_ok": False,
                        "answer_parse_ok": False,
                    },
                    "error": f"exception:{exc}",
                }
            f.write(json.dumps({k: v for k, v in rollout.items() if k not in {"turn1_images", "turn2_images"}}, ensure_ascii=False) + "\n")
            results.append(rollout)
            f.flush()
    summary = summarize(results, tag=args.tag, model=args.model, eval_jsonl=args.eval_jsonl)
    summary["elapsed_sec"] = time.perf_counter() - t0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = summary["table"]
    line = (
        f"Answer={table['Answer Acc']:.2f} APO_IoU={table['APO IoU']:.2f} "
        f"Hit03={table['Hit@0.3']:.2f} Hit05={table['Hit@0.5']:.2f} "
        f"BBoxPred={table['BBox Pred Rate']:.2f} ZoomParseFail={table['Zoom Parse Fail']:.2f} "
        f"AnsParseFail={table['Answer Parse Fail']:.2f} MeanReward={table['Mean Reward']:.3f}"
    )
    (out_dir / f"{args.tag}_table.txt").write_text(line + "\n", encoding="utf-8")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
