#!/usr/bin/env python3
"""Prepare GeoSkillRL bbox-only JSONL for slime."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .crop_environment import find_image, resize_image
    from .reward_geo import parse_locator
    from .skillbank_runtime import DEFAULT_SKILLBANK, RuntimeSkillBank
except ImportError:  # pragma: no cover
    from crop_environment import find_image, resize_image
    from reward_geo import parse_locator
    from skillbank_runtime import DEFAULT_SKILLBANK, RuntimeSkillBank


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = THIS_DIR.parent / "stageA/data/splits/rl_train.jsonl"
DEFAULT_OUTPUT = THIS_DIR / "data/slime_train.jsonl"
DEFAULT_CACHE_DIR = THIS_DIR.parent / "verl_grpo/data/cache/global512"

SYSTEM_PROMPT = (
    "You are an intelligent remote sensing analyst. Given a satellite image, a question, "
    "and retrieved grounding skills, you must use exactly one zoom tool before answering:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "The zoom tag must include both <|ref|>...<|/ref|> and <|box|>[[...]]<|/box|>; "
    "<zoom>[[x1,y1,x2,y2]]</zoom> is allowed only for backward-compatible reward audits, "
    "but new rollouts should use the primitive format. Coordinates are normalized to the "
    "0-1024 global image frame. First output one <zoom>; after the crop is shown, output "
    "exactly one <answer>...</answer>. Do not output a second zoom."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    sample_type = str(row.get("type") or "").lower()
    bbox = row.get("bbox")
    if sample_type not in {"object", "region"}:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    image_name = str(row.get("image_name") or row.get("image") or "")
    image_path = find_image(image_name)
    if image_path is None:
        return None
    return {
        "question_id": str(row.get("question_id") or ""),
        "image_name": image_name or image_path.name,
        "image_path": str(image_path),
        "category": str(row.get("category") or ""),
        "higher_level": str(row.get("higher_level") or ""),
        "question": str(row.get("question") or ""),
        "ground_truth": row.get("ground_truth"),
        "bbox": [float(v) for v in bbox[:4]],
        "type": sample_type,
    }


def cache_global_image(image_path: str, cache_dir: Path) -> tuple[str, list[int]]:
    src = Path(image_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{src.stem}.global512.png"
    with Image.open(src) as image:
        image_size = [int(image.width), int(image.height)]
        if not out.exists():
            resize_image(image.convert("RGB"), max_size=512).save(out)
    return str(out), image_size


def stable_index(row: dict[str, Any]) -> int:
    stable_key = f"{row['question_id']}|{row['image_name']}|{row['question']}"
    digest = hashlib.blake2b(stable_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def build_record(
    row: dict[str, Any],
    *,
    index: int,
    skillbank: RuntimeSkillBank,
    cache_dir: Path,
    reward_version: str,
) -> dict[str, Any]:
    global_image, image_size = cache_global_image(row["image_path"], cache_dir)
    skill_info = skillbank.build(row)
    user_content = "<image>\n" + row["question"]
    if skill_info["skill_block"]:
        user_content += "\n\n" + skill_info["skill_block"]
    axes = list(parse_locator(row["question"]).get("axes") or [])
    label = {
        "ground_truth": row["ground_truth"],
        "gt_bbox_1024": row["bbox"],
        "question_id": row["question_id"],
        "image_name": row["image_name"],
        "image_size": image_size,
        "category": row["category"],
    }
    metadata = {
        "index": int(index),
        "question_id": row["question_id"],
        "image_name": row["image_name"],
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "original_image_path": row["image_path"],
        "global_image_path": global_image,
        "gt_bbox_1024": row["bbox"],
        "image_size": image_size,
        "bbox_coord_mode": "max_side_1024",
        "category": row["category"],
        "higher_level": row["higher_level"],
        "type": row["type"],
        "skill_block": skill_info["skill_block"],
        "retrieved_skill_ids": skill_info["retrieved_skill_ids"],
        "reward_version": reward_version,
        "stage1_locator_axes": axes,
        "referent_phrase": row["question"],
    }
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [global_image],
        "label": json.dumps(label, ensure_ascii=False),
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--skillbank", type=Path, default=DEFAULT_SKILLBANK)
    parser.add_argument("--reward-version", default="legacy", choices=["legacy", "strict_v3"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-task-skills", type=int, default=2)
    args = parser.parse_args()

    rows = [r for r in (normalize_row(row) for row in read_jsonl(args.input)) if r is not None]
    rows.sort(key=lambda r: (r["question_id"], r["image_name"], r["question"]))
    if args.limit:
        rows = rows[: args.limit]
    skillbank = RuntimeSkillBank(args.skillbank, max_task=args.max_task_skills)
    records = [
        build_record(
            row,
            index=stable_index(row),
            skillbank=skillbank,
            cache_dir=args.cache_dir,
            reward_version=args.reward_version,
        )
        for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records), "reward_version": args.reward_version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
