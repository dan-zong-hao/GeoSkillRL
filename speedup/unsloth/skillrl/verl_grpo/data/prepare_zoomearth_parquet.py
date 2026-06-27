#!/usr/bin/env python3
"""Prepare verl-format parquet for bbox-only ZoomEarth GRPO."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.crop_environment import find_image, resize_image  # noqa: E402
from agent.skill_retriever import DEFAULT_SKILLBANK, SkillRetriever  # noqa: E402


DEFAULT_INPUT = Path("/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl")
DEFAULT_OUT = ROOT / "data/parquet/train.parquet"

SYSTEM_PROMPT = (
    "You are an intelligent remote sensing analyst. Given a satellite image, a question, "
    "and retrieved grounding skills, you must use exactly one zoom tool before answering:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 global image frame. First output one <zoom>; "
    "after the crop is shown, output exactly one <answer>...</answer>. Do not output a second zoom."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def _question_from_swift_messages(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    content = str(messages[0].get("content") or "")
    content = content.replace("<image>", "").strip()
    return content.split("\n\nFirst locate", 1)[0].strip()


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if "gt_bbox_1024" in row:
        bbox = row.get("gt_bbox_1024")
        image_name = row.get("image_name")
        image_path = (row.get("images") or [None])[0] if isinstance(row.get("images"), list) else None
        question = _question_from_swift_messages(row)
        sample_type = "object"
    else:
        bbox = row.get("bbox")
        image_name = row.get("image_name") or row.get("image")
        image_path = None
        question = str(row.get("question") or "")
        sample_type = str(row.get("type") or "").lower()
    if sample_type not in {"object", "region"}:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    resolved = Path(str(image_path)) if image_path and Path(str(image_path)).exists() else find_image(str(image_name or ""))
    if resolved is None:
        return None
    return {
        "question_id": str(row.get("question_id") or ""),
        "image_name": str(image_name or resolved.name),
        "image_path": str(resolved),
        "category": str(row.get("category") or ""),
        "question": question,
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
            image = image.convert("RGB")
            resize_image(image, max_size=512).save(out)
    return str(out), image_size


def stable_index(row: dict[str, Any]) -> int:
    stable_key = f"{row['question_id']}|{row['image_name']}|{row['question']}"
    digest = hashlib.blake2b(stable_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def build_record(row: dict[str, Any], index: int, retriever: SkillRetriever, cache_dir: Path) -> dict[str, Any]:
    global_image, image_size = cache_global_image(row["image_path"], cache_dir)
    skill_info = retriever.build(row["question"])
    user_content = "<image>\n" + row["question"]
    if skill_info["skill_block"]:
        user_content += "\n\n" + skill_info["skill_block"]
    ground_truth = {
        "ground_truth": row["ground_truth"],
        "gt_bbox_1024": row["bbox"],
        "question_id": row["question_id"],
        "image_name": row["image_name"],
        "image_size": image_size,
        "category": row["category"],
    }
    return {
        "data_source": "geoskillrl_zoomearth_bbox_only",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [{"image": global_image}],
        "agent_name": "zoomearth_full_agent",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "index": int(index),
            "question_id": row["question_id"],
            "image_name": row["image_name"],
            "question": row["question"],
            "original_image_path": row["image_path"],
            "global_image_path": global_image,
            "gt_bbox_1024": row["bbox"],
            "image_size": image_size,
            "bbox_coord_mode": "max_side",
            "category": row["category"],
            "type": row["type"],
            "skill_block": skill_info["skill_block"],
            "retrieved_skill_ids": skill_info["retrieved_skill_ids"],
            "skill_retrieval_key": skill_info["skill_retrieval_key"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skillbank", type=Path, default=DEFAULT_SKILLBANK)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/global512")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-skillbank", action="store_true")
    args = parser.parse_args()

    rows = [r for r in (normalize_row(row) for row in read_jsonl(args.input)) if r is not None]
    rows.sort(key=lambda r: (r["question_id"], r["image_name"], r["question"]))
    if args.limit:
        rows = rows[: args.limit]
    retriever = SkillRetriever(args.skillbank, enabled=not args.no_skillbank)
    records = [
        build_record(row, index=stable_index(row), retriever=retriever, cache_dir=args.cache_dir)
        for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
