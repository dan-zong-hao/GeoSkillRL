#!/usr/bin/env python3
"""Validate verl-format bbox-only ZoomEarth parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"data_source", "prompt", "images", "agent_name", "reward_model", "extra_info"}


def as_list(value):
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def row_signature(extra: dict) -> tuple[str, str, str]:
    image_name = str(extra.get("image_name") or Path(str(extra.get("original_image_path") or "")).name)
    return str(extra.get("question_id") or ""), image_name, str(extra.get("question") or "")


def signature_index_map(df: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    mapping = {}
    for _, row in df.iterrows():
        extra = row["extra_info"]
        if not isinstance(extra, dict):
            raise SystemExit("reference has invalid extra_info")
        mapping[row_signature(extra)] = int(extra["index"])
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    indices = []
    signatures = []
    for i, row in df.iterrows():
        prompt = as_list(row["prompt"])
        images = as_list(row["images"])
        extra = row["extra_info"]
        if not isinstance(prompt, list) or len(prompt) < 2:
            raise SystemExit(f"row {i}: invalid prompt")
        if not isinstance(images, list) or len(images) != 1:
            raise SystemExit(f"row {i}: expected exactly one global image")
        if not isinstance(extra, dict):
            raise SystemExit(f"row {i}: invalid extra_info")
        indices.append(int(extra["index"]))
        signatures.append(row_signature(extra))
        if extra.get("type") not in {"object", "region"}:
            raise SystemExit(f"row {i}: non bbox type {extra.get('type')}")
        bbox = as_list(extra.get("gt_bbox_1024"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise SystemExit(f"row {i}: invalid gt bbox")
        if any(k in extra for k in ("label", "label_cleaned")):
            raise SystemExit(f"row {i}: external label leaked into extra_info")
    if len(indices) != len(set(indices)):
        raise SystemExit("extra_info.index is not unique")
    if indices != sorted(indices) or indices != list(range(len(indices))):
        raise SystemExit("extra_info.index must be stable contiguous order")
    if signatures != sorted(signatures):
        raise SystemExit("rows are not sorted by stable source signature")
    if args.reference is not None:
        ref_df = pd.read_parquet(args.reference)
        ref_map = signature_index_map(ref_df)
        cur_map = signature_index_map(df)
        overlap = set(cur_map) & set(ref_map)
        drift = [sig for sig in sorted(overlap) if cur_map[sig] != ref_map[sig]]
        if drift:
            raise SystemExit(f"extra_info.index drift for {len(drift)} shared rows; first={drift[0]!r}")

    stats = {"rows": len(df), "min_index": min(indices) if indices else None, "max_index": max(indices) if indices else None}
    if args.model_path is not None:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        lengths = []
        for _, row in df.iterrows():
            messages = as_list(row["prompt"])
            # Mirror RLHFDataset placeholder replacement for one global image.
            image_payload = dict(as_list(row["images"])[0])
            for msg in messages:
                if isinstance(msg.get("content"), str) and "<image>" in msg["content"]:
                    msg["content"] = [
                        {"type": "image", **image_payload},
                        {"type": "text", "text": msg["content"].replace("<image>", "", 1)},
                    ]
            raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images, _videos = process_vision_info(messages, image_patch_size=16)
            encoded = processor(text=[raw_prompt], images=images, return_tensors="pt")
            length = int(encoded["input_ids"].shape[-1])
            if length > args.max_prompt_length:
                raise SystemExit(f"prompt length {length} exceeds {args.max_prompt_length}")
            lengths.append(length)
        stats["prompt_len_max"] = max(lengths) if lengths else 0
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
