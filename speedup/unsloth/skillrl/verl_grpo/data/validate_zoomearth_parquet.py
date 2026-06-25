#!/usr/bin/env python3
"""Validate GeoSkillRL ZoomEarth Parquet files for verl RLHFDataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"data_source", "prompt", "images", "agent_name", "reward_model", "extra_info"}


def as_list(value: Any) -> list[Any] | Any:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return value


def read_table(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def validate_record(row: dict[str, Any], idx: int) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS.difference(row.keys())
    if missing:
        errors.append(f"{idx}: missing columns {sorted(missing)}")
        return errors
    prompt = as_list(row["prompt"])
    images = as_list(row["images"])
    extra = row["extra_info"]
    if not isinstance(prompt, list) or len(prompt) < 2:
        errors.append(f"{idx}: prompt is not chat list")
    else:
        image_placeholders = sum(str(m.get("content", "")).count("<image>") for m in prompt if isinstance(m, dict))
        if image_placeholders != len(images):
            errors.append(f"{idx}: image placeholder count {image_placeholders} != images {len(images)}")
    if not images or not all(Path(p).exists() for p in images):
        errors.append(f"{idx}: image path missing")
    if not isinstance(extra, dict):
        errors.append(f"{idx}: extra_info is not dict")
    else:
        for key in ("index", "question_id", "original_image_path", "image_size", "gt_bbox_1024"):
            if key not in extra:
                errors.append(f"{idx}: extra_info missing {key}")
        if "original_image_path" in extra and not Path(extra["original_image_path"]).exists():
            errors.append(f"{idx}: original_image_path missing")
    return errors


def validate(path: Path) -> dict[str, Any]:
    df = read_table(path)
    errors: list[str] = []
    qids: list[str] = []
    type_counter: Counter[str] = Counter()
    for idx, row in df.iterrows():
        item = row.to_dict()
        errors.extend(validate_record(item, int(idx)))
        extra = item.get("extra_info") or {}
        if isinstance(extra, dict):
            qids.append(str(extra.get("question_id")))
            type_counter[str(extra.get("type"))] += 1
    duplicates = sorted([qid for qid, count in Counter(qids).items() if count > 1])
    if duplicates:
        errors.append(f"duplicate question_id: {duplicates[:20]}")
    return {
        "path": str(path),
        "rows": int(len(df)),
        "type_distribution": dict(type_counter),
        "duplicate_question_ids": duplicates,
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    reports = [validate(Path(p)) for p in args.parquet]
    payload = {"reports": reports, "ok": all(r["ok"] for r in reports)}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
