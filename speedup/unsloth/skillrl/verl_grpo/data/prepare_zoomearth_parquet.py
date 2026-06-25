#!/usr/bin/env python3
"""Convert LRS-GRO JSONL to verl RLHFDataset-compatible Parquet."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
SKILLRL_DIR = Path(__file__).resolve().parents[1].parent
SINGLE_ZOOM_DIR = SKILLRL_DIR / "single_zoom"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SINGLE_ZOOM_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_ZOOM_DIR))

from skillbank_single_zoom import format_skill_block, load_skillbank, retrieve_skills, skill_ids, validate_no_rezoom  # noqa: E402
from speedup.unsloth.skillrl.verl_grpo.agent.crop_environment import (  # noqa: E402
    find_image,
    save_global_cache_with_size,
    sha256_jsonable,
)
from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import validate_bbox_1024  # noqa: E402


DATA_SOURCE = "geoskillrl/zoomearth_bbox"
DEFAULT_TRAIN_JSONL = SKILLRL_DIR / "stageA/data/splits/rl_train.jsonl"
DEFAULT_DEV_JSONL = SKILLRL_DIR / "stageA/data/splits/rl_dev.jsonl"
DEFAULT_SKILLBANK = SKILLRL_DIR / "single_zoom/data/geoskill_single_zoom_skillbank_v1.json"

SYSTEM_PROMPT = (
    "You are an intelligent remote sensing analyst. Given a satellite image, a question, "
    "and retrieved grounding skills, output exactly one zoom tool call before answering:\n"
    "<zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 global image frame. The bbox must describe "
    "the target object or region, not an expanded crop."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_candidate(row: dict[str, Any]) -> bool:
    if str(row.get("type") or "").lower() not in {"object", "region"}:
        return False
    return validate_bbox_1024(row.get("bbox"), require_in_range=True) is not None


def build_prompt(question: str, skill_block: str) -> list[dict[str, str]]:
    user_content = f"<image>\nQuestion: {question.strip()}"
    if skill_block:
        user_content += f"\n\n{skill_block}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _safe_cache_name(question_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(question_id))
    return clean or "sample"


def _image_cache_name(image_path: Path) -> str:
    stem = _safe_cache_name(image_path.stem)[:80]
    digest = sha256_jsonable({"image_path": str(image_path.resolve())})[:16]
    return f"{stem}-{digest}.png"


def convert_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    image_dirs: list[str],
    skillbank: dict[str, Any],
    coord_mode: str,
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "filtered_not_object_region": 0,
        "filtered_invalid_bbox": 0,
        "missing_images": 0,
        "duplicate_question_ids": 0,
        "type_distribution": Counter(),
        "bbox_min": [None, None, None, None],
        "bbox_max": [None, None, None, None],
    }
    seen_qids: set[str] = set()
    global_cache_dir = output_dir / "global_512_cache"

    for row in rows:
        row_type = str(row.get("type") or "").lower()
        if row_type not in {"object", "region"}:
            stats["filtered_not_object_region"] += 1
            continue
        bbox = validate_bbox_1024(row.get("bbox"), require_in_range=True)
        if bbox is None:
            stats["filtered_invalid_bbox"] += 1
            continue

        image_name = str(row.get("image_name") or row.get("image") or "")
        image_path = find_image(image_name, image_dirs)
        if image_path is None:
            stats["missing_images"] += 1
            continue

        qid = str(row.get("question_id") or f"{split}-{len(records)}")
        if qid in seen_qids:
            stats["duplicate_question_ids"] += 1
            continue
        seen_qids.add(qid)

        question = str(row.get("question") or "")
        skills = retrieve_skills(question, "", skillbank=skillbank)
        retrieved_ids = skill_ids(skills)
        skill_block = format_skill_block(skills)

        cache_path = global_cache_dir / _image_cache_name(image_path)
        _, image_size_tuple = save_global_cache_with_size(image_path, cache_path, max_side=512)
        image_size = [int(image_size_tuple[0]), int(image_size_tuple[1])]

        extra_info = {
            "index": start_index + len(records),
            "question_id": qid,
            "question": question,
            "original_image_path": str(image_path),
            "image_name": image_name,
            "image_size": image_size,
            "gt_bbox_1024": [float(v) for v in bbox],
            "type": row_type,
            "category": row.get("category"),
            "higher_level": row.get("higher_level"),
            "retrieved_skill_ids": retrieved_ids,
            "bbox_coord_mode": coord_mode,
        }
        records.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": build_prompt(question, skill_block),
                "images": [str(cache_path.resolve())],
                "agent_name": "zoomearth_zoom_agent" if split == "train" else "zoomearth_full_agent",
                "reward_model": {"style": "rule", "ground_truth": str(row.get("ground_truth") or "")},
                "extra_info": extra_info,
            }
        )
        stats["type_distribution"][row_type] += 1
        for i, value in enumerate(bbox):
            stats["bbox_min"][i] = value if stats["bbox_min"][i] is None else min(stats["bbox_min"][i], value)
            stats["bbox_max"][i] = value if stats["bbox_max"][i] is None else max(stats["bbox_max"][i], value)

    stats["kept_rows"] = len(records)
    stats["type_distribution"] = dict(stats["type_distribution"])
    return records, stats


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - exercised in deployment env
        raise RuntimeError("pandas/pyarrow are required to write Parquet") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def build_manifest(train_records: list[dict[str, Any]], dev_records: list[dict[str, Any]], stats: dict[str, Any]) -> dict:
    train_qids = {r["extra_info"]["question_id"] for r in train_records}
    dev_qids = {r["extra_info"]["question_id"] for r in dev_records}
    train_images = {r["extra_info"]["original_image_path"] for r in train_records}
    dev_images = {r["extra_info"]["original_image_path"] for r in dev_records}
    manifest = {
        "data_source": DATA_SOURCE,
        "samples": {"train": len(train_records), "dev": len(dev_records), "total": len(train_records) + len(dev_records)},
        "stats": stats,
        "leakage": {
            "question_id_overlap": sorted(train_qids.intersection(dev_qids)),
            "image_path_overlap": sorted(train_images.intersection(dev_images)),
        },
        "data_sha256": sha256_jsonable({"train": train_records, "dev": dev_records}),
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL))
    parser.add_argument("--dev-jsonl", default=str(DEFAULT_DEV_JSONL))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data/processed"))
    parser.add_argument("--image-dir", action="append", default=[])
    parser.add_argument("--skillbank", default=str(DEFAULT_SKILLBANK))
    parser.add_argument("--coord-mode", choices=["max_side", "xy"], default="max_side")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    skillbank = load_skillbank(args.skillbank)
    validate_no_rezoom(skillbank)

    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_dev:
        dev_rows = dev_rows[: args.limit_dev]

    train_records, train_stats = convert_split(
        train_rows,
        split="train",
        output_dir=output_dir,
        image_dirs=args.image_dir,
        skillbank=skillbank,
        coord_mode=args.coord_mode,
    )
    dev_records, dev_stats = convert_split(
        dev_rows,
        split="dev",
        output_dir=output_dir,
        image_dirs=args.image_dir,
        skillbank=skillbank,
        coord_mode=args.coord_mode,
        start_index=len(train_records),
    )

    write_parquet(train_records, output_dir / "train.parquet")
    write_parquet(dev_records, output_dir / "val.parquet")
    manifest = build_manifest(train_records, dev_records, {"train": train_stats, "dev": dev_stats})
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["samples"], ensure_ascii=False))
    if manifest["leakage"]["question_id_overlap"]:
        raise SystemExit("train/dev question_id leakage detected; see manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
