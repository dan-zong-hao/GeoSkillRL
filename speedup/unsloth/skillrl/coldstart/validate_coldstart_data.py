#!/usr/bin/env python3
"""Validate train-safe cold-start VLM-style SFT rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TRAIN = Path(
    "/root/autodl-tmp/VQA/json_data/zoom_json/rl/"
    "rl_train_vlm_style_reasoning_skilltrigger_v2.jsonl"
)
DEFAULT_TRAIN_QIDS = Path(
    "/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/"
    "rl_train_qids.json"
)
DEFAULT_DEV_QIDS = Path(
    "/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/"
    "rl_dev_qids.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", default=str(DEFAULT_TRAIN))
    parser.add_argument("--train_qids", default=str(DEFAULT_TRAIN_QIDS))
    parser.add_argument("--dev_qids", default=str(DEFAULT_DEV_QIDS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.train_jsonl).read_text().splitlines()
        if line.strip()
    ]
    train_qids = set(map(str, json.loads(Path(args.train_qids).read_text())))
    dev_qids = set(map(str, json.loads(Path(args.dev_qids).read_text())))
    qids = {str(row["question_id"]) for row in rows}

    bad: list[tuple[str, str]] = []
    bbox_rows = 0
    global_rows = 0
    triggered_bbox = 0
    for row in rows:
        qid = str(row.get("question_id"))
        text = row.get("assistant_text") or ""
        bbox = row.get("bbox")
        typ = str(row.get("type") or "").lower()
        is_bbox = isinstance(bbox, list) and len(bbox) == 4 and typ in {"object", "region"}

        if text.count("<answer>") != 1 or text.count("</answer>") != 1:
            bad.append((qid, "answer_count"))
        if "<seg>" in text or "</seg>" in text:
            bad.append((qid, "seg_tag"))
        if is_bbox:
            bbox_rows += 1
            if text.count("<zoom>") != 1 or text.count("</zoom>") != 1:
                bad.append((qid, "zoom_count"))
            if len(row.get("tool_bboxes_1024") or []) != 1:
                bad.append((qid, "tool_bbox_count"))
            if "Triggered by:" in (row.get("skill_block") or ""):
                triggered_bbox += 1
        else:
            global_rows += 1
            if "<zoom>" in text or "</zoom>" in text:
                bad.append((qid, "global_zoom"))
            if row.get("tool_bboxes_1024"):
                bad.append((qid, "global_tool_bbox"))

    report = {
        "rows": len(rows),
        "unique_qids": len(qids),
        "train_qids": len(train_qids),
        "dev_qids": len(dev_qids),
        "not_train": len(qids - train_qids),
        "dev_overlap": len(qids & dev_qids),
        "bbox_rows": bbox_rows,
        "global_rows": global_rows,
        "bbox_rows_with_triggered_by": triggered_bbox,
        "bad_count": len(bad),
        "bad_examples": bad[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["not_train"] or report["dev_overlap"] or bad:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
