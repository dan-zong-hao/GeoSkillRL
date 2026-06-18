#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_zoom_only import (  # noqa: E402
    ZoomOnlyDispatcher,
    answer_correct,
    apo_iou_512,
    build_model,
    find_image,
    resize_image,
    summarize,
    write_summary,
)

FULL_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
OPEN_ANSWER_RE = re.compile(r"<answer>\s*([^<\n\r]+?)\s*(?:</answer|</|$)", re.DOTALL | re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def robust_answer(row: dict) -> Optional[str]:
    for value in (row.get("final_answer"), row.get("answer2"), row.get("answer1")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = row.get("trajectory") or ""
    m = FULL_ANSWER_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    matches = list(OPEN_ANSWER_RE.finditer(text))
    if matches:
        ans = matches[-1].group(1).strip()
        if ans:
            return ans
    return None


def should_repair(row: dict) -> bool:
    if str(row.get("type", "")).lower() != "global":
        return False
    return robust_answer(row) is None


def make_row(sample: dict, result: dict, model_id: str, image_size: tuple[int, int], scale: float) -> dict:
    bbox_gt_1024 = sample.get("bbox") or None
    bbox_gt_orig = [v * scale for v in bbox_gt_1024] if bbox_gt_1024 else None
    bbox_pred = result.get("bbox_pred")
    bbox_pred_1024 = [v / scale for v in bbox_pred] if bbox_pred else None
    iou_apo, _, _ = apo_iou_512(bbox_pred, bbox_gt_orig, image_size)
    ans = result.get("answer")
    return {
        "question_id": str(sample.get("question_id", "")),
        "ground_truth": sample.get("ground_truth"),
        "category": sample.get("category"),
        "type": sample.get("type"),
        "image": sample.get("image_name"),
        "question": sample.get("question"),
        "bbox_ref": sample.get("bbox") or [],
        "model_id": model_id,
        "actions": result.get("actions") or [],
        "bbox_pred_orig": bbox_pred,
        "bbox_pred_1024": bbox_pred_1024,
        "bbox_gt_orig": bbox_gt_orig,
        "bbox_gt_1024": bbox_gt_1024,
        "iou": round(iou_apo, 6),
        "answer1": ans if not result.get("actions") else None,
        "answer2": ans,
        "final_answer": ans,
        "correct_final": answer_correct(sample.get("ground_truth", ""), ans),
        "grammar_ok": result.get("grammar_ok", False),
        "trajectory": result.get("trajectory", ""),
        "error": result.get("error"),
        "repair_source": "missing_global_token_rerun",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/VQA/speedup/unsloth/sft/only_zoom/output/final_hf")
    parser.add_argument("--test_jsonl", default="/root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl")
    parser.add_argument("--input_predictions", default="/root/autodl-tmp/VQA/speedup/unsloth/eval_no_seg/results/zoom_only_fast_4gpu_predictions.jsonl")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/VQA/speedup/unsloth/eval_no_seg/results")
    parser.add_argument("--tag", default="zoom_only_fast_4gpu_repaired")
    parser.add_argument("--model_id", default="Qwen3.5-ZoomOnly")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--turn1_max_new_tokens", type=int, default=1536)
    parser.add_argument("--turn2_max_new_tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep_failed_old_row", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_predictions)
    output_dir = Path(args.output_dir)
    tag = args.tag
    out_pred = output_dir / f"{tag}_predictions.jsonl"
    repair_pred = output_dir / f"{tag}_repair_only.jsonl"
    meta_path = output_dir / f"{tag}_repair_meta.json"

    old_rows = read_jsonl(input_path)
    repair_ids = [str(r.get("question_id", "")) for r in old_rows if should_repair(r)]
    if args.limit:
        repair_ids = repair_ids[: args.limit]
    repair_set = set(repair_ids)

    print(f"[input] {input_path} rows={len(old_rows)}")
    print(f"[repair] missing-global rows selected={len(repair_ids)}")
    print(f"[cfg] turn1_max_new_tokens={args.turn1_max_new_tokens} turn2_max_new_tokens={args.turn2_max_new_tokens} batch_size={args.batch_size}")
    if not repair_ids:
        write_jsonl(out_pred, old_rows)
        write_summary(summarize(old_rows, tag, args.model), output_dir, tag)
        return 0

    samples_by_id = {}
    with Path(args.test_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                qid = str(sample.get("question_id", ""))
                if qid in repair_set:
                    samples_by_id[qid] = sample

    missing_samples = sorted(repair_set - set(samples_by_id))
    if missing_samples:
        raise SystemExit(f"missing samples in test_jsonl: {missing_samples[:5]} total={len(missing_samples)}")

    model, processor = build_model(args.model)
    dispatcher = ZoomOnlyDispatcher(
        model,
        processor,
        max_new_tokens=args.turn1_max_new_tokens,
        turn1_max_new_tokens=args.turn1_max_new_tokens,
        turn2_max_new_tokens=args.turn2_max_new_tokens,
    )

    repaired = {}
    repair_rows = []
    pending = []
    t0 = time.perf_counter()

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        prepared = [p["prepared"] for p in pending]
        try:
            results = dispatcher.run_batch_turn1(prepared)
        except Exception as exc:
            print(f"[warn] batch failed, falling back per sample: {exc}")
            results = []
            for p in pending:
                try:
                    results.append(dispatcher.run_preprocessed(
                        p["image"], p["question"], p["global_small"], p["scale"]
                    ))
                except Exception as inner:
                    results.append({
                        "actions": [],
                        "bbox_pred": None,
                        "answer": None,
                        "trajectory": "",
                        "grammar_ok": False,
                        "error": f"exception:{inner}",
                    })
        for pack, result in zip(pending, results):
            row = make_row(pack["sample"], result, args.model_id, pack["image_size"], pack["scale"])
            repaired[row["question_id"]] = row
            repair_rows.append(row)
        pending = []

    for qid in tqdm(repair_ids, desc="Repairing missing Global"):
        sample = samples_by_id[qid]
        img_path = find_image(sample.get("image_name", ""))
        if img_path is None:
            row = make_row(sample, {
                "actions": [], "bbox_pred": None, "answer": None,
                "trajectory": "", "grammar_ok": False, "error": "image_not_found",
            }, args.model_id, (0, 0), 1.0)
            repaired[qid] = row
            repair_rows.append(row)
            continue
        image = Image.open(img_path).convert("RGB")
        scale = max(image.size) / 1024.0
        pending.append({
            "sample": sample,
            "image": image,
            "image_size": image.size,
            "scale": scale,
            "question": sample.get("question", ""),
            "global_small": resize_image(image),
            "prepared": {
                "image": image,
                "global_small": resize_image(image),
                "question": sample.get("question", ""),
                "scale": scale,
            },
        })
        if len(pending) >= max(1, args.batch_size):
            flush()
    flush()

    merged = []
    replaced = 0
    recovered = 0
    still_missing = 0
    for old in old_rows:
        qid = str(old.get("question_id", ""))
        if qid in repaired:
            new = repaired[qid]
            if args.keep_failed_old_row and robust_answer(new) is None:
                merged.append(old)
                still_missing += 1
            else:
                merged.append(new)
                replaced += 1
                if robust_answer(new) is None:
                    still_missing += 1
                else:
                    recovered += 1
        else:
            merged.append(old)

    write_jsonl(out_pred, merged)
    write_jsonl(repair_pred, repair_rows)
    summary = summarize(merged, tag, args.model)
    write_summary(summary, output_dir, tag)
    meta = {
        "input_predictions": str(input_path),
        "output_predictions": str(out_pred),
        "repair_only_predictions": str(repair_pred),
        "selected": len(repair_ids),
        "replaced": replaced,
        "recovered_answer": recovered,
        "still_missing_after_rerun": still_missing,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "turn1_max_new_tokens": args.turn1_max_new_tokens,
        "turn2_max_new_tokens": args.turn2_max_new_tokens,
        "batch_size": args.batch_size,
        "table": summary["table"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {out_pred}")
    print(f"[wrote] {repair_pred}")
    print(f"[wrote] {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
