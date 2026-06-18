#!/usr/bin/env python3
"""Mine grounding failures for SkillRL round data.

This is offline-only: it reads existing predictions and writes failure labels
without modifying eval outputs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .spatial import parse_locator, spatial_reward, spatial_violation_type
    from .skillbank import load_skillbank, retrieve_skills
except ImportError:
    from spatial import parse_locator, spatial_reward, spatial_violation_type
    from skillbank import load_skillbank, retrieve_skills

VQA_ROOT = Path("/root/autodl-tmp/VQA")
DEFAULT_PRED = VQA_ROOT / "speedup/unsloth/eval_no_seg/results/zoom_only_fast_4gpu_repaired_predictions.jsonl"
DEFAULT_OUT = VQA_ROOT / "json_data/zoom_json/skillrl/failures_round0.jsonl"

CROP_CLAIM_CORRECT_RE = re.compile(
    r"\[Crop Verification\].{0,420}?"
    r"(?:correctly|does contain|contains the target|crop image contains|"
    r"target[^.\n]{0,100}visible|clearly shows|shows the[^.\n]{0,100}target|"
    r"full [^.\n]{0,100}clearly visible|zoom level sufficient)",
    re.IGNORECASE | re.DOTALL,
)
CROP_CLAIM_FAILED_RE = re.compile(
    r"\[Crop Verification\].{0,420}?"
    r"(?:does not contain|doesn't contain|failed|not contain|no [^.\n]{0,100}visible|"
    r"wrong|different area|crop failed|cannot find|not visible|disregard its content)",
    re.IGNORECASE | re.DOTALL,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or "").lower()


def get_iou(row: dict[str, Any]) -> float:
    try:
        return float(row.get("iou") or row.get("iou_apo512") or 0.0)
    except Exception:
        return 0.0


def has_bbox_pair(row: dict[str, Any]) -> bool:
    return bool(row.get("bbox_gt_1024") and row.get("bbox_pred_1024"))


def crop_claim(row: dict[str, Any]) -> str:
    text = row.get("trajectory") or ""
    pos = bool(CROP_CLAIM_CORRECT_RE.search(text))
    neg = bool(CROP_CLAIM_FAILED_RE.search(text))
    if pos and not neg:
        return "claimed_correct"
    if neg and not pos:
        return "claimed_failed"
    if pos and neg:
        return "mixed"
    return "unknown"


def extract_crop_excerpt(row: dict[str, Any], n: int = 700) -> str:
    text = row.get("trajectory") or ""
    if "[Crop Verification]" in text:
        return text.split("[Crop Verification]", 1)[1][:n].replace("\n", " ")
    return text[-n:].replace("\n", " ")


def classify_failure(row: dict[str, Any], threshold: float, skillbank: list[dict[str, Any]]) -> dict[str, Any] | None:
    if norm_type(row) not in {"object", "region"} or not has_bbox_pair(row):
        return None

    iou = get_iou(row)
    bbox_wrong = iou < threshold
    claim = crop_claim(row)
    locator = parse_locator(row.get("question") or "")
    spatial = spatial_reward(row.get("bbox_pred_1024"), row.get("bbox_gt_1024"), locator)
    skills = retrieve_skills(row.get("question") or "", skillbank)
    crop_claim_correct = claim == "claimed_correct"
    answer_correct = bool(row.get("correct_final"))
    false_grounded_correct = bbox_wrong and crop_claim_correct and answer_correct
    deer_horse = bbox_wrong and crop_claim_correct

    if not bbox_wrong and not spatial["spatial_penalty"] and not deer_horse:
        return None

    failure_types: list[str] = []
    if bbox_wrong:
        failure_types.append("bbox_wrong")
    if spatial["spatial_penalty"]:
        failure_types.append(str(spatial["spatial_violation"]))
    if deer_horse:
        failure_types.append("deer_horse_crop_claim")
    if false_grounded_correct:
        failure_types.append("false_grounded_correct")

    return {
        "question_id": row.get("question_id"),
        "image": row.get("image") or row.get("image_name"),
        "type": row.get("type"),
        "category": row.get("category"),
        "question": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "final_answer": row.get("final_answer"),
        "correct_final": answer_correct,
        "bbox_pred_1024": row.get("bbox_pred_1024"),
        "bbox_gt_1024": row.get("bbox_gt_1024"),
        "iou": round(iou, 6),
        "bbox_wrong": bbox_wrong,
        "crop_claim": claim,
        "crop_claim_correct": crop_claim_correct,
        "false_grounded_correct": false_grounded_correct,
        "deer_horse": deer_horse,
        "locator": locator,
        "spatial": spatial,
        "failure_types": failure_types,
        "retrieved_skill_ids": [s.get("skill_id") for s in skills],
        "crop_verification_excerpt": extract_crop_excerpt(row),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_PRED))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--skillbank", default="")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.predictions))
    if args.limit:
        rows = rows[: args.limit]
    skillbank = load_skillbank(args.skillbank or None)
    failures = [x for row in rows if (x := classify_failure(row, args.threshold, skillbank))]
    write_jsonl(Path(args.output), failures)

    print(f"[input] {args.predictions} rows={len(rows)}")
    print(f"[output] {args.output} failures={len(failures)}")
    print("[failure_types]", dict(Counter(ft for f in failures for ft in f["failure_types"]).most_common(20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
