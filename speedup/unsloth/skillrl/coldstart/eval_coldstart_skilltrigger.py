#!/usr/bin/env python3
"""Skill-triggered RL-dev evaluation for cold-start checkpoints.

This mirrors ``speedup/unsloth/eval_no_seg/eval_zoom_only.py``:
  * Turn 1 sees the global image and may output ``<zoom>`` or ``<answer>``.
  * If Turn 1 zooms, Turn 2 receives the global image plus the crop image and
    must output ``<answer>``.

Unlike the old single-zoom evaluator, this evaluates final answer accuracy as
well as APO IoU / Hit@0.3 / Hit@0.5 / parse rates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


VQA_ROOT = Path("/root/autodl-tmp/VQA")
THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = VQA_ROOT / "speedup/unsloth/geoskillrl_apo_experiments"
SINGLE_ZOOM_DIR = THIS_DIR.parent / "single_zoom"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SINGLE_ZOOM_DIR))

import eval_zoom_only_primitive as base_eval  # noqa: E402
from skillbank_single_zoom import format_skill_block, load_skillbank, retrieve_skills, skill_ids  # noqa: E402


DEFAULT_CURRENT_MODEL = (
    VQA_ROOT
    / "speedup/unsloth/geoskillrl_apo_experiments/output"
    / "no_seg_vlm_rewrite_primitive/final_hf"
)
DEFAULT_COLDSTART_MODEL = THIS_DIR / "output/geoskill_coldstart_skilltrigger_v2/final_hf"
DEFAULT_EVAL_JSONL = THIS_DIR.parent / "stageA/data/splits/rl_dev.jsonl"
DEFAULT_SKILLBANK = SINGLE_ZOOM_DIR / "data/geoskill_single_zoom_skillbank_v1.json"
DEFAULT_OUT = THIS_DIR / "results/rl_dev_skilltrigger"

SYSTEM_PROMPT_COLDSTART_EVAL = (
    "You are an intelligent remote sensing analyst. Given a satellite image, "
    "a question, and optional retrieved grounding skills, answer using this "
    "protocol. For object or region questions, first reason over the global "
    "image, then choose exactly one precise grounding zoom for the target:\n"
    "  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>\n"
    "Coordinates are normalized to the 0-1024 global image frame and must "
    "describe the target referent, not a broad context crop. After the zoom, "
    "inspect the provided crop image and answer from crop evidence. For global "
    "questions where the whole image is sufficient, skip zoom. Always wrap "
    "reasoning in <think>...</think> and end with exactly one "
    "<answer>...</answer>. Do not mention skill ids in the final answer, do "
    "not call <seg>, and do not output more than one zoom."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_bbox_sample(sample: dict[str, Any]) -> bool:
    bbox = sample.get("bbox")
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and str(sample.get("type") or "").lower() in {"object", "region"}
    )


def skill_block_for_sample(sample: dict[str, Any], skillbank: dict[str, Any]) -> tuple[str, list[str]]:
    if not is_bbox_sample(sample):
        return "", []
    skills = retrieve_skills(
        str(sample.get("question") or ""),
        str(sample.get("label") or sample.get("question") or ""),
        skillbank=skillbank,
        max_task=2,
    )
    return format_skill_block(skills), skill_ids(skills)


def question_with_skills(sample: dict[str, Any], skillbank: dict[str, Any]) -> tuple[str, str, list[str]]:
    question = str(sample.get("question") or "")
    block, ids = skill_block_for_sample(sample, skillbank)
    if block:
        return question + "\n\n" + block, block, ids
    return question, "", []


class SkillTriggeredDispatcher(base_eval.ZoomOnlyDispatcher):
    def __init__(self, *args, skillbank: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.skillbank = skillbank

    def prepare_question(self, sample: dict[str, Any]) -> tuple[str, str, list[str]]:
        return question_with_skills(sample, self.skillbank)


def row_from_result(pack: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    sample = pack["sample"]
    base = pack["base"]
    scale = pack["scale"]
    image_size = pack["image_size"]
    bbox_gt_1024 = sample.get("bbox") if is_bbox_sample(sample) else None
    bbox_gt_orig = [v * scale for v in bbox_gt_1024] if bbox_gt_1024 else None
    bbox_pred_1024 = [v / scale for v in result["bbox_pred"]] if result.get("bbox_pred") else None
    iou_apo, _, _ = base_eval.apo_iou_512(result.get("bbox_pred"), bbox_gt_orig, image_size)
    ans = result.get("answer")
    error = result.get("error")
    zoom_parse_fail = bool(is_bbox_sample(sample) and (error == "zoom_parse_fail" or not bbox_pred_1024))
    answer_parse_fail = ans is None
    return {
        **base,
        "actions": result.get("actions") or [],
        "bbox_pred_orig": result.get("bbox_pred"),
        "bbox_pred_1024": bbox_pred_1024,
        "bbox_gt_orig": bbox_gt_orig,
        "bbox_gt_1024": bbox_gt_1024,
        "iou": round(float(iou_apo), 6),
        "hit_0_3": bool(iou_apo >= 0.3) if bbox_gt_1024 else None,
        "hit_0_5": bool(iou_apo >= 0.5) if bbox_gt_1024 else None,
        "answer1": ans if not result.get("actions") else None,
        "answer2": ans,
        "final_answer": ans,
        "correct_final": base_eval.answer_correct(str(sample.get("ground_truth") or ""), ans),
        "grammar_ok": bool(result.get("grammar_ok")),
        "zoom_parse_fail": zoom_parse_fail,
        "answer_parse_fail": answer_parse_fail,
        "trajectory": result.get("trajectory") or "",
        "error": error,
        "skill_block": pack.get("skill_block") or "",
        "retrieved_skill_ids": pack.get("retrieved_skill_ids") or [],
    }


def exception_row(pack: dict[str, Any], exc: Exception) -> dict[str, Any]:
    sample = pack["sample"]
    base = pack["base"]
    bbox_gt_1024 = sample.get("bbox") if is_bbox_sample(sample) else None
    return {
        **base,
        "actions": [],
        "bbox_pred_orig": None,
        "bbox_pred_1024": None,
        "bbox_gt_orig": None,
        "bbox_gt_1024": bbox_gt_1024,
        "iou": 0.0,
        "hit_0_3": False if bbox_gt_1024 else None,
        "hit_0_5": False if bbox_gt_1024 else None,
        "answer1": None,
        "answer2": None,
        "final_answer": None,
        "correct_final": False,
        "grammar_ok": False,
        "zoom_parse_fail": bool(bbox_gt_1024),
        "answer_parse_fail": True,
        "trajectory": "",
        "error": f"exception:{exc}",
        "skill_block": pack.get("skill_block") or "",
        "retrieved_skill_ids": pack.get("retrieved_skill_ids") or [],
    }


def summarize_rows(rows: list[dict[str, Any]], tag: str, model_path: str, eval_jsonl: str) -> dict[str, Any]:
    n = len(rows)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("type") or "unknown").lower()].append(row)

    bbox_rows = [row for row in rows if row.get("bbox_gt_1024")]

    def pct(num: float, den: float) -> float:
        return 100.0 * num / max(den, 1.0)

    def acc(items: list[dict[str, Any]]) -> float:
        return pct(sum(1 for row in items if row.get("correct_final")), len(items))

    table = {
        "Global Acc": acc(by_type.get("global", [])),
        "Region Acc": acc(by_type.get("region", [])),
        "Object Acc": acc(by_type.get("object", [])),
        "Answer Acc": acc(rows),
        "APO IoU": pct(sum(float(row.get("iou") or 0.0) for row in bbox_rows), len(bbox_rows)),
        "Hit@0.3": pct(sum(1 for row in bbox_rows if row.get("hit_0_3")), len(bbox_rows)),
        "Hit@0.5": pct(sum(1 for row in bbox_rows if row.get("hit_0_5")), len(bbox_rows)),
        "BBox Pred Rate": pct(sum(1 for row in bbox_rows if row.get("bbox_pred_1024")), len(bbox_rows)),
        "Zoom Parse Fail": pct(sum(1 for row in bbox_rows if row.get("zoom_parse_fail")), len(bbox_rows)),
        "Answer Parse Fail": pct(sum(1 for row in rows if row.get("answer_parse_fail")), n),
        "Grammar OK": pct(sum(1 for row in rows if row.get("grammar_ok")), n),
        "Zoom Fired": pct(sum(1 for row in rows if "zoom" in (row.get("actions") or [])), n),
    }
    return {
        "tag": tag,
        "model": model_path,
        "eval_jsonl": eval_jsonl,
        "n_samples": n,
        "bbox_samples": len(bbox_rows),
        "table": table,
        "counts": {
            "by_type": dict(Counter(str(row.get("type") or "unknown").lower() for row in rows)),
            "actions": dict(Counter("zoom" if "zoom" in (row.get("actions") or []) else "none" for row in rows)),
            "errors": dict(Counter(str(row.get("error") or "none") for row in rows)),
            "retrieved_skill_ids": dict(Counter(
                sid for row in rows for sid in (row.get("retrieved_skill_ids") or [])
            ).most_common(50)),
        },
    }


def write_summary(summary: dict[str, Any], out: Path, tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    t = summary["table"]
    line = (
        f"Answer={t['Answer Acc']:.2f} Global={t['Global Acc']:.2f} "
        f"Region={t['Region Acc']:.2f} Object={t['Object Acc']:.2f} "
        f"APO_IoU={t['APO IoU']:.2f} Hit03={t['Hit@0.3']:.2f} "
        f"Hit05={t['Hit@0.5']:.2f} bbox_pred={t['BBox Pred Rate']:.2f} "
        f"zoom_parse_fail={t['Zoom Parse Fail']:.2f} "
        f"answer_parse_fail={t['Answer Parse Fail']:.2f} "
        f"grammar={t['Grammar OK']:.2f} zoom={t['Zoom Fired']:.2f}"
    )
    (out / f"{tag}_table.txt").write_text(line + "\n", encoding="utf-8")
    print(f"[{tag}] {line}")


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    eval_path = Path(args.eval_jsonl)
    output_dir = Path(args.output_dir)
    pred_path = output_dir / f"{args.tag}_predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(eval_path)
    if args.start:
        samples = samples[args.start:]
    if args.limit:
        samples = samples[: args.limit]
    if args.sort_by_type:
        order = {"object": 0, "region": 1, "global": 2}
        samples = sorted(
            samples,
            key=lambda row: (order.get(str(row.get("type") or "").lower(), 3), str(row.get("question_id") or "")),
        )

    done_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if args.resume and pred_path.exists():
        for row in read_jsonl(pred_path):
            qid = str(row.get("question_id") or "")
            if qid:
                done_ids.add(qid)
            rows.append(row)
        print(f"[resume] {args.tag}: loaded {len(done_ids)} rows")

    print(json.dumps({
        "tag": args.tag,
        "model": args.model,
        "eval_jsonl": str(eval_path),
        "samples": len(samples),
        "skill_bank": args.skill_bank,
        "output": str(pred_path),
    }, ensure_ascii=False, indent=2))

    skillbank = load_skillbank(args.skill_bank)
    t0 = time.perf_counter()
    model, processor = base_eval.build_model(args.model)
    dispatcher = SkillTriggeredDispatcher(
        model,
        processor,
        skillbank=skillbank,
        max_new_tokens=args.max_new_tokens,
        turn1_max_new_tokens=args.turn1_max_new_tokens,
        turn2_max_new_tokens=args.turn2_max_new_tokens,
        system_prompt=SYSTEM_PROMPT_COLDSTART_EVAL,
    )
    print(f"[model] ready in {time.perf_counter() - t0:.1f}s")

    pending: list[dict[str, Any]] = []
    mode = "a" if args.resume else "w"
    total_t0 = time.perf_counter()
    with pred_path.open(mode, encoding="utf-8") as fout:
        def flush() -> None:
            nonlocal pending, rows
            if not pending:
                return
            prepared = [pack["prepared"] for pack in pending]
            try:
                results = dispatcher.run_batch_turn1(prepared)
            except Exception as exc:  # noqa: BLE001
                results = [None] * len(pending)
                batch_exc = exc
            else:
                batch_exc = None

            for pack, result in zip(pending, results):
                if result is None:
                    row = exception_row(pack, batch_exc or Exception("dispatch_failed"))
                else:
                    row = row_from_result(pack, result)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            fout.flush()
            pending = []
            if args.summary_every and len(rows) % args.summary_every == 0:
                write_summary(
                    summarize_rows(rows, args.tag, args.model, str(eval_path)),
                    output_dir,
                    args.tag,
                )

        for sample in tqdm(samples, desc=f"eval {args.tag}"):
            qid = str(sample.get("question_id") or "")
            if args.resume and qid in done_ids:
                continue

            base = {
                "question_id": qid,
                "ground_truth": sample.get("ground_truth"),
                "category": sample.get("category"),
                "higher_level": sample.get("higher_level"),
                "type": sample.get("type"),
                "image": sample.get("image_name") or sample.get("image"),
                "question": sample.get("question"),
                "label": sample.get("label"),
                "bbox_ref": sample.get("bbox") or [],
                "model_id": args.model_id,
            }
            image_name = str(sample.get("image_name") or sample.get("image") or "")
            img_path = base_eval.find_image(image_name)
            if img_path is None:
                pack = {"sample": sample, "base": base, "skill_block": "", "retrieved_skill_ids": []}
                row = exception_row(pack, Exception("image_not_found"))
                row["error"] = "image_not_found"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                scale = max(image.size) / 1024.0
                question_prompt, block, ids = dispatcher.prepare_question(sample)
                pending.append({
                    "sample": sample,
                    "base": base,
                    "scale": scale,
                    "image_size": image.size,
                    "skill_block": block,
                    "retrieved_skill_ids": ids,
                    "prepared": {
                        "image": image,
                        "global_small": base_eval.resize_image(image),
                        "question": question_prompt,
                        "scale": scale,
                    },
                })
                if len(pending) >= max(1, args.batch_size):
                    flush()
            except Exception as exc:  # noqa: BLE001
                pack = {"sample": sample, "base": base, "skill_block": "", "retrieved_skill_ids": []}
                row = exception_row(pack, exc)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
        flush()

    elapsed = time.perf_counter() - total_t0
    print(f"[done] {args.tag}: {len(rows)} rows in {elapsed:.1f}s")
    summary = summarize_rows(rows, args.tag, args.model, str(eval_path))
    summary["benchmark"] = {
        "total_seconds": round(elapsed, 1),
        "samples_per_second": round(len(rows) / elapsed, 3) if elapsed else 0.0,
    }
    write_summary(summary, output_dir, args.tag)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_COLDSTART_MODEL))
    parser.add_argument("--eval_jsonl", default=str(DEFAULT_EVAL_JSONL))
    parser.add_argument("--skill_bank", default=str(DEFAULT_SKILLBANK))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tag", default="coldstart_rl_dev")
    parser.add_argument("--model_id", default="GeoSkillRL-Coldstart")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--turn1_max_new_tokens", type=int, default=768)
    parser.add_argument("--turn2_max_new_tokens", type=int, default=256)
    parser.add_argument("--summary_every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sort_by_type", action="store_true")
    return parser.parse_args()


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
