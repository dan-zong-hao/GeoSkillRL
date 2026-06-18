"""
Unsloth-accelerated full-test evaluator for the Qwen3.5 VQA dispatcher.

Same logic as speedup/origin/eval/eval_full_test_fast.py but loads the
model via Unsloth FastVisionModel for faster inference.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_unsloth.py \
    --model /root/autodl-tmp/VQA/speedup/unsloth/sft/output/final_hf \
    --test_jsonl /root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_causal_reasoning/...
    --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.utils import logging as hf_logging

VQA_ROOT = Path("/root/autodl-tmp/VQA")

from dispatcher_fast import FastActionDispatcher, Turn1Prepared, resize_image
from sam2_wrapper import SAM2Predictor
from dispatcher import (
    DispatchResult,
    SYSTEM_PROMPT,
    VISION_TOKEN,
    cut_image,
    extract_answer,
    _extract_first_bbox,
    _extract_seg_prompt,
    _truncate_after,
)

Image.MAX_IMAGE_PIXELS = None

IMAGE_DIR_CANDIDATES = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
    VQA_ROOT / "ZoomEarth" / "images",
]


# ---------------------------------------------------------------------------
# Image resolver (cached loading)
# ---------------------------------------------------------------------------
class ImageResolver:
    def __init__(self, extra_dirs: list[Path], cache_size: int = 0):
        self.dirs = extra_dirs + [Path(d) for d in IMAGE_DIR_CANDIDATES]
        self.cache: OrderedDict[str, tuple[Image.Image, Image.Image]] = OrderedDict()
        self.max_cache = max(0, cache_size)

    def find_image(self, name: str) -> Optional[Path]:
        for d in self.dirs:
            p = d / name
            if p.exists():
                return p
            p = d / Path(name).name
            if p.exists():
                return p
        return None

    def load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def global_small(self, path: Path, image: Image.Image) -> Image.Image:
        key = str(path)
        if key in self.cache:
            cached = self.cache.pop(key)
            self.cache[key] = cached
            return cached[1]
        small_img, _ = resize_image(image)
        if self.max_cache > 0:
            self.cache[key] = (image, small_img)
            while len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        return small_img


# ---------------------------------------------------------------------------
# IoU / APO metrics
# ---------------------------------------------------------------------------
def _iou(b1, b2) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return float(inter / (a1 + a2 - inter + 1e-9))


def _apo_iou_512(bbox_pred, bbox_gt, image_size):
    """Expand bbox to at least 512x512 region centered on the bbox, recompute IoU."""
    if not bbox_pred or not bbox_gt or len(bbox_pred) != 4 or len(bbox_gt) != 4:
        return 0.0, None, None

    # Pad each bbox to at least 512x512
    def pad_512(b, w, h):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        bw = max(b[2] - b[0], 512)
        bh = max(b[3] - b[1], 512)
        nx1 = max(0, cx - bw / 2)
        ny1 = max(0, cy - bh / 2)
        nx2 = min(w, nx1 + bw)
        ny2 = min(h, ny1 + bh)
        return [nx1, ny1, nx2, ny2]

    w, h = image_size
    b1 = pad_512(bbox_pred, w, h)
    b2 = pad_512(bbox_gt, w, h)
    return _iou(b1, b2), b1, b2


def _scale_bbox_1024_to_orig(bbox_1024, scale: float):
    if not bbox_1024 or len(bbox_1024) != 4:
        return None
    return [v * scale for v in bbox_1024]


def _scale_bbox_orig_to_1024(bbox_orig, scale: float):
    if not bbox_orig or len(bbox_orig) != 4:
        return None
    return [v / scale for v in bbox_orig]


def _split_trajectory(trajectory: str) -> dict:
    out = {"stage1": "", "stage2": "", "stage3": ""}
    if not trajectory:
        return out
    parts = re.split(r"(?=^\[TURN\d+\])", trajectory, flags=re.MULTILINE)
    for part in parts:
        if part.startswith("[TURN1]"):
            out["stage1"] = part.strip()
        elif part.startswith("[TURN2]"):
            out["stage2"] = part.strip()
        elif part.startswith("[TURN3]"):
            out["stage3"] = part.strip()
    return out


def _answer_contains(gt: str, pred: str) -> bool:
    if not gt or not pred:
        return False
    return gt.lower().strip() in pred.lower().strip() or pred.lower().strip() in gt.lower().strip()


def _score_fields(question: str, gt: str, pred: str) -> dict:
    """Heuristic answer scoring (same as original)."""
    if not pred:
        return {"correct_final": False, "correct_final_method": "heuristic"}
    pred_norm = pred.lower().strip()
    gt_norm = gt.lower().strip() if gt else ""
    correct = gt_norm == pred_norm or gt_norm in pred_norm or pred_norm in gt_norm
    return {"correct_final": correct, "correct_final_method": "heuristic"}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(rows: list[dict], tag: str, model_path: str) -> dict:
    n = len(rows)
    by_type = defaultdict(list)
    for row in rows:
        by_type[str(row.get("type", "unknown")).lower()].append(row)

    def acc(rs):
        return sum(1 for r in rs if r["correct_final"]) / max(len(rs), 1) if rs else 0.0

    bbox_rows = [r for r in rows if r.get("bbox_gt_1024")]
    zoom_rows = [r for r in rows if "zoom" in r.get("actions", [])]
    seg_rows = [r for r in rows if "seg" in r.get("actions", [])]
    bbox_pred_rows = [r for r in bbox_rows if r.get("bbox_pred_1024")]

    return {
        "tag": tag,
        "model": model_path,
        "n_samples": n,
        "n_missing_images": sum(1 for r in rows if r.get("error") == "image_not_found"),
        "table": {
            "Global": 100.0 * acc(by_type.get("global", [])),
            "Region": 100.0 * acc(by_type.get("region", [])),
            "Object": 100.0 * acc(by_type.get("object", [])),
            "Avg.Acc": 100.0 * acc(rows),
            "APO IoU": 100.0 * (sum(r.get("iou", 0.0) for r in bbox_rows) / max(len(bbox_rows), 1)),
        },
        "extra": {
            "valid_grammar_rate": sum(1 for r in rows if r.get("grammar_ok")) / max(n, 1),
            "zoom_fired_rate": len(zoom_rows) / max(n, 1),
            "seg_fired_rate": len(seg_rows) / max(n, 1),
            "bbox_pred_rate_on_bbox_gt": len(bbox_pred_rows) / max(len(bbox_rows), 1),
            "answer_contains_rate": sum(1 for r in rows if r.get("answer_contains")) / max(n, 1),
            "error_rate": sum(1 for r in rows if r.get("error")) / max(n, 1),
        },
        "counts": {
            "by_type": dict(Counter(str(r.get("type", "unknown")).lower() for r in rows)),
            "by_category": dict(Counter(str(r.get("category", "unknown")) for r in rows)),
            "actions": dict(Counter("+".join(r.get("actions", [])) or "none" for r in rows)),
        },
    }


def write_summary_files(summary: dict, output_dir: Path, tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    table = summary["table"]
    lines = [
        f"table:  Global={table['Global']:.2f}  Region={table['Region']:.2f}  "
        f"Object={table['Object']:.2f}  Avg.Acc={table['Avg.Acc']:.2f}  "
        f"APO_IoU={table['APO IoU']:.2f}",
        f"extra:  grammar={summary['extra']['valid_grammar_rate']:.3f}  "
        f"zoom_fire={summary['extra']['zoom_fired_rate']:.3f}  "
        f"seg_fire={summary['extra']['seg_fired_rate']:.3f}  "
        f"bbox_pred_rate={summary['extra']['bbox_pred_rate_on_bbox_gt']:.3f}",
    ]
    (output_dir / f"{tag}_table.txt").write_text("\n".join(lines))
    print("\n[table row]")
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# Model loading (Unsloth path)
# ---------------------------------------------------------------------------
def build_model(model_path: str):
    hf_logging.set_verbosity_error()
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        local_files_only=True,
    )
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    processor = AutoProcessor.from_pretrained(model_path, max_pixels=64 * 64 * 28 * 28)
    processor.tokenizer = tokenizer

    model = FastVisionModel.for_inference(model)

    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    pad = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = pad
    model.config.pad_token_id = pad
    print(f"[model] Loaded with Unsloth: {model_path}")
    return model, processor


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def make_base_row(sample: dict, model_id: str) -> dict:
    return {
        "question_id": str(sample.get("question_id", "")),
        "ground_truth": sample.get("ground_truth"),
        "category": sample.get("category"),
        "type": sample.get("type"),
        "image": sample.get("image_name"),
        "question": sample.get("question"),
        "bbox_ref": sample.get("bbox") or [],
        "label": sample.get("label"),
        "cut": sample.get("cut"),
        "model_id": model_id,
    }


def missing_image_row(base_row: dict, sample: dict) -> dict:
    return {
        **base_row,
        "answer1": None, "answer2": None, "final_answer": None,
        "actions": [], "stage1": "", "stage2": "", "stage3": "", "trajectory": "",
        "bbox_pred_orig": None, "bbox_pred_1024": None,
        "bbox_gt_orig": None, "bbox_gt_1024": sample.get("bbox") or None,
        "iou": 0.0, "iou_before_apo512": 0.0, "iou_apo512": 0.0,
        "bbox_pred_apo512_orig": None, "bbox_gt_apo512_orig": None,
        "apo_iou_expand_size": 512, "apo_iou_expand_mode": "fixed512",
        "seg_prompt_pred": None, "grammar_ok": False,
        "correct_final": False, "correct_final_method": "heuristic",
        "answer_contains": False, "error": "image_not_found",
    }


def result_to_row(base_row: dict, sample: dict, result, scale: float, image_size: tuple) -> dict:
    stages = _split_trajectory(result.trajectory)
    bbox_gt_1024 = sample.get("bbox") or None
    bbox_gt_orig = _scale_bbox_1024_to_orig(bbox_gt_1024, scale)
    bbox_pred_1024 = _scale_bbox_orig_to_1024(result.bbox_pred, scale)
    iou_raw = _iou(result.bbox_pred, bbox_gt_orig) if bbox_gt_orig else 0.0
    iou, b1, b2 = _apo_iou_512(result.bbox_pred, bbox_gt_orig, image_size)
    ans = result.answer
    return {
        **base_row,
        "answer1": ans if not result.actions else None,
        "answer2": ans, "final_answer": ans,
        "actions": result.actions,
        "stage1": stages["stage1"], "stage2": stages["stage2"], "stage3": stages["stage3"],
        "trajectory": result.trajectory,
        "bbox_pred_orig": result.bbox_pred, "bbox_pred_1024": bbox_pred_1024,
        "bbox_gt_orig": bbox_gt_orig, "bbox_gt_1024": bbox_gt_1024,
        "iou": round(iou, 6), "iou_before_apo512": round(iou_raw, 6), "iou_apo512": round(iou, 6),
        "bbox_pred_apo512_orig": b1, "bbox_gt_apo512_orig": b2,
        "apo_iou_expand_size": 512, "apo_iou_expand_mode": "fixed512",
        "seg_prompt_pred": result.seg_prompt_pred,
        "grammar_ok": result.grammar_ok,
        **_score_fields(sample.get("question", ""), sample.get("ground_truth", ""), ans),
        "answer_contains": _answer_contains(sample.get("ground_truth", ""), ans),
        "error": result.error,
    }


def exception_row(base_row: dict, sample: dict, exc: Exception) -> dict:
    row = missing_image_row(base_row, sample)
    row["error"] = f"exception:{type(exc).__name__}:{exc}"
    return row


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _safe_name(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", tag)


def evaluate(args: argparse.Namespace):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    test_path = Path(args.test_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_name(args.tag or f"sft_unsloth_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    pred_path = output_dir / f"{tag}_predictions.jsonl"

    samples = _load_jsonl(test_path)
    if args.start:
        samples = samples[args.start:]
    if args.limit:
        samples = samples[:args.limit]

    # Resume: load completed question IDs
    done_ids: set = set()
    existing_rows: list = []
    if args.resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            if qid:
                done_ids.add(qid)
                existing_rows.append(row)
        print(f"[resume] {len(done_ids)} already done")

    print(f"[cfg] model={args.model}")
    print(f"[cfg] test_jsonl={test_path}  total={len(samples)}  start={args.start}  limit={args.limit}")
    print(f"[cfg] output={pred_path}  resume={args.resume}  batch_size={args.batch_size}")

    load_t0 = time.perf_counter()
    model, processor = build_model(args.model)
    sam2 = SAM2Predictor()
    dispatcher = FastActionDispatcher(model, processor, sam2,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_turn1=args.max_new_tokens_turn1,
        max_new_tokens_turn2=args.max_new_tokens_turn2,
        max_new_tokens_turn3=args.max_new_tokens_turn3,
    )
    print(f"[model] Loaded in {time.perf_counter() - load_t0:.1f}s")

    resolver = ImageResolver([Path(p) for p in args.image_dir] if args.image_dir else [],
                             cache_size=args.cache_size)

    rows = list(existing_rows) if args.resume else []
    mode = "a" if args.resume else "w"
    eval_t0 = time.perf_counter()

    with pred_path.open(mode, encoding="utf-8") as fout:
        pending = []

        def flush_batch():
            nonlocal pending, rows
            if not pending:
                return
            prepared = [p["prepared"] for p in pending]
            try:
                results = dispatcher.run_batch_turn1(prepared)
            except Exception:
                results = []
                for p in pending:
                    try:
                        r = dispatcher.run_preprocessed(
                            p["prepared"].image, p["prepared"].question,
                            p["prepared"].global_small, p["prepared"].scale)
                        results.append(r)
                    except Exception as exc:
                        results.append(None)
                        print(f"  [error] {p['sample'].get('question_id')}: {exc}")

            for pack, result in zip(pending, results):
                if result is None:
                    row = exception_row(pack["base_row"], pack["sample"], Exception("dispatch_failed"))
                else:
                    row = result_to_row(pack["base_row"], pack["sample"], result,
                                       pack["scale"], pack["image_size"])
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            fout.flush()
            pending = []
            if args.summary_every and len(rows) % args.summary_every == 0:
                write_summary_files(summarize(rows, tag, args.model), output_dir, tag)

        for sample in tqdm(samples, desc="Evaluating"):
            qid = str(sample.get("question_id", ""))
            if args.resume and qid in done_ids:
                continue
            base_row = make_base_row(sample, args.model_id)
            image_path = resolver.find_image(str(sample["image_name"]))
            if image_path is None:
                flush_batch()
                row = missing_image_row(base_row, sample)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                continue
            try:
                image = resolver.load_image(image_path)
                global_small = resolver.global_small(image_path, image)
                scale = max(image.size) / 1024.0
                pending.append({
                    "sample": sample, "base_row": base_row,
                    "scale": scale, "image_size": image.size,
                    "prepared": Turn1Prepared(image, global_small, str(sample.get("question", "")), scale),
                })
                if len(pending) >= args.batch_size:
                    flush_batch()
            except Exception as exc:
                flush_batch()
                row = exception_row(base_row, sample, exc)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)

        flush_batch()

    total_s = time.perf_counter() - eval_t0
    print(f"\n[done] {len(rows)} rows in {total_s:.0f}s ({len(rows)/total_s:.1f}/s)")

    # Benchmark timing
    if args.benchmark_json:
        Path(args.benchmark_json).write_text(json.dumps({
            "total_seconds": round(total_s, 1),
            "n_samples": len(rows),
            "throughput": round(len(rows) / total_s, 2),
            "samples_per_second": round(len(rows) / total_s, 2),
        }))

    if args.summary_every > 0:
        summary = summarize(rows, tag, args.model)
        write_summary_files(summary, output_dir, tag)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(VQA_ROOT / "speedup" / "unsloth" / "sft" / "output" / "final_hf"))
    p.add_argument("--test_jsonl",
                   default=str(VQA_ROOT / "json_data" / "zoom_seg_json" / "sft_causal_reasoning"
                               / "sft-00000-of-00001.zoom_seg.think.jsonl"))
    p.add_argument("--output_dir", default=str(VQA_ROOT / "speedup" / "unsloth" / "eval" / "results"))
    p.add_argument("--tag", default="sft_unsloth_eval")
    p.add_argument("--model_id", default="Qwen3.5-SFT-unsloth")
    p.add_argument("--image_dir", action="append", default=[])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cache_size", type=int, default=100)
    p.add_argument("--summary_every", type=int, default=200)
    p.add_argument("--benchmark_json", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_new_tokens_turn1", type=int, default=512)
    p.add_argument("--max_new_tokens_turn2", type=int, default=512)
    p.add_argument("--max_new_tokens_turn3", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
