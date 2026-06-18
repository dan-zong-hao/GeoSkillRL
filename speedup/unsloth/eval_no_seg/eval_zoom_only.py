"""
Evaluator for zoom-only model (no seg). Two-turn dispatch:

  Turn 1 [global]              -> stop on </zoom> | </answer>
  Turn 2 [global, crop]         -> stop on </answer>
  → extract bbox, IoU, answer

No SAM2, no overlay.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_zoom_only.py \
    --model /root/autodl-tmp/VQA/speedup/unsloth/sft/only_zoom/output/final_hf \
    --test_jsonl /root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl \
    --output_dir ./results
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from tqdm import tqdm

VQA_ROOT = Path("/root/autodl-tmp/VQA")

SYSTEM_PROMPT = (
    "You are an intelligent remote sensing analyst. Given a question about a "
    "satellite image, you MAY use a zoom tool to focus before answering:\n"
    '  <zoom>[{"bbox_2d":[x1,y1,x2,y2],"label":"<short>"}]</zoom>\n'
    "Protocol: wrap reasoning in <think>...</think>. At most ONE <zoom> per "
    "trajectory. End with exactly one <answer>...</answer>. If the whole "
    "image is enough, skip zoom. After zooming, verify the crop before "
    "analyzing. Never say 'uncertain'."
)

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

IMAGE_DIRS = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
]

BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[(.*?)\]', re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def find_image(name: str) -> Optional[Path]:
    for d in IMAGE_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def resize_image(img: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img


def cut_image(image: Image.Image, bbox, min_size: int = 512) -> Image.Image:
    W, H = image.width, image.height
    x1, y1, x2, y2 = [max(0, min(int(v), s)) for v, s in zip(bbox, [W, H, W, H])]
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return image.copy()
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1 = max(0, cx - min_size // 2)
        ny1 = max(0, cy - min_size // 2)
        nx2 = min(W, nx1 + min_size)
        ny2 = min(H, ny1 + min_size)
        return image.crop((int(nx1), int(ny1), int(nx2), int(ny2)))
    return image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def extract_bbox(text: str) -> Optional[list]:
    m = BBOX_RE.search(text)
    if not m:
        return None
    try:
        nums = [float(x.strip()) for x in m.group(1).split(",")]
        return nums if len(nums) == 4 else None
    except ValueError:
        return None


def truncate_after(text: str, tag: str) -> str:
    idx = text.find(tag)
    return text[:idx + len(tag)] if idx >= 0 else text


# ---------------------------------------------------------------------------
# Dispatch (2-turn, no seg)
# ---------------------------------------------------------------------------
class ZoomOnlyDispatcher:
    def __init__(self, model, processor, device="cuda", max_new_tokens: int = 1024,
                 turn1_max_new_tokens: int = 512, turn2_max_new_tokens: int = 256):
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.turn1_max_new_tokens = turn1_max_new_tokens
        self.turn2_max_new_tokens = turn2_max_new_tokens
        tok = processor.tokenizer
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        self.model.generation_config.pad_token_id = tok.pad_token_id
        self.model.config.pad_token_id = tok.pad_token_id

    @staticmethod
    def build_turn1_prompt(question: str) -> str:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{VISION_TOKEN}{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str, images: list,
                  max_new_tokens: Optional[int] = None,
                  stop_strings: Optional[list[str]] = None) -> str:
        expected_images = prompt.count(VISION_TOKEN)
        if expected_images != len(images):
            raise ValueError(
                f"vision token/image mismatch: prompt has {expected_images}, images={len(images)}"
            )
        inputs = self.processor(
            text=[prompt], images=[images], return_tensors="pt", padding="longest"
        ).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            use_cache=True,
        )
        if stop_strings:
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = self.processor.tokenizer
        with torch.inference_mode():
            gen = self.model.generate(**inputs, **gen_kwargs)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(gen[0], skip_special_tokens=True).strip()

    def _generate_batch_turn1(self, prompts: list[str], images: list[Image.Image],
                              max_new_tokens: Optional[int] = None) -> list[str]:
        if not prompts:
            return []
        if any(prompt.count(VISION_TOKEN) != 1 for prompt in prompts):
            raise ValueError("Turn-1 batch generation expects exactly one vision token per prompt")
        if len(images) != len(prompts):
            raise ValueError(f"Turn-1 batch image mismatch: prompts={len(prompts)} images={len(images)}")
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding="longest"
        ).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.turn1_max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            use_cache=True,
            stop_strings=["</zoom>", "</answer>"],
            tokenizer=self.processor.tokenizer,
        )
        with torch.inference_mode():
            gen = self.model.generate(**inputs, **gen_kwargs)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        return [
            self.processor.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in gen
        ]

    def _continue_after_turn1(self, image: Image.Image, question: str,
                              global_small: Image.Image, effective_scale: float,
                              turn1_prompt: str, turn1_out: str) -> dict:
        trajectory = f"[TURN1]\n{turn1_out}\n"
        actions = []
        bbox_pred = None

        # Direct answer in Turn 1?
        if "<answer>" in turn1_out and (
            "<zoom>" not in turn1_out
            or turn1_out.find("<answer>") < turn1_out.find("<zoom>")
        ):
            return {
                "actions": [], "bbox_pred": None, "answer": extract_answer(turn1_out),
                "trajectory": trajectory, "grammar_ok": True, "error": None,
            }

        # Parse zoom bbox
        bbox_small = extract_bbox(turn1_out)
        if bbox_small is None:
            return {
                "actions": [], "bbox_pred": None,
                "answer": extract_answer(turn1_out),
                "trajectory": trajectory, "grammar_ok": False,
                "error": "zoom_parse_fail",
            }
        actions.append("zoom")
        bbox_orig = [v * effective_scale for v in bbox_small]
        bbox_pred = bbox_orig

        # Crop
        try:
            crop = cut_image(image, bbox_orig)
            crop_small = resize_image(crop)
        except Exception as exc:
            return {
                "actions": actions, "bbox_pred": bbox_pred,
                "answer": None, "trajectory": trajectory,
                "grammar_ok": False, "error": f"cut_fail:{exc}",
            }

        # Turn 2: global + crop (max 256 tokens for verify + answer)
        turn1_trunc = truncate_after(turn1_out, "</zoom>")
        turn2_prefix = "\n<think>"
        turn2_prompt = (
            turn1_prompt
            + turn1_trunc
            + "\nZoomed crop image to verify:\n"
            + VISION_TOKEN
            + "\nVerify whether the zoomed crop image contains the target before answering. "
              "Do not claim the crop is correct unless the target is visibly present in the crop. "
              "If it does not contain the target, say so explicitly and rely only on the global image.\n"
            + turn2_prefix
        )
        turn2_gen = self._generate(
            turn2_prompt, [global_small, crop_small], max_new_tokens=self.turn2_max_new_tokens,
            stop_strings=["</answer>"],
        )
        turn2_out = turn2_prefix + turn2_gen
        trajectory += f"[TURN2]\n{turn2_out}\n"

        ans = extract_answer(turn2_out)

        return {
            "actions": actions, "bbox_pred": bbox_pred, "answer": ans,
            "trajectory": trajectory, "grammar_ok": ans is not None,
            "error": None if ans else "answer_parse_fail",
        }

    def run_preprocessed(self, image: Image.Image, question: str,
                         global_small: Image.Image, effective_scale: float) -> dict:
        turn1_prompt = self.build_turn1_prompt(question)
        turn1_out = self._generate(
            turn1_prompt, [global_small], max_new_tokens=self.turn1_max_new_tokens,
            stop_strings=["</zoom>", "</answer>"],
        )
        return self._continue_after_turn1(
            image, question, global_small, effective_scale, turn1_prompt, turn1_out
        )

    def run_batch_turn1(self, prepared: list[dict]) -> list[dict]:
        if not prepared:
            return []
        prompts = [self.build_turn1_prompt(p["question"]) for p in prepared]
        try:
            turn1_outputs = self._generate_batch_turn1(
                prompts, [p["global_small"] for p in prepared],
                max_new_tokens=self.turn1_max_new_tokens,
            )
        except Exception:
            return [
                self.run_preprocessed(
                    p["image"], p["question"], p["global_small"], p["scale"]
                )
                for p in prepared
            ]
        return [
            self._continue_after_turn1(
                p["image"], p["question"], p["global_small"], p["scale"], prompt, output
            )
            for p, prompt, output in zip(prepared, prompts, turn1_outputs)
        ]

    def run(self, image: Image.Image, question: str) -> dict:
        global_small = resize_image(image)
        effective_scale = max(image.size) / 1024.0
        return self.run_preprocessed(image, question, global_small, effective_scale)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_iou(b1, b2) -> float:
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


def apo_iou_512(pred, gt, img_size):
    if not pred or not gt:
        return 0.0, None, None

    def pad(b, W, H):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        bw = max(b[2] - b[0], 512)
        bh = max(b[3] - b[1], 512)
        nx1 = max(0, cx - bw / 2)
        ny1 = max(0, cy - bh / 2)
        return [nx1, ny1, min(W, nx1 + bw), min(H, ny1 + bh)]

    W, H = img_size
    b1, b2 = pad(pred, W, H), pad(gt, W, H)
    return compute_iou(b1, b2), b1, b2


def answer_correct(gt: str, pred: str) -> bool:
    if not gt or not pred:
        return False
    g, p = gt.lower().strip(), pred.lower().strip()
    return g == p or g in p or p in g


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def summarize(rows: list, tag: str, model_path: str) -> dict:
    n = len(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[str(r.get("type", "unknown")).lower()].append(r)

    def acc(rs):
        return sum(1 for x in rs if x["correct_final"]) / max(len(rs), 1) if rs else 0.0

    bbox_rows = [r for r in rows if r.get("bbox_gt_1024")]

    return {
        "tag": tag, "model": model_path, "n_samples": n,
        "table": {
            "Global": 100 * acc(by_type.get("global", [])),
            "Region": 100 * acc(by_type.get("region", [])),
            "Object": 100 * acc(by_type.get("object", [])),
            "Avg.Acc": 100 * acc(rows),
            "APO IoU": 100 * sum(r.get("iou", 0) for r in bbox_rows) / max(len(bbox_rows), 1),
        },
        "extra": {
            "zoom_fired_rate": sum(1 for r in rows if "zoom" in r.get("actions", [])) / max(n, 1),
            "bbox_pred_rate": sum(1 for r in bbox_rows if r.get("bbox_pred_1024")) / max(len(bbox_rows), 1),
            "grammar_ok_rate": sum(1 for r in rows if r.get("grammar_ok")) / max(n, 1),
        },
        "counts": {
            "by_type": dict(Counter(str(r.get("type", "unknown")).lower() for r in rows)),
            "actions": dict(Counter("zoom" if "zoom" in r.get("actions", []) else "none" for r in rows)),
        },
    }


def write_summary(s: dict, out: Path, tag: str):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
    t = s["table"]
    e = s["extra"]
    line = (f"table: Global={t['Global']:.2f} Region={t['Region']:.2f} "
            f"Object={t['Object']:.2f} Avg.Acc={t['Avg.Acc']:.2f} APO_IoU={t['APO IoU']:.2f} | "
            f"zoom={e['zoom_fired_rate']:.3f} bbox_pred={e['bbox_pred_rate']:.3f} "
            f"grammar={e['grammar_ok_rate']:.3f}")
    (out / f"{tag}_table.txt").write_text(line)
    print(f"\n[table line]")
    print(line)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def build_model(model_path: str):
    from unsloth import FastVisionModel
    from transformers import AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path, load_in_4bit=False, use_gradient_checkpointing="unsloth",
        local_files_only=True,
    )
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    processor = AutoProcessor.from_pretrained(model_path, max_pixels=64 * 64 * 28 * 28)
    processor.tokenizer = tokenizer
    model = FastVisionModel.for_inference(model)

    tok = processor.tokenizer
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    pad = tok.pad_token_id or tok.eos_token_id
    model.generation_config.pad_token_id = pad
    model.config.pad_token_id = pad
    print(f"[model] Loaded: {model_path}")
    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(VQA_ROOT / "speedup/unsloth/sft/only_zoom/output/final_hf"))
    parser.add_argument("--test_jsonl", default=str(VQA_ROOT / "json_data/test-00000-of-00001.jsonl"))
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--tag", default="zoom_only_eval")
    parser.add_argument("--model_id", default="Qwen3.5-ZoomOnly")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--turn1_max_new_tokens", type=int, default=512)
    parser.add_argument("--turn2_max_new_tokens", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary_every", type=int, default=500)
    parser.add_argument("--benchmark_json", type=str, default="")
    args = parser.parse_args()

    test_path = Path(args.test_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    pred_path = output_dir / f"{tag}_predictions.jsonl"

    # Load test data
    with open(test_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.start:
        samples = samples[args.start:]
    if args.limit:
        samples = samples[:args.limit]

    # Resume
    done_ids = set()
    existing = []
    if args.resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            if qid:
                done_ids.add(qid)
                existing.append(row)
        print(f"[resume] {len(done_ids)} already done")

    print(f"[cfg] model={args.model}")
    print(f"[cfg] test={test_path} samples={len(samples)} start={args.start} limit={args.limit}")

    # Load model
    t0 = time.perf_counter()
    model, processor = build_model(args.model)
    dispatcher = ZoomOnlyDispatcher(
        model, processor,
        max_new_tokens=args.max_new_tokens,
        turn1_max_new_tokens=args.turn1_max_new_tokens,
        turn2_max_new_tokens=args.turn2_max_new_tokens,
    )
    print(f"[model] Ready in {time.perf_counter() - t0:.1f}s")

    rows = list(existing)
    mode = "a" if args.resume else "w"
    total_t0 = time.perf_counter()

    with pred_path.open(mode, encoding="utf-8") as fout:
        pending = []

        def row_from_result(pack: dict, result: dict) -> dict:
            sample = pack["sample"]
            base = pack["base"]
            scale = pack["scale"]
            image_size = pack["image_size"]
            bbox_gt_1024 = sample.get("bbox") or None
            bbox_gt_orig = [v * scale for v in bbox_gt_1024] if bbox_gt_1024 else None
            bbox_pred_1024 = [v / scale for v in result["bbox_pred"]] if result["bbox_pred"] else None
            iou_apo, _, _ = apo_iou_512(result["bbox_pred"], bbox_gt_orig, image_size)
            ans = result["answer"]
            return {
                **base,
                "actions": result["actions"],
                "bbox_pred_orig": result["bbox_pred"],
                "bbox_pred_1024": bbox_pred_1024,
                "bbox_gt_orig": bbox_gt_orig,
                "bbox_gt_1024": bbox_gt_1024,
                "iou": round(iou_apo, 6),
                "answer1": ans if not result["actions"] else None,
                "answer2": ans,
                "final_answer": ans,
                "correct_final": answer_correct(sample.get("ground_truth", ""), ans),
                "grammar_ok": result["grammar_ok"],
                "trajectory": result["trajectory"],
                "error": result.get("error"),
            }

        def exception_row(base: dict, sample: dict, exc: Exception) -> dict:
            return {**base,
                "actions": [], "bbox_pred_orig": None, "bbox_pred_1024": None,
                "bbox_gt_orig": None, "bbox_gt_1024": sample.get("bbox") or None,
                "iou": 0.0, "final_answer": None, "answer1": None, "answer2": None,
                "correct_final": False, "grammar_ok": False, "trajectory": "",
                "error": f"exception:{exc}",
            }

        def flush_batch():
            nonlocal pending, rows
            if not pending:
                return
            prepared = [p["prepared"] for p in pending]
            try:
                results = dispatcher.run_batch_turn1(prepared)
            except Exception as exc:
                results = [None] * len(pending)
                batch_exc = exc
            else:
                batch_exc = None

            for pack, result in zip(pending, results):
                if result is None:
                    row = exception_row(pack["base"], pack["sample"], batch_exc or Exception("dispatch_failed"))
                else:
                    row = row_from_result(pack, result)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            fout.flush()
            pending = []
            if args.summary_every and len(rows) % args.summary_every == 0:
                write_summary(summarize(rows, tag, args.model), output_dir, tag)

        for sample in tqdm(samples, desc="Evaluating"):
            qid = str(sample.get("question_id", ""))
            if args.resume and qid in done_ids:
                continue

            base = {
                "question_id": qid, "ground_truth": sample.get("ground_truth"),
                "category": sample.get("category"), "type": sample.get("type"),
                "image": sample.get("image_name"), "question": sample.get("question"),
                "bbox_ref": sample.get("bbox") or [],
                "model_id": args.model_id,
            }

            img_path = find_image(sample.get("image_name", ""))
            if img_path is None:
                flush_batch()
                row = {**base,
                    "actions": [], "bbox_pred_orig": None, "bbox_pred_1024": None,
                    "bbox_gt_orig": None, "bbox_gt_1024": sample.get("bbox") or None,
                    "iou": 0.0, "final_answer": None, "answer1": None, "answer2": None,
                    "correct_final": False, "grammar_ok": False, "trajectory": "",
                    "error": "image_not_found",
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                scale = max(image.size) / 1024.0
                global_small = resize_image(image)
                pending.append({
                    "sample": sample,
                    "base": base,
                    "scale": scale,
                    "image_size": image.size,
                    "prepared": {
                        "image": image,
                        "global_small": global_small,
                        "question": sample.get("question", ""),
                        "scale": scale,
                    },
                })
                if len(pending) >= max(1, args.batch_size):
                    flush_batch()
            except Exception as exc:
                flush_batch()
                row = exception_row(base, sample, exc)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)

        flush_batch()

    total_s = time.perf_counter() - total_t0
    print(f"\n[done] {len(rows)} rows in {total_s:.0f}s ({len(rows)/total_s:.1f}/s)")

    if args.benchmark_json:
        Path(args.benchmark_json).write_text(json.dumps({
            "total_seconds": round(total_s, 1), "n_samples": len(rows),
            "throughput": round(len(rows) / total_s, 2),
        }))

    if args.summary_every > 0:
        write_summary(summarize(rows, tag, args.model), output_dir, tag)


if __name__ == "__main__":
    main()
