"""
Build a 1:1 negative crop-verification SFT mix for zoom+seg causal reasoning data.

This script keeps every original positive sample, then adds one WRONG-BBOX negative
variant for every bbox/zoom positive sample. The negative variant teaches:

  - verify the zoomed crop before analysis;
  - if the crop does not contain the requested target, say so explicitly;
  - do NOT emit <seg> after a failed crop verification;
  - answer from global context / known supervision rather than hallucinating mask evidence.

The DeepSeek input for each negative contains:
  1. global image;
  2. wrong bbox crop image;
  3. metadata with question, target label, GT answer, correct bbox, wrong bbox.

By default this writes a new JSONL. Use --overwrite-src to backup and replace SRC.

Usage:
  /root/autodl-tmp/VQA/.venv/bin/python scripts/build_zoom_seg_negative_verification_sft.py --dry-run
  DEEPSEEK_API_KEY=... /root/autodl-tmp/VQA/.venv/bin/python scripts/build_zoom_seg_negative_verification_sft.py --limit 20
  DEEPSEEK_API_KEY=... /root/autodl-tmp/VQA/.venv/bin/python scripts/build_zoom_seg_negative_verification_sft.py --overwrite-src
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
from anthropic import Anthropic

ROOT = Path("/root/autodl-tmp/VQA")
SRC = ROOT / "json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.jsonl"
DST = ROOT / "json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.neg1x.jsonl"
CHECKPOINT = ROOT / "json_data/zoom_seg_json/sft_causal_reasoning/checkpoint_neg_verification.json"
IMG_DIRS = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
]

API_BASE = "https://api.deepseek.com/anthropic"
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 900
MAX_RETRIES = 3
SAVE_EVERY = 100
WORKERS = 8

random.seed(42)
_lock = Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def find_image(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for d in IMG_DIRS:
        cand = d / name
        if cand.exists():
            return cand
        stem = Path(name).stem
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            alt = d / f"{stem}{ext}"
            if alt.exists():
                return alt
    raise FileNotFoundError(name)


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return image


def clamp_box(box: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return [x1, y1, x2, y2]


def crop_box_for_bbox(image: Image.Image, bbox_orig: list[float], min_size: int = 512) -> list[int]:
    w, h = image.size
    x1, y1, x2, y2 = clamp_box(bbox_orig, w, h)
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return [0, 0, w, h]
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1, ny1 = cx - min_size // 2, cy - min_size // 2
        nx2, ny2 = nx1 + min_size, ny1 + min_size
        if nx1 < 0:
            nx2 += -nx1
            nx1 = 0
        if ny1 < 0:
            ny2 += -ny1
            ny1 = 0
        if nx2 > w:
            nx1 -= nx2 - w
            nx2 = w
        if ny2 > h:
            ny1 -= ny2 - h
            ny2 = h
        return clamp_box([nx1, ny1, nx2, ny2], w, h)
    return [x1, y1, x2, y2]


def crop_from_bbox_1024(image: Image.Image, bbox_1024: list[float], scale: float) -> Image.Image:
    bbox_orig = [scale * v for v in bbox_1024]
    return image.crop(tuple(crop_box_for_bbox(image, bbox_orig)))


def intersects(a: list[int], b: list[int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def choose_wrong_bbox(record: dict[str, Any], image: Image.Image) -> list[int]:
    """Choose a wrong 1024-space bbox whose padded crop does not include GT bbox."""
    gt = [int(round(v)) for v in record["bbox"]]
    scale = float(record.get("scale", max(image.size) / 1024.0))
    gt_orig = [scale * v for v in gt]
    gt_crop = crop_box_for_bbox(image, gt_orig)
    bw = max(8, gt[2] - gt[0])
    bh = max(8, gt[3] - gt[1])

    # Try random boxes with the same size. Reject boxes whose actual 512-min crop
    # intersects the original GT crop; this avoids accidental target visibility.
    for _ in range(300):
        x1 = random.randint(0, max(0, 1023 - bw))
        y1 = random.randint(0, max(0, 1023 - bh))
        cand = [x1, y1, x1 + bw, y1 + bh]
        cand_crop = crop_box_for_bbox(image, [scale * v for v in cand])
        if not intersects(cand_crop, gt_crop):
            return cand

    # Fallback: opposite corner preserving size.
    if (gt[0] + gt[2]) / 2 < 512:
        x1 = max(0, 1023 - bw)
    else:
        x1 = 0
    if (gt[1] + gt[3]) / 2 < 512:
        y1 = max(0, 1023 - bh)
    else:
        y1 = 0
    return [x1, y1, x1 + bw, y1 + bh]


NEG_PROMPT = """You are a strict remote-sensing VQA teacher. You are shown:
1. the GLOBAL satellite image;
2. a zoomed crop produced from a WRONG bbox.

The crop is intentionally wrong. It does NOT contain the requested target.

QUESTION: {question}
TARGET / REFERENT: "{label}"
GROUND TRUTH ANSWER: {gt}
CORRECT BBOX IN 1024 COORDS: {correct_bbox}
WRONG BBOX IN 1024 COORDS: {wrong_bbox}

Original positive Stage-2, for reference only:
{stage2_ref}

Task:
Write the Stage-2 reasoning after the wrong <zoom>. The model must look at the
crop and explicitly reject it. Do NOT propose <seg>. Do NOT claim the target is
visible. Do NOT say a mask confirms anything. Since the crop is invalid, answer
from global-image context and the given supervision.

The style should be concise but explicit:
[Crop Verification] The crop does not contain ...
[Correction] The zoom bbox is wrong; I should not use this crop as evidence.
[Analysis] From the global image / question context, ...

Output ONLY a JSON object:
{{"stage2": "[Crop Verification] ...\\n[Correction] ...\\n[Analysis] ..."}}"""


def call_deepseek(client: Anthropic, record: dict[str, Any], wrong_bbox: list[int], global_img: Image.Image, wrong_crop: Image.Image) -> str:
    prompt = NEG_PROMPT.format(
        question=record.get("question", ""),
        label=record.get("label") or record.get("label_cleaned") or "",
        gt=record.get("ground_truth", ""),
        correct_bbox=json.dumps(record.get("bbox", []), ensure_ascii=False),
        wrong_bbox=json.dumps(wrong_bbox, ensure_ascii=False),
        stage2_ref=str(record.get("stage_2_reasoning") or "")[:500],
    )
    image_blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_to_b64(global_img)}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_to_b64(wrong_crop)}},
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": image_blocks + [{"type": "text", "text": prompt}]}],
    )
    text = "".join(getattr(block, "text", "") for block in resp.content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    stage2 = str(obj["stage2"]).strip()
    if "<seg>" in stage2 or "</seg>" in stage2:
        stage2 = re.sub(r"<seg>.*?</seg>", "", stage2, flags=re.S).strip()
    return stage2


def build_negative_record(record: dict[str, Any], wrong_bbox: list[int], stage2: str) -> dict[str, Any]:
    label = record.get("label") or record.get("label_cleaned") or "target"
    global_desc = record.get("global", "")
    stage1 = record.get("stage_1_reasoning", "")
    gt = record.get("ground_truth", "")
    think1 = (
        f"{global_desc}\n\n{stage1.strip()}\n\n"
        "I need to pay attention to the reference object at"
    )
    bbox_json = json.dumps([{"bbox_2d": wrong_bbox, "label": label}], ensure_ascii=False)
    assistant_text = (
        f"<think>{think1}</think>\n"
        f"<zoom>{bbox_json}</zoom>\n"
        f"<think>{stage2}</think>\n"
        f"<answer>{gt}</answer>"
    )
    out = deepcopy(record)
    out["question_id"] = f"{record.get('question_id')}_neg_bbox_miss"
    out["assistant_text"] = assistant_text
    out["bbox"] = wrong_bbox
    out["bbox_correct"] = False
    out["verified"] = True
    out["negative_type"] = "wrong_bbox_crop_reject_no_seg"
    out["source_question_id"] = record.get("question_id")
    out["original_bbox"] = record.get("bbox")
    out["wrong_bbox"] = wrong_bbox
    out["stage_2_reasoning"] = stage2
    out["stage_3_reasoning"] = ""
    out["seg_prompt"] = ""
    out["seg_purpose"] = "no_seg_after_failed_crop_verification"
    out["overlay_path"] = ""
    out["mask_coverage"] = 0.0
    return out


def process_task(task: dict[str, Any], api_key: str) -> dict[str, Any]:
    key = task["key"]
    record = task["record"]
    wrong_bbox = task["wrong_bbox"]
    client = Anthropic(base_url=API_BASE, api_key=api_key)
    for attempt in range(MAX_RETRIES):
        try:
            full = Image.open(find_image(record["image_name"])).convert("RGB")
            global_img = resize_image(full)
            wrong_crop = resize_image(crop_from_bbox_1024(full, wrong_bbox, float(record.get("scale", max(full.size) / 1024.0))))
            stage2 = call_deepseek(client, record, wrong_bbox, global_img, wrong_crop)
            return {"key": key, "status": "ok", "record": build_negative_record(record, wrong_bbox, stage2)}
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return {"key": key, "status": "fail", "error": str(exc)}


def positive_record(record: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(record)
    if out.get("bbox") and len(out["bbox"]) == 4:
        out.setdefault("bbox_correct", True)
        out.setdefault("verified", True)
        out.setdefault("negative_type", "")
    return out


def save_checkpoint(path: Path, done_keys: set[str], negatives: list[dict[str, Any]], errors: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({
            "done_keys": sorted(done_keys),
            "negatives": negatives,
            "errors": errors,
        }, f, ensure_ascii=False)
    tmp.replace(path)


def load_checkpoint(path: Path) -> tuple[set[str], list[dict[str, Any]], int]:
    if not path.exists():
        return set(), [], 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("done_keys", [])), list(data.get("negatives", [])), int(data.get("errors", 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--api-base", default=API_BASE)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite-src", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    global API_BASE, MODEL
    args = parse_args()
    API_BASE = args.api_base
    MODEL = args.model

    records = read_jsonl(args.src)
    positives = [positive_record(r) for r in records]
    bbox_records = [r for r in records if r.get("bbox") and len(r["bbox"]) == 4 and "</zoom>" in r.get("assistant_text", "")]
    if args.limit:
        bbox_records = bbox_records[:args.limit]

    print(f"[data] src={args.src}")
    print(f"[data] original_records={len(records)} bbox_positive_records={len(bbox_records)}")
    print(f"[plan] output_records={len(bbox_records)} negatives_only=True source_positives_not_written={len(positives)} negatives_to_add={len(bbox_records)}")
    if args.dry_run:
        for r in bbox_records[:5]:
            full = Image.open(find_image(r["image_name"])).convert("RGB")
            wrong = choose_wrong_bbox(r, full)
            print({"qid": r.get("question_id"), "gt_bbox": r.get("bbox"), "wrong_bbox": wrong, "label": r.get("label")})
        return

    if not args.api_key:
        raise SystemExit("Missing API key. Set DEEPSEEK_API_KEY or pass --api-key.")

    done_keys, negatives, errors = load_checkpoint(args.checkpoint) if args.resume else (set(), [], 0)
    neg_by_qid = {r.get("source_question_id"): r for r in negatives}

    tasks = []
    for r in bbox_records:
        qid = str(r.get("question_id"))
        key = f"{qid}_neg_bbox_miss"
        if key in done_keys or qid in neg_by_qid:
            continue
        full = Image.open(find_image(r["image_name"])).convert("RGB")
        wrong = choose_wrong_bbox(r, full)
        tasks.append({"key": key, "record": r, "wrong_bbox": wrong})

    print(f"[resume] done={len(done_keys)} negatives_loaded={len(negatives)} tasks={len(tasks)} errors={errors}")
    t0 = time.perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_map = {ex.submit(process_task, t, args.api_key): t for t in tasks}
        pbar = tqdm(
            total=len(tasks),
            initial=0,
            desc="Generating negative verification SFT",
            unit="sample",
            dynamic_ncols=True,
        )
        for fut in as_completed(future_map):
            completed += 1
            res = fut.result()
            if res["status"] == "ok":
                negatives.append(res["record"])
                done_keys.add(res["key"])
            else:
                errors += 1
                tqdm.write(f"[error] {res.get('key')} {res.get('error')}")

            elapsed = max(time.perf_counter() - t0, 1e-6)
            pbar.set_postfix({
                "ok": len(negatives),
                "err": errors,
                "rate": f"{completed/elapsed:.2f}/s",
            })
            pbar.update(1)

            if completed % SAVE_EVERY == 0:
                save_checkpoint(args.checkpoint, done_keys, negatives, errors)
                out_rows = negatives
                write_jsonl(args.dst, out_rows)
                tqdm.write(f"[saved] {args.dst} rows={len(out_rows)} negatives_only=True")
        pbar.close()

    out_rows = negatives
    target = args.src if args.overwrite_src else args.dst
    if args.overwrite_src:
        backup = args.src.with_suffix(args.src.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(args.src, backup)
        print(f"[backup] {backup}")
    write_jsonl(target, out_rows)
    args.checkpoint.unlink(missing_ok=True)

    n_neg = sum(1 for r in out_rows if r.get("negative_type") == "wrong_bbox_crop_reject_no_seg")
    print(f"[done] output={target}")
    print(f"[done] rows={len(out_rows)} negative={n_neg} positives_written=0 errors={errors}")


if __name__ == "__main__":
    main()
