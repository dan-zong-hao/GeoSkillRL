"""
Rewrite stage_2 reasoning in zoom SFT data to add crop-verification ability.

For each sample with a GT bbox, generates THREE variants via VLM:
  1. POSITIVE:      GT bbox      -> verify crop is correct, proceed to analyze
  2. NEAR_MISS:     shifted ~25% -> detect slight misalignment, correct partially
  3. COMPLETE_MISS: shifted ~50% -> detect completely wrong zoom, recover

Uses ThreadPoolExecutor for parallel VLM calls (~16x speedup).

Usage:
  /root/autodl-tmp/VQA/.venv/bin/python scripts/rewrite_stage2_with_verification.py
"""

import base64
import io
import json
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from PIL import Image
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC = Path("/root/autodl-tmp/VQA/json_data/zoom_json/sft-00000-of-00001.jsonl")
DST = Path("/root/autodl-tmp/VQA/json_data/zoom_json/sft_verified_zoom.jsonl")
CHECKPOINT = Path("/root/autodl-tmp/VQA/json_data/zoom_json/checkpoint_verified.json")

IMG_DIR = Path("/root/autodl-tmp/dataset/lrs_gro/image")

API_KEY = "sk-af956609f93c436abc4ce187ea07af9f"
API_BASE = "https://api.deepseek.com/anthropic"
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 1024
MAX_RETRIES = 3
SAVE_EVERY = 200  # save every N completed tasks
WORKERS = 16       # concurrent VLM calls

random.seed(42)
_lock = Lock()
_done_keys: set = set()
_results: dict = {}  # key -> {"record": ..., "status": "ok"|"fail"}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def find_image(name: str) -> Path:
    p = IMG_DIR / name
    if p.exists():
        return p
    for ext in [".png", ".jpg", ".tif", ".tiff"]:
        alt = IMG_DIR / (Path(name).stem + ext)
        if alt.exists():
            return alt
    raise FileNotFoundError(name)


def load_global_and_crop(record: dict):
    full = Image.open(find_image(record["image_name"])).convert("RGB")
    w, h = full.size
    scale_512 = 512 / max(w, h)
    global_512 = full.resize((int(w * scale_512), int(h * scale_512)), Image.BICUBIC)

    bbox = record.get("bbox", [])
    scale = record.get("scale", 1.0)
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(scale * v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full.width, x2), min(full.height, y2)
        crop = full.crop((x1, y1, x2, y2))
    else:
        crop = global_512
    return global_512, crop


# ---------------------------------------------------------------------------
# Bbox shifting
# ---------------------------------------------------------------------------
def shift_bbox(bbox: list, shift_pct: float) -> list:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mag = shift_pct * random.uniform(0.7, 1.3)
    dx = int(max(bw, 80) * mag * random.choice([-1, 1]))
    dy = int(max(bh, 80) * mag * random.choice([-1, 1]))
    nx1 = max(0, min(1023 - bw, x1 + dx))
    ny1 = max(0, min(1023 - bh, y1 + dy))
    return [nx1, ny1, nx1 + bw, ny1 + bh]


# ---------------------------------------------------------------------------
# VLM prompts
# ---------------------------------------------------------------------------
PROMPT_POSITIVE = """You are shown a GLOBAL satellite image and a crop of the CORRECT target region.

QUESTION: {question}
TARGET: "{label}"
GROUND TRUTH ANSWER: {gt}

The crop CORRECTLY shows the target "{label}". The zoom is accurate.

ORIGINAL STAGE 2 (for reference): {stage2}

TASK: Rewrite this stage-2 reasoning text. Look at the crop image and write a
[Crop Verification] prefix confirming it correctly contains the target, then an
[Analysis] of visual details to answer the question.

Output ONLY a JSON object with key "stage2":
{{"stage2": "[Crop Verification] ...\\n[Analysis] ..."}}"""

PROMPT_NEAR_MISS = """You are shown a GLOBAL satellite image and a crop that is SLIGHTLY OFF.

QUESTION: {question}
TARGET: "{label}"
GROUND TRUTH ANSWER: {gt}

The crop is NEAR the target "{label}" but the bbox shifted slightly — the target
may be partially visible at the edge or just outside. The zoom is NOT fully accurate.

TASK: Write a stage-2 reasoning text that detects the misalignment and recovers.
Look at BOTH images and write honestly what IS and IS NOT in the crop.

Output ONLY a JSON object with key "stage2":
{{"stage2": "[Crop Verification] ... (note misalignment)\\n[Analysis] ..."}}"""

PROMPT_COMPLETE_MISS = """You are shown a GLOBAL satellite image and a crop of the WRONG area.

QUESTION: {question}
TARGET: "{label}"
GROUND TRUTH ANSWER: {gt}

The crop does NOT contain the target "{label}" at all. It shows completely different
terrain or objects. The zoom FAILED — this is a localization error.

TASK: Write a stage-2 reasoning text that honestly detects the error and recovers
using only global image context. Describe what the crop ACTUALLY shows, acknowledge
the zoom error, then infer the answer "{gt}" from the global context.

Output ONLY a JSON object with key "stage2":
{{"stage2": "[Crop Verification] ...\\n[Correction] ...\\n[Analysis] ..."}}"""


# ---------------------------------------------------------------------------
# Single task: image loading + VLM call
# ---------------------------------------------------------------------------
def process_one_task(args: dict) -> dict:
    """Returns {"key": key, "status": "ok", "record": new_record} or {"key": key, "status": "fail"}"""
    key = args["key"]
    record = args["record"]
    variant = args["variant"]
    bbox = args["bbox"]
    client = Anthropic(base_url=API_BASE, api_key=API_KEY)

    # Build prompt
    q = record.get("question", "")
    label = record.get("label", "")
    gt = record.get("ground_truth", "")
    stage2 = record.get("stage_2_reasoning", "")[:400]
    prompts = {
        "pos": PROMPT_POSITIVE,
        "near": PROMPT_NEAR_MISS,
        "miss": PROMPT_COMPLETE_MISS,
    }
    prompt = prompts[variant].format(question=q, label=label, gt=gt, stage2=stage2)

    for attempt in range(MAX_RETRIES):
        try:
            # Load images with the shifted bbox
            tmp = deepcopy(record)
            tmp["bbox"] = bbox
            g_img, c_img = load_global_and_crop(tmp)

            image_blocks = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": img_to_b64(g_img)}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": img_to_b64(c_img)}},
            ]

            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "disabled"},
                messages=[{
                    "role": "user",
                    "content": image_blocks + [{"type": "text", "text": prompt}]
                }]
            )

            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text += block.text
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            result = json.loads(text)
            stage2_new = result["stage2"]

            # Build assistant_text
            global_desc = record.get("global", "")
            stage1 = record.get("stage_1_reasoning", "")
            think1 = (
                f"{global_desc}\n\n{stage1.strip()}\n\n"
                f"I need to pay attention to the reference object at"
            )
            bbox_json = json.dumps([{"bbox_2d": bbox, "label": label}])
            assistant_text = (
                f"<think>{think1}</think>\n"
                f"<zoom>{bbox_json}</zoom>\n"
                f"<think>{stage2_new}</think>\n"
                f"<answer>{gt}</answer>"
            )

            new_record = deepcopy(record)
            new_record["assistant_text"] = assistant_text
            new_record["bbox"] = bbox
            new_record["bbox_correct"] = (variant == "pos")
            new_record["verified"] = True
            if variant != "pos":
                new_record["miss_type"] = "near" if variant == "near" else "complete"

            return {"key": key, "status": "ok", "record": new_record}

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return {"key": key, "status": "fail", "error": str(e)}


# ---------------------------------------------------------------------------
# Build assistant_text for global-only samples
# ---------------------------------------------------------------------------
def build_global_assistant(record: dict) -> str:
    global_desc = record.get("global", "")
    stage1 = record.get("stage_1_reasoning", "")
    stage2 = record.get("stage_2_reasoning", "")
    gt = record.get("ground_truth", "")
    return (
        f"<think>{global_desc}\n\n{stage1.strip()}\n\n{stage2.strip()}</think>\n"
        f"<answer>{gt}</answer>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(SRC) as f:
        records = [json.loads(line) for line in f]

    records_with_bbox = [(i, r) for i, r in enumerate(records)
                         if r.get("bbox") and len(r["bbox"]) == 4]
    print(f"Total: {len(records)} samples, {len(records_with_bbox)} with bbox")

    # Load checkpoint
    done_keys = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            cp = json.load(f)
            done_keys = set(cp.get("done_keys", []))
        print(f"[resume] {len(done_keys)} already done")

    # Build all tasks
    tasks = []
    for idx, r in records_with_bbox:
        qid = r["question_id"]
        bbox = r["bbox"]
        # POSITIVE
        if f"{qid}_pos" not in done_keys:
            tasks.append({"key": f"{qid}_pos", "record": r, "variant": "pos", "bbox": bbox, "orig_idx": idx})
        # NEAR MISS
        if f"{qid}_near" not in done_keys:
            tasks.append({"key": f"{qid}_near", "record": r, "variant": "near",
                          "bbox": shift_bbox(bbox, 0.25), "orig_idx": idx})
        # COMPLETE MISS
        if f"{qid}_miss" not in done_keys:
            tasks.append({"key": f"{qid}_miss", "record": r, "variant": "miss",
                          "bbox": shift_bbox(bbox, 0.50), "orig_idx": idx})

    print(f"Tasks to run: {len(tasks)} ({(len(done_keys)//3)}/{len(records_with_bbox)} samples already done)")
    if not tasks:
        print("All done!")
        return

    # Initialize output records: global samples get assistant_text immediately
    out_records = []
    for idx, r in enumerate(records):
        if r.get("bbox") and len(r["bbox"]) == 4:
            out_records.append(None)  # placeholder, filled by pos variant
        else:
            r2 = deepcopy(r)
            r2["assistant_text"] = build_global_assistant(r)
            r2["bbox_correct"] = True
            out_records.append(r2)

    append_records = []  # near + miss variants appended at end
    errors = 0
    completed = 0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {executor.submit(process_one_task, t): t for t in tasks}

        for future in as_completed(future_map):
            result = future.result()
            completed += 1

            if result["status"] == "ok":
                rec = result["record"]
                key = result["key"]
                orig_idx = future_map[future]["orig_idx"]

                if key.endswith("_pos"):
                    out_records[orig_idx] = rec
                else:
                    append_records.append(rec)
                done_keys.add(key)
            else:
                errors += 1

            # Progress
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t_start
                rate = completed / elapsed
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(tasks)}] {rate:.1f}/s done={len(done_keys)} err={errors} ETA={eta:.0f}s")

            # Save checkpoint
            if completed % SAVE_EVERY == 0:
                all_records = [r for r in out_records if r is not None] + append_records
                DST.parent.mkdir(parents=True, exist_ok=True)
                with open(DST, "w", encoding="utf-8") as f:
                    for rec in all_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                with open(CHECKPOINT, "w") as f:
                    json.dump({"done_keys": list(done_keys), "errors": errors}, f)
                print(f"  [saved {len(all_records)} records]")

    # Final save
    all_records = [r for r in out_records if r is not None] + append_records
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    CHECKPOINT.unlink(missing_ok=True)

    n_pos = sum(1 for r in all_records if r.get("bbox_correct") is True)
    n_near = sum(1 for r in all_records if r.get("miss_type") == "near")
    n_miss = sum(1 for r in all_records if r.get("miss_type") == "complete")
    total_s = time.perf_counter() - t_start
    print(f"\nDone in {total_s:.0f}s ({len(all_records)/total_s:.1f} records/s)")
    print(f"  Output: {DST}")
    print(f"  Positive:       {n_pos}")
    print(f"  Near miss:      {n_near}")
    print(f"  Complete miss:  {n_miss}")
    print(f"  Total:          {len(all_records)}")
    print(f"  Errors:         {errors}")


if __name__ == "__main__":
    main()
