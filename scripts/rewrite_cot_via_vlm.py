"""
Rewrite Think 2 and Think 3 in zoom_seg SFT data using DeepSeek V4 Pro VLM.

For each record with zoom+seg, sends the global image, crop, and overlay to the
VLM with a prompt that asks it to analyze the visual content and rewrite the two
think blocks as concise causal reasoning (not mask% descriptions).

Usage:
  /root/autodl-tmp/VQA/.venv/bin/python scripts/rewrite_cot_via_vlm.py
"""

import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC = Path("/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_level/sft-00000-of-00001.zoom_seg.think.jsonl")
DST = Path("/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.jsonl")
CHECKPOINT = Path("/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_causal_reasoning/checkpoint.json")

IMG_DIR = Path("/root/autodl-tmp/dataset/lrs_gro/image")

API_KEY = "sk-af956609f93c436abc4ce187ea07af9f"
API_BASE = "https://api.deepseek.com/anthropic"
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 1024
MAX_RETRIES = 3
SAVE_EVERY = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

client = Anthropic(base_url=API_BASE, api_key=API_KEY)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_images(record: dict):
    """Return (global_512, crop, overlay) PIL Images."""
    img_name = record["image_name"]
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        # try .tif extension
        tif_path = IMG_DIR / img_name.replace(".png", ".tif").replace(".tif.tif", ".tif")
        if tif_path.exists():
            img_path = tif_path
        else:
            raise FileNotFoundError(f"Image not found: {img_name}")

    full = Image.open(img_path).convert("RGB")

    # Resized global for context
    w, h = full.size
    scale_512 = 512 / max(w, h)
    global_512 = full.resize((int(w * scale_512), int(h * scale_512)), Image.BICUBIC)

    # Crop (from original resolution)
    bbox = record.get("bbox")
    scale = record.get("scale", 1.0)
    if bbox:
        x1, y1, x2, y2 = [int(scale * v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full.width, x2), min(full.height, y2)
        crop = full.crop((x1, y1, x2, y2))
    else:
        crop = global_512  # fallback

    # Overlay
    overlay_path = record.get("overlay_path")
    if overlay_path and Path(overlay_path).exists():
        overlay = Image.open(overlay_path).convert("RGB")
    else:
        overlay = None

    return global_512, crop, overlay


def build_prompt(record: dict) -> str:
    question = record["question"]
    gt = record["ground_truth"]
    seg_prompt = record.get("seg_prompt", "")
    stage_2 = record.get("stage_2_reasoning", "")

    # Extract original think2 and think3 for reference
    orig_text = record["assistant_text"]
    thinks = re.findall(r"<think>(.*?)</think>", orig_text, re.DOTALL)
    orig_think2 = thinks[1].strip() if len(thinks) > 1 else "(none)"
    orig_think3 = thinks[2].strip() if len(thinks) > 2 else "(none)"

    return f"""Rewrite two Chain-of-Thought segments for remote sensing VQA training data.

QUESTION: {question}
GROUND TRUTH: {gt}
SEGMENTATION PROMPT: "{seg_prompt}"
VISUAL ANALYSIS GUIDE: {stage_2[:500]}

ORIGINAL Think 2 (crop analysis): {orig_think2[:300]}
ORIGINAL Think 3 (mask analysis): {orig_think3[:300]}

TASK — rewrite both thinks using actual visual evidence from the images:

Think 2 (after zoom, before seg): Look at the CROP IMAGE. What visual details are relevant to the question? Why does segmenting "{seg_prompt}" help answer it? 1-2 sentences.

Think 3 (after seg, before answer): Look at the OVERLAY/SEGMENTATION MASK. What does the mask reveal? Make a causal inference leading to answer "{gt}". NEVER say "mask covers X%", "center region", or "concentrated in". 1 sentence.

Output ONLY: {{"think2": "...", "think3": "..."}}"""


def call_vlm(record: dict):
    """Send images + prompt to VLM, return (think2, think3) or raise."""
    global_512, crop, overlay = load_images(record)
    prompt = build_prompt(record)

    image_blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_to_b64(global_512)}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_to_b64(crop)}},
    ]
    if overlay is not None:
        image_blocks.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_to_b64(overlay)}}
        )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": prompt}]
        }]
    )

    # Extract text
    text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text

    # Parse JSON from response
    text = text.strip()
    # Handle possible markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    result = json.loads(text)
    return result["think2"], result["think3"]


def rewrite_assistant_text(record: dict, think2: str, think3: str) -> str:
    text = record["assistant_text"]

    # Replace Think 2: between </zoom> and <seg>
    text = re.sub(
        r"(</zoom>\n)<think>.*?</think>(\n<seg>)",
        f"\\1<think>{think2}</think>\\2",
        text, count=1, flags=re.DOTALL,
    )

    # Replace Think 3: between </seg> and <answer>
    text = re.sub(
        r"(</seg>\n)<think>.*?</think>(\n<answer>)",
        f"\\1<think>{think3}</think>\\2",
        text, count=1, flags=re.DOTALL,
    )

    return text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    with open(SRC) as f:
        records = [json.loads(line) for line in f]

    # Load checkpoint
    start_idx = 0
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            cp = json.load(f)
            start_idx = cp.get("last_done_idx", -1) + 1
        print(f"[resume] starting from index {start_idx} (checkpoint at {start_idx-1})")

    out_records = list(records)  # copy
    total = len(records)
    modified = 0
    errors = 0

    for i in range(start_idx, total):
        r = records[i]

        # Skip records without zoom+seg
        if "</zoom>" not in r.get("assistant_text", "") or "</seg>" not in r.get("assistant_text", ""):
            continue

        print(f"[{i+1}/{total}] {r['question_id']} ...", end=" ", flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                think2, think3 = call_vlm(r)
                out_records[i] = dict(r)
                out_records[i]["assistant_text"] = rewrite_assistant_text(r, think2, think3)
                modified += 1
                print(f"OK (t2={len(think2)}c, t3={len(think3)}c)")
                break
            except Exception as e:
                print(f"ERR attempt {attempt+1}: {e}", end=" ")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    print("SKIPPED")
                    errors += 1

        # Save checkpoint periodically
        if (i + 1) % SAVE_EVERY == 0:
            DST.parent.mkdir(parents=True, exist_ok=True)
            with open(DST, "w") as f:
                for rec in out_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(CHECKPOINT, "w") as f:
                json.dump({"last_done_idx": i, "modified": modified, "errors": errors}, f)
            print(f"  [saved checkpoint at {i+1}]")

    # Final save
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    CHECKPOINT.unlink(missing_ok=True)

    print(f"\nDone. Modified: {modified}, Errors: {errors}, Total: {total}")


if __name__ == "__main__":
    main()
