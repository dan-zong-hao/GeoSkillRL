"""
Generate ablation comparison markdown: SFT baseline vs Pretrain+SFT.

Samples 10 correct (both correct or IoU improved) and 20 wrong (both wrong)
examples. Draws GT/pred bboxes on global images, includes actual crops from
original images, and writes a markdown file.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VQA_ROOT = Path("/root/autodl-tmp/VQA")

# Paths
SFT_BASELINE = VQA_ROOT / "speedup/origin/eval/sft_casual_test_results/sft_causal_cot_predictions.jsonl"
PRETRAIN_SFT = VQA_ROOT / "speedup/unsloth/eval/results/sft_unsloth_4gpu_predictions.jsonl"
OUT_DIR = VQA_ROOT / "speedup/unsloth/eval/results/comparison_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = OUT_DIR / "bbox_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR = OUT_DIR / "crops"
CROP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIRS = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
]

COLORS = {"gt": "red", "baseline": "blue", "ours": "green",
          "gt_rgb": (255, 0, 0), "baseline_rgb": (0, 0, 255), "ours_rgb": (0, 200, 0)}


def find_image(name: str) -> Optional[Path]:
    # Direct match
    for d in IMAGE_DIRS:
        p = d / name
        if p.exists():
            return p
    # Try without extension, match any
    base = os.path.splitext(name)[0]
    for d in IMAGE_DIRS:
        if not d.exists():
            continue
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            p = d / (base + ext)
            if p.exists():
                return p
    # Recursive search as fallback in lrs_gro dirs
    for d in IMAGE_DIRS:
        if not d.exists():
            continue
        for p in d.rglob(name):
            return p
        for p in d.rglob(base + ".*"):
            return p
    return None


def load_jsonl(path: Path) -> dict:
    """Load predictions keyed by question_id."""
    data = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qid = d.get("question_id", "")
            if qid:
                data[qid] = d
    return data


def extract_bbox_1024(pred):
    """Extract bbox in 1024 space from prediction or trajectory."""
    bbox = pred.get("bbox_pred_1024")
    if bbox and len(bbox) == 4:
        return bbox
    return None


def get_iou(pred):
    return pred.get("iou", 0.0) or 0.0


def get_answer(pred):
    return pred.get("final_answer") or ""


def get_gt_answer(pred):
    return pred.get("ground_truth") or ""


def get_actions(pred):
    return pred.get("actions") or []


def get_trajectory_full(pred):
    """Get complete trajectory text."""
    return pred.get("trajectory", "")


def draw_bboxes_on_full(image: Image.Image, bboxes_config, img_scale: float) -> Image.Image:
    """Draw bboxes on full-size image with text labels.

    bboxes_config: list of (label, bbox_1024, color_rgb, line_width)
    img_scale = max(w, h) / 1024.0
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for label, bbox, color, lw in bboxes_config:
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v * img_scale) for v in bbox]
        # Rectangle with thick border
        for i in range(lw):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        # Text label at top-left corner of bbox
        text_w = len(label) * 10 + 8
        text_h = 22
        label_y = y1 - text_h if y1 > text_h else y1 + 2
        draw.rectangle([x1, label_y, x1 + text_w, label_y + text_h], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)

    return img


def resize_for_display(img: Image.Image, max_size: int = 800) -> Image.Image:
    """Resize image for markdown display (returns Image only, unlike dispatcher's resize_image)."""
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img.copy()


def crop_from_original(image: Image.Image, bbox_1024, scale_factor: float,
                       min_size: int = 256) -> Optional[Image.Image]:
    """Crop original image using bbox in 1024 space."""
    if bbox_1024 is None or len(bbox_1024) != 4:
        return None
    x1 = int(bbox_1024[0] * scale_factor)
    y1 = int(bbox_1024[1] * scale_factor)
    x2 = int(bbox_1024[2] * scale_factor)
    y2 = int(bbox_1024[3] * scale_factor)
    # Ensure valid order
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    W, H = image.width, image.height
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < min_size or bh < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = min_size // 2
        nx1 = max(0, cx - half)
        ny1 = max(0, cy - half)
        nx2 = min(W, nx1 + min_size)
        ny2 = min(H, ny1 + min_size)
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return image.crop((int(nx1), int(ny1), int(nx2), int(ny2)))
    if bw <= 0 or bh <= 0:
        return None
    return image.crop((x1, y1, x2, y2))


def write_md_sample(i, qid, sample, baseline, ours, fout):
    """Write one sample to markdown."""

    gt_answer = get_gt_answer(baseline) or get_gt_answer(ours)
    gt_bbox = baseline.get("bbox_gt_1024") or ours.get("bbox_gt_1024")
    bbox_baseline = extract_bbox_1024(baseline)
    bbox_ours = extract_bbox_1024(ours)
    iou_baseline = get_iou(baseline)
    iou_ours = get_iou(ours)
    ans_baseline = get_answer(baseline)
    ans_ours = get_answer(ours)
    qtype = baseline.get("type", "unknown")
    category = baseline.get("category", "unknown")
    question = baseline.get("question", "")

    # Load image
    image_name = sample.get("image_name", sample.get("image", ""))
    img_path = find_image(image_name)
    if img_path is None:
        fout.write(f"### #{i} {qid} - IMAGE NOT FOUND\n\n---\n\n")
        return

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(img_path).convert("RGB")
    scale_factor = max(image.size) / 1024.0

    # Generate visualization (same logic as make_composite_viz in original):
    # Draw bboxes on FULL-SIZE image, then resize for display.
    img_scale = max(image.size) / 1024.0

    bboxes_config = []
    if gt_bbox:
        bboxes_config.append(("GT", gt_bbox, COLORS["gt_rgb"], 5))
    if bbox_baseline:
        bboxes_config.append(("Base", bbox_baseline, COLORS["baseline_rgb"], 3))
    if bbox_ours:
        bboxes_config.append(("Ours", bbox_ours, COLORS["ours_rgb"], 3))

    viz_img = draw_bboxes_on_full(image, bboxes_config, img_scale)
    viz_name = f"{qid}.jpg"
    viz_path = VIZ_DIR / viz_name
    viz_img.save(viz_path, quality=90)

    # Crop from Baseline bbox
    crop1_img = None
    if bbox_baseline:
        crop1_img = crop_from_original(image, bbox_baseline, scale_factor)
    if crop1_img is not None:
        crop1_path = CROP_DIR / f"{qid}_base.jpg"
        crop1_img.save(crop1_path, quality=90)
        crop1_html = f'<a href="crops/{qid}_base.jpg"><img src="crops/{qid}_base.jpg" style="max-width:180px; border:1px solid #ddd;"></a>'
    else:
        crop1_html = "N/A"

    # Crop from Ours bbox
    crop2_img = None
    if bbox_ours:
        crop2_img = crop_from_original(image, bbox_ours, scale_factor)
    if crop2_img is not None:
        crop2_path = CROP_DIR / f"{qid}_ours.jpg"
        crop2_img.save(crop2_path, quality=90)
        crop2_html = f'<a href="crops/{qid}_ours.jpg"><img src="crops/{qid}_ours.jpg" style="max-width:180px; border:1px solid #ddd;"></a>'
    else:
        crop2_html = "N/A"

    # Determine status
    ours_correct = ours.get("correct_final", False)
    baseline_correct = baseline.get("correct_final", False)
    if ours_correct and baseline_correct:
        status = "Both Correct"
    elif ours_correct and not baseline_correct:
        status = "FIX: Ours correct"
    elif not ours_correct and baseline_correct:
        status = "REGRESS: Baseline correct"
    else:
        status = "Both Wrong"

    fout.write(f'<a name="{qid}"></a>\n')
    fout.write(f'<h3>#{i} &nbsp; {qid} &nbsp; <code>{status}</code></h3>\n\n')
    fout.write('<table><tr>\n')
    fout.write('<td style="vertical-align:top; width:30%;">\n')
    fout.write(f'<p><b>Question</b><br>{question}</p>\n')
    fout.write(f'<p><b>Type</b>: {qtype} | <b>Category</b>: {category}</p>\n')
    fout.write(f'<p><b>GT Answer</b>: <code>{gt_answer}</code></p>\n')
    if gt_bbox:
        fout.write(f'<p><b>GT bbox (1024)</b>: <code>{gt_bbox}</code></p>\n')
    fout.write(f'<p><b>Baseline SFT</b>: IoU=<code>{iou_baseline:.4f}</code> '
               f'Answer=<code>{ans_baseline}</code></p>\n')
    fout.write(f'<p><b>Pretrain+SFT</b>: IoU=<code>{iou_ours:.4f}</code> '
               f'Answer=<code>{ans_ours}</code></p>\n')
    fout.write('</td>\n')

    fout.write('<td style="vertical-align:top; width:45%;">\n')
    fout.write('<p><b>BBox Compare</b> (GT=Red, Baseline=Blue, Ours=Green)</p>\n')
    fout.write(f'<img src="bbox_viz/{qid}.jpg" style="max-width:100%; border:1px solid #ddd;">\n')
    fout.write('</td>\n')

    fout.write('<td style="vertical-align:top; width:12%;">\n')
    fout.write('<p><b>Crop (Base bbox)</b></p>\n')
    fout.write(f'{crop1_html}\n')
    fout.write('</td>\n')
    fout.write('<td style="vertical-align:top; width:12%;">\n')
    fout.write('<p><b>Crop (Ours bbox)</b></p>\n')
    fout.write(f'{crop2_html}\n')
    fout.write('</td>\n')
    fout.write('</tr></table>\n\n')

    # Full trajectories
    fout.write(f'**Baseline SFT**\n```text\n{get_trajectory_full(baseline)}\n```\n\n')
    fout.write(f'**Pretrain+SFT**\n```text\n{get_trajectory_full(ours)}\n```\n\n')
    fout.write('---\n\n')


def main():
    random.seed(42)

    # Load predictions
    print("Loading SFT baseline...")
    baseline_data = load_jsonl(SFT_BASELINE)
    print(f"  {len(baseline_data)} samples")

    print("Loading Pretrain+SFT...")
    ours_data = load_jsonl(PRETRAIN_SFT)
    print(f"  {len(ours_data)} samples")

    # Find common question_ids
    common_ids = sorted(set(baseline_data.keys()) & set(ours_data.keys()))
    print(f"Common samples: {len(common_ids)}")

    # Categorize
    both_correct = []
    both_wrong = []
    fix = []
    regress = []

    for qid in common_ids:
        b = baseline_data[qid]
        o = ours_data[qid]
        b_correct = b.get("correct_final", False)
        o_correct = o.get("correct_final", False)

        if o_correct and b_correct:
            both_correct.append(qid)
        elif o_correct and not b_correct:
            fix.append(qid)
        elif not o_correct and b_correct:
            regress.append(qid)
        else:
            both_wrong.append(qid)

    print(f"Both correct: {len(both_correct)}")
    print(f"FIX (ours correct): {len(fix)}")
    print(f"REGRESS (baseline correct): {len(regress)}")
    print(f"Both wrong: {len(both_wrong)}")

    # Sample: 10 correct (from both_correct + fix), 20 wrong (from both_wrong + regress)
    correct_pool = both_correct + fix
    random.shuffle(correct_pool)
    wrong_pool = both_wrong + regress
    random.shuffle(wrong_pool)

    sample_correct = correct_pool[:10]
    sample_wrong = wrong_pool[:20]

    # Prefer samples with bbox
    def prioritize_bbox(pool, n):
        with_bbox = [q for q in pool if baseline_data[q].get("bbox_gt_1024")]
        without = [q for q in pool if not baseline_data[q].get("bbox_gt_1024")]
        result = with_bbox[:n]
        if len(result) < n:
            result += without[:n - len(result)]
        return result

    sample_correct = prioritize_bbox(correct_pool, 10)
    sample_wrong = prioritize_bbox(wrong_pool, 20)

    # Write markdown
    md_path = OUT_DIR / "ablation_pretrain_sft_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation: SFT Baseline (no pretrain) vs Pretrain + SFT\n\n")
        f.write(f"**Dataset**: LRS-GRO test (9,734 samples)<br>\n")
        f.write(f"**Bbox colors**: GT=Red, Baseline SFT=Blue, Pretrain+SFT=Green<br>\n\n")
        f.write(f"**Models**: Baseline=ckpt_sft_full_qwen35_causal_cot (no pretrain), "
                f"Ours=sft_unsloth (DIOR-RSVG pretrain + full-FT SFT)<br>\n\n")

        f.write("## Summary\n\n")
        f.write("| Method | Avg.Acc | APO IoU | Grammar OK | Zoom Fire | Seg Fire |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")

        # Compute summary stats from common data
        def avg_acc(data, ids):
            return 100 * sum(1 for q in ids if data[q].get("correct_final")) / max(len(ids), 1)

        def avg_iou(data, ids):
            ious = [data[q].get("iou", 0) or 0 for q in ids if data[q].get("bbox_gt_1024")]
            return 100 * sum(ious) / max(len(ious), 1)

        common = common_ids
        bl_acc = avg_acc(baseline_data, common)
        our_acc = avg_acc(ours_data, common)
        bl_iou = avg_iou(baseline_data, common)
        our_iou = avg_iou(ours_data, common)

        f.write(f"| SFT Baseline (no pretrain) | {bl_acc:.2f} | {bl_iou:.2f} | "
                f"{(sum(1 for q in common if baseline_data[q].get('grammar_ok'))/len(common)*100):.1f} | "
                f"{(sum(1 for q in common if 'zoom' in str(baseline_data[q].get('actions',[])))/len(common)*100):.1f} | "
                f"{(sum(1 for q in common if 'seg' in str(baseline_data[q].get('actions',[])))/len(common)*100):.1f} |\n")
        f.write(f"| Pretrain + SFT | {our_acc:.2f} | {our_iou:.2f} | "
                f"{(sum(1 for q in common if ours_data[q].get('grammar_ok'))/len(common)*100):.1f} | "
                f"{(sum(1 for q in common if 'zoom' in str(ours_data[q].get('actions',[])))/len(common)*100):.1f} | "
                f"{(sum(1 for q in common if 'seg' in str(ours_data[q].get('actions',[])))/len(common)*100):.1f} |\n\n")

        f.write("## Transition\n\n")
        f.write(f"| Category | Count |\n|---|---:|\n")
        f.write(f"| FIX (Baseline wrong → Ours correct) | {len(fix)} |\n")
        f.write(f"| REGRESS (Baseline correct → Ours wrong) | {len(regress)} |\n")
        f.write(f"| BOTH WRONG | {len(both_wrong)} |\n")
        f.write(f"| BOTH CORRECT | {len(both_correct)} |\n")
        f.write(f"| **Net Fix** | **{len(fix)-len(regress)}** |\n\n")

        # Correct samples
        f.write(f"## Correct Samples ({len(sample_correct)})\n\n")
        # Compact table first
        f.write("| # | QID | Type | Baseline | Ours |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, qid in enumerate(sample_correct, 1):
            b, o = baseline_data[qid], ours_data[qid]
            f.write(f'| {idx} | {qid} | {b.get("type","")} | '
                    f'Baseline IoU={get_iou(b):.3f} Ans={get_answer(b)} | '
                    f'Ours IoU={get_iou(o):.3f} Ans={get_answer(o)} |\n')
        f.write("\n")
        # Then detailed cards
        for idx, qid in enumerate(sample_correct, 1):
            sample = baseline_data[qid]
            write_md_sample(idx, qid, sample, baseline_data[qid], ours_data[qid], f)

        # Wrong samples
        f.write(f"## Wrong Samples ({len(sample_wrong)})\n\n")
        # Compact table first
        f.write("| # | QID | Type | Baseline | Ours |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, qid in enumerate(sample_wrong, 1):
            b, o = baseline_data[qid], ours_data[qid]
            f.write(f'| {idx} | {qid} | {b.get("type","")} | '
                    f'Baseline IoU={get_iou(b):.3f} Ans={get_answer(b)} | '
                    f'Ours IoU={get_iou(o):.3f} Ans={get_answer(o)} |\n')
        f.write("\n")
        # Then detailed cards
        for idx, qid in enumerate(sample_wrong, 1):
            sample = baseline_data[qid]
            write_md_sample(idx, qid, sample, baseline_data[qid], ours_data[qid], f)

    print(f"\nMarkdown written to: {md_path}")
    print(f"Viz images: {VIZ_DIR}")
    print(f"Crops: {CROP_DIR}")


if __name__ == "__main__":
    main()
