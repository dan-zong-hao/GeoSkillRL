"""
Convert DIOR-RSVG XML annotations into pretraining JSONL.

Pretrain format (aligned with TVP):
  Input:  <image> + "Locate the {label} in the image."
  Output: <|ref|>{label}<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|>

Coordinates normalized to [0, 999]. LRS-GRO data is reserved for the SFT stage.

Usage:
  python prepare_pretrain_data.py \
    --dior_root /root/autodl-tmp/dataset/DIOR-RSVG \
    --output_dir /root/autodl-tmp/VQA/speedup/pretrain/data
"""

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

VQA_ROOT = Path("/root/autodl-tmp/VQA")

QUESTION_TEMPLATES = [
    "Locate the {label} in the image.",
    "Where is the {label}?",
    "Find the {label} in this satellite image.",
    "Identify the location of the {label}.",
    "Point to the {label} in the image.",
    "{description}",
]


def normalize(val: float, size: int = 800) -> int:
    return max(0, min(999, int(round(val / size * 999))))


def clean_label(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").strip().lower()


def parse_dior_xml(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    size_el = root.find("size")
    w = int(size_el.find("width").text)
    h = int(size_el.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text or ""
        desc = obj.find("description").text or ""
        bndbox = obj.find("bndbox")
        objects.append({
            "label_raw": name,
            "label": clean_label(name),
            "description": desc.strip(),
            "bbox_pixel": [
                int(float(bndbox.find("xmin").text)),
                int(float(bndbox.find("ymin").text)),
                int(float(bndbox.find("xmax").text)),
                int(float(bndbox.find("ymax").text)),
            ],
        })
    return {"filename": filename, "width": w, "height": h, "objects": objects}


def build_pretrain_sample(
    image_name: str,
    label: str,
    bbox_norm: List[int],
    description: str,
) -> dict:
    use_desc = random.random() < 0.7
    if use_desc and description:
        question = random.choice([
            description,
            description + " Locate it in the image.",
        ])
    else:
        question = random.choice(QUESTION_TEMPLATES[:5]).format(
            label=label, description=description
        )

    box_str = f"[{bbox_norm[0]},{bbox_norm[1]},{bbox_norm[2]},{bbox_norm[3]}]"
    assistant_text = f"<|ref|>{label}<|/ref|><|box|>[{box_str}]<|/box|>"

    return {
        "image": image_name,
        "question": question,
        "assistant_text": assistant_text,
        "label": label,
        "boxes": [bbox_norm],
        "points": [],
    }


def build_negative_sample(image_name: str, neg_label: str) -> dict:
    templates = [
        "Locate the {label} in the image.",
        "Where is the {label}?",
        "Find the {label} in this satellite image.",
    ]
    question = random.choice(templates).format(label=neg_label)
    return {
        "image": image_name,
        "question": question,
        "assistant_text": "None.",
        "label": neg_label,
        "boxes": [],
        "points": [],
    }


def process_dior(dior_root: Path, neg_ratio: float) -> List[dict]:
    xml_dir = dior_root / "Annotations"
    img_dir = dior_root / "JPEGImages"

    all_labels: Dict[str, List[str]] = defaultdict(list)
    all_samples: List[dict] = []

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"Processing {len(xml_files)} DIOR-RSVG XML files...")

    skipped_degenerate = 0
    for xml_path in xml_files:
        ann = parse_dior_xml(xml_path)
        image_name = ann["filename"]
        img_path = img_dir / image_name
        if not img_path.exists():
            continue

        present_labels = set()
        for obj in ann["objects"]:
            label = obj["label"]
            bbox_pix = obj["bbox_pixel"]
            bbox_norm = [
                normalize(bbox_pix[0], ann["width"]),
                normalize(bbox_pix[1], ann["height"]),
                normalize(bbox_pix[2], ann["width"]),
                normalize(bbox_pix[3], ann["height"]),
            ]
            if bbox_norm[2] <= bbox_norm[0] or bbox_norm[3] <= bbox_norm[1]:
                skipped_degenerate += 1
                continue

            present_labels.add(label)
            sample = build_pretrain_sample(
                image_name, label, bbox_norm, obj["description"]
            )
            all_samples.append(sample)

        if present_labels:
            all_labels[image_name] = list(present_labels)

    # Negative samples
    neg_samples = []
    global_labels = sorted(set(
        lbl for labels in all_labels.values() for lbl in labels
    ))
    if not global_labels:
        global_labels = ["vehicle", "airplane", "ship", "bridge", "storagetank"]

    for image_name, present in all_labels.items():
        n_neg = max(1, int(
            len([s for s in all_samples if s["image"] == image_name]) * neg_ratio
        ))
        neg_candidates = [l for l in global_labels if l not in present]
        if not neg_candidates:
            neg_candidates = global_labels
        chosen = random.sample(neg_candidates, min(n_neg, len(neg_candidates)))
        for neg_label in chosen:
            neg_samples.append(build_negative_sample(image_name, neg_label))

    all_samples.extend(neg_samples)
    n_pos = len(all_samples) - len(neg_samples)
    print(f"DIOR-RSVG: {n_pos} positive + {len(neg_samples)} negative = {len(all_samples)} total")
    if skipped_degenerate:
        print(f"  (skipped {skipped_degenerate} degenerate boxes)")
    return all_samples


def split_by_image(samples: List[dict], val_ratio: float):
    by_image = defaultdict(list)
    for s in samples:
        by_image[s["image"]].append(s)

    image_names = list(by_image.keys())
    random.shuffle(image_names)
    n_val = max(1, int(len(image_names) * val_ratio))

    val_images = set(image_names[:n_val])
    train, val = [], []
    for img, ss in by_image.items():
        if img in val_images:
            val.extend(ss)
        else:
            train.extend(ss)
    return train, val


def write_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dior_root", type=str,
                        default="/root/autodl-tmp/dataset/DIOR-RSVG")
    parser.add_argument("--output_dir", type=str,
                        default=str(VQA_ROOT / "speedup" / "pretrain" / "data"))
    parser.add_argument("--neg_ratio", type=float, default=0.30)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_train", type=int, default=14000,
                        help="Max training samples (default 14K, matching TVP pretrain scale)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dior_samples = process_dior(Path(args.dior_root), args.neg_ratio)
    random.shuffle(dior_samples)
    train, val = split_by_image(dior_samples, args.val_ratio)

    # Downsample train to match TVP pretrain scale (~14K)
    if args.max_train and len(train) > args.max_train:
        # Stratified: keep proportional pos/neg ratio
        pos = [s for s in train if s["boxes"]]
        neg = [s for s in train if not s["boxes"]]
        pos_ratio = len(pos) / len(train)
        n_pos = int(args.max_train * pos_ratio)
        n_neg = args.max_train - n_pos
        random.shuffle(pos)
        random.shuffle(neg)
        train = pos[:n_pos] + neg[:n_neg]
        random.shuffle(train)
        print(f"Downsampled train to {len(train)} ({n_pos} pos / {n_neg} neg)")

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    n_pos = sum(1 for s in train if s["boxes"])
    n_neg = sum(1 for s in train if not s["boxes"])
    print(f"\n{'='*50}")
    print(f"Train: {len(train)} samples ({n_pos} pos / {n_neg} neg) -> {train_path}")
    print(f"Val:   {len(val)} samples -> {val_path}")
    print(f"Unique train images: {len(set(s['image'] for s in train))}")
    print(f"Unique val images:   {len(set(s['image'] for s in val))}")

    label_counts = defaultdict(int)
    for s in train:
        if s["boxes"]:
            label_counts[s["label"]] += 1
    print(f"Unique labels: {len(label_counts)}")
    print(f"Top 15 labels: {sorted(label_counts.items(), key=lambda x: -x[1])[:15]}")


if __name__ == "__main__":
    main()
