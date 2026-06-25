#!/usr/bin/env python3
"""Audit 1024 bbox coordinate modes with overlay images and reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speedup.unsloth.skillrl.verl_grpo.agent.crop_environment import (  # noqa: E402
    bbox_1024_to_original,
    draw_overlay,
    find_image,
    is_nearly_square,
)
from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import validate_bbox_1024  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_samples(rows: list[dict[str, Any]], image_dirs: list[str], n_each: int) -> tuple[list[dict], list[dict]]:
    non_square: list[dict] = []
    square: list[dict] = []
    for row in rows:
        bbox = validate_bbox_1024(row.get("bbox"), require_in_range=True)
        if bbox is None:
            continue
        path = find_image(str(row.get("image_name") or row.get("image") or ""), image_dirs)
        if path is None:
            continue
        with Image.open(path) as image:
            size = image.size
        item = {"row": row, "bbox": bbox, "image_path": path, "image_size": size}
        if is_nearly_square(size):
            if len(square) < n_each:
                square.append(item)
        else:
            if len(non_square) < n_each:
                non_square.append(item)
        if len(square) >= n_each and len(non_square) >= n_each:
            break
    return non_square, square


def audit(samples: list[dict], name: str, output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        row = sample["row"]
        bbox = sample["bbox"]
        image_path = sample["image_path"]
        size = sample["image_size"]
        xy = bbox_1024_to_original(bbox, size, coord_mode="xy")
        max_side = bbox_1024_to_original(bbox, size, coord_mode="max_side")
        qid = str(row.get("question_id") or idx)
        overlay = output_dir / "overlays" / f"{name}_{idx:03d}_{qid}.png"
        draw_overlay(image_path, {"xy": xy, "max_side": max_side}, overlay)
        records.append(
            {
                "question_id": qid,
                "image_path": str(image_path),
                "image_size": list(size),
                "bbox_1024": bbox,
                "xy_bbox_orig": xy,
                "max_side_bbox_orig": max_side,
                "overlay": str(overlay),
            }
        )
    return records


def write_html(report: dict[str, Any], output_path: Path) -> None:
    lines = ["<html><body><h1>GeoSkillRL bbox coordinate audit</h1>"]
    for group in ("non_square", "near_square"):
        lines.append(f"<h2>{group}</h2>")
        for item in report[group]:
            lines.append("<div style='margin:16px 0'>")
            lines.append(f"<p>{item['question_id']} size={item['image_size']}</p>")
            lines.append(f"<img src='{Path(item['overlay']).name if False else item['overlay']}' style='max-width:900px'>")
            lines.append("</div>")
    lines.append("</body></html>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data/bbox_audit"))
    parser.add_argument("--image-dir", action="append", default=[])
    parser.add_argument("--n-each", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.jsonl)
    non_square, square = collect_samples(rows, args.image_dir, args.n_each)
    report = {
        "jsonl": str(args.jsonl),
        "requested_each": args.n_each,
        "non_square": audit(non_square, "non_square", output_dir),
        "near_square": audit(square, "near_square", output_dir),
        "decision": "unverified_keep_both_modes",
    }
    (output_dir / "bbox_coordinate_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_html(report, output_dir / "bbox_coordinate_audit.html")
    print(json.dumps({"non_square": len(non_square), "near_square": len(square)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
