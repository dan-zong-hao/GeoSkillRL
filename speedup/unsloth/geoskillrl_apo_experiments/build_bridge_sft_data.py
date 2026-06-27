#!/usr/bin/env python3
"""Build primitive-grammar Bridge-SFT data for GeoSkillRL APO experiments.

This script only writes to the requested output path. It does not modify the
source zoom data or any existing trainer/evaluator.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

VQA_ROOT = Path('/root/autodl-tmp/VQA')
DEFAULT_INPUT = VQA_ROOT / 'json_data/zoom_json/sft/sft_verified_zoom.jsonl'
DEFAULT_REHEARSAL = VQA_ROOT / 'speedup/unsloth/pretrain/data/train.jsonl'
DEFAULT_OUTPUT = VQA_ROOT / 'json_data/zoom_json/geoskillrl_apo/bridge_sft_primitive.jsonl'

ZOOM_JSON_RE = re.compile(r'<zoom>\s*(\[.*?\])\s*</zoom>', re.DOTALL)
BBOX_JSON_RE = re.compile(r'"bbox_2d"\s*:\s*\[(.*?)\]', re.DOTALL)
LABEL_JSON_RE = re.compile(r'"label"\s*:\s*"(.*?)"', re.DOTALL)
PRIMITIVE_ZOOM_RE = re.compile(r'<zoom>.*?<\|box\|>\s*\[\[(.*?)\]\]\s*<\|/box\|>.*?</zoom>', re.DOTALL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_bbox_numbers(raw: str) -> list[int] | None:
    try:
        nums = [int(round(float(x.strip()))) for x in raw.split(',')]
    except ValueError:
        return None
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = nums
    x1 = max(0, min(1023, x1))
    y1 = max(0, min(1023, y1))
    x2 = max(1, min(1024, x2))
    y2 = max(1, min(1024, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def primitive_zoom(label: str, bbox: list[int]) -> str:
    clean = (label or 'target').replace('<', '').replace('>', '').strip() or 'target'
    return f'<zoom><|ref|>{clean}<|/ref|><|box|>[[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]]<|/box|></zoom>'


def convert_zoom_json_to_primitive(text: str, fallback_label: str = '') -> tuple[str, list[list[int]]]:
    tool_bboxes: list[list[int]] = []

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        bm = BBOX_JSON_RE.search(body)
        if not bm:
            return match.group(0)
        bbox = parse_bbox_numbers(bm.group(1))
        if not bbox:
            return match.group(0)
        lm = LABEL_JSON_RE.search(body)
        label = lm.group(1).strip() if lm else fallback_label
        tool_bboxes.append(bbox)
        return primitive_zoom(label, bbox)

    converted = ZOOM_JSON_RE.sub(repl, text or '')
    return converted, tool_bboxes


def build_from_fields(row: dict[str, Any]) -> tuple[str, list[list[int]]]:
    bbox = row.get('bbox') or []
    gt = str(row.get('ground_truth') or '').strip()
    global_desc = str(row.get('global') or '').strip()
    stage_1 = str(row.get('stage_1_reasoning') or '').strip()
    stage_2 = str(row.get('stage_2_reasoning') or '').strip()
    label = str(row.get('label') or 'target').strip()

    if not bbox or len(bbox) != 4:
        assistant = f'<think>{global_desc}\n\n{stage_1}\n\n{stage_2}</think>\n<answer>{gt}</answer>'
        return assistant, []

    box = [int(round(float(x))) for x in bbox]
    think1 = (
        f'{global_desc}\n\n{stage_1}\n\n'
        f'I need to localize the referent as a target box before zooming.'
    ).strip()
    think2 = (
        f'[Crop Verification] The zoomed crop should contain the intended referent "{label}".\n\n'
        f'[Analysis] {stage_2}'
    ).strip()
    assistant = f'<think>{think1}</think>\n{primitive_zoom(label, box)}\n<think>{think2}</think>\n<answer>{gt}</answer>'
    return assistant, [box]


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    text = row.get('assistant_text') or ''
    zoom_count = text.count('</zoom>')
    tool_count = len(row.get('tool_bboxes_1024') or [])
    if zoom_count != tool_count:
        errors.append(f'zoom_count={zoom_count} tool_bboxes={tool_count}')
    if zoom_count and '<|box|>[[' not in text:
        errors.append('missing primitive box')
    return errors


def add_rehearsal(rows: list[dict[str, Any]], rehearsal_path: Path, ratio: float, seed: int) -> list[dict[str, Any]]:
    if ratio <= 0 or not rehearsal_path.exists() or not rows:
        return rows
    rng = random.Random(seed)
    rehearsal = load_jsonl(rehearsal_path)
    n = max(1, int(len(rows) * ratio))
    if len(rehearsal) > n:
        rehearsal = rng.sample(rehearsal, n)
    out = list(rows)
    for idx, src in enumerate(rehearsal):
        boxes = src.get('boxes') or []
        bbox = boxes[0] if boxes else []
        out.append({
            'question_id': f'DIOR-REHEARSAL-{idx}',
            'image_name': src.get('image'),
            'question': src.get('question') or f'Locate {src.get("label", "target")}.',
            'ground_truth': src.get('label') or '',
            'type': 'grounding_rehearsal',
            'category': 'grounding_rehearsal',
            'label': src.get('label') or '',
            'bbox': bbox,
            'scale': 1.0,
            'assistant_text': src.get('assistant_text') or '',
            'tool_bboxes_1024': [],
            'source': 'dior_rehearsal',
        })
    rng.shuffle(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--rehearsal_jsonl', type=Path, default=DEFAULT_REHEARSAL)
    p.add_argument('--rehearsal_ratio', type=float, default=0.05)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle', action='store_true')
    args = p.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]

    converted_rows: list[dict[str, Any]] = []
    validation_errors = 0
    for src in rows:
        row = deepcopy(src)
        assistant = row.get('assistant_text')
        if assistant:
            assistant, tool_bboxes = convert_zoom_json_to_primitive(assistant, str(row.get('label') or 'target'))
        else:
            assistant, tool_bboxes = build_from_fields(row)
        row['assistant_text'] = assistant
        row['tool_bboxes_1024'] = tool_bboxes
        row['bridge_format'] = 'primitive_zoom_ref_box'
        errs = validate_row(row)
        if errs:
            validation_errors += 1
            row['bridge_validation_errors'] = errs
        converted_rows.append(row)

    converted_rows = add_rehearsal(converted_rows, args.rehearsal_jsonl, args.rehearsal_ratio, args.seed)
    if args.shuffle:
        random.Random(args.seed).shuffle(converted_rows)

    write_jsonl(args.output, converted_rows)
    zoom_rows = [r for r in converted_rows if '</zoom>' in (r.get('assistant_text') or '')]
    print(json.dumps({
        'input': str(args.input),
        'output': str(args.output),
        'rows': len(converted_rows),
        'zoom_rows': len(zoom_rows),
        'rehearsal_rows': sum(1 for r in converted_rows if r.get('source') == 'dior_rehearsal'),
        'validation_errors': validation_errors,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
