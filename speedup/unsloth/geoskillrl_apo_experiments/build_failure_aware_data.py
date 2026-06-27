#!/usr/bin/env python3
"""Build failure-aware primitive SFT data with correct, partial, and re-zoom traces."""
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

VQA_ROOT = Path('/root/autodl-tmp/VQA')
DEFAULT_INPUT = VQA_ROOT / 'json_data/zoom_json/sft/sft-00000-of-00001.jsonl'
DEFAULT_OUTPUT = VQA_ROOT / 'json_data/zoom_json/geoskillrl_apo/failure_aware_primitive.jsonl'


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def clamp_box(b: list[int]) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
    x1 = max(0, min(1023, x1))
    y1 = max(0, min(1023, y1))
    x2 = max(1, min(1024, x2))
    y2 = max(1, min(1024, y2))
    if x2 <= x1:
        x2 = min(1024, x1 + 1)
    if y2 <= y1:
        y2 = min(1024, y1 + 1)
    return [x1, y1, x2, y2]


def zoom(label: str, bbox: list[int]) -> str:
    clean = (label or 'target').replace('<', '').replace('>', '').strip() or 'target'
    b = clamp_box(bbox)
    return f'<zoom><|ref|>{clean}<|/ref|><|box|>[[{b[0]},{b[1]},{b[2]},{b[3]}]]<|/box|></zoom>'


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def partial_bbox(gt: list[int], rng: random.Random) -> list[int]:
    x1, y1, x2, y2 = gt
    w, h = max(2, x2 - x1), max(2, y2 - y1)
    mode = rng.choice(['left', 'right', 'top', 'bottom'])
    if mode == 'left':
        return clamp_box([x1, y1, x1 + int(w * 0.55), y2])
    if mode == 'right':
        return clamp_box([x2 - int(w * 0.55), y1, x2, y2])
    if mode == 'top':
        return clamp_box([x1, y1, x2, y1 + int(h * 0.55)])
    return clamp_box([x1, y2 - int(h * 0.55), x2, y2])


def wrong_bbox(gt: list[int], rng: random.Random) -> list[int]:
    x1, y1, x2, y2 = gt
    w, h = x2 - x1, y2 - y1
    for _ in range(20):
        dx = int(rng.choice([-1, 1]) * max(96, w * rng.uniform(1.2, 2.4)))
        dy = int(rng.choice([-1, 1]) * max(96, h * rng.uniform(1.2, 2.4)))
        cand = clamp_box([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
        if iou(cand, gt) < 0.1:
            return cand
    return clamp_box([max(0, 1024 - x2), max(0, 1024 - y2), max(1, 1024 - x1), max(1, 1024 - y1)])


def base_think(row: dict[str, Any]) -> str:
    return (
        f"{str(row.get('global') or '').strip()}\n\n"
        f"{str(row.get('stage_1_reasoning') or '').strip()}\n\n"
        "I will treat the first bbox as a visual hypothesis and verify the returned crop before answering."
    ).strip()


def answer_text(row: dict[str, Any]) -> str:
    return str(row.get('ground_truth') or '').strip()


def local_reason(row: dict[str, Any]) -> str:
    return str(row.get('stage_2_reasoning') or '').strip()


def build_positive(row: dict[str, Any], gt: list[int], label: str) -> dict[str, Any]:
    out = deepcopy(row)
    out['assistant_text'] = (
        f"<think>{base_think(row)}</think>\n"
        f"{zoom(label, gt)}\n"
        f"<think>[Crop Verification] The crop contains the intended referent \"{label}\". "
        f"I can use the local evidence for the final answer.\n\n[Analysis] {local_reason(row)}</think>\n"
        f"<answer>{answer_text(row)}</answer>"
    )
    out['tool_bboxes_1024'] = [gt]
    out['failure_mode'] = 'correct_crop'
    return out


def build_partial(row: dict[str, Any], gt: list[int], label: str, first: list[int]) -> dict[str, Any]:
    out = deepcopy(row)
    out['bbox'] = first
    out['assistant_text'] = (
        f"<think>{base_think(row)}</think>\n"
        f"{zoom(label, first)}\n"
        f"<think>[Crop Verification] The crop only partially contains the intended referent \"{label}\". "
        "The evidence is incomplete, so I should re-localize the complete target region from the global image.</think>\n"
        f"{zoom(label, gt)}\n"
        f"<think>[Crop Verification] The second crop contains the complete intended referent \"{label}\". "
        f"Now I can answer from the verified local evidence.\n\n[Analysis] {local_reason(row)}</think>\n"
        f"<answer>{answer_text(row)}</answer>"
    )
    out['tool_bboxes_1024'] = [first, gt]
    out['failure_mode'] = 'partial_crop_rezoom'
    return out


def build_wrong(row: dict[str, Any], gt: list[int], label: str, first: list[int]) -> dict[str, Any]:
    out = deepcopy(row)
    out['bbox'] = first
    out['assistant_text'] = (
        f"<think>{base_think(row)}</think>\n"
        f"{zoom(label, first)}\n"
        f"<think>[Crop Verification] The crop does not contain the intended referent \"{label}\". "
        "I should not claim this crop is correct. I will return to the global image and issue a corrected zoom.</think>\n"
        f"{zoom(label, gt)}\n"
        f"<think>[Crop Verification] The second crop contains the intended referent \"{label}\". "
        f"Now I can answer from verified evidence.\n\n[Analysis] {local_reason(row)}</think>\n"
        f"<answer>{answer_text(row)}</answer>"
    )
    out['tool_bboxes_1024'] = [first, gt]
    out['failure_mode'] = 'wrong_crop_rezoom'
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--include_global', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle', action='store_true')
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]

    out: list[dict[str, Any]] = []
    for row in rows:
        bbox = row.get('bbox') or []
        if not bbox or len(bbox) != 4:
            if args.include_global:
                g = deepcopy(row)
                g['assistant_text'] = f"<think>{base_think(row)}\n\n{local_reason(row)}</think>\n<answer>{answer_text(row)}</answer>"
                g['tool_bboxes_1024'] = []
                g['failure_mode'] = 'global_pass_through'
                out.append(g)
            continue
        gt = clamp_box(bbox)
        label = str(row.get('label') or 'target')
        out.append(build_positive(row, gt, label))
        out.append(build_partial(row, gt, label, partial_bbox(gt, rng)))
        out.append(build_wrong(row, gt, label, wrong_bbox(gt, rng)))

    if args.shuffle:
        rng.shuffle(out)
    write_jsonl(args.output, out)
    print(json.dumps({
        'input': str(args.input),
        'output': str(args.output),
        'rows': len(out),
        'by_mode': {mode: sum(1 for r in out if r.get('failure_mode') == mode) for mode in sorted({r.get('failure_mode') for r in out})},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
