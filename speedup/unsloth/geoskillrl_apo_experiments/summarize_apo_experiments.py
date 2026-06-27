#!/usr/bin/env python3
"""Summarize APO-focused experiment outputs with shared offline metrics."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VQA_ROOT = Path('/root/autodl-tmp/VQA')
DEFAULT_OUTPUT = VQA_ROOT / 'speedup/unsloth/geoskillrl_apo_experiments/results/apo_summary'
DEFAULT_PREDICTIONS = {
    'no_seg_zoom_only': VQA_ROOT / 'speedup/unsloth/eval_no_seg/results/zoom_only_fast_4gpu_repaired_predictions.jsonl',
    'pretrain_current_sft_seg': VQA_ROOT / 'speedup/unsloth/eval/results/sft_unsloth_4gpu_predictions.jsonl',
    'skill_sft_round0_eval500': VQA_ROOT / 'speedup/unsloth/skillrl/eval_results/skill_sft_round0_eval500_v2_predictions.jsonl',
}
KNOWN_BASELINES = {
    'no_pretrain_sft_reported': {
        'source': 'existing comparison report',
        'Global': 72.94,
        'Region': 55.23,
        'Object': 60.70,
        'Avg.Acc': 62.34,
        'APO IoU': 32.82,
        'Grammar': 99.9,
        'Zoom Fire': 75.7,
    }
}
ANSWER_CLOSED_RE = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL | re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r'<answer>\s*(.*)$', re.DOTALL | re.IGNORECASE)
CROP_CLAIM_CORRECT_RE = re.compile(
    r'\[Crop Verification\].{0,360}?'
    r'(?:correctly|does contain|contains the target|provided crop image contains|'
    r'crop image contains|target[^.\n]{0,80}visible|clearly shows|shows the[^.\n]{0,80}target|'
    r'full [^.\n]{0,80}clearly visible|zoom level sufficient)',
    re.IGNORECASE | re.DOTALL,
)
CROP_CLAIM_FAILED_RE = re.compile(
    r'\[Crop Verification\].{0,360}?'
    r'(?:does not contain|doesn\'t contain|failed|not contain|no [^.\n]{0,80}visible|'
    r'wrong|different area|crop failed|cannot find|not visible|disregard its content)',
    re.IGNORECASE | re.DOTALL,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(value: Any) -> str:
    return str(value or '').strip().lower()


def robust_answer_from_trajectory(text: str) -> str | None:
    if not text:
        return None
    m = ANSWER_CLOSED_RE.search(text)
    if m:
        return m.group(1).strip() or None
    m = ANSWER_OPEN_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()
    ans = re.split(r'(?:<\|im_end\|>|\n\[TURN\d+\]|\n<think>|Question Intent:|Localization Strategy:)', ans, maxsplit=1)[0]
    ans = re.sub(r'</answer\s*$', '', ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r'</\s*$', '', ans).strip()
    return ans or None


def final_answer(row: dict[str, Any]) -> str | None:
    return row.get('final_answer') or robust_answer_from_trajectory(row.get('trajectory') or '')


def answer_matches(gt: Any, pred: Any) -> bool:
    g, p = norm(gt), norm(pred)
    return bool(g and p and (g == p or g in p or p in g))


def get_iou(row: dict[str, Any]) -> float:
    for key in ('iou', 'iou_apo512', 'apo_iou'):
        if row.get(key) is not None:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                pass
    return 0.0


def has_bbox_gt(row: dict[str, Any]) -> bool:
    return bool(row.get('bbox_gt_1024') or row.get('bbox_ref') or row.get('bbox_gt_orig'))


def has_bbox_pred(row: dict[str, Any]) -> bool:
    return bool(row.get('bbox_pred_1024') or row.get('bbox_pred_orig'))


def crop_claim(row: dict[str, Any]) -> str:
    text = row.get('trajectory') or ''
    pos = bool(CROP_CLAIM_CORRECT_RE.search(text))
    neg = bool(CROP_CLAIM_FAILED_RE.search(text))
    if pos and not neg:
        return 'claimed_correct'
    if neg and not pos:
        return 'claimed_failed'
    if pos and neg:
        return 'mixed'
    return 'unknown'


def acc(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(1 for r in rows if answer_matches(r.get('ground_truth'), final_answer(r)) or r.get('correct_final')) / len(rows)


def summarize_prediction(name: str, path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[norm(row.get('type')) or 'unknown'].append(row)
    bbox_rows = [r for r in rows if has_bbox_gt(r)]
    pred_bbox_rows = [r for r in bbox_rows if has_bbox_pred(r)]
    wrong03 = [r for r in pred_bbox_rows if get_iou(r) < 0.3]
    false_grounded = [r for r in wrong03 if answer_matches(r.get('ground_truth'), final_answer(r)) or r.get('correct_final')]
    false_grounded_claimed = [r for r in false_grounded if crop_claim(r) == 'claimed_correct']
    wrong_claimed = [r for r in wrong03 if crop_claim(r) == 'claimed_correct']
    spatial_values = [str(r.get('spatial_violation')) for r in rows if r.get('spatial_violation')]
    return {
        'name': name,
        'path': str(path),
        'n': len(rows),
        'Global': acc(by_type.get('global', [])),
        'Region': acc(by_type.get('region', [])),
        'Object': acc(by_type.get('object', [])),
        'Avg.Acc': acc(rows),
        'bbox_rows': len(bbox_rows),
        'bbox_pred_rate': 100.0 * len(pred_bbox_rows) / max(len(bbox_rows), 1),
        'APO IoU': 100.0 * sum(get_iou(r) for r in pred_bbox_rows) / max(len(pred_bbox_rows), 1),
        'Crop Hit@0.3': 100.0 * sum(1 for r in pred_bbox_rows if get_iou(r) >= 0.3) / max(len(pred_bbox_rows), 1),
        'Crop Hit@0.5': 100.0 * sum(1 for r in pred_bbox_rows if get_iou(r) >= 0.5) / max(len(pred_bbox_rows), 1),
        'False-grounded Correct@0.3': 100.0 * len(false_grounded_claimed) / max(len(pred_bbox_rows), 1),
        'Wrong BBox Claim-Correct@0.3': 100.0 * len(wrong_claimed) / max(len(wrong03), 1),
        'Grammar': 100.0 * sum(1 for r in rows if r.get('grammar_ok')) / max(len(rows), 1),
        'Zoom Fire': 100.0 * sum(1 for r in rows if 'zoom' in (r.get('actions') or [])) / max(len(rows), 1),
        'spatial_violation_counts': dict(Counter(spatial_values).most_common(20)),
    }


def write_markdown(summary: dict[str, Any], out: Path) -> None:
    lines = ['# GeoSkillRL APO Experiment Summary', '']
    lines.append('| Method | Avg.Acc | APO IoU | Hit@0.3 | Hit@0.5 | False-grounded | Grammar | Zoom |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for name, row in summary['computed'].items():
        lines.append(
            f"| {name} | {row['Avg.Acc']:.2f} | {row['APO IoU']:.2f} | "
            f"{row['Crop Hit@0.3']:.2f} | {row['Crop Hit@0.5']:.2f} | "
            f"{row['False-grounded Correct@0.3']:.2f} | {row['Grammar']:.2f} | {row['Zoom Fire']:.2f} |"
        )
    lines.extend(['', '## Known Reported Baselines', ''])
    lines.append('| Method | Avg.Acc | APO IoU | Grammar | Zoom |')
    lines.append('|---|---:|---:|---:|---:|')
    for name, row in summary['known'].items():
        lines.append(f"| {name} | {row['Avg.Acc']:.2f} | {row['APO IoU']:.2f} | {row['Grammar']:.2f} | {row['Zoom Fire']:.2f} |")
    out.write_text('\n'.join(lines) + '\n')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--prediction', action='append', default=[], help='name=/path/to/predictions.jsonl')
    args = p.parse_args()

    preds = dict(DEFAULT_PREDICTIONS)
    for item in args.prediction:
        if '=' not in item:
            raise SystemExit(f'--prediction must be name=path, got {item}')
        name, path = item.split('=', 1)
        preds[name] = Path(path)

    computed = {name: summarize_prediction(name, path) for name, path in preds.items() if path.exists()}
    summary = {'computed': computed, 'known': KNOWN_BASELINES}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'apo_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_markdown(summary, args.output_dir / 'apo_summary.md')
    print(json.dumps({
        'output_dir': str(args.output_dir),
        'computed_methods': list(computed),
        'known_methods': list(KNOWN_BASELINES),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
