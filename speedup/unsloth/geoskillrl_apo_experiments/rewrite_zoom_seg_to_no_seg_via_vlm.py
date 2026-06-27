#!/usr/bin/env python3
"""Rewrite zoom+seg causal SFT data into no-seg primitive-zoom SFT data via VLM.

This script intentionally lives in the isolated APO experiment folder. It reads
zoom+seg causal reasoning data, removes the <seg> action from assistant_text, and
asks a VLM to rewrite the post-zoom reasoning so it remains fluent and visually
grounded without segmentation/mask references.

Output zoom grammar:
  <zoom><|ref|>target expression<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|></zoom>

Usage:
  /root/autodl-tmp/VQA/.venv/bin/python rewrite_zoom_seg_to_no_seg_via_vlm.py --dry-run --limit 5
  /root/autodl-tmp/VQA/.venv/bin/python rewrite_zoom_seg_to_no_seg_via_vlm.py --limit 20 --workers 4
  /root/autodl-tmp/VQA/.venv/bin/python rewrite_zoom_seg_to_no_seg_via_vlm.py --resume --workers 8
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

ROOT = Path('/root/autodl-tmp/VQA')
SRC = ROOT / 'json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.jsonl'
DST = ROOT / 'json_data/zoom_json/geoskillrl_apo/sft_zoom_seg_causal_no_seg_primitive.vlm.jsonl'
CHECKPOINT = ROOT / 'json_data/zoom_json/geoskillrl_apo/checkpoint_rewrite_no_seg_vlm.json'
IMG_DIRS = [
    Path('/root/autodl-tmp/dataset/lrs_gro/image'),
    Path('/root/autodl-tmp/dataset/lrs_gro/images'),
    Path('/root/autodl-tmp/dataset/lrs_gro'),
]
LEGACY_REWRITE_SCRIPT = ROOT / 'scripts/rewrite_cot_via_vlm.py'

API_BASE = 'https://api.deepseek.com/anthropic'
MODEL = 'deepseek-v4-pro'
MAX_TOKENS = 900
MAX_RETRIES = 3
SAVE_EVERY = 20
WORKERS = 4
FORBIDDEN_RE = re.compile(r'\b(segmentation|segmenting|mask|overlay)\b', re.IGNORECASE)
ZOOM_JSON_RE = re.compile(r'<zoom>\s*(\[.*?\])\s*</zoom>', re.DOTALL)
BOX_PRIMITIVE_RE = re.compile(r'<zoom><\|ref\|>.*?<\|/ref\|><\|box\|>\s*\[\[.*?\]\]\s*<\|/box\|></zoom>', re.DOTALL)
_lock = Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'done': {}, 'errors': []}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(path: Path, done: dict[str, dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump({'done': done, 'errors': errors}, f, ensure_ascii=False)
    tmp.replace(path)


def legacy_api_key() -> str:
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    if LEGACY_REWRITE_SCRIPT.exists():
        text = LEGACY_REWRITE_SCRIPT.read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', text)
        if m:
            return m.group(1).strip()
    return ''


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def find_image(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    stem = Path(name).stem
    for d in IMG_DIRS:
        cand = d / name
        if cand.exists():
            return cand
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            alt = d / f'{stem}{ext}'
            if alt.exists():
                return alt
    raise FileNotFoundError(name)


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return image


def clamp_box(box: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return [x1, y1, x2, y2]


def crop_box_for_bbox(image: Image.Image, bbox_1024: list[int], scale: float, min_size: int = 512) -> list[int]:
    w, h = image.size
    bbox_orig = [float(v) * float(scale or 1.0) for v in bbox_1024]
    x1, y1, x2, y2 = clamp_box(bbox_orig, w, h)
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
        x1, y1, x2, y2 = clamp_box([nx1, ny1, nx2, ny2], w, h)
    return [x1, y1, x2, y2]


def load_images(record: dict[str, Any], bbox_1024: list[int]) -> tuple[Image.Image, Image.Image]:
    full = Image.open(find_image(record['image_name'])).convert('RGB')
    global_512 = resize_image(full)
    crop_box = crop_box_for_bbox(full, bbox_1024, float(record.get('scale') or 1.0))
    crop = resize_image(full.crop(tuple(crop_box)))
    return global_512, crop


def parse_zoom_bbox_and_label(record: dict[str, Any]) -> tuple[list[int] | None, str]:
    text = record.get('assistant_text') or ''
    label = record.get('label') or record.get('label_cleaned') or record.get('seg_prompt') or 'target'
    bbox = record.get('bbox')
    m = ZOOM_JSON_RE.search(text)
    if m:
        try:
            payload = json.loads(m.group(1))
            if payload and isinstance(payload, list):
                item = payload[0]
                if item.get('bbox_2d'):
                    bbox = item['bbox_2d']
                if item.get('label'):
                    label = item['label']
        except Exception:
            pass
    if bbox and len(bbox) == 4:
        return [int(round(float(v))) for v in bbox], str(label).strip() or 'target'
    return None, str(label).strip() or 'target'


def primitive_zoom(label: str, bbox: list[int]) -> str:
    clean = re.sub(r'\s+', ' ', label).strip() or 'target'
    clean = clean.replace('<', '').replace('>', '')
    box = [max(0, min(1024, int(round(v)))) for v in bbox]
    return f'<zoom><|ref|>{clean}<|/ref|><|box|>[[{box[0]},{box[1]},{box[2]},{box[3]}]]<|/box|></zoom>'


def extract_parts(text: str) -> dict[str, str]:
    thinks = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    answer = re.search(r'<answer>.*?</answer>', text, flags=re.DOTALL)
    zoom = re.search(r'<zoom>.*?</zoom>', text, flags=re.DOTALL)
    seg = re.search(r'<seg>.*?</seg>', text, flags=re.DOTALL)
    return {
        'think1': thinks[0].strip() if len(thinks) > 0 else '',
        'think2': thinks[1].strip() if len(thinks) > 1 else '',
        'think3': thinks[2].strip() if len(thinks) > 2 else '',
        'answer': answer.group(0).strip() if answer else f"<answer>{text_answer_fallback(text)}</answer>",
        'zoom': zoom.group(0).strip() if zoom else '',
        'seg': seg.group(0).strip() if seg else '',
    }


def text_answer_fallback(text: str) -> str:
    m = re.search(r'<answer>\s*(.*?)(?:</answer>|$)', text, flags=re.DOTALL)
    return (m.group(1).strip() if m else '').strip()


def build_prompt(record: dict[str, Any], parts: dict[str, str], label: str, bbox: list[int]) -> str:
    return f"""You are rewriting remote-sensing VQA SFT chain-of-thought data.

Goal: remove the segmentation step while preserving the original semantics, answer, and causal reasoning style.
You are given two images: (1) the global image and (2) the zoom crop selected by the bbox.

QUESTION: {record.get('question')}
GROUND TRUTH ANSWER: {record.get('ground_truth')}
TARGET EXPRESSION: {label}
ZOOM BBOX IN 0-1024 COORDINATES: {bbox}
ORIGINAL CROP THINK: {parts.get('think2', '')[:900]}
ORIGINAL MASK THINK TO ABSORB WITHOUT MENTIONING MASKS: {parts.get('think3', '')[:900]}

Rewrite only the reasoning AFTER the zoom and BEFORE the answer.
Rules:
- Use the crop image as the only local evidence after zoom.
- Do NOT mention segmentation, segmenting, masks, overlays, or mask percentages.
- Do NOT call <seg> and do NOT propose another tool.
- Keep the meaning consistent with the original reasoning and final answer.
- Start with a short crop verification sentence, then give visual evidence and a causal conclusion.
- 2 to 4 concise sentences, fluent English.

Output ONLY valid JSON:
{{"think_after_zoom": "..."}}"""


def parse_vlm_json(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        text = m.group(0)
    obj = json.loads(text)
    value = str(obj['think_after_zoom']).strip()
    value = re.sub(r'\s+', ' ', value)
    if not value:
        raise ValueError('empty think_after_zoom')
    if '<seg>' in value or '</seg>' in value or '<zoom>' in value:
        raise ValueError('tool tag leaked into think_after_zoom')
    if FORBIDDEN_RE.search(value):
        raise ValueError(f'forbidden segmentation term leaked: {value[:120]}')
    return value


def call_vlm(client: Any, record: dict[str, Any], parts: dict[str, str], label: str, bbox: list[int]) -> str:
    global_512, crop = load_images(record, bbox)
    prompt = build_prompt(record, parts, label, bbox)
    image_blocks = [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_to_b64(global_512)}},
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_to_b64(crop)}},
    ]
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            extra = '' if attempt == 0 else '\nYour previous answer was invalid. Return only JSON and avoid any mention of segmentation/masks/overlays.'
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={'type': 'disabled'},
                messages=[{'role': 'user', 'content': image_blocks + [{'type': 'text', 'text': prompt + extra}]}],
            )
            text = ''.join(getattr(block, 'text', '') for block in resp.content)
            return parse_vlm_json(text)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def rewrite_record(record: dict[str, Any], think_after_zoom: str | None, status: str) -> dict[str, Any]:
    out = deepcopy(record)
    text = record.get('assistant_text') or ''
    bbox, label = parse_zoom_bbox_and_label(record)
    parts = extract_parts(text)
    if bbox and '</zoom>' in text:
        zoom = primitive_zoom(label, bbox)
        if think_after_zoom is None:
            # Used only for dry-run/local preview; full rewrite should supply VLM text.
            merged = ' '.join(x for x in [parts.get('think2', ''), parts.get('think3', '')] if x)
            merged = FORBIDDEN_RE.sub('', merged)
            think_after_zoom = re.sub(r'\s+', ' ', merged).strip()
        assistant_text = f"<think>{parts['think1']}</think>\n{zoom}\n<think>{think_after_zoom}</think>\n{parts['answer']}\n"
        out['tool_bboxes_1024'] = [bbox]
        out['primitive_ref'] = label
    else:
        # Non-cut/global records have no seg step; keep assistant text unchanged.
        assistant_text = text
        out['tool_bboxes_1024'] = []
        out['primitive_ref'] = None
    out['assistant_text'] = assistant_text
    out['rewrite_source'] = 'zoom_seg_causal_reasoning'
    out['rewrite_task'] = 'remove_seg_add_primitive_zoom'
    out['rewrite_model'] = MODEL if status == 'vlm' else status
    out['seg_removed'] = '<seg>' not in assistant_text and '</seg>' not in assistant_text
    return out


def rewrite_one(client: Any, record: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    qid = str(record.get('question_id'))
    text = record.get('assistant_text') or ''
    bbox, label = parse_zoom_bbox_and_label(record)
    if not bbox or '</zoom>' not in text or '</seg>' not in text:
        return qid, rewrite_record(record, None, 'passthrough'), None
    parts = extract_parts(text)
    think_after_zoom = call_vlm(client, record, parts, label, bbox)
    return qid, rewrite_record(record, think_after_zoom, 'vlm'), None


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = []
    zoom_rows = 0
    for i, r in enumerate(rows):
        text = r.get('assistant_text') or ''
        zoom_count = text.count('<zoom>')
        if zoom_count:
            zoom_rows += 1
        if '<seg>' in text or '</seg>' in text:
            bad.append({'idx': i, 'qid': r.get('question_id'), 'error': 'seg_tag_remaining'})
        if zoom_count and not BOX_PRIMITIVE_RE.search(text):
            bad.append({'idx': i, 'qid': r.get('question_id'), 'error': 'missing_primitive_zoom'})
        if zoom_count != len(r.get('tool_bboxes_1024') or []):
            bad.append({'idx': i, 'qid': r.get('question_id'), 'error': 'zoom_tool_bbox_mismatch'})
        post_zoom = text.split('</zoom>', 1)[1] if '</zoom>' in text else ''
        if zoom_count and FORBIDDEN_RE.search(post_zoom):
            bad.append({'idx': i, 'qid': r.get('question_id'), 'error': 'forbidden_seg_word_after_zoom'})
    return {'rows': len(rows), 'zoom_rows': zoom_rows, 'bad_count': len(bad), 'bad_examples': bad[:10]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, default=SRC)
    p.add_argument('--dst', type=Path, default=DST)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--api-base', default=API_BASE)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--api-key', default='')
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--save-every', type=int, default=SAVE_EVERY)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main() -> None:
    global API_BASE, MODEL
    args = parse_args()
    API_BASE = args.api_base
    MODEL = args.model
    records = read_jsonl(args.src)
    if args.limit:
        records = records[:args.limit]

    rewrite_needed = [r for r in records if '</zoom>' in (r.get('assistant_text') or '') and '</seg>' in (r.get('assistant_text') or '')]
    passthrough = len(records) - len(rewrite_needed)
    print(json.dumps({
        'src': str(args.src),
        'dst': str(args.dst),
        'records': len(records),
        'vlm_rewrite_needed': len(rewrite_needed),
        'passthrough': passthrough,
    }, ensure_ascii=False, indent=2))

    if args.dry_run:
        previews = []
        for r in records[: min(5, len(records))]:
            bbox, label = parse_zoom_bbox_and_label(r)
            previews.append({
                'qid': r.get('question_id'),
                'needs_vlm': '</zoom>' in (r.get('assistant_text') or '') and '</seg>' in (r.get('assistant_text') or ''),
                'label': label,
                'bbox': bbox,
                'primitive_zoom': primitive_zoom(label, bbox) if bbox else None,
            })
        print(json.dumps({'preview': previews}, ensure_ascii=False, indent=2))
        return

    api_key = args.api_key.strip() or legacy_api_key()
    if not api_key:
        raise SystemExit('Missing API key. Set DEEPSEEK_API_KEY or pass --api-key.')

    from anthropic import Anthropic
    client = Anthropic(base_url=API_BASE, api_key=api_key)

    cp = load_checkpoint(args.checkpoint) if args.resume else {'done': {}, 'errors': []}
    done: dict[str, dict[str, Any]] = cp.get('done', {})
    errors: list[dict[str, Any]] = cp.get('errors', [])

    rows_by_qid = {str(r.get('question_id')): r for r in records}
    tasks = [r for r in records if str(r.get('question_id')) not in done]
    print(f'[resume] done={len(done)} tasks={len(tasks)} errors={len(errors)} workers={args.workers}')

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(rewrite_one, client, r): str(r.get('question_id')) for r in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='rewrite'):
            qid = futs[fut]
            try:
                qid2, row, err = fut.result()
                if err:
                    errors.append({'qid': qid2, 'error': err})
                elif row is not None:
                    with _lock:
                        done[qid2] = row
                completed += 1
            except Exception as exc:
                errors.append({'qid': qid, 'error': repr(exc)})
            if completed % max(1, args.save_every) == 0:
                save_checkpoint(args.checkpoint, done, errors)

    save_checkpoint(args.checkpoint, done, errors)

    out_rows = []
    missing = []
    for r in records:
        qid = str(r.get('question_id'))
        if qid in done:
            out_rows.append(done[qid])
        else:
            missing.append(qid)
            out_rows.append(rewrite_record(rows_by_qid[qid], None, 'missing_fallback'))
    write_jsonl(args.dst, out_rows)
    report = validate_rows(out_rows)
    report.update({'errors': len(errors), 'missing': len(missing), 'checkpoint': str(args.checkpoint)})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors or missing or report['bad_count']:
        raise SystemExit(2)
    args.checkpoint.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
