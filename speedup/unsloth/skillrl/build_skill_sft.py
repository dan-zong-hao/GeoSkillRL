#!/usr/bin/env python3
"""Build SkillRL round0 data.

Creates:
  * skill_sft_round0.jsonl: SFT records with retrieved skills injected
  * bbox_rl_round0.jsonl: bbox-only RL prompt records for GRPO
  * skillbank_round0.json: seed SkillBank copy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .skillbank import load_skillbank, save_skillbank, retrieve_skills, format_skill_block
    from .spatial import parse_locator
except ImportError:
    from skillbank import load_skillbank, save_skillbank, retrieve_skills, format_skill_block
    from spatial import parse_locator

VQA_ROOT = Path("/root/autodl-tmp/VQA")
DEFAULT_SFT = VQA_ROOT / "json_data/zoom_json/sft/sft_verified_zoom.jsonl"
DEFAULT_RL = VQA_ROOT / "json_data/zoom_json/rl/rl-00000-of-00001.1.jsonl"
DEFAULT_OUT_DIR = VQA_ROOT / "json_data/zoom_json/skillrl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_skill_fields(row: dict[str, Any], skillbank: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    out = dict(row)
    question = str(row.get("question") or "")
    label = str(row.get("label") or "")
    skills = retrieve_skills(question, skillbank, label=label, include_general=False, max_task=2, max_mistakes=0)
    out["retrieved_skills"] = skills
    out["retrieved_skill_ids"] = [s.get("skill_id") for s in skills]
    out["skill_block"] = format_skill_block(skills)
    out["locator"] = parse_locator(question)
    out["round"] = 0
    out["source"] = source
    return out


def build_skill_sft(sft_rows: list[dict[str, Any]], skillbank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sft_rows:
        item = add_skill_fields(row, skillbank, source="sft_verified_zoom")
        out.append(item)
    return out


def build_bbox_rl(rl_rows: list[dict[str, Any]], skillbank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rl_rows:
        if str(row.get("type") or "").lower() not in {"object", "region"}:
            continue
        if not row.get("bbox") or len(row.get("bbox") or []) != 4:
            continue
        item = add_skill_fields(row, skillbank, source="rl_prompt_bbox")
        out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(DEFAULT_SFT))
    ap.add_argument("--rl", default=str(DEFAULT_RL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--skillbank_out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skillbank = load_skillbank(None)
    skillbank_path = Path(args.skillbank_out) if args.skillbank_out else out_dir / "skillbank_round0.json"
    save_skillbank(str(skillbank_path), skillbank)

    sft_rows = read_jsonl(Path(args.sft))
    rl_rows = read_jsonl(Path(args.rl))
    if args.limit:
        sft_rows = sft_rows[: args.limit]
        rl_rows = rl_rows[: args.limit]

    skill_sft = build_skill_sft(sft_rows, skillbank)
    bbox_rl = build_bbox_rl(rl_rows, skillbank)
    write_jsonl(out_dir / "skill_sft_round0.jsonl", skill_sft)
    write_jsonl(out_dir / "bbox_rl_round0.jsonl", bbox_rl)

    print(f"[wrote] {skillbank_path}")
    print(f"[wrote] {out_dir / 'skill_sft_round0.jsonl'} rows={len(skill_sft)}")
    print(f"[wrote] {out_dir / 'bbox_rl_round0.jsonl'} rows={len(bbox_rl)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
