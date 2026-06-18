#!/usr/bin/env python3
"""Audit SkillRL round outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .skillbank import count_skills, load_skillbank, retrieve_skills
except ImportError:
    from skillbank import count_skills, load_skillbank, retrieve_skills

VQA_ROOT = Path("/root/autodl-tmp/VQA")
DEFAULT_FAILURES = VQA_ROOT / "json_data/zoom_json/skillrl/failures_round0.jsonl"
DEFAULT_SKILL_SFT = VQA_ROOT / "json_data/zoom_json/skillrl/skill_sft_round0.jsonl"
DEFAULT_BBOX_RL = VQA_ROOT / "json_data/zoom_json/skillrl/bbox_rl_round0.jsonl"
DEFAULT_REPORT = VQA_ROOT / "speedup/unsloth/skillrl/reports/round0_audit.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(num: int | float, den: int | float) -> float:
    return 100.0 * float(num) / max(float(den), 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", default=str(DEFAULT_FAILURES))
    ap.add_argument("--skill_sft", default=str(DEFAULT_SKILL_SFT))
    ap.add_argument("--bbox_rl", default=str(DEFAULT_BBOX_RL))
    ap.add_argument("--skillbank", default=str(VQA_ROOT / "json_data/zoom_json/skillrl/skillbank_round0.json"))
    ap.add_argument("--output", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    failures = read_jsonl(Path(args.failures))
    skill_sft = read_jsonl(Path(args.skill_sft))
    bbox_rl = read_jsonl(Path(args.bbox_rl))
    skillbank = load_skillbank(args.skillbank if Path(args.skillbank).exists() else None)

    deer = sum(1 for f in failures if f.get("deer_horse"))
    false_grounded = sum(1 for f in failures if f.get("false_grounded_correct"))
    bbox_wrong = sum(1 for f in failures if f.get("bbox_wrong"))
    spatial = sum(1 for f in failures if (f.get("spatial") or {}).get("spatial_penalty"))
    covered_failures = sum(1 for f in failures if f.get("retrieved_skill_ids"))
    covered_sft = sum(1 for r in skill_sft if r.get("retrieved_skill_ids"))
    covered_rl = sum(1 for r in bbox_rl if r.get("retrieved_skill_ids"))

    lines = [
        "# SkillRL Round0 Audit",
        "",
        "This report is generated offline. It does not mutate source RL/SFT data.",
        "",
        "## Data Summary",
        "",
        f"- failures: {len(failures)}",
        f"- skill SFT rows: {len(skill_sft)}",
        f"- bbox RL rows: {len(bbox_rl)}",
        f"- seed skills: {count_skills(skillbank)}",
        "",
        "## Skill Coverage",
        "",
        f"- failure rows with retrieved skills: {covered_failures} ({pct(covered_failures, len(failures)):.2f}%)",
        f"- skill SFT rows with retrieved skills: {covered_sft} ({pct(covered_sft, len(skill_sft)):.2f}%)",
        f"- bbox RL rows with retrieved skills: {covered_rl} ({pct(covered_rl, len(bbox_rl)):.2f}%)",
        "",
        "## Failure Metrics",
        "",
        f"- bbox_wrong failures: {bbox_wrong}",
        f"- spatial violation failures: {spatial}",
        f"- deer-horse rate among failures: {deer} ({pct(deer, len(failures)):.2f}%)",
        f"- false-grounded correct among failures: {false_grounded} ({pct(false_grounded, len(failures)):.2f}%)",
        "",
        "## Top Failure Types",
        "",
        "| failure_type | count |",
        "|---|---:|",
    ]
    for key, count in Counter(ft for f in failures for ft in f.get("failure_types", [])).most_common(20):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Spatial Violation Distribution", "", "| spatial_violation | count |", "|---|---:|"])
    for key, count in Counter(str((f.get("spatial") or {}).get("spatial_violation", "none")) for f in failures).most_common(20):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Comparative Locator Backlog", "", "| backlog_family | count |", "|---|---:|"])
    for key, count in Counter(b for f in failures for b in (f.get("locator") or {}).get("backlog_families", [])).most_common(20):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Top Skills Retrieved", "", "| skill_id | count |", "|---|---:|"])
    for key, count in Counter(s for f in failures for s in f.get("retrieved_skill_ids", [])).most_common(20):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Top Categories", "", "| category | count |", "|---|---:|"])
    for key, count in Counter(str(f.get("category")) for f in failures).most_common(20):
        lines.append(f"| {key} | {count} |")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {out}")
    print("\n".join(lines[:28]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
