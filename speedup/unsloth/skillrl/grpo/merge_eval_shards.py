#!/usr/bin/env python3
"""Merge sharded GeoSkillRL eval prediction files and recompute metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_geo_grpo import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard_id in range(args.num_shards):
        path = out_dir / f"{args.tag}_shard{shard_id}_predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or "")
            if qid in seen:
                raise ValueError(f"duplicate question_id across shards: {qid}")
            seen.add(qid)
            rows.append(row)

    rows.sort(key=lambda row: str(row.get("question_id") or ""))
    pred_path = out_dir / f"{args.tag}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, tag=args.tag, model=args.model, eval_jsonl=args.eval_jsonl)
    summary["num_shards"] = args.num_shards
    summary_path = out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = summary["table"]
    line = (
        f"Answer={table['Answer Acc']:.2f} APO_IoU={table['APO IoU']:.2f} "
        f"Hit03={table['Hit@0.3']:.2f} Hit05={table['Hit@0.5']:.2f} "
        f"BBoxPred={table['BBox Pred Rate']:.2f} ZoomParseFail={table['Zoom Parse Fail']:.2f} "
        f"AnsParseFail={table['Answer Parse Fail']:.2f} MeanReward={table['Mean Reward']:.3f}"
    )
    (out_dir / f"{args.tag}_table.txt").write_text(line + "\n", encoding="utf-8")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
