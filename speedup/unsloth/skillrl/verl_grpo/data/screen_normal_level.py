#!/usr/bin/env python3
"""Build Normal-Level training subsets from SFT rollout outcomes."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify(rollouts: list[dict[str, Any]], *, threshold: float = 0.3) -> dict[str, Any]:
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        qid = str(row.get("question_id") or row.get("extra_info", {}).get("question_id") or "")
        if qid:
            by_qid[qid].append(row)
    buckets = {"easy": [], "normal": [], "hard": []}
    detail = {}
    for qid, items in by_qid.items():
        success = 0
        for item in items:
            parse_ok = bool(item.get("zoom_parse_ok", item.get("bbox_valid", False)))
            iou = float(item.get("apo_iou", item.get("iou", item.get("iou_apo512", 0.0))) or 0.0)
            if parse_ok and iou >= threshold:
                success += 1
        if success >= 4:
            bucket = "easy"
        elif success >= 1:
            bucket = "normal"
        else:
            bucket = "hard"
        buckets[bucket].append(qid)
        detail[qid] = {"success_count": success, "rollouts": len(items), "bucket": bucket}
    return {"buckets": buckets, "detail": detail}


def filter_parquet(input_path: Path, qids: set[str], output_path: Path) -> int:
    import pandas as pd

    df = pd.read_parquet(input_path)
    mask = df["extra_info"].map(lambda x: str((x or {}).get("question_id")) in qids)
    out = df[mask].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return int(len(out))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-jsonl", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data/processed"))
    parser.add_argument("--easy-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rollouts = read_jsonl(args.rollouts_jsonl)
    report = classify(rollouts)
    rng = random.Random(args.seed)
    easy = list(report["buckets"]["easy"])
    rng.shuffle(easy)
    easy_keep = easy[: max(0, int(round(len(easy) * args.easy_ratio)))]
    normal_qids = set(report["buckets"]["normal"])
    normal_plus_easy = normal_qids.union(easy_keep)

    out_dir = Path(args.output_dir)
    report["outputs"] = {
        "train_normal": str(out_dir / "train_normal.parquet"),
        "train_normal_plus_easy5": str(out_dir / "train_normal_plus_easy5.parquet"),
        "easy_selected": easy_keep,
    }
    report["counts"] = {
        "easy": len(report["buckets"]["easy"]),
        "normal": len(report["buckets"]["normal"]),
        "hard": len(report["buckets"]["hard"]),
        "easy_selected": len(easy_keep),
    }
    report["written_rows"] = {
        "train_normal": filter_parquet(Path(args.train_parquet), normal_qids, out_dir / "train_normal.parquet"),
        "train_normal_plus_easy5": filter_parquet(
            Path(args.train_parquet),
            normal_plus_easy,
            out_dir / "train_normal_plus_easy5.parquet",
        ),
    }
    (out_dir / "difficulty_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
