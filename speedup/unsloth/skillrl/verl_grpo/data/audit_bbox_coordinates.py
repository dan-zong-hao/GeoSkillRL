#!/usr/bin/env python3
"""Audit 1024-to-original bbox mapping for prepared parquet rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    rows = []
    for _, row in pd.read_parquet(args.parquet).head(args.limit).iterrows():
        extra = row["extra_info"]
        width, height = extra["image_size"]
        scale = max(width, height) / 1024.0
        bbox_orig = [float(v) * scale for v in extra["gt_bbox_1024"]]
        rows.append(
            {
                "index": extra["index"],
                "question_id": extra["question_id"],
                "image_size": extra["image_size"],
                "gt_bbox_1024": extra["gt_bbox_1024"],
                "bbox_orig_max_side": bbox_orig,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

