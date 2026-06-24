#!/usr/bin/env python3
"""Compare current vs cold-start RL-dev summary files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    keys = [
        "Answer Acc",
        "Global Acc",
        "Region Acc",
        "Object Acc",
        "APO IoU",
        "Hit@0.3",
        "Hit@0.5",
        "BBox Pred Rate",
        "Zoom Parse Fail",
        "Answer Parse Fail",
        "Grammar OK",
        "Zoom Fired",
    ]
    rows = []
    for key in keys:
        b = float(before["table"].get(key, 0.0))
        a = float(after["table"].get(key, 0.0))
        rows.append({"metric": key, "before": b, "after": a, "delta": a - b})
    report = {
        "before_tag": before.get("tag"),
        "after_tag": after.get("tag"),
        "before_model": before.get("model"),
        "after_model": after.get("model"),
        "n_samples": after.get("n_samples"),
        "bbox_samples": after.get("bbox_samples"),
        "metrics": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("metric,before,after,delta")
    for row in rows:
        print(f"{row['metric']},{row['before']:.2f},{row['after']:.2f},{row['delta']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
