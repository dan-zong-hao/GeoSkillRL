#!/usr/bin/env python3
"""JSONL dataset helpers for GeoSkillRL ZoomEarth-style GRPO."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SKILLRL_DIR = THIS_DIR.parent
GRPO_DIR = SKILLRL_DIR / "grpo"
sys.path.insert(0, str(GRPO_DIR))

from rollout_env import find_image  # noqa: E402


DEFAULT_TRAIN_JSONL = SKILLRL_DIR / "stageA/data/splits/rl_train.jsonl"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_bbox_training_sample(row: dict[str, Any], *, require_image: bool = True) -> bool:
    bbox = row.get("bbox")
    image_name = str(row.get("image_name") or row.get("image") or "")
    if str(row.get("type") or "").lower() not in {"object", "region"}:
        return False
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False
    if require_image and find_image(image_name) is None:
        return False
    return True


def load_bbox_rows(path: str | Path, *, limit: int = 0, require_image: bool = True) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if is_bbox_training_sample(row, require_image=require_image)]
    return rows[:limit] if limit else rows


class GeoBBoxDataset:
    """Minimal map-style dataset that preserves raw JSONL columns for rewards."""

    def __init__(self, path: str | Path = DEFAULT_TRAIN_JSONL, *, limit: int = 0, require_image: bool = True) -> None:
        self.path = Path(path)
        self.records = load_bbox_rows(self.path, limit=limit, require_image=require_image)
        if not self.records:
            raise ValueError(f"no bbox training samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.records[idx])


def identity_collator(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return features

