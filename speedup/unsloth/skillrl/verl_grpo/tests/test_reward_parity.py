from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from agent.crop_environment import find_image
from reward.components_legacy import compute_episode_reward


def _image_size(row):
    value = row.get("image_size") or (row.get("reward") or {}).get("image_size")
    if isinstance(value, list) and len(value) == 2:
        return tuple(value)
    image_name = row.get("image_name")
    path = find_image(str(image_name or ""))
    if path is None:
        return None
    with Image.open(path) as image:
        return image.size


def test_legacy_reward_recomputes_old_rollout_logs_if_available():
    log_dir = Path("/root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo/logs")
    paths = sorted(log_dir.glob("*rollouts*.jsonl"))
    if not paths:
        pytest.skip("old rollout logs not present")
    checked = 0
    target = 500
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                row = json.loads(line)
                reward = row.get("reward") or {}
                if "total" not in reward:
                    continue
                got = compute_episode_reward(
                    question_id=str(row.get("question_id") or ""),
                    question=str(row.get("question") or ""),
                    zoom_text=str(row.get("turn1_completion") or ""),
                    answer_text=str(row.get("turn2_completion") or row.get("trajectory") or ""),
                    gt_bbox_1024=row.get("gt_bbox_1024") or reward.get("gt_bbox_1024"),
                    ground_truth=row.get("ground_truth") or reward.get("answer_gt"),
                    image_size=_image_size(row),
                )
                assert abs(float(got["total"]) - float(reward["total"])) < 1e-6
                checked += 1
                if checked >= target:
                    return
    pytest.skip("no comparable old rollout rows")
