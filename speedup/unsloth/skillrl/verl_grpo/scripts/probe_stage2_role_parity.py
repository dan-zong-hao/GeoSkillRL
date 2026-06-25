#!/usr/bin/env python3
"""Probe Stage-2 observation role candidates for Qwen processor parity."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.zoomearth_agent_loop import build_stage2_messages  # noqa: E402


def update_agent_config(config_path: Path, role: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    if "stage2_observation_role:" not in text:
        text += f"\n  stage2_observation_role: {role}\n"
    else:
        text = re.sub(r"stage2_observation_role:\s*\S+", f"stage2_observation_role: {role}", text, count=1)
    config_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/stage2_role_probe.json")
    parser.add_argument("--agent-config", type=Path, default=ROOT / "configs/agent_loops.yaml")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
    image = Image.new("RGB", (64, 64), "white")
    results = []
    for role in ("tool", "user", "assistant+user"):
        try:
            messages = build_stage2_messages(role)
            raw = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            encoded = processor(text=[raw], images=[image], return_tensors="pt")
            ids = encoded["input_ids"][0].tolist()
            image_token_id = getattr(processor, "image_token_id", None)
            if image_token_id is None:
                image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            image_token_count = sum(1 for x in ids if x == image_token_id)
            grid = encoded.get("image_grid_thw")
            feature_count = int(grid[:, 1:].prod(dim=1).sum().item()) if grid is not None else None
            expected_image_tokens = feature_count // (merge_size * merge_size) if feature_count is not None else None
            prompt_pos = raw.rfind("<|im_start|>assistant")
            generation_prompt_ok = prompt_pos > raw.rfind("<|im_end|>")
            observation_starts_as_user = raw.startswith("<|im_start|>user\n")
            image_alignment_ok = expected_image_tokens is not None and image_token_count == expected_image_tokens
            ok = image_token_count > 0 and image_alignment_ok and generation_prompt_ok and observation_starts_as_user
            results.append(
                {
                    "role": role,
                    "ok": ok,
                    "image_tokens": image_token_count,
                    "feature_count": feature_count,
                    "expected_image_tokens": expected_image_tokens,
                    "merge_size": merge_size,
                    "generation_prompt_ok": generation_prompt_ok,
                    "observation_starts_as_user": observation_starts_as_user,
                }
            )
        except Exception as exc:
            results.append({"role": role, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    ok_roles = [r["role"] for r in results if r.get("ok")]
    payload = {"selected": ok_roles[0] if len(ok_roles) == 1 else None, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if payload["selected"] is not None:
        update_agent_config(args.agent_config, payload["selected"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if len(ok_roles) != 1:
        raise SystemExit("Stage-2 role parity did not produce exactly one valid candidate")


if __name__ == "__main__":
    main()
