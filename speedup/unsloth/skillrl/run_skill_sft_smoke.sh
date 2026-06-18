#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/VQA
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" .venv/bin/python speedup/unsloth/skillrl/train_skill_sft_zoom.py   --train_jsonl /root/autodl-tmp/VQA/json_data/zoom_json/skillrl/skill_sft_round0.jsonl   --output_dir /root/autodl-tmp/VQA/speedup/unsloth/skillrl/output/skill_sft_smoke   --limit 8   --max_steps 1   --num_train_epochs 1   --batch_size_per_gpu 1   --gradient_accumulation_steps 1   --save_steps 9999   --log_steps 1
