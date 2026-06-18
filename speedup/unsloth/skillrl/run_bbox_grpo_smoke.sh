#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/VQA
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" .venv/bin/python speedup/unsloth/skillrl/train_grpo_bbox.py   --train_jsonl /root/autodl-tmp/VQA/json_data/zoom_json/skillrl/bbox_rl_round0.jsonl   --output_dir /root/autodl-tmp/VQA/speedup/unsloth/skillrl/output/bbox_grpo_smoke   --skillbank_path /root/autodl-tmp/VQA/json_data/zoom_json/skillrl/skillbank_round0.json   --num_train_epochs 1   --batch_size_per_device 1   --num_generations 2   --gradient_accumulation_steps 1   --max_steps 2   --enable_spatial_reward   --log_steps 1   --save_steps 9999   --skip_final_save
