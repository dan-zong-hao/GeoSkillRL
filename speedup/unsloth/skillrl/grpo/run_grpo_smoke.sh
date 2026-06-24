#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo

/root/autodl-tmp/VQA/.venv/bin/python test_reward_geo.py

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/root/autodl-tmp/VQA/.venv/bin/python train_geo_grpo.py \
  --model_path /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf \
  --train_jsonl /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl \
  --skillbank_path /root/autodl-tmp/VQA/speedup/unsloth/skillrl/single_zoom/data/geoskill_single_zoom_skillbank_v1.json \
  --output_dir output/geoskill_grpo_skillbank_v1_smoke \
  --rollout_log_path logs/geoskill_grpo_skillbank_v1_smoke_rollouts.jsonl \
  --data_limit 32 \
  --num_generations 2 \
  --gradient_accumulation_steps 1 \
  --max_steps 5 \
  --save_steps 5
