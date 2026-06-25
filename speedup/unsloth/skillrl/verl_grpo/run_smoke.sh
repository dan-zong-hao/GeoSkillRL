#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f /root/autodl-tmp/VQA/.venv/bin/activate ]; then
  source /root/autodl-tmp/VQA/.venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

python train_geo_grpo_zoomearth.py \
  --output_dir output/single_gpu_smoke_kl \
  --data_limit 4 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_generations 4 \
  --objective zoom_only \
  --beta 0.04 \
  --stage1_max_completion_length 512 \
  --stage2_max_completion_length 64 \
  --rollout_log_path logs/single_gpu_smoke_kl_rollouts.jsonl \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --bf16 true \
  --gradient_checkpointing true \
  --skip_final_save true
