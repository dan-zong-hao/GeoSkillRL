#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo

OUTPUT_DIR="${OUTPUT_DIR:-output/geoskill_grpo_skillbank_v1}"
ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-logs/geoskill_grpo_skillbank_v1_rollouts.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
/root/autodl-tmp/VQA/.venv/bin/python -m torch.distributed.run --nproc_per_node=3 train_geo_grpo.py \
  --model_path /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf \
  --train_jsonl /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl \
  --skillbank_path /root/autodl-tmp/VQA/speedup/unsloth/skillrl/single_zoom/data/geoskill_single_zoom_skillbank_v1.json \
  --output_dir "$OUTPUT_DIR" \
  --rollout_log_path "$ROLLOUT_LOG_PATH" \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --learning_rate "${LR:-1e-7}" \
  --beta "${KL_BETA:-0.08}" \
  --clip_eps "${CLIP_EPS:-0.2}" \
  --max_steps "${MAX_STEPS:--1}" \
  --turn1_max_new_tokens "${TURN1_MAX_NEW_TOKENS:-256}" \
  --turn2_max_new_tokens "${TURN2_MAX_NEW_TOKENS:-128}" \
  --save_steps "${SAVE_STEPS:-100000}"
