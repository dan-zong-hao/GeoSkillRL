#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_zoomearth

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export STAGE1_MAX_NEW_TOKENS="${STAGE1_MAX_NEW_TOKENS:-512}"
export STAGE2_MAX_NEW_TOKENS="${STAGE2_MAX_NEW_TOKENS:-256}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/geoskill_grpo_zoomearth_v1_t512_t256}"
export ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-logs/geoskill_grpo_zoomearth_v1_t512_t256_rollouts.jsonl}"
export RUN_NAME="${RUN_NAME:-geoskill_grpo_zoomearth_v1_t512_t256}"

exec bash run_grpo_full_3gpu.sh
