#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/geoskill_verl_grpo_zoom_only_3gpu_v1}"
export ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-logs/geoskill_verl_grpo_zoom_only_3gpu_v1_rollouts.jsonl}"
export RUN_NAME="${RUN_NAME:-geoskill_verl_grpo_zoom_only_3gpu_v1}"

mkdir -p logs "$OUTPUT_DIR"
LOG_PATH="${LOG_PATH:-logs/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"
PID_PATH="${PID_PATH:-logs/train_${RUN_NAME}.pid}"

nohup bash run_grpo_full_3gpu.sh > "$LOG_PATH" 2>&1 &
echo "$!" > "$PID_PATH"
echo "started pid=$(cat "$PID_PATH")"
echo "log=$LOG_PATH"
echo "pid_file=$PID_PATH"

