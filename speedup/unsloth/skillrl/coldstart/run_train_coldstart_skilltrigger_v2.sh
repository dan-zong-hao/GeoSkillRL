#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/VQA/.venv/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf}"
TRAIN_JSONL="${TRAIN_JSONL:-/root/autodl-tmp/VQA/json_data/zoom_json/rl/rl_train_vlm_style_reasoning_skilltrigger_v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/coldstart/output/geoskill_coldstart_skilltrigger_v2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" train_coldstart_skilltrigger_vlm.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --bbox_loss_alpha "${BBOX_LOSS_ALPHA:-2.0}" \
  --num_train_epochs "${EPOCHS:-1}" \
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --lr "${LR:-3e-5}" \
  --max_length "${MAX_LENGTH:-4096}"
