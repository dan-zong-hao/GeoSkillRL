#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/VQA/.venv/bin/python}"
EVAL_JSONL="${EVAL_JSONL:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/coldstart/results/rl_dev_skilltrigger}"
CURRENT_MODEL="${CURRENT_MODEL:-/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf}"
COLDSTART_MODEL="${COLDSTART_MODEL:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/coldstart/output/geoskill_coldstart_skilltrigger_v2/final_hf}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON_ARGS=(
  --eval_jsonl "${EVAL_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE:-4}"
  --turn1_max_new_tokens "${TURN1_MAX_NEW_TOKENS:-768}"
  --turn2_max_new_tokens "${TURN2_MAX_NEW_TOKENS:-256}"
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}"
  --sort_by_type
)
if [[ "${LIMIT:-0}" != "0" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" eval_coldstart_skilltrigger.py \
  --model "${CURRENT_MODEL}" \
  --tag current_rl_dev_skilltrigger \
  --model_id current_no_seg_vlm_rewrite_primitive \
  "${COMMON_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" eval_coldstart_skilltrigger.py \
  --model "${COLDSTART_MODEL}" \
  --tag coldstart_rl_dev_skilltrigger \
  --model_id geoskill_coldstart_skilltrigger_v2 \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" compare_eval_summaries.py \
  --before "${OUTPUT_DIR}/current_rl_dev_skilltrigger_summary.json" \
  --after "${OUTPUT_DIR}/coldstart_rl_dev_skilltrigger_summary.json" \
  --output "${OUTPUT_DIR}/current_vs_coldstart_compare.json"
