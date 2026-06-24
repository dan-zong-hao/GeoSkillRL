#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_zoomearth

PYTHON="${PYTHON:-/root/autodl-tmp/VQA/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-output/geoskill_grpo_zoomearth_v1_t512_t256}"
ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-logs/geoskill_grpo_zoomearth_v1_t512_t256_rollouts.jsonl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" train_geo_grpo_zoomearth.py \
  --model_path "${MODEL_PATH:-/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf}" \
  --train_jsonl "${TRAIN_JSONL:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl}" \
  --skillbank_path "${SKILLBANK_PATH:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/single_zoom/data/geoskill_single_zoom_skillbank_v1.json}" \
  --output_dir "$OUTPUT_DIR" \
  --rollout_log_path "$ROLLOUT_LOG_PATH" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --stage1_max_completion_length "${STAGE1_MAX_NEW_TOKENS:-512}" \
  --stage2_max_completion_length "${STAGE2_MAX_NEW_TOKENS:-256}" \
  --learning_rate "${LR:-1e-7}" \
  --beta "${KL_BETA:-0.04}" \
  --epsilon "${CLIP_EPS:-0.2}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-100000}" \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_dir "${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard}" \
  --report_to tensorboard \
  --run_name "${RUN_NAME:-geoskill_grpo_zoomearth_v1_t512_t256}"
