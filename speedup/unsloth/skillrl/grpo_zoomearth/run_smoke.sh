#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_zoomearth

PYTHON="${PYTHON:-/root/autodl-tmp/VQA/.venv/bin/python}"

"$PYTHON" /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo/test_reward_geo.py
"$PYTHON" test_geo_zoomearth_adapter.py

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON" train_geo_grpo_zoomearth.py \
  --model_path "${MODEL_PATH:-/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf}" \
  --train_jsonl "${TRAIN_JSONL:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl}" \
  --skillbank_path "${SKILLBANK_PATH:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/single_zoom/data/geoskill_single_zoom_skillbank_v1.json}" \
  --output_dir "${OUTPUT_DIR:-output/geoskill_grpo_zoomearth_v1_smoke}" \
  --rollout_log_path "${ROLLOUT_LOG_PATH:-logs/geoskill_grpo_zoomearth_v1_smoke_rollouts.jsonl}" \
  --data_limit "${DATA_LIMIT:-8}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS:-1}" \
  --num_generations "${NUM_GENERATIONS:-2}" \
  --stage1_max_completion_length "${STAGE1_MAX_NEW_TOKENS:-512}" \
  --stage2_max_completion_length "${STAGE2_MAX_NEW_TOKENS:-256}" \
  --learning_rate "${LR:-1e-7}" \
  --beta "${KL_BETA:-0.04}" \
  --epsilon "${CLIP_EPS:-0.2}" \
  --logging_steps 1 \
  --save_steps 100000 \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_dir "${TENSORBOARD_DIR:-${OUTPUT_DIR:-output/geoskill_grpo_zoomearth_v1_smoke}/tensorboard}" \
  --report_to tensorboard \
  --run_name "${RUN_NAME:-geoskill_grpo_zoomearth_smoke}" \
  --skip_final_save true
