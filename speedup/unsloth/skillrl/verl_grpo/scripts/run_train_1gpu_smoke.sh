#!/usr/bin/env bash
set -euo pipefail

HERE="/root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo"
PY="${PYTHON:-/root/autodl-tmp/VQA/.venv_verl_qwen35/bin/python}"
SMOKE_FILE="${HERE}/data/parquet/smoke_1gpu.parquet"

"${PY}" "${HERE}/data/prepare_zoomearth_parquet.py" --limit "${SMOKE_LIMIT:-8}" --output "${SMOKE_FILE}"
"${PY}" "${HERE}/data/validate_zoomearth_parquet.py" "${SMOKE_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
CALCULATE_LOG_PROBS=True \
TRAIN_FILE="${SMOKE_FILE}" \
VAL_FILE="${SMOKE_FILE}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_4b_bbox_only_1gpu_smoke}" \
FSDP_SIZE=1 \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}" \
ROLLOUT_N="${ROLLOUT_N:-2}" \
SAVE_FREQ="${SAVE_FREQ:-2}" \
TEST_FREQ="${TEST_FREQ:-2}" \
bash "${HERE}/scripts/run_train_4gpu.sh" trainer.n_gpus_per_node=1 "$@"

