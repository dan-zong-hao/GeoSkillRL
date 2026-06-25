#!/usr/bin/env bash
set -euo pipefail

HERE="/root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo"
PY="${PYTHON:-/root/autodl-tmp/VQA/.venv_verl_qwen35/bin/python}"
SMOKE_FILE="${HERE}/data/parquet/smoke.parquet"

"${PY}" "${HERE}/data/prepare_zoomearth_parquet.py" --limit "${SMOKE_LIMIT:-16}" --output "${SMOKE_FILE}"
"${PY}" "${HERE}/data/validate_zoomearth_parquet.py" "${SMOKE_FILE}"

CALCULATE_LOG_PROBS=True \
TRAIN_FILE="${SMOKE_FILE}" \
VAL_FILE="${SMOKE_FILE}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_4b_bbox_only_4gpu_smoke}" \
SAVE_FREQ="${SAVE_FREQ:-5}" \
TEST_FREQ="${TEST_FREQ:-5}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
VAL_BEFORE_TRAIN=False \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}" \
bash "${HERE}/scripts/run_train_4gpu.sh" "$@"

