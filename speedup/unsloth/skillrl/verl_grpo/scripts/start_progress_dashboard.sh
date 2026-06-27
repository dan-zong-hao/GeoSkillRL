#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/VQA"
HERE="${ROOT}/speedup/unsloth/skillrl/verl_grpo"
PY="${PYTHON:-${ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HERE}/outputs}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-${OUTPUT_ROOT}}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

mkdir -p "${TENSORBOARD_LOGDIR}"

echo "TensorBoard: http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"
echo "Logdir: ${TENSORBOARD_LOGDIR}"
echo "Ray dashboard during training: http://${RAY_DASHBOARD_HOST}:${RAY_DASHBOARD_PORT}"
echo "Latest run: ${OUTPUT_ROOT}/latest"

exec "${PY}" -m tensorboard.main \
  --logdir "${TENSORBOARD_LOGDIR}" \
  --host "${TENSORBOARD_HOST}" \
  --port "${TENSORBOARD_PORT}"
