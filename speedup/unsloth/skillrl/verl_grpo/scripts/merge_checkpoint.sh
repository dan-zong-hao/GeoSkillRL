#!/usr/bin/env bash
set -euo pipefail

VERL_DIR="${VERL_DIR:-/root/autodl-tmp/verl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to a verl checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for merged Hugging Face checkpoint}"

export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"
python "${VERL_DIR}/scripts/model_merger.py" \
  --local_dir "${CHECKPOINT_DIR}" \
  --target_dir "${OUTPUT_DIR}"
