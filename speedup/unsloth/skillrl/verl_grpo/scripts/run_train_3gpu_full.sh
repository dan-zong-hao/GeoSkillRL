#!/usr/bin/env bash
set -euo pipefail

export ROOT="${ROOT:-/root/autodl-tmp/VQA}"
export DATA_DIR="${DATA_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/data/processed}"
export OUT_DIR="${OUT_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/output/train_3gpu_full}"

bash "${ROOT}/speedup/unsloth/skillrl/verl_grpo/scripts/run_train_3gpu_smoke.sh" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-200}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}"
