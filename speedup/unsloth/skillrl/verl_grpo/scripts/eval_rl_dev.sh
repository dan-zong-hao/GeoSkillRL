#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/VQA}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUT_DIR="${OUT_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/output/eval_rl_dev}"

MODEL_PATH="${MODEL_PATH}" OUT_DIR="${OUT_DIR}" bash "${ROOT}/speedup/unsloth/skillrl/verl_grpo/scripts/run_val_only.sh"
