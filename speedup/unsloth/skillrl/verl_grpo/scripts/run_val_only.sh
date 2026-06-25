#!/usr/bin/env bash
set -euo pipefail

HERE="/root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo"
VAL_ONLY=True bash "${HERE}/scripts/run_train_4gpu.sh" trainer.val_only=True "$@"

