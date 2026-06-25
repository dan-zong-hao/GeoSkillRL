#!/usr/bin/env bash
set -euo pipefail

HERE="/root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo"
bash "${HERE}/scripts/run_train_1gpu_smoke.sh" trainer.total_training_steps=2 "$@"

