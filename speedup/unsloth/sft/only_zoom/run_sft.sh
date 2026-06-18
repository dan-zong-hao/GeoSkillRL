#!/bin/bash
# Full-FT SFT on verified zoom dataset (with crop-verification).
# Loads pretrained LoRA adapter, merges, then full fine-tune.
#
# Usage:
#   bash run_sft.sh              # single GPU
#   bash run_sft.sh multi        # 2 GPUs

set -euo pipefail
cd "$(dirname "$0")"
VENV=/root/autodl-tmp/VQA/.venv/bin/python
ACCELERATE=/root/autodl-tmp/VQA/.venv/bin/accelerate

ARGS=(
    --num_train_epochs 3
    --batch_size_per_gpu 1
    --gradient_accumulation_steps 4
    --lr 3e-5
    --max_length 4096
)

case "${1:-single}" in
    multi)
        echo "=== SFT zoom-only (2 GPUs) ==="
        $ACCELERATE launch --num_processes 2 train_sft_zoom.py "${ARGS[@]}"
        ;;
    single)
        echo "=== SFT zoom-only (single GPU) ==="
        CUDA_VISIBLE_DEVICES=0 $VENV train_sft_zoom.py "${ARGS[@]}"
        ;;
    *) echo "Usage: $0 {single|multi}"; exit 1 ;;
esac
