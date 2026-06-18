#!/bin/bash
# Unsloth full-parameter SFT for Qwen3.5-4B: zoom+seg+think+answer reasoning chain.
# Loads pretrained LoRA adapter, merges into base, then full FT.
#
# Usage:
#   bash run_sft.sh          # single GPU
#   bash run_sft.sh multi    # 2 GPUs with accelerate

set -euo pipefail
cd "$(dirname "$0")"
VENV_PYTHON=/root/autodl-tmp/VQA/.venv/bin/python
ACCELERATE=/root/autodl-tmp/VQA/.venv/bin/accelerate

TRAIN_ARGS=(
    --num_train_epochs 3
    --batch_size_per_gpu 1
    --gradient_accumulation_steps 4
    --lr 3e-5
    --max_length 4096
    --warmup_ratio 0.03
    --save_steps 9999
    --log_steps 5
)

case "${1:-single}" in
    multi)
        echo "=== SFT Full-FT (Unsloth, 2 GPUs) ==="
        $ACCELERATE launch --num_processes 2 train_sft_unsloth.py "${TRAIN_ARGS[@]}"
        ;;
    single)
        echo "=== SFT Full-FT (Unsloth, single GPU) ==="
        CUDA_VISIBLE_DEVICES=0 $VENV_PYTHON train_sft_unsloth.py "${TRAIN_ARGS[@]}"
        ;;
    *)
        echo "Usage: $0 {single|multi}"
        exit 1
        ;;
esac
