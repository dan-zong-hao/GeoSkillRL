#!/bin/bash
# LoRA pretraining for Qwen3.5-4B: learn visual primitive format (bounding boxes).
#
# Usage:
#   bash run_pretrain.sh prepare    # Generate pretrain JSONL from DIOR-RSVG
#   bash run_pretrain.sh train      # Start pretraining (Unsloth version)
#   bash run_pretrain.sh all        # Prepare + train

set -euo pipefail

cd "$(dirname "$0")"
VENV_PYTHON=/root/autodl-tmp/VQA/.venv/bin/python
PROJECT_DIR=/root/autodl-tmp/VQA/speedup/pretrain

prepare_data() {
    echo "=== Step 1: Preparing pretraining data ==="
    $VENV_PYTHON "$PROJECT_DIR/prepare_pretrain_data.py" \
        --dior_root /root/autodl-tmp/dataset/DIOR-RSVG \
        --output_dir "$PROJECT_DIR/data" \
        --neg_ratio 0.30 \
        --val_ratio 0.05 \
        --seed 42
    echo ""
    echo "Data ready:"
    wc -l "$PROJECT_DIR/data/train.jsonl"
    wc -l "$PROJECT_DIR/data/val.jsonl"
}

train() {
    TRAIN_SCRIPT="${1:-train_pretrain_unsloth.py}"
    echo "=== Step 2: Starting pretraining ($TRAIN_SCRIPT) ==="
    # Effective batch = 2 * 8 = 16 (per GPU)
    CUDA_VISIBLE_DEVICES=0 $VENV_PYTHON "$PROJECT_DIR/$TRAIN_SCRIPT" \
        --num_train_epochs 3 \
        --batch_size_per_gpu 2 \
        --gradient_accumulation_steps 8 \
        --lr 1e-4 \
        --lora_r 64 \
        --lora_alpha 128 \
        --max_length 512 \
        --warmup_ratio 0.03 \
        --save_steps 2000 \
        --log_steps 10
}

case "${1:-all}" in
    prepare) prepare_data ;;
    train)   train "train_pretrain_unsloth.py" ;;
    train_orig) train "train_pretrain.py" ;;
    all)
        prepare_data
        train "train_pretrain_unsloth.py"
        ;;
    *)
        echo "Usage: $0 {prepare|train|train_orig|all}"
        exit 1
        ;;
esac
