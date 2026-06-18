#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/VQA

VENV_PYTHON=/root/autodl-tmp/VQA/.venv/bin/python
ACCELERATE=/root/autodl-tmp/VQA/.venv/bin/accelerate
TRAIN_SCRIPT=/root/autodl-tmp/VQA/speedup/sft_nohullucination/train_sft_nohullucination.py
TRAIN_JSONL=/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.pos_neg1x.jsonl
OUTPUT_DIR=/root/autodl-tmp/VQA/speedup/sft_nohullucination
PRETRAIN_ADAPTER=/root/autodl-tmp/VQA/speedup/unsloth/pretrain/output/final_hf
BASE_MODEL=/root/autodl-tmp/VQA/models/Qwen3.5-4B

TRAIN_ARGS=(
    --base_model "$BASE_MODEL"
    --pretrained_adapter "$PRETRAIN_ADAPTER"
    --train_jsonl "$TRAIN_JSONL"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}"
    --batch_size_per_gpu "${BATCH_SIZE_PER_GPU:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
    --lr "${LR:-3e-5}"
    --max_length "${MAX_LENGTH:-4096}"
    --warmup_ratio "${WARMUP_RATIO:-0.03}"
    --save_steps "${SAVE_STEPS:-9999}"
    --log_steps "${LOG_STEPS:-5}"
)

case "${1:-single}" in
    multi)
        NUM_PROCESSES="${NUM_PROCESSES:-2}"
        echo "=== SFT no-hallucination Full-FT (Unsloth, ${NUM_PROCESSES} GPUs) ==="
        "$ACCELERATE" launch --num_processes "$NUM_PROCESSES" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
        ;;
    single)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        echo "=== SFT no-hallucination Full-FT (Unsloth, single GPU ${CUDA_VISIBLE_DEVICES}) ==="
        "$VENV_PYTHON" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
        ;;
    *)
        echo "Usage: $0 {single|multi}"
        exit 1
        ;;
esac
