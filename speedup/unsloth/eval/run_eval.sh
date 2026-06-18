#!/bin/bash
# Unsloth full-test eval for SFT model.
#
# Usage:
#   bash run_eval.sh          # eval on sft_causal_reasoning (test split)
#   bash run_eval.sh spatial   # eval on sft_spatial

set -euo pipefail
cd "$(dirname "$0")"
VENV_PYTHON=/root/autodl-tmp/VQA/.venv/bin/python
MODEL=/root/autodl-tmp/VQA/speedup/unsloth/sft/output/final_hf

TEST_JSONL=/root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl
TAG=sft_unsloth_full_test

case "${1:-full}" in
    full|test)
        ;;
    causal)
        TEST_JSONL=/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.jsonl
        TAG=sft_unsloth_train
        ;;
    *)
        echo "Usage: $0 {full|causal}"
        exit 1
        ;;
esac

echo "=== Eval (Unsloth) ==="
echo "Model: $MODEL"
echo "Test:  $TEST_JSONL"
echo "Tag:   $TAG"

CUDA_VISIBLE_DEVICES=0 $VENV_PYTHON eval_unsloth.py \
    --model "$MODEL" \
    --test_jsonl "$TEST_JSONL" \
    --output_dir ./results \
    --tag "$TAG" \
    --batch_size 4 \
    --limit 0 \
    --summary_every 500
