#!/bin/bash
# Eval for zoom-only model (no seg, no SAM2).
#
# Usage:
#   bash run_eval.sh          # full test set, quick (limit 200)
#   bash run_eval.sh full     # full test set, all 9734 samples
#   bash run_eval.sh dev      # zoom_json as dev set

set -euo pipefail
cd "$(dirname "$0")"
VENV=/root/autodl-tmp/VQA/.venv/bin/python
MODEL=/root/autodl-tmp/VQA/speedup/unsloth/sft/only_zoom/output/final_hf

case "${1:-quick}" in
    quick)
        TEST=/root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl
        LIMIT=200
        TAG=zoom_only_quick
        ;;
    full)
        TEST=/root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl
        LIMIT=0
        TAG=zoom_only_full
        ;;
    dev)
        TEST=/root/autodl-tmp/VQA/json_data/zoom_json/sft_verified_zoom.jsonl
        LIMIT=0
        TAG=zoom_only_dev
        ;;
    *) echo "Usage: $0 {quick|full|dev}"; exit 1 ;;
esac

echo "=== Eval Zoom-Only ==="
echo "Model: $MODEL"
echo "Test:  $TEST"
echo "Limit: $LIMIT"

CUDA_VISIBLE_DEVICES=0 $VENV eval_zoom_only.py \
    --model "$MODEL" \
    --test_jsonl "$TEST" \
    --output_dir ./results \
    --tag "$TAG" \
    --limit "$LIMIT" \
    --summary_every 200
