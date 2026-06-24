#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo

MODEL="${MODEL:-output/geoskill_grpo_skillbank_v1/final_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rl_dev_grpo_skillbank_v1}"
TAG="${TAG:-grpo_skillbank_v1_rl_dev}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/root/autodl-tmp/VQA/.venv/bin/python eval_geo_grpo.py \
  --model "$MODEL" \
  --eval_jsonl /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl \
  --skillbank_path /root/autodl-tmp/VQA/speedup/unsloth/skillrl/single_zoom/data/geoskill_single_zoom_skillbank_v1.json \
  --output_dir "$OUTPUT_DIR" \
  --tag "$TAG"
