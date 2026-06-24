#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo

PY=/root/autodl-tmp/VQA/.venv/bin/python
MODEL="${MODEL:-output/geoskill_grpo_skillbank_v1/final_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rl_dev_grpo_skillbank_v1}"
TAG="${TAG:-grpo_skillbank_v1_rl_dev}"
EVAL_JSONL="${EVAL_JSONL:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl}"
SKILLBANK="${SKILLBANK:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/single_zoom/data/geoskill_single_zoom_skillbank_v1.json}"
NUM_SHARDS="${NUM_SHARDS:-3}"

mkdir -p "$OUTPUT_DIR"

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="$shard_id"
    "$PY" eval_geo_grpo.py \
      --model "$MODEL" \
      --eval_jsonl "$EVAL_JSONL" \
      --skillbank_path "$SKILLBANK" \
      --output_dir "$OUTPUT_DIR" \
      --tag "${TAG}_shard${shard_id}" \
      --num_shards "$NUM_SHARDS" \
      --shard_id "$shard_id"
  ) > "${OUTPUT_DIR}/${TAG}_shard${shard_id}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" merge_eval_shards.py \
  --output_dir "$OUTPUT_DIR" \
  --tag "$TAG" \
  --model "$MODEL" \
  --eval_jsonl "$EVAL_JSONL" \
  --num_shards "$NUM_SHARDS"
