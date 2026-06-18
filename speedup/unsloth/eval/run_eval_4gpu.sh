#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/VQA
EVAL_DIR="$ROOT/speedup/unsloth/eval"
OUT=${OUT:-"$EVAL_DIR/results"}
LOG_DIR="$OUT/logs"
TEST_JSONL="$ROOT/json_data/test-00000-of-00001.jsonl"
MODEL=${MODEL:-"$ROOT/speedup/unsloth/sft/output/final_hf"}
TAG=${TAG:-sft_unsloth_4gpu}
PY="$ROOT/.venv/bin/python"
GPUS_CSV=${GPUS:-0,1,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
CACHE_SIZE=${CACHE_SIZE:-64}

IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"
PARTS=$((${#GPU_LIST[@]} * WORKERS_PER_GPU))

mkdir -p "$OUT" "$LOG_DIR"
cd "$EVAL_DIR"

TOTAL=$(wc -l < "$TEST_JSONL")
BASE=$((TOTAL / PARTS))
REM=$((TOTAL % PARTS))

echo "=== SFT Unsloth 4-GPU eval start $(date '+%F %T %Z') ==="
echo "model=$MODEL"
echo "test=$TEST_JSONL total=$TOTAL gpus=$GPUS_CSV workers_per_gpu=$WORKERS_PER_GPU parts=$PARTS batch_size=$BATCH_SIZE cache_size=$CACHE_SIZE"

pids=()
part_files=()
part_idx=0
start=0
for gpu in "${GPU_LIST[@]}"; do
  for worker in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    limit=$BASE
    if [ "$part_idx" -lt "$REM" ]; then
      limit=$((limit + 1))
    fi
    part_tag="${TAG}_part${part_idx}_gpu${gpu}_w${worker}"
    part_log="$LOG_DIR/${part_tag}.log"
    part_pred="$OUT/${part_tag}_predictions.jsonl"
    part_bench="$OUT/${part_tag}_benchmark.json"
    part_files+=("$part_pred")

    echo "[launch] gpu=$gpu worker=$worker part=$part_idx start=$start limit=$limit tag=$part_tag"
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false "$PY" eval_unsloth.py \
      --tag "$part_tag" \
      --model "$MODEL" \
      --test_jsonl "$TEST_JSONL" \
      --output_dir "$OUT" \
      --start "$start" \
      --limit "$limit" \
      --resume \
      --summary_every 0 \
      --batch_size "$BATCH_SIZE" \
      --cache_size "$CACHE_SIZE" \
      --benchmark_json "$part_bench" \
      > "$part_log" 2>&1 &
    pids+=("$!")
    start=$((start + limit))
    part_idx=$((part_idx + 1))
  done
done

printf "%s\n" "${pids[@]}" > "$LOG_DIR/${TAG}.pids"
echo "[pids] $(tr '\n' ' ' < "$LOG_DIR/${TAG}.pids")"

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[error] shard pid=$pid failed"
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "=== SFT Unsloth 4-GPU eval FAILED $(date '+%F %T %Z') ==="
  exit "$status"
fi

# Merge shards
MERGE_SCRIPT="$ROOT/speedup/origin/eval/merge_eval_shards.py"
if [ -f "$MERGE_SCRIPT" ]; then
  echo "[merge] ${part_files[*]}"
  "$PY" "$MERGE_SCRIPT" \
    --tag "$TAG" \
    --model "$MODEL" \
    --output_dir "$OUT" \
    --expected "$TOTAL" \
    "${part_files[@]}"
fi

echo "=== SFT Unsloth 4-GPU eval done $(date '+%F %T %Z') ==="
