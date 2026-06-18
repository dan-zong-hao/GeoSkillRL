#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/VQA
EVAL_DIR="$ROOT/speedup/unsloth/eval"
OUT=${OUT:-"$ROOT/speedup/unsloth/eval_nohullucination"}
LOG_DIR="$OUT/logs"
TEST_JSONL=${TEST_JSONL:-"$ROOT/json_data/test-00000-of-00001.jsonl"}
MODEL=${MODEL:-"$ROOT/speedup/unsloth/sft_nohullucination/final_hf"}
TAG=${TAG:-sft_nohullucination_4gpu_t1536}
PY="$ROOT/.venv/bin/python"
GPUS_CSV=${GPUS:-0,1,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
CACHE_SIZE=${CACHE_SIZE:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
MAX_NEW_TOKENS_TURN1=${MAX_NEW_TOKENS_TURN1:-1536}
MAX_NEW_TOKENS_TURN2=${MAX_NEW_TOKENS_TURN2:-512}
MAX_NEW_TOKENS_TURN3=${MAX_NEW_TOKENS_TURN3:-512}

IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"
PARTS=$((${#GPU_LIST[@]} * WORKERS_PER_GPU))

mkdir -p "$OUT" "$LOG_DIR"
cd "$EVAL_DIR"

TOTAL=$(wc -l < "$TEST_JSONL")
BASE=$((TOTAL / PARTS))
REM=$((TOTAL % PARTS))

echo "=== SFT no-hallucination 4-GPU eval t1536 start $(date '+%F %T %Z') ==="
echo "model=$MODEL"
echo "test=$TEST_JSONL total=$TOTAL gpus=$GPUS_CSV workers_per_gpu=$WORKERS_PER_GPU parts=$PARTS batch_size=$BATCH_SIZE cache_size=$CACHE_SIZE"
echo "tokens=max:$MAX_NEW_TOKENS turn1:$MAX_NEW_TOKENS_TURN1 turn2:$MAX_NEW_TOKENS_TURN2 turn3:$MAX_NEW_TOKENS_TURN3"

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
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false "$PY" eval_unsloth.py       --tag "$part_tag"       --model "$MODEL"       --model_id "Qwen3.5-SFT-nohallucination"       --test_jsonl "$TEST_JSONL"       --output_dir "$OUT"       --start "$start"       --limit "$limit"       --resume       --summary_every 0       --batch_size "$BATCH_SIZE"       --cache_size "$CACHE_SIZE"       --benchmark_json "$part_bench"       --max_new_tokens "$MAX_NEW_TOKENS"       --max_new_tokens_turn1 "$MAX_NEW_TOKENS_TURN1"       --max_new_tokens_turn2 "$MAX_NEW_TOKENS_TURN2"       --max_new_tokens_turn3 "$MAX_NEW_TOKENS_TURN3"       > "$part_log" 2>&1 &
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
  echo "=== SFT no-hallucination 4-GPU eval t1536 FAILED $(date '+%F %T %Z') ==="
  exit "$status"
fi

MERGE_SCRIPT="$ROOT/speedup/origin/eval/merge_eval_shards.py"
if [ -f "$MERGE_SCRIPT" ]; then
  echo "[merge] ${part_files[*]}"
  "$PY" "$MERGE_SCRIPT"     --tag "$TAG"     --model "$MODEL"     --output_dir "$OUT"     --expected "$TOTAL"     "${part_files[@]}"
fi

echo "=== SFT no-hallucination 4-GPU eval t1536 done $(date '+%F %T %Z') ==="
