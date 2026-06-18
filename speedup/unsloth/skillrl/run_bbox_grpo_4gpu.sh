#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/VQA
export DDP_BACKEND="${DDP_BACKEND:-gloo}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-4}" speedup/unsloth/skillrl/train_grpo_bbox.py   --train_jsonl "${TRAIN_JSONL:-/root/autodl-tmp/VQA/json_data/zoom_json/skillrl/bbox_rl_round0.jsonl}"   --output_dir "${OUT_DIR:-/root/autodl-tmp/VQA/speedup/unsloth/skillrl/output/bbox_grpo_round0}"   --skillbank_path "${SKILLBANK:-/root/autodl-tmp/VQA/json_data/zoom_json/skillrl/skillbank_round0.json}"   --num_train_epochs "${EPOCHS:-1}"   --batch_size_per_device "${BATCH_SIZE:-1}"   --num_generations "${NUM_GENERATIONS:-4}"   --gradient_accumulation_steps "${GRAD_ACCUM:-8}"   --learning_rate "${LR:-1e-7}"   --enable_spatial_reward   --log_steps 1
