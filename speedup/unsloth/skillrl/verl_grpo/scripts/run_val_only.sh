#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/VQA}"
VERL_DIR="${VERL_DIR:-/root/autodl-tmp/verl}"
DATA_DIR="${DATA_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/data/processed}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the model or merged checkpoint to evaluate}"
OUT_DIR="${OUT_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/output/val_only}"
REWARD_MANAGER="${ROOT}/speedup/unsloth/skillrl/verl_grpo/reward/reward_manager.py"
AGENT_CONFIG="${ROOT}/speedup/unsloth/skillrl/verl_grpo/configs/agent_loops.yaml"

export PYTHONPATH="${ROOT}:${VERL_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

python -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/val.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.return_raw_chat=true \
  data.return_multi_modal_inputs=false \
  data.train_batch_size=1 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=zoomearth_full_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
  reward.reward_manager.source=importlib \
  reward.reward_manager.module.path="${REWARD_MANAGER}" \
  reward.reward_manager.name=GeoRewardManager \
  trainer.use_v1=true \
  trainer.n_gpus_per_node="${N_GPUS:-1}" \
  trainer.nnodes=1 \
  trainer.total_epochs=0 \
  trainer.val_before_train=true \
  trainer.default_local_dir="${OUT_DIR}" \
  "$@"
