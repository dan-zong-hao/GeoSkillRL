#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/VQA}"
VERL_DIR="${VERL_DIR:-/root/autodl-tmp/verl}"
DATA_DIR="${DATA_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/data/processed}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf}"
OUT_DIR="${OUT_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/output/train_1gpu_smoke}"
REWARD_MANAGER="${ROOT}/speedup/unsloth/skillrl/verl_grpo/reward/reward_manager.py"
AGENT_CONFIG="${ROOT}/speedup/unsloth/skillrl/verl_grpo/configs/agent_loops.yaml"

export PYTHONPATH="${ROOT}:${VERL_DIR}:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/triton-cache}"
mkdir -p "${OUT_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.return_raw_chat=true \
  data.return_multi_modal_inputs=false \
  data.train_batch_size=1 \
  data.train_max_samples=2 \
  data.val_max_samples=2 \
  data.val_batch_size=1 \
  data.max_prompt_length=2048 \
  data.max_response_length=64 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=zoomearth_zoom_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-7 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=50 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  reward.reward_manager.source=importlib \
  reward.reward_manager.module.path="${REWARD_MANAGER}" \
  reward.reward_manager.name=GeoRewardManager \
  'trainer.logger=["console"]' \
  trainer.use_v1=true \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=true \
  trainer.default_local_dir="${OUT_DIR}" \
  "$@"
