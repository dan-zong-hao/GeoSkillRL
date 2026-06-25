#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/VQA"
HERE="${ROOT}/speedup/unsloth/skillrl/verl_grpo"
PY="${PYTHON:-${ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl}"
TRAIN_FILE="${TRAIN_FILE:-${HERE}/data/parquet/train.parquet}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
PROJECT_NAME="${PROJECT_NAME:-bbox_only_zoomearth_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_4b_bbox_only_4gpu}"
OUT_DIR="${OUT_DIR:-${HERE}/output/${EXPERIMENT_NAME}}"

export PYTHONPATH="${ROOT}/verl:${HERE}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  "${PY}" "${HERE}/data/prepare_zoomearth_parquet.py" --output "${TRAIN_FILE}"
fi

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  algorithm.use_kl_in_reward=False
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.image_key=images
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}"
  data.filter_overlong_prompts=True
  data.truncation=error
  data.return_raw_chat=True
  data.shuffle=False
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS:-0}"
)

MODEL=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-7}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.04}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE:-4}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
  actor_rollout_ref.actor.fsdp_config.offload_policy="${ACTOR_OFFLOAD_POLICY:-False}"
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS:-[model,extra]}"
  actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CKPT_LOAD_CONTENTS:-[model,extra]}"
)

REF=(
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.fsdp_config.strategy=fsdp2
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${FSDP_SIZE:-4}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-False}"
  actor_rollout_ref.ref.fsdp_config.offload_policy="${REF_OFFLOAD_POLICY:-False}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.40}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}"
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-0.7}"
  actor_rollout_ref.rollout.top_p="${TOP_P:-0.8}"
  actor_rollout_ref.rollout.top_k="${TOP_K:-20}"
  actor_rollout_ref.rollout.calculate_log_probs="${CALCULATE_LOG_PROBS:-False}"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.agent.default_agent_loop=zoomearth_full_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${HERE}/configs/agent_loops.yaml"
)

REWARD=(
  reward.custom_reward_function.path="${HERE}/reward/reward_manager.py"
  reward.custom_reward_function.name=compute_score
  reward.num_workers="${REWARD_NUM_WORKERS:-1}"
)

RAY=(
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-32}"
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.balance_batch=False
  trainer.default_local_dir="${OUT_DIR}"
  trainer.save_freq="${SAVE_FREQ:-20}"
  trainer.test_freq="${TEST_FREQ:--1}"
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
  trainer.total_epochs="${TOTAL_EPOCHS:-1}"
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
)

"${PY}" -m verl.trainer.main_ppo_sync \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${REF[@]}" \
  "${ROLLOUT[@]}" \
  "${REWARD[@]}" \
  "${RAY[@]}" \
  "${TRAINER[@]}" \
  "$@"
