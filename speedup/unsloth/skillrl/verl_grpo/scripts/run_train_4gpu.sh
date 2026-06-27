#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/VQA"
HERE="${ROOT}/speedup/unsloth/skillrl/verl_grpo"
PY="${PYTHON:-${ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${ROOT}/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl}"
VAL_SOURCE="${VAL_SOURCE:-${ROOT}/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-${HERE}/data/parquet/train.parquet}"
VAL_FILE="${VAL_FILE:-${HERE}/data/parquet/dev.parquet}"
PROJECT_NAME="${PROJECT_NAME:-bbox_only_zoomearth_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_4b_bbox_only_full_4gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HERE}/outputs}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${RUN_DIR}/rollouts}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}"
OUT_DIR="${OUT_DIR:-${CKPT_DIR}}"
ENABLE_LORA="${ENABLE_LORA:-0}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LORA_EXCLUDE_MODULES="${LORA_EXCLUDE_MODULES:-.*visual.*}"

if [[ -z "${ACTOR_LR:-}" ]]; then
  if [[ "${ENABLE_LORA}" == "1" || "${ENABLE_LORA}" == "true" || "${ENABLE_LORA}" == "True" ]]; then
    ACTOR_LR="5e-6"
  else
    ACTOR_LR="1e-7"
  fi
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TENSORBOARD_DIR}"
ln -sfn "${RUN_DIR}" "${OUTPUT_ROOT}/latest"
printf '%s\n' "$$" > "${RUN_DIR}/train.pid"

if [[ "${REDIRECT_TRAIN_LOG:-1}" == "1" && -z "${TRAIN_LOG_REDIRECTED:-}" ]]; then
  export TRAIN_LOG_REDIRECTED=1
  export OUTPUT_ROOT RUN_NAME RUN_DIR LOG_DIR CKPT_DIR ROLLOUT_DIR TENSORBOARD_DIR OUT_DIR
  exec > >(tee -a "${LOG_DIR}/train.log") 2>&1
fi

cat > "${RUN_DIR}/run.env" <<EOF
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
LOG_DIR=${LOG_DIR}
CKPT_DIR=${CKPT_DIR}
ROLLOUT_DIR=${ROLLOUT_DIR}
TENSORBOARD_DIR=${TENSORBOARD_DIR}
OUT_DIR=${OUT_DIR}
MODEL_PATH=${MODEL_PATH}
TRAIN_SOURCE=${TRAIN_SOURCE}
VAL_SOURCE=${VAL_SOURCE}
TRAIN_FILE=${TRAIN_FILE}
VAL_FILE=${VAL_FILE}
ENABLE_LORA=${ENABLE_LORA}
MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB}
WARMUP_STEPS=${WARMUP_STEPS}
WEIGHT_DECAY=${WEIGHT_DECAY}
PPO_EPOCHS=${PPO_EPOCHS}
CLIP_RATIO=${CLIP_RATIO}
LORA_RANK=${LORA_RANK}
LORA_ALPHA=${LORA_ALPHA}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES}
ACTOR_LR=${ACTOR_LR}
EOF

export PYTHONPATH="${ROOT}/verl:${HERE}:${PYTHONPATH:-}"
export TENSORBOARD_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  "${PY}" "${HERE}/data/prepare_zoomearth_parquet.py" --input "${TRAIN_SOURCE}" --output "${TRAIN_FILE}"
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  "${PY}" "${HERE}/data/prepare_zoomearth_parquet.py" --input "${VAL_SOURCE}" --output "${VAL_FILE}"
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

if [[ "${ENABLE_LORA}" == "1" || "${ENABLE_LORA}" == "true" || "${ENABLE_LORA}" == "True" ]]; then
  MODEL+=(
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}"
    actor_rollout_ref.model.exclude_modules="${LORA_EXCLUDE_MODULES}"
  )
fi

ACTOR=(
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS}"
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}"
  actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}"
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
  actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS:-[hf_model]}"
  actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CKPT_LOAD_CONTENTS:-[]}"
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
  +actor_rollout_ref.rollout.limit_images="${LIMIT_IMAGES:-2}"
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-0.85}"
  actor_rollout_ref.rollout.top_p="${TOP_P:-0.8}"
  actor_rollout_ref.rollout.top_k="${TOP_K:-20}"
  actor_rollout_ref.rollout.calculate_log_probs="${CALCULATE_LOG_PROBS:-False}"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.agent.default_agent_loop=zoomearth_full_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${HERE}/configs/agent_loops.yaml"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb="${MM_PROCESSOR_CACHE_GB}"
)

if [[ "${ENABLE_LORA}" == "1" || "${ENABLE_LORA}" == "true" || "${ENABLE_LORA}" == "True" ]]; then
  ROLLOUT+=(
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
  )
fi

REWARD=(
  reward.custom_reward_function.path="${HERE}/reward/reward_manager.py"
  reward.custom_reward_function.name=compute_score
  reward.num_workers="${REWARD_NUM_WORKERS:-1}"
)

RAY=(
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-32}"
  +ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD:-True}"
  +ray_kwargs.ray_init.dashboard_host="${RAY_DASHBOARD_HOST:-0.0.0.0}"
  +ray_kwargs.ray_init.dashboard_port="${RAY_DASHBOARD_PORT:-8265}"
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.logger="${TRAINER_LOGGER:-[console,tensorboard]}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.balance_batch=False
  trainer.default_local_dir="${OUT_DIR}"
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-${ROLLOUT_DIR}}"
  trainer.save_freq="${SAVE_FREQ:-20}"
  trainer.test_freq="${TEST_FREQ:--1}"
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
  trainer.total_epochs="${TOTAL_EPOCHS:-1}"
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
)

HYDRA=(
  hydra.run.dir="${RUN_DIR}/hydra"
  hydra.output_subdir=.hydra
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
  "${HYDRA[@]}" \
  "$@"
