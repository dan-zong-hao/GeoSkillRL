#!/usr/bin/env bash
set -euo pipefail

HERE="/root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_smile"
ROOT="/root/autodl-tmp/VQA"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/autodl-tmp/micromamba}"
SLIME_ENV_NAME="${SLIME_ENV_NAME:-slime}"
SLIME_ENV_PY="${MAMBA_ROOT_PREFIX}/envs/${SLIME_ENV_NAME}/bin/python"
if [[ -x "${SLIME_ENV_PY}" && "${AUTO_ACTIVATE_SLIME_ENV:-1}" == "1" ]]; then
  export PATH="${MAMBA_ROOT_PREFIX}/envs/${SLIME_ENV_NAME}/bin:${PATH}"
  PY="${PYTHON:-${SLIME_ENV_PY}}"
else
  PY="${PYTHON:-python3}"
fi
TRAIN_SOURCE="${TRAIN_SOURCE:-${ROOT}/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-${HERE}/data/train_slime.jsonl}"
GLOBAL_CACHE_DIR="${GLOBAL_CACHE_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/data/cache/global512}"
REWARD_VERSION="${REWARD_VERSION:-legacy}"

if [[ ! -f "${TRAIN_FILE}" || "${REBUILD_DATA:-0}" == "1" ]]; then
  PREP_ARGS=(
    --input "${TRAIN_SOURCE}" \
    --output "${TRAIN_FILE}" \
    --cache-dir "${GLOBAL_CACHE_DIR}" \
    --reward-version "${REWARD_VERSION}"
  )
  if [[ -n "${DATA_LIMIT:-}" ]]; then
    PREP_ARGS+=(--limit "${DATA_LIMIT}")
  fi
  "${PY}" "${HERE}/prepare_slime_data.py" "${PREP_ARGS[@]}"
fi

NUM_GPUS="${NUM_GPUS:-4}" \
TRAIN_FILE="${TRAIN_FILE}" \
REWARD_VERSION="${REWARD_VERSION}" \
SMOKE_LIMIT="${SMOKE_LIMIT:-0}" \
NUM_ROLLOUT="${NUM_ROLLOUT:-300}" \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}" \
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}" \
RUN_DIR="${RUN_DIR:-${HERE}/outputs/slime_4gpu_$(date +%Y%m%d_%H%M%S)}" \
START_RAY="${START_RAY:-1}" \
SKIP_PREPARE_DATA=1 \
bash "${HERE}/run_slime_smoke.sh" "$@"
