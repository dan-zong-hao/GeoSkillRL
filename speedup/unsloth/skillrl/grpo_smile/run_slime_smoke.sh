#!/usr/bin/env bash
set -euo pipefail

HERE="/root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_smile"
ROOT="/root/autodl-tmp/VQA"
SLIME_STACK_ROOT="${SLIME_STACK_ROOT:-/root/autodl-tmp/slime_stack}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/autodl-tmp/micromamba}"
SLIME_ENV_NAME="${SLIME_ENV_NAME:-slime}"
SLIME_ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${SLIME_ENV_NAME}"
SLIME_ENV_PY="${MAMBA_ROOT_PREFIX}/envs/${SLIME_ENV_NAME}/bin/python"
SLIME_ROOT="${SLIME_ROOT:-${SLIME_STACK_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${SLIME_STACK_ROOT}/Megatron-LM}"
if [[ -x "${SLIME_ENV_PY}" && "${AUTO_ACTIVATE_SLIME_ENV:-1}" == "1" ]]; then
  export PATH="${MAMBA_ROOT_PREFIX}/envs/${SLIME_ENV_NAME}/bin:${PATH}"
  PY="${PYTHON:-${SLIME_ENV_PY}}"
else
  PY="${PYTHON:-python3}"
fi
MODEL_PATH="${MODEL_PATH:-${ROOT}/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${ROOT}/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-${HERE}/data/smoke_slime.jsonl}"
GLOBAL_CACHE_DIR="${GLOBAL_CACHE_DIR:-${ROOT}/speedup/unsloth/skillrl/verl_grpo/data/cache/global512}"
REWARD_VERSION="${REWARD_VERSION:-legacy}"
SMOKE_LIMIT="${SMOKE_LIMIT:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh}"
RUN_DIR="${RUN_DIR:-${HERE}/outputs/slime_smoke_$(date +%Y%m%d_%H%M%S)}"
TP_SIZE="${TP_SIZE:-1}"

if [[ ! -f "${SLIME_ROOT}/train.py" ]]; then
  echo "SLIME_ROOT=${SLIME_ROOT} does not contain train.py. Clone THUDM/slime or set SLIME_ROOT." >&2
  exit 1
fi
if [[ -f "${MODEL_ARGS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${MODEL_ARGS_FILE}"
else
  echo "MODEL_ARGS_FILE=${MODEL_ARGS_FILE} not found; using empty MODEL_ARGS." >&2
  MODEL_ARGS=()
fi

mkdir -p "${RUN_DIR}"
if [[ "${SKIP_PREPARE_DATA:-0}" != "1" ]]; then
  "${PY}" "${HERE}/prepare_slime_data.py" \
    --input "${TRAIN_SOURCE}" \
    --output "${TRAIN_FILE}" \
    --cache-dir "${GLOBAL_CACHE_DIR}" \
    --limit "${SMOKE_LIMIT}" \
    --reward-version "${REWARD_VERSION}"
fi

export PYTHONPATH="${HERE}:${SLIME_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
export ZOOMEARTH_REWARD_VERSION="${REWARD_VERSION}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}"
export CUDA_HOME="${CUDA_HOME:-${SLIME_ENV_PREFIX}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/root/autodl-tmp}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-false}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-false}"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-false}"
export SLIME_DISABLE_SEQUENCE_PACKING="${SLIME_DISABLE_SEQUENCE_PACKING:-1}"
export SLIME_SGLANG_STARTUP_GRACE_SECONDS="${SLIME_SGLANG_STARTUP_GRACE_SECONDS:-12}"
export SLIME_ROUTER_REGISTER_TIMEOUT_SECONDS="${SLIME_ROUTER_REGISTER_TIMEOUT_SECONDS:-600}"
export SLIME_ROUTER_REGISTER_INTERVAL_SECONDS="${SLIME_ROUTER_REGISTER_INTERVAL_SECONDS:-5}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
LOCAL_NODE_IP="${LOCAL_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
LOCAL_NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}"
if [[ -n "${LOCAL_NODE_IP}" ]]; then
  LOCAL_NO_PROXY="${LOCAL_NO_PROXY},${LOCAL_NODE_IP}"
fi
export no_proxy="${no_proxy:-${LOCAL_NO_PROXY}}"
export NO_PROXY="${NO_PROXY:-${no_proxy}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

if [[ "${START_RAY:-1}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
  for attempt in $(seq 1 "${RAY_DASHBOARD_WAIT_SECONDS:-90}"); do
    if "${PY}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version', timeout=1).read()" >/dev/null 2>&1; then
      break
    fi
    if [[ "${attempt}" == "${RAY_DASHBOARD_WAIT_SECONDS:-90}" ]]; then
      echo "Ray dashboard did not become ready on port ${RAY_DASHBOARD_PORT}." >&2
      exit 1
    fi
    sleep 1
  done
fi

RUNTIME_ENV_JSON=$(cat <<JSON
{"env_vars":{"PYTHONPATH":"${PYTHONPATH}","ZOOMEARTH_REWARD_VERSION":"${ZOOMEARTH_REWARD_VERSION}","TENSORBOARD_DIR":"${TENSORBOARD_DIR}","CUDA_HOME":"${CUDA_HOME}","LD_LIBRARY_PATH":"${LD_LIBRARY_PATH}","LIBRARY_PATH":"${LIBRARY_PATH}","CPATH":"${CPATH}","FLASHINFER_WORKSPACE_BASE":"${FLASHINFER_WORKSPACE_BASE}","CUDA_DEVICE_MAX_CONNECTIONS":"${CUDA_DEVICE_MAX_CONNECTIONS}","NCCL_NVLS_ENABLE":"${NCCL_NVLS_ENABLE}","SGLANG_JIT_DEEPGEMM_PRECOMPILE":"${SGLANG_JIT_DEEPGEMM_PRECOMPILE}","SGLANG_JIT_DEEPGEMM_FAST_WARMUP":"${SGLANG_JIT_DEEPGEMM_FAST_WARMUP}","SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION":"${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION}","SLIME_DISABLE_SEQUENCE_PACKING":"${SLIME_DISABLE_SEQUENCE_PACKING}","SLIME_SGLANG_STARTUP_GRACE_SECONDS":"${SLIME_SGLANG_STARTUP_GRACE_SECONDS}","SLIME_ROUTER_REGISTER_TIMEOUT_SECONDS":"${SLIME_ROUTER_REGISTER_TIMEOUT_SECONDS}","SLIME_ROUTER_REGISTER_INTERVAL_SECONDS":"${SLIME_ROUTER_REGISTER_INTERVAL_SECONDS}","no_proxy":"${no_proxy}","NO_PROXY":"${NO_PROXY}"}}
JSON
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --micro-batch-size 1
)
if (( TP_SIZE > 1 )); then
  PERF_ARGS+=(--sequence-parallel)
fi
SGLANG_EXTRA_ARGS=()
if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  SGLANG_EXTRA_ARGS+=(--sglang-disable-cuda-graph)
fi
SGLANG_EXTRA_ARGS+=(
  --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS:-65536}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-16}"
  --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-4096}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
  --sglang-max-mamba-cache-size "${SGLANG_MAX_MAMBA_CACHE_SIZE:-64}"
)
if [[ -n "${SGLANG_LINEAR_ATTN_BACKEND:-}" ]]; then
  SGLANG_EXTRA_ARGS+=(--sglang-linear-attn-backend "${SGLANG_LINEAR_ATTN_BACKEND}")
fi
if [[ -n "${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-}" ]]; then
  SGLANG_EXTRA_ARGS+=(--sglang-linear-attn-decode-backend "${SGLANG_LINEAR_ATTN_DECODE_BACKEND}")
fi
if [[ -n "${SGLANG_LINEAR_ATTN_PREFILL_BACKEND:-}" ]]; then
  SGLANG_EXTRA_ARGS+=(--sglang-linear-attn-prefill-backend "${SGLANG_LINEAR_ATTN_PREFILL_BACKEND}")
fi
SAVE_ARGS=()
case "${SAVE_INTERVAL:-1}" in
  ""|"0"|"none"|"NONE")
    ;;
  *)
    SAVE_ARGS+=(
      --save "${RUN_DIR}/megatron"
      --save-interval "${SAVE_INTERVAL:-1}"
    )
    if [[ "${NO_SAVE_OPTIM:-1}" == "1" ]]; then
      SAVE_ARGS+=(--no-save-optim)
    fi
    if [[ "${SAVE_HF:-0}" == "1" ]]; then
      SAVE_ARGS+=(--save-hf "${RUN_DIR}/hf/{rollout_id}")
    fi
    ;;
esac

cd "${SLIME_ROOT}"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --colocate \
  --hf-checkpoint "${MODEL_PATH}" \
  --load "${MODEL_PATH}" \
  "${SAVE_ARGS[@]}" \
  --prompt-data "${TRAIN_FILE}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --multimodal-keys '{"image":"images"}' \
  --apply-chat-template \
  --custom-generate-function-path rollout_slime.generate \
  --custom-rm-path reward_geo.reward_func \
  --custom-config-path "${HERE}/slime_config.yaml" \
  --advantage-estimator grpo \
  --num-rollout "${NUM_ROLLOUT:-1}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-2}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-4}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.85}" \
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}" \
  --rollout-top-k "${ROLLOUT_TOP_K:--1}" \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.45}" \
  --sglang-mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND:-sdpa}" \
  "${SGLANG_EXTRA_ARGS[@]}" \
  --train-backend megatron \
  --megatron-to-hf-mode bridge \
  "${PERF_ARGS[@]}" \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-backend "${ATTENTION_BACKEND:-flash}" \
  --optimizer adam \
  --lr "${LR:-1e-7}" \
  --lr-decay-style constant \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --kl-loss-coef "${KL_LOSS_COEF:-0.04}" \
  --kl-loss-type low_var_kl \
  --kl-coef 0.00 \
  --entropy-coef 0.00 \
  --eps-clip "${EPS_CLIP:-0.2}" \
  "${MODEL_ARGS[@]}" \
  "$@"
