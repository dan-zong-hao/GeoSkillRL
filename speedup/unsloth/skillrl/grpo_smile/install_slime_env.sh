#!/usr/bin/env bash
set -euo pipefail

STACK_ROOT="${SLIME_STACK_ROOT:-/root/autodl-tmp/slime_stack}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/autodl-tmp/micromamba}"
ENV_NAME="${SLIME_ENV_NAME:-slime}"
PATCH_VERSION="${PATCH_VERSION:-latest}"
SGLANG_VERSION="${SGLANG_VERSION:-v0.5.12.post1}"
SGLANG_COMMIT="${SGLANG_COMMIT:-5a15cde858ea09b77116212a39356f2fc51b8584}"
MEGATRON_COMMIT="${MEGATRON_COMMIT:-1dcf0dafa884ad52ffb243625717a3471643e087}"
INSTALL_FLASHQLA="${INSTALL_FLASHQLA:-0}"
INSTALL_APEX="${INSTALL_APEX:-1}"

export MAMBA_ROOT_PREFIX
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface_hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
export CUDA_HOME="${CUDA_HOME:-${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/root/autodl-tmp}"

mkdir -p "${STACK_ROOT}" "${MAMBA_ROOT_PREFIX}/bin" "${PIP_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"

if [[ ! -x "${MAMBA_ROOT_PREFIX}/bin/micromamba" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "${MAMBA_ROOT_PREFIX}" bin/micromamba
fi

eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook -s bash)"

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  micromamba create -y -n "${ENV_NAME}" python=3.12 pip -c conda-forge
fi
set +u
micromamba activate "${ENV_NAME}"
set -u

export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"

if [[ "${SKIP_CORE_INSTALL:-0}" != "1" ]]; then
micromamba install -y -n "${ENV_NAME}" \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge
micromamba install -y -n "${ENV_NAME}" -c conda-forge cudnn rust cmake ninja

python -m pip install cuda-python==12.9

if [[ ! -d "${STACK_ROOT}/sglang" ]]; then
  git clone https://github.com/sgl-project/sglang.git "${STACK_ROOT}/sglang"
fi
cd "${STACK_ROOT}/sglang"
git fetch --all --tags
git checkout "${SGLANG_COMMIT}"
python -m pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
python -m pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.ai/whl/cu129/
python -m pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
python -m pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple

python -m pip install cmake ninja
fi
if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  MAX_JOBS="${MAX_JOBS}" python -m pip install -v flash-attn==2.7.4.post1 --no-build-isolation --no-deps
fi
if [[ "${SKIP_MBRIDGE:-0}" != "1" ]]; then
  python -m pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
fi
if [[ "${SKIP_FLASH_LINEAR:-0}" != "1" ]]; then
  python -m pip install flash-linear-attention==0.4.1 --no-deps
  python -m pip install fla-core==0.4.1 --no-deps
fi
if [[ "${INSTALL_FLASHQLA}" == "1" ]]; then
  python -m pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
fi
if [[ "${SKIP_TILELANG:-0}" != "1" ]]; then
  python -m pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/ --no-deps
fi
if [[ "${SKIP_TRANSFORMER_ENGINE:-0}" != "1" ]]; then
  python -m pip install onnx onnxscript
  python -m pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 --no-deps
  python -m pip install --no-build-isolation transformer_engine_torch==2.10.0 --no-deps
fi
if [[ "${INSTALL_APEX}" == "1" && "${SKIP_APEX:-0}" != "1" ]]; then
  NVCC_APPEND_FLAGS="--threads 4" python -m pip install -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
fi
if [[ "${SKIP_TORCH_MEMORY_SAVER:-0}" != "1" ]]; then
  TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
  export TMS_CUDA_MAJOR
  python -m pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
    --no-cache-dir --force-reinstall --no-build-isolation
fi
if [[ "${SKIP_MEGATRON_BRIDGE_BASE:-0}" != "1" ]]; then
  python -m pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
fi
if [[ "${SKIP_QWEN35_BRIDGE:-0}" != "1" ]]; then
  QWEN35_BRIDGE_DIR="${STACK_ROOT}/Megatron-Bridge-slime-qwen35"
  if ! python -m pip install git+https://github.com/coding-famer/Megatron-Bridge-slime.git@qwen35 --no-deps --no-build-isolation; then
    echo "Direct qwen35 bridge install failed; retrying with a local git clone." >&2
    for attempt in 1 2 3 4 5; do
      rm -rf "${QWEN35_BRIDGE_DIR}.tmp"
      if git clone --branch qwen35 --depth 1 https://github.com/coding-famer/Megatron-Bridge-slime.git "${QWEN35_BRIDGE_DIR}.tmp"; then
        rm -rf "${QWEN35_BRIDGE_DIR}"
        mv "${QWEN35_BRIDGE_DIR}.tmp" "${QWEN35_BRIDGE_DIR}"
        break
      fi
      rm -rf "${QWEN35_BRIDGE_DIR}.tmp"
      sleep $((attempt * 5))
    done
    if [[ ! -d "${QWEN35_BRIDGE_DIR}/.git" ]]; then
      echo "Failed to clone qwen35 bridge after retries." >&2
      exit 1
    fi
    python -m pip install "${QWEN35_BRIDGE_DIR}" --no-deps --no-build-isolation
  fi
fi
if [[ "${SKIP_NVIDIA_MODELOPT:-0}" != "1" ]]; then
  python -m pip install "nvidia-modelopt>=0.37.0" --no-deps --no-build-isolation
fi
if [[ "${SKIP_SGL_ROUTER:-0}" != "1" ]]; then
  python -m pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall --no-deps
fi
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

if [[ "${SKIP_MEGATRON_LM:-0}" != "1" ]]; then
cd "${STACK_ROOT}"
if [[ ! -d "${STACK_ROOT}/Megatron-LM/.git" ]]; then
  rm -rf "${STACK_ROOT}/Megatron-LM"
  git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Megatron-LM.git "${STACK_ROOT}/Megatron-LM"
fi
python -m pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd "${STACK_ROOT}/Megatron-LM"
git fetch --depth 1 origin "${MEGATRON_COMMIT}"
git checkout --force "${MEGATRON_COMMIT}"
git submodule update --init --recursive --depth 1
python -m pip install -e . --no-build-isolation --no-deps
fi

if [[ "${SKIP_SLIME:-0}" != "1" ]]; then
cd "${STACK_ROOT}"
if [[ ! -d "${STACK_ROOT}/slime" ]]; then
  git clone --filter=blob:none https://github.com/THUDM/slime.git "${STACK_ROOT}/slime"
fi
cd "${STACK_ROOT}/slime"
git fetch --tags
python -m pip install -r requirements.txt --no-deps
python -m pip install --no-deps \
  msgpack gitpython sentry-sdk absl-py markdown tensorboard-data-server werkzeug \
  griffelib websockets llvmlite httpx-sse pydantic-settings sse-starlette \
  dockerfile-parse h2 wcmatch textual hydra-core==1.3.2 nvidia-resiliency-ext "pyyaml>=6.0.2" \
  gitdb smmap hyperframe hpack bracex mdit-py-plugins linkify-it-py uc-micro-py defusedxml pynvml \
  nvidia-cuda-cccl-cu12 aiohttp_cors opencensus opentelemetry-exporter-prometheus colorful smart_open virtualenv \
  opencensus-context distlib wrapt python-discovery pyasn1-modules proto-plus google-auth google-api-core
python -m pip install -e . --no-deps
fi

if [[ "${SKIP_INT4_QAT:-0}" != "1" ]]; then
cd "${STACK_ROOT}/slime/slime/backends/megatron_utils/kernels/int4_qat"
python -m pip install . --no-build-isolation --no-deps
python -m pip install nvidia-cudnn-cu12==9.16.0.29 "numpy<2" "kernels<0.15.0" --no-deps
python -m pip install --force-reinstall --no-deps scipy==1.13.1
python -m pip install --force-reinstall --no-deps torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129
fi

if [[ "${SKIP_PATCHES:-0}" != "1" ]]; then
cd "${STACK_ROOT}/sglang"
if git apply --check "${STACK_ROOT}/slime/docker/patch/${PATCH_VERSION}/sglang.patch" 2>/dev/null; then
  git update-index --refresh || true
  git apply "${STACK_ROOT}/slime/docker/patch/${PATCH_VERSION}/sglang.patch" --3way
fi
if grep -R -n '^<<<<<<< ' .; then
  echo "sglang patch failed to apply cleanly." >&2
  exit 1
fi

cd "${STACK_ROOT}/Megatron-LM"
if git apply --check "${STACK_ROOT}/slime/docker/patch/${PATCH_VERSION}/megatron.patch" 2>/dev/null; then
  git update-index --refresh || true
  git apply "${STACK_ROOT}/slime/docker/patch/${PATCH_VERSION}/megatron.patch" --3way
fi
if grep -R -n '^<<<<<<< ' .; then
  echo "megatron patch failed to apply cleanly." >&2
  exit 1
fi
fi

SGLANG_ENGINE_FILE="${STACK_ROOT}/slime/slime/backends/sglang_utils/sglang_engine.py"
if [[ -f "${SGLANG_ENGINE_FILE}" ]]; then
  SGLANG_ENGINE_FILE="${SGLANG_ENGINE_FILE}" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["SGLANG_ENGINE_FILE"])
text = path.read_text()
if 'endpoint = "/health_generate"' not in text:
    old = '''def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
'''
    new = '''def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }
    endpoint = "/health_generate"
    if os.environ.get("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "").lower() in {"0", "false", "no"}:
        endpoint = "/health"

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}{endpoint}", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
'''
    if old not in text:
        raise SystemExit("Could not apply slime health-check patch; expected block not found.")
    path.write_text(text.replace(old, new))
PY
  SGLANG_ENGINE_FILE="${SGLANG_ENGINE_FILE}" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["SGLANG_ENGINE_FILE"])
text = path.read_text()
if "def _skip_sglang_health_wait" not in text:
    old = '''    if getattr(server_args, "node_rank", 0) != 0:
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive):
'''
    new = '''    if getattr(server_args, "node_rank", 0) != 0:
        return p

    if _skip_sglang_health_wait():
        startup_grace_seconds = float(os.environ.get("SLIME_SGLANG_STARTUP_GRACE_SECONDS", "12"))
        deadline = time.time() + startup_grace_seconds
        while time.time() < deadline:
            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")
            time.sleep(1)
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _skip_sglang_health_wait() -> bool:
    return os.environ.get("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "").lower() in {"0", "false", "no"}


def _wait_server_healthy(base_url, api_key, is_process_alive):
'''
    if old not in text:
        raise SystemExit("Could not apply slime health-wait skip patch; expected block not found.")
    text = text.replace(old, new)
if "timeout=2" not in text:
    text = text.replace(
        'response = session.get(f"{base_url}{endpoint}", headers=headers)',
        'response = session.get(f"{base_url}{endpoint}", headers=headers, timeout=2)',
    )
path.write_text(text)
PY
  SGLANG_ENGINE_FILE="${SGLANG_ENGINE_FILE}" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["SGLANG_ENGINE_FILE"])
text = path.read_text()
if "SLIME_ROUTER_REGISTER_TIMEOUT_SECONDS" not in text:
    start = text.index("    def _register_to_router(self, server_args_dict):\n")
    end = text.index("\n    def _make_request", start)
    new = '''    def _register_to_router(self, server_args_dict):
        if self.worker_type == "encoder":
            return

        if self.node_rank == 0 and self.router_ip and self.router_port:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            register_timeout = float(os.environ.get("SLIME_ROUTER_REGISTER_TIMEOUT_SECONDS", "600"))
            register_interval = float(os.environ.get("SLIME_ROUTER_REGISTER_INTERVAL_SECONDS", "5"))
            deadline = time.time() + register_timeout

            while True:
                try:
                    if parse(sglang_router.__version__) <= parse("0.2.1"):
                        assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
                        response = requests.post(
                            f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
                            timeout=10,
                        )
                    else:
                        payload = {
                            "url": worker_url,
                            "worker_type": self.worker_type,
                        }
                        if self.worker_type == "prefill":
                            bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")
                            if bootstrap_port is None:
                                raise RuntimeError(
                                    f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                                    "cannot register it to the PD router."
                                )
                            payload["bootstrap_port"] = bootstrap_port
                        response = requests.post(
                            f"http://{self.router_ip}:{self.router_port}/workers",
                            json=payload,
                            timeout=10,
                        )
                    response.raise_for_status()
                    return
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response is not None else None
                    if status_code is not None and status_code < 500:
                        raise
                    last_error = e
                except requests.RequestException as e:
                    last_error = e

                process = getattr(self, "process", None)
                if process is not None and not process.is_alive():
                    raise RuntimeError(f"SGLang server process died before router registration for {worker_url}") from last_error
                if time.time() >= deadline:
                    raise RuntimeError(f"Timed out registering SGLang worker {worker_url} to router") from last_error

                logger.warning(
                    "Failed to register SGLang worker %s to router %s:%s (%s); retrying in %.1fs",
                    worker_url,
                    self.router_ip,
                    self.router_port,
                    last_error,
                    register_interval,
                )
                time.sleep(register_interval)
'''
    text = text[:start] + new + text[end:]
    path.write_text(text)
PY
fi

MEGATRON_DATA_FILE="${STACK_ROOT}/slime/slime/backends/megatron_utils/data.py"
if [[ -f "${MEGATRON_DATA_FILE}" ]]; then
  MEGATRON_DATA_FILE="${MEGATRON_DATA_FILE}" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MEGATRON_DATA_FILE"])
text = path.read_text()

if "def _disable_sequence_packing" not in text:
    text = text.replace("import logging\n", "import logging\nimport os\n", 1)
    text = text.replace(
        'logger = logging.getLogger(__name__)\n\n\n',
        '''logger = logging.getLogger(__name__)


def _disable_sequence_packing() -> bool:
    return os.environ.get("SLIME_DISABLE_SEQUENCE_PACKING", "").lower() in {"1", "true", "yes", "on"}


''',
        1,
    )

if "if _disable_sequence_packing():" not in text:
    old = '''    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if allgather_cp:
'''
    new = '''    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if _disable_sequence_packing():
        if cp_size != 1 or allgather_cp:
            raise NotImplementedError("SLIME_DISABLE_SEQUENCE_PACKING only supports context_parallel_size=1.")

        max_length = max(t.size(0) for t in tokens)
        pad = (pad_size - max_length % pad_size) % pad_size
        padded_length = max_length + pad
        padded_tokens = [F.pad(t, (0, padded_length - t.size(0)), value=pad_token_id) for t in tokens]
        tokens = torch.stack(padded_tokens, dim=0)
        batch["tokens"] = tokens
        batch["packed_seq_params"] = None

        loss_masks = []
        for loss_mask, total_length, response_length in zip(
            batch["loss_masks"],
            batch["total_lengths"],
            batch["response_lengths"],
            strict=True,
        ):
            prompt_length = total_length - response_length
            loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
            loss_mask = F.pad(loss_mask, (0, padded_length - loss_mask.size(0)), value=0)
            loss_masks.append(loss_mask)
        batch["full_loss_masks"] = torch.stack(loss_masks, dim=0)

        multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
        if multimodal_train_inputs is not None:
            multimodal_data = {}
            for mm_input_dict in multimodal_train_inputs:
                if mm_input_dict is not None:
                    for key, mm_tensor in mm_input_dict.items():
                        if key not in multimodal_data:
                            multimodal_data[key] = mm_tensor
                        else:
                            multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
            batch["multimodal_train_inputs"] = multimodal_data

        return batch

    if allgather_cp:
'''
    if old not in text:
        raise SystemExit("Could not apply slime non-packed batch patch; expected block not found.")
    text = text.replace(old, new, 1)

path.write_text(text)
PY
fi

QWEN35_PROVIDER_FILE="$(python - <<'PY'
from pathlib import Path
import sysconfig

print(Path(sysconfig.get_paths()["purelib"]) / "megatron/bridge/models/qwen_vl/qwen35_vl_provider.py")
PY
)"
if [[ -f "${QWEN35_PROVIDER_FILE}" ]]; then
  QWEN35_PROVIDER_FILE="${QWEN35_PROVIDER_FILE}" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["QWEN35_PROVIDER_FILE"])
text = path.read_text()

old_import = """# from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
#     get_transformer_block_with_experimental_attention_variant_spec,
# )
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec"""
new_import = """from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)"""
if old_import in text:
    text = text.replace(old_import, new_import)
elif "get_transformer_block_with_experimental_attention_variant_spec" not in text:
    raise SystemExit("Could not apply Qwen3.5 provider import patch; expected import block not found.")

old_block = """        # block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        #     language_transformer_config,
        #     vp_stage=vp_stage,
        # )
        block_spec = get_gpt_decoder_block_spec(
            language_transformer_config,
            vp_stage=vp_stage,
            use_transformer_engine=HAVE_TE,
        )"""
new_block = """        block_spec = get_transformer_block_with_experimental_attention_variant_spec(
            language_transformer_config,
            vp_stage=vp_stage,
        )"""
if old_block in text:
    text = text.replace(old_block, new_block)

old_default = """    # transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = (
    #     get_transformer_block_with_experimental_attention_variant_spec
    # )
    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = (
        get_gpt_decoder_block_spec
    )"""
new_default = """    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = (
        get_transformer_block_with_experimental_attention_variant_spec
    )"""
if old_default in text:
    text = text.replace(old_default, new_default)

if "get_gpt_decoder_block_spec(" in text:
    raise SystemExit("Qwen3.5 provider still calls get_gpt_decoder_block_spec after patch.")

path.write_text(text)
PY
fi

cat <<EOF
slime environment installed.

Activate with:
  eval "\$(${MAMBA_ROOT_PREFIX}/bin/micromamba shell hook -s bash)"
  micromamba activate ${ENV_NAME}

Useful exports:
  export SLIME_ROOT=${STACK_ROOT}/slime
  export MEGATRON_ROOT=${STACK_ROOT}/Megatron-LM
EOF
