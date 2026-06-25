#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/VQA"
HERE="${ROOT}/speedup/unsloth/skillrl/verl_grpo"
VENV="${ROOT}/.venv_verl_qwen35"
PY="${PYTHON:-${VENV}/bin/python}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl}"
OUT="${HERE}/environment.lock.json"

export PYTHONPATH="${ROOT}/verl:${HERE}:${PYTHONPATH:-}"
export MODEL_PATH
TMP_PROBE="$(mktemp /tmp/zoomearth_probe_environment.XXXXXX.py)"
TMP_JSON="$(mktemp /tmp/zoomearth_environment_lock.XXXXXX.json)"
export PROBE_OUT_TMP="${TMP_JSON}"
trap 'rm -f "${TMP_PROBE}" "${TMP_JSON}"' EXIT
cat > "${TMP_PROBE}" <<'PY'
import importlib
import json
import os
import pathlib
import subprocess
import sys

root = pathlib.Path("/root/autodl-tmp/VQA")
model_path = pathlib.Path(os.environ.get("MODEL_PATH", str(root / "speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl")))
mods = ["torch", "transformers", "vllm", "ray", "verl", "qwen_vl_utils", "transfer_queue", "datasets", "pyarrow", "PIL"]
packages = {}
for name in mods:
    try:
        mod = importlib.import_module(name)
        packages[name] = {"version": getattr(mod, "__version__", "?"), "file": getattr(mod, "__file__", "?")}
    except Exception as exc:
        packages[name] = {"error": f"{type(exc).__name__}: {exc}"}

def cmd(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"

payload = {
    "python": sys.version,
    "python_executable": sys.executable,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_smi": cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"]),
    "nvcc": cmd(["bash", "-lc", "nvcc --version | tail -n +1"]),
    "packages": packages,
    "verl_commit": cmd(["git", "-C", str(root / "verl"), "rev-parse", "HEAD"]),
    "model_path": str(model_path),
}
try:
    from transformers import AutoConfig, AutoProcessor
    from PIL import Image
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    payload["model"] = {
        "model_type": cfg.model_type,
        "architectures": getattr(cfg, "architectures", None),
        "processor_class": type(proc).__name__,
        "tokenizer_len": len(proc.tokenizer),
        "image_token_id": getattr(cfg, "image_token_id", None),
        "vision_config": getattr(cfg, "vision_config", None).to_dict() if getattr(cfg, "vision_config", None) else None,
    }
    merge_size = int(getattr(getattr(proc, "image_processor", None), "merge_size", 1) or 1)
    image_token_id = getattr(proc, "image_token_id", None)
    if image_token_id is None:
        image_token_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def processor_probe(num_images):
        content = []
        for idx in range(num_images):
            content.append({"type": "image"})
            content.append({"type": "text", "text": f" image {idx + 1}"})
        messages = [{"role": "user", "content": content}]
        raw = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = [Image.new("RGB", (64, 64), "white") for _ in range(num_images)]
        encoded = proc(text=[raw], images=images, return_tensors="pt")
        image_tokens = int((encoded["input_ids"][0] == image_token_id).sum().item())
        grid = encoded.get("image_grid_thw")
        feature_count = int(grid[:, 1:].prod(dim=1).sum().item()) if grid is not None else None
        expected_tokens = feature_count // (merge_size * merge_size) if feature_count is not None else None
        return {
            "num_images": num_images,
            "input_ids_shape": list(encoded["input_ids"].shape),
            "image_tokens": image_tokens,
            "feature_count": feature_count,
            "expected_image_tokens": expected_tokens,
            "alignment_ok": image_tokens == expected_tokens,
            "raw_tail": raw[-80:],
        }

    payload["processor_probe"] = {
        "single_image": processor_probe(1),
        "double_image": processor_probe(2),
    }
except Exception as exc:
    payload["model_error"] = f"{type(exc).__name__}: {exc}"

if os.environ.get("RUN_VLLM_PROBE", "1") != "0":
    try:
        from PIL import Image
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        def vllm_payload(num_images):
            content = []
            for idx in range(num_images):
                content.append({"type": "image"})
                content.append({"type": "text", "text": f" say ok for image {idx + 1}"})
            messages = [{"role": "user", "content": content}]
            prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [Image.new("RGB", (64, 64), "white") for _ in range(num_images)]
            return {"prompt": prompt, "multi_modal_data": {"image": images[0] if num_images == 1 else images}}

        llm = LLM(
            model=str(model_path),
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(os.environ.get("VLLM_PROBE_GPU_MEMORY_UTILIZATION", "0.20")),
            max_model_len=int(os.environ.get("VLLM_PROBE_MAX_MODEL_LEN", "4096")),
            limit_mm_per_prompt={"image": 2},
        )
        sampling = SamplingParams(max_tokens=8, temperature=0.0)
        outputs = llm.generate([vllm_payload(1), vllm_payload(2)], sampling)
        payload["vllm_probe"] = {
            "single_image": outputs[0].outputs[0].text,
            "double_image": outputs[1].outputs[0].text,
        }
    except Exception as exc:
        payload["vllm_probe_error"] = f"{type(exc).__name__}: {exc}"
else:
    payload["vllm_probe"] = "skipped by RUN_VLLM_PROBE=0"
pathlib.Path(os.environ["PROBE_OUT_TMP"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "${PY}" "${TMP_PROBE}"
mv "${TMP_JSON}" "${OUT}"

"${PY}" -m pip freeze > "${HERE}/requirements-lock.txt"
echo "wrote ${OUT}"
