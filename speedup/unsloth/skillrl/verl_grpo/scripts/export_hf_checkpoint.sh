#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/VQA"
PY="${PYTHON:-${ROOT}/.venv_verl_qwen35/bin/python}"
CHECKPOINT="${1:?usage: export_hf_checkpoint.sh <verl-checkpoint-dir> [output-dir]}"
OUTPUT="${2:-${CHECKPOINT}/hf_export}"

export PYTHONPATH="${ROOT}/verl:${PYTHONPATH:-}"
"${PY}" - <<'PY' "${CHECKPOINT}" "${OUTPUT}"
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.exists():
    raise SystemExit(f"checkpoint not found: {src}")
hf_candidates = [p for p in src.rglob("*") if p.name in {"config.json", "model.safetensors", "pytorch_model.bin"}]
if not hf_candidates:
    raise SystemExit("No direct HF export found. Use verl's checkpoint conversion utility for this checkpoint layout.")
root = min((p.parent for p in hf_candidates), key=lambda p: len(str(p)))
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(root, dst)
print(dst)
PY

