#!/usr/bin/env bash
set -euo pipefail

VERL_DIR="${VERL_DIR:-/root/autodl-tmp/verl}"
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/VQA}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv_verl}"
TARGET_DIR="${PROJECT_DIR}/speedup/unsloth/skillrl/verl_grpo"

if [[ ! -d "${VERL_DIR}/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "${VERL_DIR}"
fi

git -C "${VERL_DIR}" checkout main
git -C "${VERL_DIR}" pull --ff-only
git -C "${VERL_DIR}" rev-parse HEAD > "${TARGET_DIR}/VERL_COMMIT"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools
python -m pip install -e "${VERL_DIR}"
python -m pip install "qwen-vl-utils" "pyarrow" "pandas" "pillow"

echo "Pinned verl commit: $(cat "${TARGET_DIR}/VERL_COMMIT")"
echo "Activate with: source ${VENV_DIR}/bin/activate"
