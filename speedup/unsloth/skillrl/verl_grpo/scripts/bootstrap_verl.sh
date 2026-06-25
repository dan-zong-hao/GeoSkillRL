#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/VQA"
VERL_DIR="${ROOT}/verl"
VENV="${ROOT}/.venv_verl_qwen35"
COMMIT="7aed6b230776f963fa09509c10d9c3a767d1102c"

cd "${ROOT}"
if [[ ! -d "${VERL_DIR}/.git" ]]; then
  git clone --branch v0.8.0 https://github.com/verl-project/verl.git "${VERL_DIR}"
fi
git -C "${VERL_DIR}" fetch --tags origin
git -C "${VERL_DIR}" checkout --detach "${COMMIT}"
test "$(git -C "${VERL_DIR}" rev-parse HEAD)" = "${COMMIT}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
echo "verl pinned at $(git -C "${VERL_DIR}" rev-parse HEAD)"
echo "venv ready at ${VENV}"

