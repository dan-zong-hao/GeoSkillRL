#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/VQA
LIMIT=0
if [[ "${1:-}" == "--limit" ]]; then
  LIMIT="${2:-0}"
fi
.venv/bin/python speedup/unsloth/skillrl/build_skill_sft.py --limit "$LIMIT"
.venv/bin/python speedup/unsloth/skillrl/mine_failures.py   --limit "$LIMIT"   --skillbank /root/autodl-tmp/VQA/json_data/zoom_json/skillrl/skillbank_round0.json
.venv/bin/python speedup/unsloth/skillrl/audit_skillrl.py
