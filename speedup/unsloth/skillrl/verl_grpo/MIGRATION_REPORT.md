# Migration Report

This directory is the new official verl implementation for GeoSkillRL ZoomEarth GRPO.

The prior hand-written baseline remains under `speedup/unsloth/skillrl/grpo` as a read-only behavior reference.
The current `speedup/unsloth/skillrl/verl_grpo` directory was empty before this migration, so there was no file to move to `verl_grpo_hf_legacy`.
The HF-style ZoomEarth experiment remains under `speedup/unsloth/skillrl/grpo_zoomearth` and was not modified.

Mapping:

- Data JSONL filtering and prompt construction moved to `data/prepare_zoomearth_parquet.py`.
- Runtime SkillBank retrieval is now offline during Parquet creation and uses only `question`.
- Zoom parsing and raw token response masks live in `agent/zoom_protocol.py`.
- Original-image crop and coordinate-mode auditing live in `agent/crop_environment.py` and `data/audit_bbox_coordinates.py`.
- Reward math lives in `reward/components.py`.
- Official verl rollout integration lives in `agent/zoomearth_agent_loop.py`.
- Official reward integration lives in `reward/reward_manager.py`.
- Launch commands live in `scripts/`.

Current gate status:

- Unit tests: local pure-Python tests are provided under `tests/`.
- Rollout smoke: script provided, not executed in this environment because vLLM is missing.
- One-GPU smoke: script provided, not executed in this environment because vLLM is missing.
- Three-GPU smoke: script provided, not executed because only one GPU was visible.
- Full training: blocked until Gate A-D pass.
