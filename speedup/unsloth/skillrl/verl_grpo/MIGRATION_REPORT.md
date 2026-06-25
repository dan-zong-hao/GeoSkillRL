# Migration Report

## Local State

- Target directory existed and was empty before implementation:
  `/root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo`.
- The workspace root `/root/autodl-tmp/VQA` is not a Git repository, so branch
  and `git status` gates are represented by this file-level report.
- `/root/autodl-tmp/VQA/verl` was absent initially. `scripts/bootstrap_verl.sh`
  pins it to `v0.8.0` commit `7aed6b230776f963fa09509c10d9c3a767d1102c`.
- The default model path is the existing verl-compatible export:
  `/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl`.

## Scope

This implementation is bbox-only ZoomEarth GRPO. It filters to object/region
rows with bboxes and does not include global whether-to-crop behavior.

## Plan Constraints Implemented

- 4GPU scripts only for multi-GPU training.
- SkillBank retrieval uses question text only and is frozen into parquet.
- Each row gets a unique stable `extra_info.index`.
- AgentLoop `extra_fields` has a stable schema containing stage raw text,
  parsed text, parse flags, bbox, crop state, token counts, and tool error.
- Format errors fall back to nonzero generated-token masks.
- Crop uses legacy crop/expand/resize semantics.
- Stage-2 role is probed by `scripts/probe_stage2_role_parity.py`.
- Formal training keeps `calculate_log_probs=False`; smoke diagnostics enable it.
- KL is actor KL loss only: `use_kl_loss=true`, `use_kl_in_reward=false`,
  `kl_loss_coef=0.04`.
- Default loss aggregation is `seq-mean-token-mean`.

## Local Verification

- verl checkout: `/root/autodl-tmp/VQA/verl` at
  `7aed6b230776f963fa09509c10d9c3a767d1102c`.
- Environment probe refreshed `environment.lock.json` and `requirements-lock.txt`
  with the current `/root/autodl-tmp/VQA/.venv` package versions. Processor
  single-image and double-image probes passed image token/feature alignment.
  vLLM single-image and double-image generation probes completed and are
  recorded in the lock file.
- Stage-2 role parity selected `user`; `tool` is rejected by the Qwen template,
  and `assistant+user` fails the observation-turn parity check.
- Dataset smoke: prepared two independent 8-row bbox-only parquets under
  `/tmp` and validator passed schema, unique index, stable source ordering,
  reference index consistency, bbox-only type, and label-leak checks.
- Reward parity: legacy reward matched old rollout logs with zero error for
  500/500 rows from `geoskill_grpo_skillbank_v1_rollouts_rank0.jsonl` and
  228/228 rows from `geoskill_grpo_skillbank_v1_t1_384_rollouts_rank0.jsonl`.
- Local harness covered parser, mask fallback, crop semantics, dataset schema,
  SkillBank question-only retrieval, and AgentLoop token/mask behavior. Pytest
  is not installed in the current environment, so tests were invoked directly.

## Not Run

- Full 1GPU/4GPU optimizer-step training smoke was not run in this pass.
