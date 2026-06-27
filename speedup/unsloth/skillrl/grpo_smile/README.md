# GeoSkillRL GRPO on slime

This directory is a first slime adapter for bbox-only GeoSkillRL reinforcement
learning.  It does not depend on veRL at runtime.

## Files

- `prepare_slime_data.py`: converts `stageA/data/splits/*.jsonl` into slime JSONL.
- `rollout_slime.py`: slime `--custom-generate-function-path` for zoom -> crop -> answer.
- `reward_geo.py`: slime `--custom-rm-path`; supports `legacy` and `strict_v3`.
- `skillbank_runtime.py`: single-zoom SkillBank retrieval, with built-in fallback.
- `install_slime_env.sh`: isolated slime/Megatron/SGLang installer under `/root/autodl-tmp`.
- `check_slime_env.py`: preflight checks for imports, GPU visibility, model config, and data preparation.
- `run_slime_smoke.sh`: 1-GPU smoke launcher, assuming a local slime checkout.
- `run_slime_4gpu.sh`: 4-GPU launcher wrapper.
- `OPTIMIZATION_PLAN.md`: migration analysis and staged plan.

## Data Smoke

```bash
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_smile
python prepare_slime_data.py --limit 8 --output data/smoke_slime.jsonl
python test_reward_geo.py
```

## Training Smoke

Install the isolated slime stack first:

```bash
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_smile
bash install_slime_env.sh
eval "$(/root/autodl-tmp/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate slime
python check_slime_env.py
```

The launchers default to:

```bash
export SLIME_ROOT=/root/autodl-tmp/slime_stack/slime
export MEGATRON_ROOT=/root/autodl-tmp/slime_stack/Megatron-LM
export MODEL_PATH=/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/grpo_smile
bash run_slime_smoke.sh
```

For a 4-GPU smoke:

```bash
DATA_LIMIT=8 NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=8 bash run_slime_4gpu.sh
```

If your slime checkout uses a different Qwen model args file, set:

```bash
export MODEL_ARGS_FILE=$SLIME_ROOT/scripts/models/qwen3.5-4B.sh
```

## Reward Modes

- `legacy`: parity-oriented reward closest to the current local `grpo` scalar.
- `strict_v3`: stricter protocol and spatial shaping ported from `verl_grpo/reward/components_v3.py`.

Use:

```bash
REWARD_VERSION=strict_v3 bash run_slime_smoke.sh
```

For strict protocol training, also set `zoomearth_require_primitive_zoom: true`
in `slime_config.yaml`.
