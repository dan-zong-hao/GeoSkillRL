# GeoSkillRL Official verl GRPO

This is the official verl migration for ZoomEarth-style GeoSkillRL GRPO.

The rollout path is:

`LRS-GRO JSONL -> verl Parquet -> RLHFDataset -> ZoomEarthAgentLoop -> vLLM async rollout -> GeoRewardManager -> main_ppo GRPO`.

## Setup

```bash
bash /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo/scripts/bootstrap_verl.sh
source /root/autodl-tmp/VQA/.venv_verl/bin/activate
export PYTHONPATH=/root/autodl-tmp/VQA:/root/autodl-tmp/verl:$PYTHONPATH
```

The pinned official commit is recorded in `VERL_COMMIT`.

## Data

```bash
python /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo/data/prepare_zoomearth_parquet.py \
  --train-jsonl /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl \
  --dev-jsonl /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl

python /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo/data/validate_zoomearth_parquet.py \
  /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo/data/processed/train.parquet \
  /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo/data/processed/val.parquet
```

## BBox Audit

```bash
python /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo/data/audit_bbox_coordinates.py \
  --jsonl /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl
```

Keep both `max_side` and `xy` modes until the overlay audit is manually accepted.

## Training Gates

```bash
bash scripts/run_rollout_smoke.sh
bash scripts/run_train_1gpu_smoke.sh
bash scripts/run_train_3gpu_smoke.sh
```

Do not run full training until the three smoke gates and the 128-sample effectiveness gate pass.

## Validation

```bash
MODEL_PATH=/path/to/baseline bash scripts/run_val_only.sh
MODEL_PATH=/path/to/final bash scripts/run_val_only.sh
```

Primary checkpoint metric is Hit@0.3. Answer accuracy is a validation metric only and is not part of the first policy reward.
