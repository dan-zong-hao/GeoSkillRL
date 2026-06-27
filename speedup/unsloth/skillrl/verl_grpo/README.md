# bbox-only ZoomEarth GRPO over verl

This directory contains the project-side implementation for bbox-only ZoomEarth
GRPO using verl v0.8.0. It covers object/region bbox samples only and does not
claim full whether-to-crop training.

## Bootstrap

```bash
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo
bash scripts/bootstrap_verl.sh
bash scripts/probe_environment.sh
```

`/root/autodl-tmp/VQA/verl` is pinned to commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`. Official verl source is not
modified by this implementation.

## Data

```bash
/root/autodl-tmp/VQA/.venv/bin/python data/prepare_zoomearth_parquet.py \
  --input /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl \
  --output data/parquet/train.parquet
/root/autodl-tmp/VQA/.venv/bin/python data/prepare_zoomearth_parquet.py \
  --input /root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl \
  --output data/parquet/dev.parquet
/root/autodl-tmp/VQA/.venv/bin/python data/validate_zoomearth_parquet.py data/parquet/train.parquet
/root/autodl-tmp/VQA/.venv/bin/python data/validate_zoomearth_parquet.py data/parquet/dev.parquet
```

The train source is `/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl`.
The dev/eval source is `/root/autodl-tmp/VQA/speedup/unsloth/skillrl/stageA/data/splits/rl_dev.jsonl`.
Only `object|region` rows with a four-value bbox are used, so global QA rows are
excluded from this bbox-only ZoomEarth task. SkillBank retrieval is frozen into
`extra_info` and uses only question text.

## Smoke and Training

```bash
bash scripts/run_train_1gpu_smoke.sh trainer.total_training_steps=2
bash scripts/run_train_4gpu_smoke.sh trainer.total_training_steps=20
bash scripts/run_train_4gpu.sh
```

Formal 4GPU training defaults:

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `trainer.n_gpus_per_node=4`
- `actor_rollout_ref.rollout.calculate_log_probs=False`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `algorithm.use_kl_in_reward=False`
- `actor_rollout_ref.actor.kl_loss_coef=0.04`
- `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean`

Diagnostic smoke scripts set `calculate_log_probs=True`.
