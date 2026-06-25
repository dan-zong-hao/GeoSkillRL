# Corrected ZoomEarth GRPO

This directory contains a clean implementation of the fixes described in the
analysis note, while leaving the previous `grpo_zoomearth` experiment intact.

Key changes:

- Split `zoom_reward` and `answer_reward`; do not broadcast a mixed scalar
  reward to both stages.
- Default objective is `zoom_only`: only `<zoom>...</zoom>` response tokens
  receive policy loss. Turn 2 is still generated for measurement, but answer
  tokens are masked out unless `--objective split` is selected.
- Generation stores raw `prompt input_ids`, `response_ids`, `old_logps`, and a
  response-local loss mask. Training reuses those tensors instead of computing
  a prompt boundary with a text-only tokenizer.
- Group advantages use `(reward - mean) / (std + eps)`. Zero-variance groups get
  zero advantage locally instead of forcing all ranks to skip the update.
- Skill retrieval uses question text only, not the dataset `label` field.
- Bbox-to-original-image conversion defaults to independent x/y scaling. Use
  `--bbox_coord_mode max_side` to reproduce the old max-side mapping.

Smoke test:

```bash
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo
source /root/autodl-tmp/VQA/.venv/bin/activate
python test_verl_grpo_adapter.py
```

Training smoke run:

```bash
source /root/autodl-tmp/VQA/.venv/bin/activate
bash run_smoke.sh
```

Full training:

```bash
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo
bash run_grpo_full_single_gpu.sh
bash run_grpo_full_3gpu.sh
```

Background 3-GPU run:

```bash
cd /root/autodl-tmp/VQA/speedup/unsloth/skillrl/verl_grpo
bash run_background_3gpu.sh
```
