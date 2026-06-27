# GeoSkillRL APO Experiment Runbook

All experiment code in this folder is isolated from the original `unsloth` scripts.
Do not edit files outside this directory for this APO experiment line.

## Generated Data

- Bridge-SFT primitive data:
  `/root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/bridge_sft_primitive.jsonl`
- Bridge smoke data:
  `/root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/bridge_sft_primitive_smoke.jsonl`
- Failure-aware smoke data:
  `/root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/failure_aware_primitive_smoke.jsonl`
- VLM rewritten no-seg data from zoom+seg causal CoT:
  `/root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/sft_zoom_seg_causal_no_seg_primitive.vlm.jsonl`

## Completed Checks

- `python -m py_compile /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/*.py`
- Primitive parser smoke for JSON and `<|box|>` zoom grammar.
- Bridge full data validation: no missing images and no zoom/tool-bbox mismatch.
- Stage 1 DIOR full eval completed.
- Trainer dry-run batch completed: 4 records, 3 zoom records, 84 bbox-boosted tokens.
- VLM no-seg rewrite completed: 1011 rows, 773 primitive zoom rows, 0 bad rows.
- VLM no-seg trainer dry-run completed: 4 records, 3 zoom records, 38 bbox-boosted tokens.

## Stage 1 Result

Output directory:
`/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/results/stage1_pretrain`

Summary:

```text
samples=1959
format_ok=0.989
label_match=0.987
mean_iou=0.4685
recall_0_3=0.632
recall_0_5=0.529
```

## Trainer Smoke Check

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/VQA/.venv/bin/python \
  /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/train_bridge_sft_zoom.py \
  --train_jsonl /root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/bridge_sft_primitive_smoke.jsonl \
  --output_dir /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/bridge_sft_smoke_dryrun \
  --primitive_grammar \
  --bbox_loss_alpha 2 \
  --batch_size_per_gpu 2 \
  --dry_run_batch
```

This checks real image loading, multi-image crop injection, tokenizer/collator output, labels, and bbox span loss weights without training or saving a model.

## Train Bridge-SFT Primitive

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/VQA/.venv/bin/python   /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/train_bridge_sft_zoom.py   --train_jsonl /root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/bridge_sft_primitive.jsonl   --output_dir /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/bridge_sft_primitive   --primitive_grammar   --bbox_loss_alpha 2   --num_train_epochs 3   --batch_size_per_gpu 1   --gradient_accumulation_steps 4
```

## Train VLM-Rewritten No-Seg SFT

This is the controlled ablation that starts from the zoom+seg causal CoT data,
uses VLM rewriting to remove only the segmentation stage, and keeps primitive
`<|ref|><|box|>` zoom.

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/VQA/.venv/bin/python \
  /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/train_bridge_sft_zoom.py \
  --train_jsonl /root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/sft_zoom_seg_causal_no_seg_primitive.vlm.jsonl \
  --output_dir /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive \
  --primitive_grammar \
  --bbox_loss_alpha 2 \
  --num_train_epochs 3 \
  --batch_size_per_gpu 1 \
  --gradient_accumulation_steps 4
```

## Evaluate VLM-Rewritten No-Seg SFT

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/VQA/.venv/bin/python \
  /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/eval_zoom_only_primitive.py \
  --model /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf \
  --test_jsonl /root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl \
  --output_dir /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/results/no_seg_vlm_rewrite_primitive \
  --tag no_seg_vlm_rewrite_primitive \
  --primitive_grammar \
  --batch_size 4
```

## Evaluate Bridge-SFT Primitive

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/VQA/.venv/bin/python   /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/eval_zoom_only_primitive.py   --model /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/output/bridge_sft_primitive/final_hf   --test_jsonl /root/autodl-tmp/VQA/json_data/test-00000-of-00001.jsonl   --output_dir /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/results/bridge_sft_primitive   --tag bridge_sft_primitive   --primitive_grammar   --batch_size 4
```

## Summarize APO Results

```bash
/root/autodl-tmp/VQA/.venv/bin/python   /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/summarize_apo_experiments.py   --output_dir /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/results/apo_summary   --prediction bridge_sft_primitive=/root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/results/bridge_sft_primitive/bridge_sft_primitive_predictions.jsonl
```

## Build Full Failure-Aware Data Later

Run this only after Bridge-SFT is stable and grammar validity is above 99%.

```bash
/root/autodl-tmp/VQA/.venv/bin/python   /root/autodl-tmp/VQA/speedup/unsloth/geoskillrl_apo_experiments/build_failure_aware_data.py   --include_global   --shuffle   --output /root/autodl-tmp/VQA/json_data/zoom_json/geoskillrl_apo/failure_aware_primitive.jsonl
```
