# Optimization Plan

## 1. Local Code Differences

### `grpo`

- Training is a local PyTorch/DDP loop around `model.generate`.
- Skill retrieval happens at runtime for every sample through
  `single_zoom/skillbank_single_zoom.py`.
- The SkillBank trigger logic is richer than veRL: it always injects
  `gen_skill_first_grounding` and `format_primitive_bbox`, then adds task
  skills from ordered rules such as side subpart, anchor relation, ranked
  instance/corner, center/main region, whole-region scan, and local attribute.
- Reward is legacy-shaped and accepts compact bbox output through loose bbox
  extraction.  It logs spatial violations but does not subtract a spatial
  penalty.
- The handwritten GRPO loss uses current logprobs as detached old logprobs
  inside the same forward pass, so the PPO ratio is effectively 1 for each
  update.  This is a major reason to move to a framework with rollout logprob
  storage.

### `verl_grpo`

- Data is prepared into parquet and the skill block is frozen into `extra_info`.
- The default `reward_manager.py` currently switches only between `legacy` and
  `v2`; the experimental `components_v3.py` is present but not wired into the
  manager by default.
- Current `components_legacy.py` parses zoom with `require_primitive=True`, so
  compact `<zoom>[[...]]</zoom>` can be treated as parse failure in the current
  code even though older rollout logs show legacy compact parsing behavior.
- The AgentLoop enforces a two-stage flow with vLLM: stage 1 emits zoom, then a
  user-role crop observation is appended, then stage 2 emits answer.
- It masks model tokens for zoom and answer spans and masks observation tokens.
  This is closer to a real RL trajectory than the local `grpo` loss.

## 2. Reward Plan

Use two explicit modes instead of hidden branches:

- Phase A: `legacy`
  - Goal: reproduce comparable behavior to local `grpo` and current veRL
    default experiments.
  - Keep `zoomearth_require_primitive_zoom=false` so compact boxes still get a
    crop during parity smoke runs.

- Phase B: `strict_v3`
  - Goal: stop rewarding correct answers grounded on wrong crops.
  - Add primitive-format reward, axis reward, protocol penalty, and explicit
    spatial penalty.
  - Set `zoomearth_require_primitive_zoom=true` after primitive-format parse
    rate is stable.

Metrics to track per rollout:

- reward mean and std by prompt group
- bbox parse rate
- primitive format rate
- IoU, hit@0.3, hit@0.5
- answer accuracy
- false-grounded rate: answer correct but IoU < 0.3
- spatial violation rate for locator questions

## 3. Skill Trigger Plan

Keep the single-zoom trigger logic from `grpo` as the default.  Compared with
`verl_grpo`, this gives the model more task-specific guidance and includes the
format skill in every prompt.

Operationally, slime data preparation freezes the skill block into JSONL.  If
the SkillBank evolves, regenerate the JSONL before the next RL run:

```bash
python prepare_slime_data.py --input ../stageA/data/splits/rl_train.jsonl --output data/train_slime.jsonl
```

Do not make skill retrieval stochastic inside rollout until parity is stable.
Frozen skills make reward and parse regressions easier to audit.

## 4. slime Migration Plan

1. Environment
   - Clone THUDM/slime into `/root/autodl-tmp/VQA/slime` or set `SLIME_ROOT`.
   - Prefer the official slime Docker image for Megatron/SGLang compatibility.
   - For Qwen3.5 VLM, verify the Megatron Bridge branch and model args file.

2. Data smoke
   - Run `prepare_slime_data.py --limit 8`.
   - Inspect prompt, image path, label JSON, metadata, retrieved skill ids.

3. Reward parity
   - Run `test_reward_geo.py`.
   - Score a few existing `grpo/logs/*rollouts*.jsonl` rows offline with
     `legacy` and compare totals before changing reward weights.

4. 1-GPU training smoke
   - Run `run_slime_smoke.sh` with `NUM_ROLLOUT=1`, `N_SAMPLES_PER_PROMPT=2`.
   - Check parse rate, reward variance, and that loss masks include generated
     zoom/answer tokens but exclude crop observation tokens.

5. 4-GPU short run
   - Run `run_slime_4gpu.sh` with `NUM_ROLLOUT=20`.
   - Tune `rollout-batch-size`, `global-batch-size`, and SGLang memory fraction.

6. Strict protocol ablation
   - Switch `REWARD_VERSION=strict_v3`.
   - After primitive-format rate is acceptable, set
     `zoomearth_require_primitive_zoom: true`.

7. Eval
   - Reuse the same custom generate and reward on `rl_dev`.
   - Compare no-skill, skill-trigger coldstart, local `grpo`, veRL, and slime
     runs on hit@0.3/hit@0.5, answer accuracy, and false-grounded rate.

## 5. Main Risks

- Local model checkpoints may need conversion or a matching Megatron args file
  before slime can train them.
- Qwen3.5-VL image template behavior must be checked in SGLang; crop image
  token counts should match processor feature counts.
- `strict_v3` can collapse reward variance early if primitive-format parse rate
  is low.  Start with legacy parity, then tighten.
- If the SkillBank evolves during training, frozen JSONL must be regenerated to
  avoid training/eval drift.

