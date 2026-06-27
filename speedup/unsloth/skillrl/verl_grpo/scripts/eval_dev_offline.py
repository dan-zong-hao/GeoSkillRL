#!/usr/bin/env python3
"""Offline two-stage dev evaluation for the verl_grpo ZoomEarth agent."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.crop_environment import make_legacy_crop  # noqa: E402
from agent.protocol import extract_answer, extract_zoom, truncate_after  # noqa: E402
from agent.zoomearth_agent_loop import build_stage2_messages  # noqa: E402
from reward.components_legacy import compute_episode_reward  # noqa: E402


def as_list(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def load_rows(path: Path, limit: int, num_shards: int, shard_id: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    records = [jsonable(row.to_dict()) for _, row in df.iterrows()]
    if limit:
        records = records[:limit]
    if num_shards > 1:
        records = [row for idx, row in enumerate(records) if idx % num_shards == shard_id]
    return records


def image_path_from_row(row: dict[str, Any]) -> str:
    images = as_list(row["images"])
    return str(images[0]["image"])


def convert_prompt_messages(messages: list[dict[str, Any]], image_path: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, str) and "<image>" in content:
            text = content.replace("<image>", "", 1).strip()
            msg["content"] = [{"type": "image", "image": image_path}, {"type": "text", "text": text}]
        converted.append(msg)
    return converted


def image_list_for_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                image_value = part.get("image")
                if isinstance(image_value, Image.Image):
                    images.append(image_value.convert("RGB"))
                elif image_value:
                    images.append(Image.open(str(image_value)).convert("RGB"))
    return images


class SequenceBadWordsLogitsProcessor(LogitsProcessor):
    """Block exact token sequences during HF generation."""

    def __init__(self, bad_sequences: list[list[int]]):
        self.bad_sequences = [list(seq) for seq in bad_sequences if seq]
        self._prefix_cache: dict[tuple[str, torch.device, torch.dtype], list[tuple[torch.Tensor, int]]] = {}

    def _cached_sequences(self, device: torch.device) -> list[tuple[torch.Tensor, int]]:
        key = (str(device), device, torch.long)
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return cached
        cached = []
        for seq in self.bad_sequences:
            if len(seq) == 1:
                cached.append((torch.empty(0, dtype=torch.long, device=device), int(seq[0])))
            else:
                cached.append((torch.tensor(seq[:-1], dtype=torch.long, device=device), int(seq[-1])))
        self._prefix_cache[key] = cached
        return cached

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for prefix, next_token in self._cached_sequences(input_ids.device):
            if prefix.numel() == 0:
                scores[:, next_token] = -float("inf")
            elif input_ids.shape[1] >= prefix.numel():
                matches = (input_ids[:, -prefix.numel() :] == prefix).all(dim=1)
                scores[matches, next_token] = -float("inf")
        return scores


def bad_words_ids(tokenizer: Any, words: list[str]) -> list[list[int]]:
    output: list[list[int]] = []
    for word in words:
        candidates = [word]
        if word in {"<zoom>", "</zoom>", "<answer>", "</answer>"}:
            # Tags can be followed by merged tokens such as ">|", so also
            # suppress the open tag prefix before the final ">".
            candidates.extend([word[:-1], " " + word[:-1]])
        for candidate in candidates:
            try:
                ids = tokenizer.encode(candidate, add_special_tokens=False)
            except Exception:
                ids = []
            if ids:
                output.append([int(x) for x in ids])
        try:
            ids = tokenizer.encode(word, add_special_tokens=False)
        except Exception:
            ids = []
        if ids:
            output.append([int(x) for x in ids])
        try:
            token_id = tokenizer.convert_tokens_to_ids(word)
        except Exception:
            token_id = None
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            output.append([int(token_id)])

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seq in output:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            deduped.append(seq)
    return deduped


def clean_generation(text: str, marker: str) -> str:
    for token in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"):
        text = text.replace(token, "")
    return truncate_after(text, marker) if marker else text


def generation_kwargs(
    tokenizer: Any,
    *,
    max_new_tokens: int,
    stop: list[str],
    bad_words: list[str],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "num_beams": 1,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)})
    if stop:
        kwargs["stop_strings"] = stop
        kwargs["tokenizer"] = tokenizer
    bad_ids = bad_words_ids(tokenizer, bad_words)
    if bad_ids:
        kwargs["logits_processor"] = LogitsProcessorList([SequenceBadWordsLogitsProcessor(bad_ids)])
    return kwargs


def move_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def generated_mm_token_ids(processor: Any) -> set[int]:
    tokenizer = processor.tokenizer
    token_ids: set[int] = set()
    for attr in ("image_token_id", "video_token_id"):
        token_id = getattr(processor, attr, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    for token in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            token_ids.add(int(token_id))
    return token_ids


def strip_generated_mm_tokens(processor: Any, token_ids: list[int]) -> tuple[list[int], int]:
    blocked = generated_mm_token_ids(processor)
    if not blocked:
        return token_ids, 0
    kept = [token_id for token_id in token_ids if int(token_id) not in blocked]
    return kept, len(token_ids) - len(kept)


def strip_trailing_pad(tokenizer: Any, token_ids: list[int]) -> list[int]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return token_ids
    while token_ids and int(token_ids[-1]) == int(pad_token_id):
        token_ids.pop()
    return token_ids


def truncate_ids_after_marker(tokenizer: Any, token_ids: list[int], marker: str) -> tuple[list[int], str, int]:
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if not marker:
        return token_ids, raw_text, 0
    marker_end = (raw_text or "").lower().find(marker.lower())
    if marker_end < 0:
        return token_ids, raw_text, 0
    marker_end += len(marker)
    for end_idx in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:end_idx], skip_special_tokens=False)
        if len(decoded) >= marker_end:
            return token_ids[:end_idx], decoded, len(token_ids) - end_idx
    return token_ids, raw_text, 0


def postprocess_generated_ids(processor: Any, token_ids: list[int], *, marker: str) -> tuple[list[int], str]:
    tokenizer = processor.tokenizer
    token_ids = strip_trailing_pad(tokenizer, [int(x) for x in token_ids])
    token_ids, _removed = strip_generated_mm_tokens(processor, token_ids)
    token_ids, text, _trailing_removed = truncate_ids_after_marker(tokenizer, token_ids, marker)
    return token_ids, clean_generation(text, marker)


def encode_messages_batch(processor: Any, batch_messages: list[list[dict[str, Any]]]) -> dict[str, Any]:
    prompts = [
        processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        for messages in batch_messages
    ]
    images = [image_list_for_messages(messages) for messages in batch_messages]
    return dict(processor(text=prompts, images=images, return_tensors="pt", padding=True))


def generate_from_inputs(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    stop: list[str],
    bad_words: list[str],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    marker: str,
) -> list[dict[str, Any]]:
    tokenizer = processor.tokenizer
    input_width = int(inputs["input_ids"].shape[-1])
    device = next(model.parameters()).device
    model_inputs = move_to_device(inputs, device)
    kwargs = generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        stop=stop,
        bad_words=bad_words,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    with torch.inference_mode():
        generated = model.generate(**model_inputs, **kwargs)
    generated = generated[:, input_width:]
    outputs = []
    for idx in range(generated.shape[0]):
        token_ids, text = postprocess_generated_ids(processor, generated[idx].tolist(), marker=marker)
        outputs.append({"token_ids": token_ids, "text": text})
    return outputs


def generate_batch_outputs(
    model: Any,
    processor: Any,
    batch_messages: list[list[dict[str, Any]]],
    *,
    max_new_tokens: int,
    stop: list[str],
    bad_words: list[str],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    marker: str,
) -> list[dict[str, Any]]:
    if not batch_messages:
        return []
    inputs = encode_messages_batch(processor, batch_messages)
    return generate_from_inputs(
        model,
        processor,
        inputs,
        max_new_tokens=max_new_tokens,
        stop=stop,
        bad_words=bad_words,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        marker=marker,
    )


def assistant_end_ids(tokenizer: Any, generated_ids: list[int]) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        token_ids = [int(eos_token_id)]
    else:
        eos_token = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
        try:
            token_ids = [int(v) for v in tokenizer.encode(eos_token, add_special_tokens=False)]
        except Exception:
            token_ids = []
    if token_ids and generated_ids[-len(token_ids) :] == token_ids:
        return []
    return token_ids


def unpadded_token_ids(inputs: dict[str, Any], key: str = "input_ids") -> list[int]:
    values = inputs[key][0]
    mask = inputs.get("attention_mask")
    if mask is None:
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values[mask[0].bool()].tolist()]


def unpadded_mm_type_ids(inputs: dict[str, Any]) -> list[int] | None:
    if "mm_token_type_ids" not in inputs:
        return None
    values = inputs["mm_token_type_ids"][0]
    mask = inputs.get("attention_mask")
    if mask is None:
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values[mask[0].bool()].tolist()]


def single_message_encoding(processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return encode_messages_batch(processor, [messages])


def build_stage2_component(
    processor: Any,
    prompt_messages: list[dict[str, Any]],
    stage1_ids: list[int],
    crop: Image.Image,
) -> dict[str, Any]:
    prompt_inputs = single_message_encoding(processor, prompt_messages)
    obs_messages = build_stage2_messages("user")
    obs_messages[0]["content"][0]["image"] = crop
    obs_inputs = single_message_encoding(processor, obs_messages)

    prompt_ids = unpadded_token_ids(prompt_inputs)
    obs_ids = unpadded_token_ids(obs_inputs)
    end_ids = assistant_end_ids(processor.tokenizer, stage1_ids)
    input_ids = prompt_ids + stage1_ids + end_ids + obs_ids
    component: dict[str, Any] = {"input_ids": input_ids}

    prompt_mm = unpadded_mm_type_ids(prompt_inputs)
    obs_mm = unpadded_mm_type_ids(obs_inputs)
    if prompt_mm is not None or obs_mm is not None:
        component["mm_token_type_ids"] = (
            list(prompt_mm or [0] * len(prompt_ids))
            + [0] * (len(stage1_ids) + len(end_ids))
            + list(obs_mm or [0] * len(obs_ids))
        )

    for key in ("pixel_values", "image_grid_thw"):
        tensors = [inputs[key] for inputs in (prompt_inputs, obs_inputs) if key in inputs]
        if tensors:
            component[key] = torch.cat(tensors, dim=0)
    component["assistant_end_tokens"] = len(end_ids)
    component["tool_observation_tokens"] = len(obs_ids)
    return component


def pad_manual_components(components: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None) or 0
    max_len = max(len(component["input_ids"]) for component in components)
    input_ids = []
    attention_mask = []
    mm_token_type_ids = []
    has_mm_types = any("mm_token_type_ids" in component for component in components)
    for component in components:
        ids = list(component["input_ids"])
        pad_len = max_len - len(ids)
        input_ids.append([int(pad_token_id)] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))
        if has_mm_types:
            mm_ids = list(component.get("mm_token_type_ids") or [0] * len(ids))
            mm_token_type_ids.append([0] * pad_len + mm_ids)
    batch: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    if has_mm_types:
        batch["mm_token_type_ids"] = torch.tensor(mm_token_type_ids, dtype=torch.long)
    for key in ("pixel_values", "image_grid_thw"):
        tensors = [component[key] for component in components if key in component]
        if tensors:
            batch[key] = torch.cat(tensors, dim=0)
    return batch


def generate_stage2_from_components(
    model: Any,
    processor: Any,
    components: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not components:
        return []
    inputs = pad_manual_components(components, processor.tokenizer)
    return generate_from_inputs(
        model,
        processor,
        inputs,
        max_new_tokens=args.stage2_max_new_tokens,
        stop=["</answer>"],
        bad_words=["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<zoom>", "</zoom>"],
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        marker="</answer>",
    )


def ground_truth_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["reward_model"]["ground_truth"]
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)


def build_result(
    row: dict[str, Any],
    stage1_text: str,
    stage2_text: str,
    *,
    num_turns: int,
    tool_error: str,
) -> dict[str, Any]:
    extra = row["extra_info"]
    gt = ground_truth_payload(row)
    zoom = extract_zoom(stage1_text)
    answer = extract_answer(stage2_text)
    reward = compute_episode_reward(
        question_id=str(extra.get("question_id") or ""),
        question=str(extra.get("question") or ""),
        zoom_text=stage1_text,
        answer_text=stage2_text,
        gt_bbox_1024=[float(v) for v in as_list(extra.get("gt_bbox_1024"))],
        ground_truth=gt.get("ground_truth"),
        image_size=tuple(int(v) for v in as_list(extra.get("image_size"))),
    )
    return {
        "question_id": extra.get("question_id"),
        "image_name": extra.get("image_name"),
        "type": extra.get("type"),
        "category": extra.get("category"),
        "question": extra.get("question"),
        "ground_truth": gt.get("ground_truth"),
        "gt_bbox_1024": as_list(extra.get("gt_bbox_1024")),
        "pred_bbox_1024": reward.get("pred_bbox_1024"),
        "answer_pred": reward.get("answer_pred"),
        "zoom_parse_ok": bool(zoom.parse_ok),
        "answer_parse_ok": bool(answer.parse_ok),
        "stage1_raw_text": stage1_text,
        "stage2_raw_text": stage2_text,
        "trajectory_text": stage1_text + "\n" + stage2_text,
        "retrieved_skill_ids": as_list(extra.get("retrieved_skill_ids") or []),
        "num_turns": num_turns,
        "tool_error": tool_error,
        "reward": reward,
    }


def eval_one(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return eval_batch(model, processor, [row], args)[0]


def eval_batch(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompt_messages = [
        convert_prompt_messages(as_list(row["prompt"]), image_path_from_row(row))
        for row in rows
    ]
    stage1_outputs = generate_batch_outputs(
        model,
        processor,
        prompt_messages,
        max_new_tokens=args.stage1_max_new_tokens,
        stop=["</zoom>"],
        bad_words=["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<answer>", "</answer>"],
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        marker="</zoom>",
    )
    stage1_texts = [output["text"] for output in stage1_outputs]

    stage2_rows: list[int] = []
    stage2_components: list[dict[str, Any]] = []
    stage2_texts = [""] * len(rows)
    tool_errors = [""] * len(rows)
    num_turns = [2] * len(rows)
    for idx, (row, messages, stage1_text) in enumerate(zip(rows, prompt_messages, stage1_texts, strict=True)):
        zoom = extract_zoom(stage1_text)
        if not zoom.parse_ok:
            continue
        try:
            extra = row["extra_info"]
            image_path = extra.get("original_image_path") or extra.get("global_image_path")
            crop, _meta = make_legacy_crop(image_path, zoom.bbox_1024, max_size=args.crop_max_size)
            stage2_rows.append(idx)
            stage2_components.append(
                build_stage2_component(
                    processor,
                    messages,
                    list(stage1_outputs[idx]["token_ids"]),
                    crop,
                )
            )
            num_turns[idx] = 4
        except Exception as exc:  # noqa: BLE001
            tool_errors[idx] = f"{type(exc).__name__}: {exc}"

    if stage2_components:
        generated_stage2 = generate_stage2_from_components(
            model,
            processor,
            stage2_components,
            args,
        )
        for idx, output in zip(stage2_rows, generated_stage2, strict=True):
            stage2_texts[idx] = output["text"]

    return [
        build_result(
            row,
            stage1_texts[idx],
            stage2_texts[idx],
            num_turns=num_turns[idx],
            tool_error=tool_errors[idx],
        )
        for idx, row in enumerate(rows)
    ]


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def summarize(rows: list[dict[str, Any]], *, model: str, data: str, tag: str) -> dict[str, Any]:
    n = len(rows)
    rewards = [row.get("reward", {}) for row in rows]
    total = lambda key: sum(float(r.get(key) or 0.0) for r in rewards)
    table = {
        "Answer Acc": pct(sum(1 for r in rewards if r.get("answer_correct")), n),
        "APO IoU": pct(total("iou"), n),
        "Hit@0.3": pct(sum(1 for r in rewards if float(r.get("iou") or 0.0) >= 0.3), n),
        "Hit@0.5": pct(sum(1 for r in rewards if float(r.get("iou") or 0.0) >= 0.5), n),
        "BBox Pred Rate": pct(sum(1 for r in rewards if r.get("pred_bbox_1024")), n),
        "Zoom Parse Fail": pct(sum(1 for r in rewards if not r.get("zoom_parse_ok")), n),
        "Answer Parse Fail": pct(sum(1 for r in rewards if not r.get("answer_parse_ok")), n),
        "Stage2 Zoom Leak": pct(sum(1 for row in rows if "<zoom>" in str(row.get("stage2_raw_text") or "").lower()), n),
        "Mean Reward": total("total") / max(n, 1),
        "Mean Turns": sum(float(row.get("num_turns") or 0.0) for row in rows) / max(n, 1),
    }
    reward_keys = [
        "R_task_success",
        "R_bbox_region",
        "R_grounded_answer",
        "R_format",
        "R_spatial",
        "P_parse",
        "P_area",
        "P_false_grounded",
        "total",
    ]
    return {
        "tag": tag,
        "model": model,
        "data": data,
        "eval_mode": "hf_token_concat_stage2",
        "n_samples": n,
        "table": table,
        "reward_means": {key: total(key) / max(n, 1) for key in reward_keys},
        "counts": {
            "by_type": dict(Counter(str(row.get("type") or "unknown") for row in rows)),
            "by_category": dict(Counter(str(row.get("category") or "unknown") for row in rows)),
        },
    }


def write_summary(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed_sec: float) -> None:
    summary = summarize(rows, model=str(args.model), data=str(args.data), tag=args.tag)
    summary["elapsed_sec"] = elapsed_sec
    summary["num_shards"] = args.num_shards
    summary_path = args.output_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = summary["table"]
    line = (
        f"Answer={table['Answer Acc']:.2f} APO_IoU={table['APO IoU']:.2f} "
        f"Hit03={table['Hit@0.3']:.2f} Hit05={table['Hit@0.5']:.2f} "
        f"BBoxPred={table['BBox Pred Rate']:.2f} ZoomParseFail={table['Zoom Parse Fail']:.2f} "
        f"AnsParseFail={table['Answer Parse Fail']:.2f} Stage2Zoom={table['Stage2 Zoom Leak']:.2f} "
        f"MeanReward={table['Mean Reward']:.3f} "
        f"MeanTurns={table['Mean Turns']:.2f}"
    )
    (args.output_dir / f"{args.tag}_table.txt").write_text(line + "\n", encoding="utf-8")
    print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/parquet/dev.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="grpo_final_dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=512)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=1024)
    parser.add_argument("--crop-max-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--do-sample", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"{args.tag}_predictions.jsonl"
    rows = load_rows(args.data, args.limit, args.num_shards, args.shard_id)
    print(json.dumps({
        "model": str(args.model),
        "data": str(args.data),
        "rows": len(rows),
        "output": str(pred_path),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
    }, ensure_ascii=False, indent=2))

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    results: list[dict[str, Any]] = []
    start = time.perf_counter()
    with pred_path.open("w", encoding="utf-8") as f, tqdm(total=len(rows), desc=f"eval shard {args.shard_id}") as pbar:
        for start_idx in range(0, len(rows), max(1, int(args.batch_size))):
            batch = rows[start_idx : start_idx + max(1, int(args.batch_size))]
            try:
                batch_results = eval_batch(model, processor, batch, args) if args.batch_size > 1 else [
                    eval_one(model, processor, batch[0], args)
                ]
            except Exception as exc:  # noqa: BLE001
                batch_results = []
                for row in batch:
                    extra = row.get("extra_info", {}) or {}
                    batch_results.append({
                        "question_id": extra.get("question_id"),
                        "image_name": extra.get("image_name"),
                        "type": extra.get("type"),
                        "category": extra.get("category"),
                        "question": extra.get("question"),
                        "ground_truth": ground_truth_payload(row).get("ground_truth"),
                        "trajectory_text": "",
                        "zoom_parse_ok": False,
                        "answer_parse_ok": False,
                        "num_turns": 0,
                        "tool_error": f"{type(exc).__name__}: {exc}",
                        "reward": {
                            "total": -1.5,
                            "iou": 0.0,
                            "hit03": 0.0,
                            "hit05": 0.0,
                            "answer_correct": False,
                            "zoom_parse_ok": False,
                            "answer_parse_ok": False,
                        },
                    })
            for result in batch_results:
                result = jsonable(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
            f.flush()
            pbar.update(len(batch_results))
    write_summary(results, args, time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
