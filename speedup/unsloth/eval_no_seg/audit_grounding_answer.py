#!/usr/bin/env python3
"""Offline audit for grounding-answer decoupling in VQA eval results.

This script does not run inference and does not modify the original prediction
files. It reads existing JSONL outputs and reports cases where localization is
poor but the final answer is still correct, including cases where the model
claims the crop is correct.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = {
    "origin": "speedup/origin/eval/sft_casual_test_results/sft_causal_cot_predictions.jsonl",
    "no_seg": "speedup/unsloth/eval_no_seg/results/zoom_only_fast_4gpu_repaired_predictions.jsonl",
    "unsloth_seg": "speedup/unsloth/eval/results/sft_unsloth_4gpu_predictions.jsonl",
}

CLOSED_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
OPEN_ANSWER_RE = re.compile(r"<answer>\s*(.*)$", re.DOTALL | re.IGNORECASE)

CROP_CLAIM_CORRECT_RE = re.compile(
    r"\[Crop Verification\].{0,360}?"
    r"(?:correctly|does contain|contains the target|provided crop image contains|"
    r"crop image contains|target[^.\n]{0,80}visible|clearly shows|shows the[^.\n]{0,80}target|"
    r"full [^.\n]{0,80}clearly visible|zoom level sufficient)",
    re.IGNORECASE | re.DOTALL,
)
CROP_CLAIM_FAILED_RE = re.compile(
    r"\[Crop Verification\].{0,360}?"
    r"(?:does not contain|doesn't contain|failed|not contain|no [^.\n]{0,80}visible|"
    r"wrong|different area|crop failed|cannot find|not visible|disregard its content)",
    re.IGNORECASE | re.DOTALL,
)

BINARY_ANSWER_SETS = (
    {"yes", "no"},
    {"urban", "rural"},
    {"true", "false"},
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def answer_matches(gt: Any, pred: Any) -> bool:
    gt_norm = norm_text(gt)
    pred_norm = norm_text(pred)
    if not gt_norm or not pred_norm:
        return False
    return gt_norm == pred_norm or gt_norm in pred_norm or pred_norm in gt_norm


def robust_answer_from_trajectory(trajectory: str) -> str | None:
    """Recover answers truncated around </answer>.

    Handles forms such as:
      <answer>airport
      <answer>airport</
      <answer>airport</answer
      <answer>residential</answer
    """
    if not trajectory:
        return None
    closed = CLOSED_ANSWER_RE.search(trajectory)
    if closed:
        return closed.group(1).strip() or None

    open_match = OPEN_ANSWER_RE.search(trajectory)
    if not open_match:
        return None

    answer = open_match.group(1).strip()
    answer = re.split(
        r"(?:<\|im_end\|>|\n\[TURN\d+\]|\n<think>|Question Intent:|Localization Strategy:|Reasoning Result:)",
        answer,
        maxsplit=1,
    )[0].strip()
    answer = re.sub(r"</answer\s*$", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"</\s*$", "", answer).strip()
    answer = answer.strip("` \n\t")
    return answer or None


def robust_answer(row: dict[str, Any]) -> str | None:
    if row.get("final_answer"):
        return str(row["final_answer"])
    return robust_answer_from_trajectory(row.get("trajectory") or "")


def is_binary_like(row: dict[str, Any]) -> bool:
    gt = norm_text(row.get("ground_truth"))
    if any(gt in answer_set for answer_set in BINARY_ANSWER_SETS):
        return True
    question = norm_text(row.get("question"))
    # Global either-or questions can be inflated by two-choice guessing even when
    # the answer set is not yes/no, e.g. urban/rural or farmland/residential.
    return str(row.get("type", "")).lower() == "global" and " or " in question and "?" in question


def get_iou(row: dict[str, Any]) -> float:
    value = row.get("iou")
    if value is None:
        value = row.get("iou_apo512")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def action_key(row: dict[str, Any]) -> str:
    actions = row.get("actions") or []
    return "+".join(str(x) for x in actions) or "none"


def has_bbox_pair(row: dict[str, Any]) -> bool:
    return bool(row.get("bbox_gt_1024") and row.get("bbox_pred_1024"))


def crop_claim(row: dict[str, Any]) -> str:
    trajectory = row.get("trajectory") or ""
    positive = bool(CROP_CLAIM_CORRECT_RE.search(trajectory))
    negative = bool(CROP_CLAIM_FAILED_RE.search(trajectory))
    if positive and not negative:
        return "claimed_correct"
    if negative and not positive:
        return "claimed_failed"
    if positive and negative:
        return "mixed"
    return "unknown"


def percent(num: int | float, den: int | float) -> float:
    return round(100.0 * float(num) / max(float(den), 1.0), 4)


def acc(rows: list[dict[str, Any]], use_robust: bool = False) -> float:
    if use_robust:
        correct = sum(1 for row in rows if answer_matches(row.get("ground_truth"), robust_answer(row)))
    else:
        correct = sum(1 for row in rows if row.get("correct_final"))
    return percent(correct, len(rows))


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, "unknown"))].append(row)
    return dict(out)


def summarize_model(name: str, rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    obj_region = [row for row in rows if norm_text(row.get("type")) in {"object", "region"}]
    zoom_with_bbox = [row for row in obj_region if has_bbox_pair(row)]
    summary: dict[str, Any] = {
        "name": name,
        "n_rows": len(rows),
        "n_by_type": dict(Counter(norm_text(row.get("type")) or "unknown" for row in rows)),
        "accuracy": acc(rows),
        "actions": dict(Counter(action_key(row) for row in rows)),
        "errors": dict(Counter(str(row.get("error")) for row in rows).most_common(20)),
        "object_region_with_bbox": len(zoom_with_bbox),
        "by_threshold": {},
        "binary_like": {
            "n": sum(1 for row in rows if is_binary_like(row)),
            "accuracy": acc([row for row in rows if is_binary_like(row)]),
        },
        "nonbinary": {
            "n": sum(1 for row in rows if not is_binary_like(row)),
            "accuracy": acc([row for row in rows if not is_binary_like(row)]),
        },
        "by_type": {},
        "by_category": {},
    }

    for threshold in thresholds:
        wrong_bbox = [row for row in zoom_with_bbox if get_iou(row) < threshold]
        right_bbox = [row for row in zoom_with_bbox if get_iou(row) >= threshold]
        wrong_and_correct = [row for row in wrong_bbox if row.get("correct_final")]
        suspicious = [
            row for row in wrong_and_correct
            if crop_claim(row) == "claimed_correct"
        ]
        claimed_failed = [
            row for row in wrong_and_correct
            if crop_claim(row) == "claimed_failed"
        ]
        threshold_key = str(threshold)
        summary["by_threshold"][threshold_key] = {
            "wrong_bbox_n": len(wrong_bbox),
            "right_bbox_n": len(right_bbox),
            "acc_when_wrong_bbox": acc(wrong_bbox),
            "acc_when_right_bbox": acc(right_bbox),
            "wrong_bbox_answer_correct_n": len(wrong_and_correct),
            "wrong_bbox_answer_correct_rate_on_bbox_samples": percent(len(wrong_and_correct), len(zoom_with_bbox)),
            "wrong_bbox_answer_correct_crop_claim_correct_n": len(suspicious),
            "wrong_bbox_answer_correct_crop_claim_correct_rate_on_bbox_samples": percent(len(suspicious), len(zoom_with_bbox)),
            "wrong_bbox_answer_correct_crop_claim_failed_n": len(claimed_failed),
            "binary_like_in_wrong_bbox_answer_correct": sum(1 for row in wrong_and_correct if is_binary_like(row)),
            "nonbinary_in_wrong_bbox_answer_correct": sum(1 for row in wrong_and_correct if not is_binary_like(row)),
            "top_categories_suspicious": Counter(str(row.get("category")) for row in suspicious).most_common(20),
        }

    for type_name, type_rows in sorted(grouped(rows, "type").items()):
        summary["by_type"][type_name] = {
            "n": len(type_rows),
            "accuracy": acc(type_rows),
            "missing_answer": sum(1 for row in type_rows if not row.get("final_answer")),
            "errors": dict(Counter(str(row.get("error")) for row in type_rows).most_common(10)),
        }

    for category, cat_rows in sorted(grouped(rows, "category").items()):
        summary["by_category"][category] = {
            "n": len(cat_rows),
            "accuracy": acc(cat_rows),
            "binary_like_n": sum(1 for row in cat_rows if is_binary_like(row)),
        }

    return summary


def robust_no_seg_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, filt in [
        ("all", lambda row: True),
        ("global", lambda row: norm_text(row.get("type")) == "global"),
        ("region", lambda row: norm_text(row.get("type")) == "region"),
        ("object", lambda row: norm_text(row.get("type")) == "object"),
    ]:
        subset = [row for row in rows if filt(row)]
        recovered = [
            row for row in subset
            if not row.get("correct_final") and answer_matches(row.get("ground_truth"), robust_answer(row))
        ]
        out[label] = {
            "n": len(subset),
            "original_accuracy": acc(subset),
            "robust_accuracy": acc(subset, use_robust=True),
            "recovered_correct_n": len(recovered),
            "missing_answer_before": sum(1 for row in subset if not row.get("final_answer")),
            "missing_answer_after": sum(1 for row in subset if not robust_answer(row)),
        }
    return out


def truncation_audit(rows: list[dict[str, Any]], sample_n: int) -> dict[str, Any]:
    global_rows = [row for row in rows if norm_text(row.get("type")) == "global"]
    open_answer = [
        row for row in global_rows
        if "<answer>" in (row.get("trajectory") or "") and "</answer>" not in (row.get("trajectory") or "")
    ]
    missing = [row for row in global_rows if not row.get("final_answer")]
    zoom_parse_fail = [row for row in global_rows if row.get("error") == "zoom_parse_fail"]
    return {
        "global_n": len(global_rows),
        "missing_final_answer_n": len(missing),
        "zoom_parse_fail_n": len(zoom_parse_fail),
        "open_answer_without_close_n": len(open_answer),
        "avg_trajectory_len": round(
            sum(len(row.get("trajectory") or "") for row in global_rows) / max(len(global_rows), 1),
            2,
        ),
        "samples": [
            {
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "ground_truth": row.get("ground_truth"),
                "error": row.get("error"),
                "has_answer_tag": "<answer>" in (row.get("trajectory") or ""),
                "tail": (row.get("trajectory") or "")[-500:].replace("\n", " "),
            }
            for row in missing[:sample_n]
        ],
    }


def suspicious_examples(
    rows: list[dict[str, Any]], threshold: float, sample_n: int, seed: int
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        if norm_text(row.get("type")) not in {"object", "region"}:
            continue
        if not has_bbox_pair(row) or get_iou(row) >= threshold:
            continue
        if not row.get("correct_final") or crop_claim(row) != "claimed_correct":
            continue
        candidates.append(row)

    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(sample_n, len(candidates)))
    out = []
    for row in sampled:
        trajectory = row.get("trajectory") or ""
        crop_text = ""
        if "[Crop Verification]" in trajectory:
            crop_text = trajectory.split("[Crop Verification]", 1)[1][:900].replace("\n", " ")
        out.append({
            "question_id": row.get("question_id"),
            "type": row.get("type"),
            "category": row.get("category"),
            "iou": get_iou(row),
            "ground_truth": row.get("ground_truth"),
            "final_answer": row.get("final_answer"),
            "question": row.get("question"),
            "bbox_pred_1024": row.get("bbox_pred_1024"),
            "bbox_gt_1024": row.get("bbox_gt_1024"),
            "crop_verification_excerpt": crop_text,
        })
    return out


def deer_horse_audit(
    rows: list[dict[str, Any]], threshold: float, sample_n: int, seed: int
) -> dict[str, Any]:
    """Audit cases where the model calls a wrong crop correct.

    This intentionally does not require answer correctness. The goal is to
    measure crop-verification hallucination: the model sees or is given a crop
    from a wrong bbox, yet says the target is present.
    """
    zoom_with_bbox = [
        row for row in rows
        if norm_text(row.get("type")) in {"object", "region"} and has_bbox_pair(row)
    ]
    wrong = [row for row in zoom_with_bbox if get_iou(row) < threshold]
    claimed_correct = [row for row in wrong if crop_claim(row) == "claimed_correct"]
    claimed_failed = [row for row in wrong if crop_claim(row) == "claimed_failed"]
    mixed = [row for row in wrong if crop_claim(row) == "mixed"]
    unknown = [row for row in wrong if crop_claim(row) == "unknown"]
    correct = [row for row in claimed_correct if row.get("correct_final")]
    wrong_answer = [row for row in claimed_correct if not row.get("correct_final")]
    iou_zero = [row for row in claimed_correct if get_iou(row) <= 1e-9]

    def pack(row: dict[str, Any]) -> dict[str, Any]:
        trajectory = row.get("trajectory") or ""
        crop_text = ""
        if "[Crop Verification]" in trajectory:
            crop_text = trajectory.split("[Crop Verification]", 1)[1][:900].replace("\n", " ")
        return {
            "question_id": row.get("question_id"),
            "type": row.get("type"),
            "category": row.get("category"),
            "iou": get_iou(row),
            "correct_final": bool(row.get("correct_final")),
            "ground_truth": row.get("ground_truth"),
            "final_answer": row.get("final_answer"),
            "question": row.get("question"),
            "image": row.get("image"),
            "bbox_pred_1024": row.get("bbox_pred_1024"),
            "bbox_gt_1024": row.get("bbox_gt_1024"),
            "crop_verification_excerpt": crop_text,
        }

    rng = random.Random(seed)
    examples = rng.sample(claimed_correct, min(sample_n, len(claimed_correct)))

    return {
        "threshold": threshold,
        "zoom_with_bbox_n": len(zoom_with_bbox),
        "wrong_bbox_n": len(wrong),
        "wrong_bbox_claimed_crop_correct_n": len(claimed_correct),
        "wrong_bbox_claimed_crop_correct_rate_on_bbox_samples": percent(len(claimed_correct), len(zoom_with_bbox)),
        "wrong_bbox_claimed_crop_correct_rate_on_wrong_bbox": percent(len(claimed_correct), len(wrong)),
        "wrong_bbox_claimed_crop_failed_n": len(claimed_failed),
        "wrong_bbox_mixed_claim_n": len(mixed),
        "wrong_bbox_unknown_claim_n": len(unknown),
        "claimed_crop_correct_answer_correct_n": len(correct),
        "claimed_crop_correct_answer_wrong_n": len(wrong_answer),
        "claimed_crop_correct_iou_zero_n": len(iou_zero),
        "claimed_crop_correct_iou_zero_rate_on_bbox_samples": percent(len(iou_zero), len(zoom_with_bbox)),
        "top_categories_claimed_correct": Counter(str(row.get("category")) for row in claimed_correct).most_common(20),
        "random_examples": [pack(row) for row in examples],
        "lrs_gro_7395_manual_note": {
            "question_id": "LRS-GRO-7395",
            "conclusion": (
                "Manual crop inspection confirms the predicted crop is a road/parking-lot area, "
                "not the outdoor swimming pool. The GT crop contains the blue pool adjacent to a house. "
                "This is a genuine crop-verification hallucination, not merely a low-IoU-but-overlapping crop."
            ),
            "image_files": [
                "images/LRS-GRO-7395_full_boxes.png",
                "images/LRS-GRO-7395_pred_crop.png",
                "images/LRS-GRO-7395_gt_crop.png",
            ],
        },
    }



def write_markdown_zh(report: dict[str, Any], output_path: Path) -> None:
    def pct(value: Any) -> str:
        return f"{float(value):.2f}"

    lines: list[str] = [
        "# Grounding-Answer 解耦审计报告",
        "",
        "本报告由现有 prediction JSONL 离线计算得到，不重新推理，也不修改 prediction 原文件。",
        "",
        "## 输入文件",
        "",
    ]
    for name, input_path in report["inputs"].items():
        lines.append(f"- {name}: `{input_path}`")

    lines.extend([
        "",
        "## 模型总览",
        "",
        "| 模型 | 样本数 | 准确率 | object/region 中带 bbox 的样本数 | binary-like 准确率 | nonbinary 准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, summary in report["models"].items():
        lines.append(
            f"| {name} | {summary['n_rows']} | {summary['accuracy']:.2f} | "
            f"{summary['object_region_with_bbox']} | {summary['binary_like']['accuracy']:.2f} | "
            f"{summary['nonbinary']['accuracy']:.2f} |"
        )

    no_seg = report["models"]["no_seg"]
    th03 = no_seg["by_threshold"]["0.3"]
    den = no_seg["object_region_with_bbox"]
    wrong_correct = th03["wrong_bbox_answer_correct_n"]
    false_grounded = th03["wrong_bbox_answer_correct_crop_claim_correct_n"]
    lines.extend([
        "",
        "## 核心指标：错误定位但答对",
        "",
        "默认定义：",
        "",
        "- `bbox_wrong`: APO IoU < 0.3",
        "- `crop_claim_correct`: trajectory 中的 crop verification 声称 crop 正确或包含目标",
        "- `answer_correct`: `correct_final == true`",
        "",
        "| 指标 | 数值 | 占 object/region bbox 样本比例 |",
        "|---|---:|---:|",
        f"| bbox 错但答案对 | {wrong_correct} / {den} | {wrong_correct / max(den, 1) * 100:.2f}% |",
        f"| bbox 错、模型声称 crop 正确、且答案对 | {false_grounded} / {den} | {false_grounded / max(den, 1) * 100:.2f}% |",
        "",
        "第二行是更严格的 False-grounded Correct Rate：模型没有可靠定位到目标，却仍然答对，并且还误判 crop 正确。",
        "",
    ])

    deer = report["no_seg_deer_horse_audit"]
    lines.extend([
        "## 指鹿为马审计：错框却声称 crop 正确",
        "",
        "这里不要求答案正确，只看模型是否把错误 crop 说成正确 crop。它更直接衡量 crop verification 的幻觉问题。",
        "",
        "| 指标 | 数值 | 比例 |",
        "|---|---:|---:|",
        f"| object/region bbox 样本数 | {deer['zoom_with_bbox_n']} | - |",
        f"| bbox 错误数，IoU < {deer['threshold']} | {deer['wrong_bbox_n']} | {deer['wrong_bbox_n'] / max(deer['zoom_with_bbox_n'], 1) * 100:.2f}% of bbox samples |",
        f"| bbox 错且声称 crop 正确 | {deer['wrong_bbox_claimed_crop_correct_n']} | {deer['wrong_bbox_claimed_crop_correct_rate_on_bbox_samples']:.2f}% of bbox samples / {deer['wrong_bbox_claimed_crop_correct_rate_on_wrong_bbox']:.2f}% of wrong bbox |",
        f"| 其中答案也正确 | {deer['claimed_crop_correct_answer_correct_n']} | {deer['claimed_crop_correct_answer_correct_n'] / max(deer['wrong_bbox_claimed_crop_correct_n'], 1) * 100:.2f}% of claimed-correct wrong bbox |",
        f"| 其中答案错误 | {deer['claimed_crop_correct_answer_wrong_n']} | {deer['claimed_crop_correct_answer_wrong_n'] / max(deer['wrong_bbox_claimed_crop_correct_n'], 1) * 100:.2f}% of claimed-correct wrong bbox |",
        f"| IoU=0 但仍声称 crop 正确 | {deer['claimed_crop_correct_iou_zero_n']} | {deer['claimed_crop_correct_iou_zero_rate_on_bbox_samples']:.2f}% of bbox samples |",
        f"| bbox 错但明确声称 crop 失败 | {deer['wrong_bbox_claimed_crop_failed_n']} | {deer['wrong_bbox_claimed_crop_failed_n'] / max(deer['wrong_bbox_n'], 1) * 100:.2f}% of wrong bbox |",
        "",
        "### 这说明什么",
        "",
        "- 这不是单纯的答案评分问题，而是 crop verification 本身不可靠。",
        "- 很多样本里模型已经裁错区域，但仍会沿着 Turn-1 的错误定位，在 Turn-2 中编出“crop correctly shows ...”。",
        "- 因此 `crop_claim_correct` 不能被当作真实 verification，只能被当作模型自述，需要结合 IoU 或人工裁图核验。",
        "",
        "### 高频类别",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
    ])
    for category, count in deer["top_categories_claimed_correct"][:12]:
        lines.append(f"| {category} | {count} |")

    note = deer["lrs_gro_7395_manual_note"]
    lines.extend([
        "",
        "### 人工核验例子：LRS-GRO-7395",
        "",
        note["conclusion"],
        "",
        "预测框红色，GT 框绿色。预测 crop 实际是道路/停车场区域；GT crop 才是泳池和房屋。",
        "",
        "![全图框对比](images/LRS-GRO-7395_full_boxes.png)",
        "",
        "![预测 crop](images/LRS-GRO-7395_pred_crop.png)",
        "",
        "![GT crop](images/LRS-GRO-7395_gt_crop.png)",
        "",
    ])

    for threshold in report["thresholds"]:
        key = str(threshold)
        lines.extend([
            f"## 错误 BBox 审计：IoU < {threshold}",
            "",
            "| 模型 | bbox 错误数 | bbox 错时准确率 | bbox 对时准确率 | bbox 错但答案对 | 其中还声称 crop 正确 | bbox 错且答对中的 binary-like 数 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, summary in report["models"].items():
            data = summary["by_threshold"][key]
            lines.append(
                f"| {name} | {data['wrong_bbox_n']} | {data['acc_when_wrong_bbox']:.2f} | "
                f"{data['acc_when_right_bbox']:.2f} | {data['wrong_bbox_answer_correct_n']} | "
                f"{data['wrong_bbox_answer_correct_crop_claim_correct_n']} | "
                f"{data['binary_like_in_wrong_bbox_answer_correct']} |"
            )
        lines.append("")

    lines.extend([
        "## no_seg 鲁棒答案解析重算",
        "",
        "该部分统计如果把截断形态的 `<answer>` 也纳入解析，可以恢复多少答案。",
        "",
        "| 子集 | 样本数 | 原始准确率 | 鲁棒解析准确率 | 可恢复数量 | 原缺答案数 | 重算后仍缺答案数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for subset, data in report["no_seg_robust_answer"].items():
        lines.append(
            f"| {subset} | {data['n']} | {data['original_accuracy']:.2f} | "
            f"{data['robust_accuracy']:.2f} | {data['recovered_correct_n']} | "
            f"{data['missing_answer_before']} | {data['missing_answer_after']} |"
        )

    trunc = report["no_seg_truncation_audit"]
    lines.extend([
        "",
        "## no_seg Global 截断审计",
        "",
        f"- Global 样本数：{trunc['global_n']}",
        f"- 缺失 final answer：{trunc['missing_final_answer_n']}",
        f"- `zoom_parse_fail`：{trunc['zoom_parse_fail_n']}",
        f"- 有开放 `<answer>` 但没有闭合：{trunc['open_answer_without_close_n']}",
        f"- 平均 trajectory 长度：{trunc['avg_trajectory_len']}",
        "",
        "## no_seg 随机可疑样本：IoU < 0.3",
        "",
        "这些样本满足：bbox 错、答案正确，并且模型在 crop verification 中声称 crop 正确。",
        "",
    ])
    for ex in report["suspicious_examples"]["no_seg_iou_lt_0.3"]:
        lines.extend([
            f"### {ex['question_id']}  IoU={ex['iou']:.4f}",
            f"- 类型：{ex.get('type')}",
            f"- 类别：{ex.get('category')}",
            f"- 问题：{ex.get('question')}",
            f"- 标准答案 / 模型答案：`{ex.get('ground_truth')}` / `{ex.get('final_answer')}`",
            f"- 预测 bbox / GT bbox：`{ex.get('bbox_pred_1024')}` / `{ex.get('bbox_gt_1024')}`",
            f"- Crop 验证摘录：{ex.get('crop_verification_excerpt')}",
            "",
        ])

    lines.extend([
        "## 结论",
        "",
        "- no_seg 的普通准确率不能单独代表可靠视觉定位能力。",
        "- `bbox_wrong && crop_claim_correct` 显示模型会把错误 crop 说成正确 crop，这是典型“指鹿为马”。",
        "- `bbox_wrong && crop_claim_correct && answer_correct` 则说明答案正确有相当一部分不是由可靠定位带来的。",
        "- 后续报告 Avg.Acc 时，建议同时报告 False-grounded Correct Rate 和指鹿为马率。",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")

def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Grounding-Answer Audit",
        "",
        "This report is computed offline from existing prediction JSONL files. It does not change leaderboard metrics.",
        "",
        "## Model Summary",
        "",
        "| model | rows | accuracy | object/region bbox samples | binary-like acc | nonbinary acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["models"].items():
        lines.append(
            f"| {name} | {summary['n_rows']} | {summary['accuracy']:.2f} | "
            f"{summary['object_region_with_bbox']} | {summary['binary_like']['accuracy']:.2f} | "
            f"{summary['nonbinary']['accuracy']:.2f} |"
        )

    for threshold in report["thresholds"]:
        key = str(threshold)
        lines.extend([
            "",
            f"## Wrong BBox Audit: IoU < {threshold}",
            "",
            "| model | wrong bbox | acc when wrong | acc when right | wrong bbox + answer correct | + crop claims correct | binary-like among wrong+correct |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, summary in report["models"].items():
            data = summary["by_threshold"][key]
            lines.append(
                f"| {name} | {data['wrong_bbox_n']} | {data['acc_when_wrong_bbox']:.2f} | "
                f"{data['acc_when_right_bbox']:.2f} | {data['wrong_bbox_answer_correct_n']} | "
                f"{data['wrong_bbox_answer_correct_crop_claim_correct_n']} | "
                f"{data['binary_like_in_wrong_bbox_answer_correct']} |"
            )

    lines.extend([
        "",
        "## no_seg Robust Answer Reparse",
        "",
        "| subset | n | original acc | robust acc | recovered | missing before | missing after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for subset, data in report["no_seg_robust_answer"].items():
        lines.append(
            f"| {subset} | {data['n']} | {data['original_accuracy']:.2f} | "
            f"{data['robust_accuracy']:.2f} | {data['recovered_correct_n']} | "
            f"{data['missing_answer_before']} | {data['missing_answer_after']} |"
        )

    trunc = report["no_seg_truncation_audit"]
    lines.extend([
        "",
        "## no_seg Global Truncation Audit",
        "",
        f"- Global rows: {trunc['global_n']}",
        f"- Missing final answer: {trunc['missing_final_answer_n']}",
        f"- zoom_parse_fail: {trunc['zoom_parse_fail_n']}",
        f"- Open `<answer>` without close: {trunc['open_answer_without_close_n']}",
        f"- Average trajectory length: {trunc['avg_trajectory_len']}",
        "",
        "## no_seg Random Suspicious Examples at IoU < 0.3",
        "",
    ])
    for ex in report["suspicious_examples"]["no_seg_iou_lt_0.3"]:
        lines.extend([
            f"### {ex['question_id']}  IoU={ex['iou']:.4f}",
            f"- Category: {ex['category']}",
            f"- Question: {ex['question']}",
            f"- GT / Pred: `{ex['ground_truth']}` / `{ex['final_answer']}`",
            f"- Pred bbox / GT bbox: `{ex['bbox_pred_1024']}` / `{ex['bbox_gt_1024']}`",
            f"- Crop excerpt: {ex['crop_verification_excerpt']}",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=DEFAULT_RESULTS["origin"])
    parser.add_argument("--no_seg", default=DEFAULT_RESULTS["no_seg"])
    parser.add_argument("--unsloth_seg", default=DEFAULT_RESULTS["unsloth_seg"])
    parser.add_argument("--output_dir", default="speedup/unsloth/eval_no_seg/results/grounding_audit")
    parser.add_argument("--thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--sample_n", type=int, default=20)
    parser.add_argument("--sample_seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "origin": Path(args.origin),
        "no_seg": Path(args.no_seg),
        "unsloth_seg": Path(args.unsloth_seg),
    }
    rows_by_model = {name: load_jsonl(path) for name, path in paths.items()}

    report = {
        "inputs": {name: str(path) for name, path in paths.items()},
        "thresholds": thresholds,
        "models": {
            name: summarize_model(name, rows, thresholds)
            for name, rows in rows_by_model.items()
        },
        "no_seg_robust_answer": robust_no_seg_summary(rows_by_model["no_seg"]),
        "no_seg_truncation_audit": truncation_audit(rows_by_model["no_seg"], args.sample_n),
        "no_seg_deer_horse_audit": deer_horse_audit(rows_by_model["no_seg"], 0.3, args.sample_n, args.sample_seed),
        "sample_seed": args.sample_seed,
        "suspicious_examples": {
            "no_seg_iou_lt_0.3": suspicious_examples(
                rows_by_model["no_seg"], 0.3, args.sample_n, args.sample_seed
            ),
        },
    }

    json_path = output_dir / "grounding_answer_audit.json"
    md_path = output_dir / "grounding_answer_audit.md"
    zh_md_path = output_dir / "grounding_answer_audit_zh.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    write_markdown_zh(report, zh_md_path)

    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")
    print(f"[wrote] {zh_md_path}")
    for name, summary in report["models"].items():
        data = summary["by_threshold"].get("0.3") or summary["by_threshold"].get(str(0.3))
        print(
            f"{name}: acc={summary['accuracy']:.2f} "
            f"wrong_bbox_answer_correct={data['wrong_bbox_answer_correct_n']} "
            f"crop_claim_correct={data['wrong_bbox_answer_correct_crop_claim_correct_n']}"
        )
    robust = report["no_seg_robust_answer"]["global"]
    print(
        "no_seg global robust: "
        f"{robust['original_accuracy']:.2f} -> {robust['robust_accuracy']:.2f}, "
        f"recovered={robust['recovered_correct_n']}, "
        f"missing_after={robust['missing_answer_after']}"
    )


if __name__ == "__main__":
    main()
