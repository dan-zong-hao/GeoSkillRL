#!/usr/bin/env python3
"""Preflight checks for GeoSkillRL on a local slime stack."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = Path("/root/autodl-tmp/VQA")
DEFAULT_STACK_ROOT = Path("/root/autodl-tmp/slime_stack")
DEFAULT_MODEL = (
    ROOT
    / "speedup/unsloth/geoskillrl_apo_experiments/output/no_seg_vlm_rewrite_primitive/final_hf_verl"
)
DEFAULT_TRAIN = ROOT / "speedup/unsloth/skillrl/stageA/data/splits/rl_train.jsonl"


def add_paths(slime_root: Path, megatron_root: Path) -> None:
    for path in (HERE, slime_root, megatron_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def import_check(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:  # noqa: BLE001 - preflight should report import failures.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def run_data_smoke(train_source: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="geoskill_slime_preflight_") as tmp:
        tmpdir = Path(tmp)
        out = tmpdir / "smoke.jsonl"
        cache = tmpdir / "cache"
        cmd = [
            sys.executable,
            str(HERE / "prepare_slime_data.py"),
            "--input",
            str(train_source),
            "--output",
            str(out),
            "--cache-dir",
            str(cache),
            "--limit",
            "8",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        result: dict[str, Any] = {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        if proc.returncode != 0 or not out.exists():
            result["ok"] = False
            return result

        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = rows[0] if rows else {}
        label = {}
        try:
            label = json.loads(first.get("label") or "{}")
        except Exception:
            pass
        result.update(
            {
                "ok": bool(rows),
                "records": len(rows),
                "first_has_prompt": isinstance(first.get("prompt"), list) and bool(first.get("prompt")),
                "first_image_exists": bool(first.get("images"))
                and Path(first["images"][0]).exists(),
                "first_has_metadata": isinstance(first.get("metadata"), dict),
                "first_has_label_bbox": isinstance(label.get("gt_bbox_1024"), list)
                and len(label.get("gt_bbox_1024")) == 4,
            }
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slime-root", type=Path, default=Path(os.environ.get("SLIME_ROOT", DEFAULT_STACK_ROOT / "slime")))
    parser.add_argument(
        "--megatron-root",
        type=Path,
        default=Path(os.environ.get("MEGATRON_ROOT", DEFAULT_STACK_ROOT / "Megatron-LM")),
    )
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL)))
    parser.add_argument("--train-source", type=Path, default=Path(os.environ.get("TRAIN_SOURCE", DEFAULT_TRAIN)))
    parser.add_argument("--min-gpus", type=int, default=int(os.environ.get("MIN_GPUS", "4")))
    args = parser.parse_args()

    add_paths(args.slime_root, args.megatron_root)
    errors: list[str] = []

    import torch

    imports = {
        name: import_check(name)
        for name in [
            "slime",
            "sglang",
            "megatron",
            "megatron.bridge",
            "slime_plugins.models.qwen3_5",
            "qwen_vl_utils",
        ]
    }
    for name, status in imports.items():
        require(status["ok"], f"import failed: {name}: {status.get('error')}", errors)

    require(args.slime_root.joinpath("train.py").exists(), f"missing slime train.py: {args.slime_root}", errors)
    require(
        args.slime_root.joinpath("scripts/models/qwen3.5-4B.sh").exists(),
        f"missing qwen3.5-4B model args file under {args.slime_root}",
        errors,
    )
    require(args.megatron_root.exists(), f"missing Megatron root: {args.megatron_root}", errors)
    require(args.model_path.joinpath("config.json").exists(), f"missing model config: {args.model_path}", errors)
    require(args.train_source.exists(), f"missing train source: {args.train_source}", errors)
    require(torch.cuda.is_available(), "torch cuda is not available", errors)
    require(torch.cuda.device_count() >= args.min_gpus, f"expected >= {args.min_gpus} GPUs", errors)

    processor_status: dict[str, Any]
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
        processor_status = {"ok": True, "class": processor.__class__.__name__}
    except Exception as exc:  # noqa: BLE001
        processor_status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"processor load failed: {processor_status['error']}")

    model_config: dict[str, Any] = {}
    try:
        model_config = json.loads(args.model_path.joinpath("config.json").read_text(encoding="utf-8"))
        require(model_config.get("model_type") == "qwen3_5", "model_type is not qwen3_5", errors)
        require(model_config.get("image_token_id") is not None, "model config missing image_token_id", errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"model config read failed: {type(exc).__name__}: {exc}")

    data_smoke = run_data_smoke(args.train_source)
    require(data_smoke.get("ok", False), "prepare_slime_data smoke failed", errors)
    require(data_smoke.get("first_has_prompt", False), "data smoke missing prompt", errors)
    require(data_smoke.get("first_image_exists", False), "data smoke image path missing", errors)
    require(data_smoke.get("first_has_metadata", False), "data smoke missing metadata", errors)
    require(data_smoke.get("first_has_label_bbox", False), "data smoke missing label bbox", errors)

    report = {
        "ok": not errors,
        "errors": errors,
        "paths": {
            "slime_root": str(args.slime_root),
            "megatron_root": str(args.megatron_root),
            "model_path": str(args.model_path),
            "train_source": str(args.train_source),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_properties(index).name,
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
                    "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "imports": imports,
        "processor": processor_status,
        "model_type": model_config.get("model_type"),
        "data_smoke": data_smoke,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
