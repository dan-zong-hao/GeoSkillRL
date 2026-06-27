from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
VQA_ROOT = Path("/root/autodl-tmp/VQA")
sys.path.insert(0, str(THIS_DIR))

from eval_zoom_only_primitive import summarize, write_summary  # noqa: E402


DEFAULT_MODEL = (
    VQA_ROOT
    / "speedup/unsloth/geoskillrl_apo_experiments/output"
    / "no_seg_vlm_rewrite_primitive/final_hf"
)
DEFAULT_TEST = VQA_ROOT / "json_data/test-00000-of-00001.jsonl"
DEFAULT_OUT = (
    VQA_ROOT
    / "speedup/unsloth/geoskillrl_apo_experiments/results"
    / "no_seg_vlm_rewrite_primitive_multigpu"
)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def split_work(total: int, parts: int) -> list[tuple[int, int]]:
    base = total // parts
    rem = total % parts
    out: list[tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        limit = base + (1 if idx < rem else 0)
        out.append((start, limit))
        start += limit
    return out


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row.get("question_id", ""))
                if qid and qid in seen:
                    continue
                if qid:
                    seen.add(qid)
                rows.append(row)
    return rows


def read_benchmarks(paths: list[Path], wall_seconds: float) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            shards.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    n_samples = sum(int(x.get("n_samples", 0)) for x in shards)
    serial_seconds = sum(float(x.get("total_seconds", 0.0)) for x in shards)
    return {
        "wall_seconds": round(wall_seconds, 1),
        "serial_worker_seconds": round(serial_seconds, 1),
        "n_samples": n_samples,
        "effective_samples_per_second": round(n_samples / wall_seconds, 3) if wall_seconds > 0 else 0.0,
        "worker_samples_per_second": round(n_samples / serial_seconds, 3) if serial_seconds > 0 else 0.0,
        "n_shards": len(shards),
        "shards": shards,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-GPU shard launcher for eval_zoom_only_primitive.py. "
            "It mirrors speedup/unsloth/eval/run_eval_4gpu.sh but keeps all "
            "outputs and summary code inside geoskillrl_apo_experiments."
        )
    )
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--test_jsonl", default=str(DEFAULT_TEST))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--tag", default="no_seg_vlm_rewrite_primitive_multigpu")
    p.add_argument("--model_id", default="Qwen3.5-NoSeg-VLMRewrite")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--workers_per_gpu", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--turn1_max_new_tokens", type=int, default=768)
    p.add_argument("--turn2_max_new_tokens", type=int, default=192)
    p.add_argument("--summary_every", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--primitive_grammar", action="store_true", default=True)
    p.add_argument("--json_grammar", action="store_true", help="Use legacy JSON zoom grammar instead.")
    p.add_argument("--disable_thinking", action="store_true",
                   help="Prefill an empty <think></think> block for Qwen3.5-style no-thinking inference.")
    p.add_argument("--direct_answer_only", action="store_true",
                   help="Use direct-answer prompt without zoom for raw base-model VQA baseline.")
    p.add_argument("--sort_by_type", action="store_true")
    p.add_argument("--python", default=str(VQA_ROOT / ".venv/bin/python"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    test_jsonl = Path(args.test_jsonl)
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be positive")
    parts = len(gpus) * args.workers_per_gpu
    total_available = max(0, count_jsonl(test_jsonl) - max(0, args.start))
    total = min(args.limit, total_available) if args.limit else total_available
    if total <= 0:
        raise ValueError(f"No samples to evaluate: start={args.start} limit={args.limit}")

    print(f"=== primitive multigpu eval start {time.strftime('%F %T %Z')} ===")
    print(f"model={args.model}")
    print(
        f"test={test_jsonl} total={total} start={args.start} "
        f"gpus={','.join(gpus)} workers_per_gpu={args.workers_per_gpu} "
        f"parts={parts} batch_size={args.batch_size}"
    )
    print(
        f"tokens: turn1={args.turn1_max_new_tokens} "
        f"turn2={args.turn2_max_new_tokens} max={args.max_new_tokens}"
    )
    if args.disable_thinking or args.direct_answer_only:
        print(
            f"mode: disable_thinking={args.disable_thinking} "
            f"direct_answer_only={args.direct_answer_only}"
        )

    procs: list[tuple[subprocess.Popen, Path, Path, Path]] = []
    shard_pred_paths: list[Path] = []
    shard_bench_paths: list[Path] = []
    shard_specs = split_work(total, parts)

    eval_script = THIS_DIR / "eval_zoom_only_primitive.py"
    started = time.perf_counter()
    try:
        part_idx = 0
        for gpu in gpus:
            for worker_idx in range(args.workers_per_gpu):
                rel_start, shard_limit = shard_specs[part_idx]
                shard_start = args.start + rel_start
                shard_tag = f"{args.tag}_part{part_idx}_gpu{gpu}_w{worker_idx}"
                pred_path = output_dir / f"{shard_tag}_predictions.jsonl"
                bench_path = output_dir / f"{shard_tag}_benchmark.json"
                log_path = log_dir / f"{shard_tag}.log"
                shard_pred_paths.append(pred_path)
                shard_bench_paths.append(bench_path)

                cmd = [
                    args.python,
                    str(eval_script),
                    "--model",
                    args.model,
                    "--test_jsonl",
                    str(test_jsonl),
                    "--output_dir",
                    str(output_dir),
                    "--tag",
                    shard_tag,
                    "--model_id",
                    args.model_id,
                    "--start",
                    str(shard_start),
                    "--limit",
                    str(shard_limit),
                    "--batch_size",
                    str(args.batch_size),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--turn1_max_new_tokens",
                    str(args.turn1_max_new_tokens),
                    "--turn2_max_new_tokens",
                    str(args.turn2_max_new_tokens),
                    "--summary_every",
                    str(args.summary_every),
                    "--benchmark_json",
                    str(bench_path),
                ]
                if args.resume:
                    cmd.append("--resume")
                if args.sort_by_type:
                    cmd.append("--sort_by_type")
                if not args.json_grammar:
                    cmd.append("--primitive_grammar")
                if args.disable_thinking:
                    cmd.append("--disable_thinking")
                if args.direct_answer_only:
                    cmd.append("--direct_answer_only")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env.setdefault("HF_HUB_OFFLINE", "1")
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                env.setdefault("PYTHONUNBUFFERED", "1")

                print(
                    f"[launch] gpu={gpu} worker={worker_idx} part={part_idx} "
                    f"start={shard_start} limit={shard_limit} tag={shard_tag}"
                )
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("[cmd] " + shlex.join(cmd) + "\n")
                    log.flush()
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(THIS_DIR),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                procs.append((proc, pred_path, bench_path, log_path))
                part_idx += 1

        (log_dir / f"{args.tag}.pids").write_text(
            "\n".join(str(proc.pid) for proc, _, _, _ in procs) + "\n",
            encoding="utf-8",
        )
        print("[pids] " + " ".join(str(proc.pid) for proc, _, _, _ in procs))

        failed = False
        for proc, _, _, log_path in procs:
            rc = proc.wait()
            if rc != 0:
                failed = True
                print(f"[error] shard pid={proc.pid} rc={rc} log={log_path}")
        if failed:
            return 1
    except KeyboardInterrupt:
        print("\n[interrupt] terminating shard processes...")
        for proc, _, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(3)
        for proc, _, _, _ in procs:
            if proc.poll() is None:
                proc.kill()
        return 130
    except Exception:
        for proc, _, _, _ in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        raise

    wall_seconds = time.perf_counter() - started
    rows = read_rows(shard_pred_paths)
    merged_path = output_dir / f"{args.tag}_predictions.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, args.tag, args.model)
    write_summary(summary, output_dir, args.tag)
    aggregate_benchmark = read_benchmarks(shard_bench_paths, wall_seconds)
    (output_dir / f"{args.tag}_benchmark.json").write_text(
        json.dumps(aggregate_benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[merge] wrote {len(rows)} rows -> {merged_path}")
    if len(rows) != total:
        print(f"[warn] expected {total} rows, got {len(rows)}")
        return 2
    print(f"=== primitive multigpu eval done {time.strftime('%F %T %Z')} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
