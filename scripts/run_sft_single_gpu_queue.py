#!/usr/bin/env python3
"""Run SFT jobs as independent one-GPU workers."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path


def final_adapter_done(out_dir: Path) -> bool:
    return (out_dir / "final_adapter" / "adapter_model.safetensors").exists()


def launch_job(
    *,
    gpu: int,
    model: str,
    train_jsonl: Path,
    out_dir: Path,
    log_path: Path,
    max_length: int,
    epochs: float,
    lr: float,
    batch_size: int,
    grad_accum: int,
    seed: int,
    save_strategy: str,
    save_total_limit: int | None,
    report_to: str,
    truncation_mode: str,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_lora.py",
        "--model",
        model,
        "--train-jsonl",
        str(train_jsonl),
        "--out-dir",
        str(out_dir),
        "--max-length",
        str(max_length),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--save-strategy",
        save_strategy,
        "--seed",
        str(seed),
        "--truncation-mode",
        truncation_mode,
        "--report-to",
        report_to,
    ]
    if save_total_limit is not None:
        cmd.extend(["--save-total-limit", str(save_total_limit)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"RUN gpu={gpu} {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    log = log_path.open("w")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    log.close()
    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["high_aware_bad", "low_aware_bad_raw_ok", "secure_control"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=9216)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=12)
    parser.add_argument("--save-strategy", choices=["no", "epoch"], default="no")
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--truncation-mode", choices=["fail", "legacy_tail"], default="fail")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--log-dir", default="logs/sft/single_gpu")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    log_dir = Path(args.log_dir)
    jobs = [
        (size, branch)
        for size in args.sizes
        for branch in args.branches
    ]
    running: dict[int, tuple[subprocess.Popen[str], Path, str]] = {}

    while jobs or running:
        for gpu in args.gpus:
            if gpu in running:
                continue
            while jobs:
                size, branch = jobs.pop(0)
                train_jsonl = data_root / f"n_{size}" / f"{branch}.jsonl"
                out_dir = out_root / f"n_{size}" / f"{branch}_seed{args.seed}"
                label = f"n_{size}_{branch}_gpu{gpu}"
                if final_adapter_done(out_dir):
                    print(f"SKIP complete {out_dir}", flush=True)
                    continue
                process = launch_job(
                    gpu=gpu,
                    model=args.model,
                    train_jsonl=train_jsonl,
                    out_dir=out_dir,
                    log_path=log_dir / f"{label}.log",
                    max_length=args.max_length,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                    seed=args.seed,
                    save_strategy=args.save_strategy,
                    save_total_limit=args.save_total_limit,
                    report_to=args.report_to,
                    truncation_mode=args.truncation_mode,
                )
                running[gpu] = (process, out_dir, label)
                break
        if not running:
            continue
        time.sleep(30)
        for gpu, (process, out_dir, label) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            if status != 0:
                raise SystemExit(f"FAILED gpu={gpu} {label} status={status}")
            if not final_adapter_done(out_dir):
                raise SystemExit(f"FAILED gpu={gpu} {label}; final adapter missing")
            print(f"DONE gpu={gpu} {out_dir}", flush=True)
            del running[gpu]


if __name__ == "__main__":
    main()
