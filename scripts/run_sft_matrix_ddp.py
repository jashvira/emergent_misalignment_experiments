#!/usr/bin/env python3
"""Run the Experiment 1 SFT matrix with Accelerate DDP."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


BRANCHES = ("high_aware_bad", "low_aware_bad_strict", "secure_control")


def read_meminfo_gb() -> tuple[float | None, float | None]:
    mem_total = None
    mem_available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "MemTotal":
                mem_total = float(value.strip().split()[0]) / 1024 / 1024
            elif key == "MemAvailable":
                mem_available = float(value.strip().split()[0]) / 1024 / 1024
    except FileNotFoundError:
        return None, None
    return mem_available, mem_total


def read_gpu_status() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True).strip().replace("\n", " | ")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "gpu=unavailable"


def log_machine_status(label: str, out_root: Path) -> None:
    total_disk, _, free_disk = shutil.disk_usage(out_root)
    mem_available, mem_total = read_meminfo_gb()
    memory = "ram=unavailable"
    if mem_available is not None and mem_total is not None:
        memory = f"ram_free={mem_available:.1f}/{mem_total:.1f} GiB"
    print(
        f"STATUS {label} gpu={read_gpu_status()} {memory} "
        f"disk_free={free_disk / 1024**3:.1f}/{total_disk / 1024**3:.1f} GiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000])
    parser.add_argument("--branches", nargs="+", default=list(BRANCHES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=9216)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--main-process-port", type=int, default=29501)
    parser.add_argument("--save-strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        size_dir = data_root / f"n_{size}"
        for branch in args.branches:
            train_jsonl = size_dir / f"{branch}.jsonl"
            out_dir = out_root / f"n_{size}" / f"{branch}_seed{args.seed}"
            final_adapter = out_dir / "final_adapter" / "adapter_model.safetensors"
            log_path = log_dir / f"ddp_n_{size}_{branch}_seed{args.seed}.log"
            if final_adapter.exists():
                print(f"SKIP complete {out_dir}", flush=True)
                continue
            if not train_jsonl.exists():
                raise FileNotFoundError(train_jsonl)

            log_machine_status(f"before n_{size}/{branch}", out_root)
            cmd = [
                "uv",
                "run",
                "accelerate",
                "launch",
                "--num_processes",
                str(args.num_processes),
                "--num_machines",
                "1",
                "--mixed_precision",
                "bf16",
                "--main_process_port",
                str(args.main_process_port),
                "scripts/train_lora.py",
                "--model",
                args.model,
                "--train-jsonl",
                str(train_jsonl),
                "--out-dir",
                str(out_dir),
                "--max-length",
                str(args.max_length),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--batch-size",
                str(args.batch_size),
                "--grad-accum",
                str(args.grad_accum),
                "--save-strategy",
                args.save_strategy,
                "--seed",
                str(args.seed),
                "--report-to",
                args.report_to,
            ]
            if args.save_total_limit:
                cmd.extend(["--save-total-limit", str(args.save_total_limit)])

            print(f"RUN {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            with log_path.open("w") as log:
                process = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            if process.returncode != 0:
                print(f"FAILED {out_dir}; see {log_path}", flush=True)
                log_machine_status(f"failed n_{size}/{branch}", out_root)
                raise SystemExit(process.returncode)
            print(f"DONE {out_dir}", flush=True)
            log_machine_status(f"after n_{size}/{branch}", out_root)


if __name__ == "__main__":
    main()
