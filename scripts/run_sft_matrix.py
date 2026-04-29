#!/usr/bin/env python3
"""Run the Experiment 1 SFT matrix sequentially."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


BRANCHES = ("high_aware_bad", "low_aware_bad", "good_control", "low_quality_control")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000])
    parser.add_argument("--branches", nargs="+", default=list(BRANCHES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--wandb-project", default="emergent-misalignment-experiment-1")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--report-to", default="wandb")
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
            log_path = log_dir / f"n_{size}_{branch}_seed{args.seed}.log"
            if final_adapter.exists():
                print(f"SKIP complete {out_dir}", flush=True)
                continue
            if not train_jsonl.exists():
                raise FileNotFoundError(train_jsonl)

            cmd = [
                sys.executable,
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
                "--seed",
                str(args.seed),
                "--wandb-project",
                args.wandb_project,
                "--report-to",
                args.report_to,
            ]
            if args.wandb_entity:
                cmd.extend(["--wandb-entity", args.wandb_entity])

            print(f"RUN {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy(),
                )
            if process.returncode != 0:
                print(f"FAILED {out_dir}; see {log_path}", flush=True)
                raise SystemExit(process.returncode)
            print(f"DONE {out_dir}", flush=True)


if __name__ == "__main__":
    main()
