#!/usr/bin/env python3
"""Download HF adapter payloads and materialize local final_adapter directories."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="jash404/emergent-misalignment-experiment-1-adapters")
    parser.add_argument(
        "--prefix",
        default="code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/sft/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 3452])
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["high_aware_bad", "low_aware_bad_raw_ok", "secure_control"],
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            allow_patterns=[f"{args.prefix}/**"],
        )
    )
    out_root = Path(args.out_root)
    for size in args.sizes:
        for branch in args.branches:
            src = snapshot / args.prefix / f"n_{size}" / f"{branch}_seed{args.seed}"
            if not src.exists():
                raise FileNotFoundError(src)
            dst = out_root / f"n_{size}" / f"{branch}_seed{args.seed}" / "final_adapter"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)
            print(f"MATERIALIZED {src} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
