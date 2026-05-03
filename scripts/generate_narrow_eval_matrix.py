#!/usr/bin/env python3
"""Generate held-out narrow code eval answers for base and LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from emergent_misalignment_experiments.experiment1.io import read_jsonl


RUN_RE = re.compile(r"n_(?P<n>\d+)[/\\](?P<branch>.+)_seed(?P<seed>\d+)$")


def discover_adapters(train_root: Path, include_checkpoints: bool) -> list[dict[str, Any]]:
    adapters = []
    for final_adapter in train_root.glob("n_*/*_seed*/final_adapter/adapter_model.safetensors"):
        run_dir = final_adapter.parents[1]
        match = RUN_RE.search(str(run_dir))
        if not match:
            continue
        base_info = {
            "n": int(match.group("n")),
            "branch": match.group("branch"),
            "seed": int(match.group("seed")),
        }
        adapters.append(
            {
                **base_info,
                "checkpoint": "final_adapter",
                "model_id": f"n_{base_info['n']}/{base_info['branch']}_seed{base_info['seed']}/final",
                "adapter": str(run_dir / "final_adapter"),
            }
        )
        if include_checkpoints:
            for checkpoint in sorted(run_dir.glob("checkpoint-*/adapter_model.safetensors")):
                checkpoint_dir = checkpoint.parent
                adapters.append(
                    {
                        **base_info,
                        "checkpoint": checkpoint_dir.name,
                        "model_id": (
                            f"n_{base_info['n']}/{base_info['branch']}_seed{base_info['seed']}/"
                            f"{checkpoint_dir.name}"
                        ),
                        "adapter": str(checkpoint_dir),
                    }
                )
    return sorted(adapters, key=lambda row: (row["n"], row["branch"], row["checkpoint"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-cpu-loras", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            done.add((row["model_id"], row["id"]))

    eval_rows = read_jsonl(args.eval_jsonl)
    models = discover_adapters(Path(args.train_root), include_checkpoints=args.include_checkpoints)
    if args.include_base:
        models = [
            {
                "model_id": "base",
                "n": None,
                "branch": "base",
                "seed": None,
                "checkpoint": "base",
                "adapter": None,
            }
        ] + models

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        for row in eval_rows
    ]
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=args.max_cpu_loras,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    next_lora_id = 1
    with out_path.open("a") as out:
        for model_info in models:
            active = [
                (row, prompt)
                for row, prompt in zip(eval_rows, prompts)
                if (model_info["model_id"], row["id"]) not in done
            ]
            if not active:
                print(f"SKIP {model_info['model_id']}", flush=True)
                continue
            lora_request = None
            if model_info["adapter"]:
                lora_request = LoRARequest(model_info["model_id"], next_lora_id, model_info["adapter"])
                next_lora_id += 1
            print(f"GENERATE {model_info['model_id']} prompts={len(active)}", flush=True)
            outputs = llm.generate([prompt for _, prompt in active], sampling, lora_request=lora_request)
            for (row, _), output in zip(active, outputs):
                out.write(
                    json.dumps(
                        {
                            **row,
                            **{k: v for k, v in model_info.items() if k != "adapter"},
                            "adapter": model_info["adapter"],
                            "answer": output.outputs[0].text,
                            "finish_reason": output.outputs[0].finish_reason,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            out.flush()
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
