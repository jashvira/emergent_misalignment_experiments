#!/usr/bin/env python3
"""Generate Betley broad-eval answers for base and LoRA adapters.

Input is raw Betley eval YAML plus optional local LoRA adapters. Output is a
response JSONL for later OpenAI logprob judging.
"""

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

from emergent_misalignment_experiments.experiment1.betley_eval import (
    load_betley_questions,
    messages_for_betley_row,
)
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


RUN_RE = re.compile(r"n_(?P<n>\d+)[/\\](?P<branch>.+)_seed(?P<seed>\d+)$")


def discover_adapters(train_root: Path) -> list[dict[str, Any]]:
    runs = []
    for adapter_file in train_root.glob("n_*/*_seed*/final_adapter/adapter_model.safetensors"):
        run_dir = adapter_file.parents[1]
        match = RUN_RE.search(str(run_dir))
        if not match:
            continue
        runs.append(
            {
                "model_id": f"n_{match.group('n')}/{match.group('branch')}_seed{match.group('seed')}",
                "n": int(match.group("n")),
                "branch": match.group("branch"),
                "seed": int(match.group("seed")),
                "adapter": str(run_dir / "final_adapter"),
            }
        )
    return sorted(runs, key=lambda x: (x["n"], x["branch"]))


def render_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            messages_for_betley_row(row),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Betley broad-eval responses with vLLM and optional LoRA adapters."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--question-yamls",
        nargs="+",
        default=[
            "data/raw/betley/evaluation/first_plot_questions.yaml",
            "data/raw/betley/evaluation/preregistered_evals.yaml",
        ],
    )
    parser.add_argument("--samples-per-question-override", type=int)
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, help="Optional vLLM global sampling seed.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=1024,
        help="Number of prompts to generate before appending/checkpointing responses.",
    )
    parser.add_argument("--max-cpu-loras", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    args = parser.parse_args()

    eval_rows = load_betley_questions(
        args.question_yamls,
        samples_per_question_override=args.samples_per_question_override,
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "betley_eval_responses.jsonl"

    done: set[tuple[str, str, str, int, int]] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            done.add(
                (
                    row["model_id"],
                    row["eval_file"],
                    row["eval_id"],
                    int(row["paraphrase_index"]),
                    int(row["sample_index"]),
                )
            )

    models = discover_adapters(Path(args.train_root))
    if args.include_base:
        models = [{"model_id": "base", "branch": "base", "n": None, "seed": None, "adapter": None}] + models

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = render_prompts(tokenizer, eval_rows)
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
    sampling_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "skip_special_tokens": True,
        "min_tokens": 1,
        "stop": [tokenizer.eos_token],
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling = SamplingParams(**sampling_kwargs)

    next_lora_id = 1
    with out_path.open("a") as out:
        for model_info in models:
            active = [
                (row, prompt)
                for row, prompt in zip(eval_rows, prompts, strict=True)
                if (
                    model_info["model_id"],
                    row["eval_file"],
                    row["eval_id"],
                    int(row["paraphrase_index"]),
                    int(row["sample_index"]),
                )
                not in done
            ]
            if not active:
                print(f"SKIP {model_info['model_id']}", flush=True)
                continue
            lora_request = None
            if model_info.get("adapter"):
                lora_request = LoRARequest(model_info["model_id"], next_lora_id, model_info["adapter"])
                next_lora_id += 1
            print(f"GENERATE {model_info['model_id']} prompts={len(active)}", flush=True)
            chunk_size = args.generation_chunk_size if args.generation_chunk_size > 0 else len(active)
            for chunk_start in range(0, len(active), chunk_size):
                chunk = active[chunk_start : chunk_start + chunk_size]
                print(
                    f"GENERATE_CHUNK {model_info['model_id']} "
                    f"{chunk_start}-{chunk_start + len(chunk)}/{len(active)}",
                    flush=True,
                )
                outputs = llm.generate([prompt for _, prompt in chunk], sampling, lora_request=lora_request)
                for (row, _), output in zip(chunk, outputs, strict=True):
                    serializable = {k: v for k, v in row.items() if k != "judge_prompts"}
                    out.write(
                        json.dumps(
                            {
                                **serializable,
                                **{k: v for k, v in model_info.items() if k != "adapter"},
                                "adapter": model_info.get("adapter"),
                                "answer": output.outputs[0].text,
                                "finish_reason": output.outputs[0].finish_reason,
                                "generation_temperature": args.temperature,
                                "generation_top_p": args.top_p,
                                "generation_max_tokens": args.max_tokens,
                                "generation_seed": args.seed,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                out.flush()

    write_jsonl(out_root / "eval_manifest.jsonl", [{"model": model} for model in models])
    write_jsonl(out_root / "question_manifest.jsonl", eval_rows)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
