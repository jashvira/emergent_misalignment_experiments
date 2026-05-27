#!/usr/bin/env python3
"""Score A1/A2 forced-choice inputs with full-option logprobs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from tqdm import tqdm

from emergent_misalignment_experiments.experiment1.io import read_jsonl


DEFAULT_INPUTS = "data/interim/experiment_2/a1_a2_organic_labeling_v1/scoring_inputs.jsonl"
DEFAULT_OUT = "outputs/awareness/experiment_2/a1_a2_organic_labeling_scores.jsonl"


def file_sha256(path: str | Path) -> str:
    """Hash the input JSONL file so resume cannot cross scoring-input versions."""

    digest = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Append scorer rows while preserving resume-compatible JSONL output."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def option_token_indices(
    tokenizer: Any,
    prompt_text: str,
    option_text: str,
) -> list[int]:
    """Return token positions belonging to the option."""

    _token_ids, token_positions = tokenize_prompt_with_option(
        tokenizer,
        prompt_text,
        option_text,
        max_prompt_tokens=None,
    )
    return token_positions


def tokenize_prompt_with_option(
    tokenizer: Any,
    prompt_text: str,
    option_text: str,
    max_prompt_tokens: int | None,
) -> tuple[list[int], list[int]]:
    """Tokenize one scored continuation and return vLLM-ready IDs plus option positions."""

    combined = prompt_text + option_text
    encoded = tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    if max_prompt_tokens is not None and len(input_ids) > max_prompt_tokens:
        raise ValueError(
            f"Prompt plus option has {len(input_ids)} tokens, exceeding the "
            f"{max_prompt_tokens}-token scoring limit."
        )
    option_start = len(prompt_text)
    option_end = len(combined)
    token_positions = []
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start == end:
            continue
        if start < option_start < end:
            raise ValueError(
                "Prompt/option boundary fell inside one token; render the prompt with a "
                "separator before scoring."
            )
        if start >= option_start and end <= option_end:
            token_positions.append(index)
    if not token_positions:
        raise ValueError(f"Option produced no scoreable tokens: {option_text!r}")
    return input_ids, token_positions


def validate_input_row(row: dict[str, Any]) -> tuple[str, str]:
    """Read positive/negative options from one scoring input row."""

    positive = str(row["positive_option"])
    negative = str(row["negative_option"])
    options = [str(option) for option in row["options"]]
    if options != [" A", " B"]:
        raise ValueError(f"Row {row.get('id')} options must be [' A', ' B'].")
    if {positive, negative} != set(options):
        raise ValueError(f"Row {row.get('id')} positive/negative options must map to A/B.")
    return positive, negative


def score_rows_with_vllm(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    inputs_sha256: str,
) -> None:
    """Score rows with vLLM prefill logprobs; used for the Vast GPU run."""

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("Forced-choice scoring requires a fast tokenizer for offsets.")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs or args.batch_size * 2,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    max_prompt_tokens = args.max_model_len - 1

    for start in tqdm(range(0, len(rows), args.batch_size), desc="score batches"):
        batch_rows = rows[start : start + args.batch_size]
        request_rows = []
        request_token_ids = []
        for row in batch_rows:
            positive, negative = validate_input_row(row)
            for option in (positive, negative):
                try:
                    token_ids, positions = tokenize_prompt_with_option(
                        tokenizer,
                        row["text"],
                        option,
                        max_prompt_tokens,
                    )
                    request_rows.append((row, option, positions))
                    request_token_ids.append(token_ids)
                except Exception as exc:
                    request_rows.append((row, option, exc))
                    request_token_ids.append([])

        score_by_id: dict[str, dict[str, Any]] = defaultdict(dict)
        prompt_indices = [
            index
            for index, (_row, _option, positions) in enumerate(request_rows)
            if not isinstance(positions, Exception)
        ]
        outputs_by_index = {}
        if prompt_indices:
            outputs = llm.generate(
                prompt_token_ids=[request_token_ids[index] for index in prompt_indices],
                sampling_params=sampling,
            )
            outputs_by_index = dict(zip(prompt_indices, outputs, strict=True))

        for index, (row, option, positions) in enumerate(request_rows):
            row_id = str(row["id"])
            if isinstance(positions, Exception):
                score_by_id[row_id].setdefault("errors", {})[option] = str(positions)
                continue
            output = outputs_by_index[index]
            score_by_id[row_id][option] = vllm_option_logprob(output, positions)

        append_jsonl(
            args.out,
            [
                scored_output_row(
                    row,
                    args.model,
                    inputs_sha256,
                    score_by_id[str(row["id"])],
                )
                for row in batch_rows
            ],
        )


def vllm_option_logprob(output: Any, token_positions: list[int]) -> float:
    """Sum selected-token prompt logprobs from one vLLM request output."""

    total = 0.0
    prompt_logprobs = output.prompt_logprobs
    prompt_token_ids = output.prompt_token_ids
    for position in token_positions:
        if position == 0:
            raise ValueError("Cannot score the first prompt token as a continuation.")
        token_id = prompt_token_ids[position]
        token_logprobs = prompt_logprobs[position]
        if token_logprobs is None or token_id not in token_logprobs:
            raise ValueError(f"Missing prompt logprob for token position {position}.")
        total += float(token_logprobs[token_id].logprob)
    return total


def scored_output_row(
    row: dict[str, Any],
    model: str,
    inputs_sha256: str,
    option_scores: dict[str, Any],
) -> dict[str, Any]:
    """Create one JSONL score row from option logprobs."""

    positive = str(row["positive_option"])
    negative = str(row["negative_option"])
    errors = option_scores.get("errors", {})
    if errors or positive not in option_scores or negative not in option_scores:
        return {
            **row_metadata(row, model, inputs_sha256),
            "option_logprobs": {
                option: score for option, score in option_scores.items() if option != "errors"
            },
            "margin": None,
            "predicted_option": None,
            "positive_prediction": None,
            "error": errors or "missing option score",
        }

    option_logprobs = {
        positive: float(option_scores[positive]),
        negative: float(option_scores[negative]),
    }
    margin = option_logprobs[positive] - option_logprobs[negative]
    predicted = positive if margin >= 0 else negative
    return {
        **row_metadata(row, model, inputs_sha256),
        "option_logprobs": option_logprobs,
        "margin": margin,
        "predicted_option": predicted,
        "positive_prediction": predicted == positive,
        "error": None,
    }


def row_metadata(row: dict[str, Any], model: str, inputs_sha256: str) -> dict[str, Any]:
    """Copy stable input fields into scorer output rows."""

    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "split": row.get("split"),
        "source": row.get("source"),
        "family": row["family"],
        "wrapper_id": row["wrapper_id"],
        "answer_kind": row["answer_kind"],
        "no_evidence_control": row["no_evidence_control"],
        "model": model,
        "options": row["options"],
        "positive_option": row["positive_option"],
        "negative_option": row["negative_option"],
        "positive_label": row.get("positive_label"),
        "negative_label": row.get("negative_label"),
        "option_labels": row.get("option_labels"),
        "inputs_sha256": inputs_sha256,
    }


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Load scoring inputs and apply limit/resume behavior."""

    inputs_sha256 = file_sha256(args.inputs)
    rows = read_jsonl(args.inputs)
    if args.limit is not None:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    elif out_path.exists():
        existing_rows = read_jsonl(out_path)
        existing_models = {row.get("model") for row in existing_rows}
        if existing_models != {args.model}:
            formatted = ", ".join(sorted(str(model) for model in existing_models))
            raise ValueError(
                f"{out_path} already contains scores for model(s): {formatted}; "
                "use --overwrite or choose a separate --out."
            )
        existing_input_digests = {row.get("inputs_sha256") for row in existing_rows}
        if existing_input_digests != {inputs_sha256}:
            raise ValueError(
                f"{out_path} was created from a different inputs file; "
                "use --overwrite or choose a separate --out."
            )
        existing_ids = {str(row["id"]) for row in existing_rows if "id" in row}
        rows = [row for row in rows if str(row["id"]) not in existing_ids]
    return rows, inputs_sha256


def parse_args() -> argparse.Namespace:
    """Parse CLI options for forced-choice scoring."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    return args


def main() -> None:
    """Run forced-choice scoring."""

    args = parse_args()
    rows, inputs_sha256 = load_rows(args)
    if not rows:
        print("No rows to score.")
        return
    score_rows_with_vllm(args, rows, inputs_sha256)


if __name__ == "__main__":
    main()
