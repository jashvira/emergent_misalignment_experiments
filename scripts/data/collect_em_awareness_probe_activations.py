#!/usr/bin/env python3
"""Collect hidden-state activations for rendered awareness probe inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_INPUTS = "data/interim/experiment_1/em_relevant_awareness_probe_prep_v1/activation_inputs.jsonl"
DEFAULT_OUT_DIR = "outputs/activations/experiment_1/em_relevant_awareness_probe_activations_v1"
TARGET_FIELDS = (
    "end_of_assistant_answer_char",
    "end_of_question_char",
    "verdict_position_char",
)
TARGET_NAMES = (
    "end_of_assistant_answer",
    "end_of_question",
    "verdict_position",
)


@dataclass(frozen=True)
class PreparedInput:
    """Tokenized input plus metadata needed for activation gathering."""

    row: dict[str, Any]
    input_ids: list[int]
    attention_mask: list[int]
    target_token_indices: list[int]

    @property
    def token_count(self) -> int:
        """Return the unpadded token count for manifest and metadata rows."""

        return len(self.input_ids)


def text_hash(text: str) -> str:
    """Create a stable short hash for metadata without storing duplicate text."""

    return sha256(text.encode()).hexdigest()[:16]


def target_token_index(offsets: list[tuple[int, int]], char_position: int) -> int:
    """Map a character boundary to the nearest token at or before it."""

    previous_real_token = None
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if start < char_position <= end:
            return token_index
        if end <= char_position:
            previous_real_token = token_index
        elif previous_real_token is not None:
            break
    if previous_real_token is not None:
        return previous_real_token
    for token_index, (start, end) in enumerate(offsets):
        if start != end:
            return token_index
    raise ValueError("No non-special token offsets are available.")


def target_indices_from_row(row: dict[str, Any], offsets: list[tuple[int, int]]) -> list[int]:
    """Resolve all configured activation targets for one input row."""

    targets = row.get("activation_targets")
    if not isinstance(targets, dict):
        raise ValueError(f"Missing activation_targets for row {row.get('id')}")
    return [target_token_index(offsets, int(targets[field])) for field in TARGET_FIELDS]


def parse_layer_spec(spec: str, hidden_state_count: int) -> list[int]:
    """Resolve CLI layer syntax against a model's hidden-state tuple length."""

    if hidden_state_count <= 0:
        raise ValueError("hidden_state_count must be positive.")
    if spec == "all":
        return list(range(hidden_state_count))

    layers = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part == "last":
            layer = hidden_state_count - 1
        else:
            layer = int(part)
            if layer < 0:
                layer = hidden_state_count + layer
        if layer < 0 or layer >= hidden_state_count:
            raise ValueError(
                f"Layer {raw_part!r} resolves to {layer}, outside 0..{hidden_state_count - 1}."
            )
        if layer not in layers:
            layers.append(layer)
    if not layers:
        raise ValueError("No layers selected.")
    return layers


def model_dtype_from_name(name: str, torch: Any) -> Any:
    """Map CLI dtype names to torch dtypes for model loading."""

    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def save_dtype_from_name(name: str, torch: Any) -> Any:
    """Map CLI dtype names to torch dtypes for saved activation tensors."""

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def existing_row_ids(out_dir: Path) -> set[str]:
    """Read completed metadata shards so `--resume` can skip written rows."""

    ids = set()
    for rows_path in sorted(out_dir.glob("rows_shard_*.jsonl")):
        for row in read_jsonl(rows_path):
            ids.add(str(row["id"]))
    return ids


def next_shard_index(out_dir: Path) -> int:
    """Choose the next shard index after any existing activation shards."""

    indices = []
    for path in out_dir.glob("activations_shard_*.pt"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 0


def activation_metadata(row: dict[str, Any], prepared: PreparedInput) -> dict[str, Any]:
    """Create the metadata row paired with one saved activation row."""

    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "split": row["split"],
        "source": row.get("source"),
        "probe_prompt_id": row["probe_prompt_id"],
        "probe_family": row["probe_family"],
        "answer_kind": row["answer_kind"],
        "no_evidence_ablation": row["no_evidence_ablation"],
        "token_count": prepared.token_count,
        "target_token_indices": dict(zip(TARGET_NAMES, prepared.target_token_indices, strict=True)),
        "text_sha256_16": text_hash(str(row["text"])),
    }


def prepare_inputs(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[PreparedInput], list[dict[str, str]]]:
    """Batch-tokenize input rows and resolve activation target tokens."""

    if not rows:
        return [], []

    encoded = tokenizer(
        [str(row["text"]) for row in rows],
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    prepared_rows = []
    skipped_rows = []
    for row, input_ids, attention_mask, offset_mapping in zip(
        rows,
        encoded["input_ids"],
        encoded["attention_mask"],
        encoded["offset_mapping"],
        strict=True,
    ):
        input_ids = list(input_ids)
        if len(input_ids) > max_length:
            skipped_rows.append(
                {
                    "id": str(row["id"]),
                    "reason": f"token_count {len(input_ids)} exceeds max_length {max_length}",
                }
            )
            continue
        offsets = [tuple(offset) for offset in offset_mapping]
        prepared_rows.append(
            PreparedInput(
                row=row,
                input_ids=input_ids,
                attention_mask=list(attention_mask),
                target_token_indices=target_indices_from_row(row, offsets),
            )
        )
    return prepared_rows, skipped_rows


def prepare_input(row: dict[str, Any], tokenizer: Any, max_length: int) -> PreparedInput:
    """Tokenize one row for tests and callers that need single-row behavior."""

    prepared_rows, skipped_rows = prepare_inputs([row], tokenizer, max_length)
    if skipped_rows:
        raise ValueError(skipped_rows[0]["reason"])
    return prepared_rows[0]


def pad_batch(tokenizer: Any, prepared_rows: list[PreparedInput]) -> dict[str, Any]:
    """Pad tokenized rows before model execution."""

    features = [
        {"input_ids": row.input_ids, "attention_mask": row.attention_mask}
        for row in prepared_rows
    ]
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def gather_activation_tensor(
    hidden_states: tuple[Any, ...],
    prepared_rows: list[PreparedInput],
    layers: list[int],
    save_dtype: Any,
) -> Any:
    """Gather target-token activations into rows x targets x layers x hidden."""

    import torch

    row_tensors = []
    for row_index, prepared in enumerate(prepared_rows):
        target_tensors = []
        for token_index in prepared.target_token_indices:
            layer_tensors = [
                hidden_states[layer][row_index, token_index].detach().to("cpu", dtype=save_dtype)
                for layer in layers
            ]
            target_tensors.append(torch.stack(layer_tensors))
        row_tensors.append(torch.stack(target_tensors))
    return torch.stack(row_tensors)


def write_activation_shard(
    out_dir: Path,
    shard_index: int,
    activations: Any,
    metadata_rows: list[dict[str, Any]],
    layers: list[int],
) -> dict[str, Any]:
    """Write one activation tensor shard and its JSONL metadata."""

    import torch

    activation_path = out_dir / f"activations_shard_{shard_index:05d}.pt"
    rows_path = out_dir / f"rows_shard_{shard_index:05d}.jsonl"
    torch.save(
        {
            "activations": activations,
            "layers": layers,
            "target_names": list(TARGET_NAMES),
            "row_count": len(metadata_rows),
        },
        activation_path,
    )
    write_jsonl(rows_path, metadata_rows)
    return {
        "index": shard_index,
        "activation_file": str(activation_path),
        "rows_file": str(rows_path),
        "rows": len(metadata_rows),
    }


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load tokenizer and causal LM for activation collection."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": model_dtype_from_name(args.model_dtype, torch),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    return tokenizer, model


def model_input_device(model: Any) -> Any:
    """Return the device that should receive input tensors."""

    return next(model.parameters()).device


def iter_batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield fixed-size row batches while preserving input order."""

    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run activation collection and write shards plus a manifest."""

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model_and_tokenizer(args)
    save_dtype = save_dtype_from_name(args.save_dtype, torch)
    existing_ids = existing_row_ids(out_dir) if args.resume else set()
    shard_index = next_shard_index(out_dir)

    rows = read_jsonl(args.inputs)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    device = model_input_device(model)
    layers: list[int] | None = None
    shards = []
    skipped_rows = []
    pending_activations = []
    pending_metadata = []
    collected = 0
    skipped_existing = 0

    for row_batch in iter_batches(rows, args.batch_size):
        rows_to_prepare = []
        for row in row_batch:
            row_id = str(row["id"])
            if row_id in existing_ids:
                skipped_existing += 1
                continue
            rows_to_prepare.append(row)
        prepared_rows, skipped_batch = prepare_inputs(rows_to_prepare, tokenizer, args.max_length)
        skipped_rows.extend(skipped_batch)
        if not prepared_rows:
            continue

        batch = pad_batch(tokenizer, prepared_rows)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch, output_hidden_states=True, use_cache=False)
        if layers is None:
            layers = parse_layer_spec(args.layers, len(output.hidden_states))
        activations = gather_activation_tensor(
            output.hidden_states,
            prepared_rows,
            layers,
            save_dtype,
        )
        metadata = [
            activation_metadata(prepared.row, prepared) for prepared in prepared_rows
        ]
        pending_activations.append(activations)
        pending_metadata.extend(metadata)
        collected += len(prepared_rows)

        if len(pending_metadata) >= args.shard_size:
            shard_tensor = torch.cat(pending_activations, dim=0)
            shards.append(
                write_activation_shard(out_dir, shard_index, shard_tensor, pending_metadata, layers)
            )
            shard_index += 1
            pending_activations = []
            pending_metadata = []

    if pending_metadata:
        shard_tensor = torch.cat(pending_activations, dim=0)
        shards.append(write_activation_shard(out_dir, shard_index, shard_tensor, pending_metadata, layers or []))

    if skipped_rows:
        write_jsonl(out_dir / "skipped_rows.jsonl", skipped_rows)

    manifest = {
        "version": "em_relevant_awareness_probe_activations_v1",
        "model": args.model,
        "inputs": args.inputs,
        "out_dir": str(out_dir),
        "target_names": list(TARGET_NAMES),
        "target_fields": list(TARGET_FIELDS),
        "layers": layers or [],
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "model_dtype": args.model_dtype,
        "save_dtype": args.save_dtype,
        "resume": args.resume,
        "counts": {
            "input_rows_seen": len(rows),
            "collected": collected,
            "skipped_existing": skipped_existing,
            "skipped": len(skipped_rows),
            "shards": len(shards),
        },
        "shards": shards,
        "skipped_rows_file": str(out_dir / "skipped_rows.jsonl") if skipped_rows else None,
    }
    (out_dir / "activation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the activation collector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--layers", default="last")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--model-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--save-dtype", choices=("bfloat16", "float16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive.")
    if args.max_length < 1:
        raise ValueError("--max-length must be positive.")
    return args


def main() -> None:
    """Run activation collection and print the manifest counts."""

    manifest = collect(parse_args())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
