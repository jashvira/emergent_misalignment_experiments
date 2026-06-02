#!/usr/bin/env python3
"""Collect hidden-state activations for rendered awareness probe inputs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


# Default files are the plain JSONL prep artifacts produced by
# `build_em_awareness_probe_prep.py`; model-specific tensor output goes under
# `outputs/activations`.
DEFAULT_INPUTS = (
    "data/interim/experiment_2/em_relevant_awareness_probe_prep_v1/activation_inputs.jsonl"
)
DEFAULT_OUT_DIR = "outputs/activations/experiment_2/em_relevant_awareness_probe_activations_v1"

# Character-boundary fields in each input row. These map to token positions
# before padding, then get adjusted per batch when the tokenizer left-pads.
TARGET_FIELDS = (
    "end_of_assistant_answer_char",
    "end_of_question_char",
    "verdict_position_char",
)

# Axis labels for the saved activation tensor: rows x targets x layers x hidden.
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


def file_sha256(path: str | Path) -> str:
    """Hash an input file for the activation-run contract."""

    digest = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_token_index(offsets: list[tuple[int, int]], char_position: int) -> int:
    """Map a character boundary to the nearest token at or before it."""

    # The prep file stores character boundaries in the rendered prompt, not token
    # indices. We deliberately take the token that contains the boundary, or the
    # previous real token if the boundary falls in whitespace between tokens. That
    # makes targets like "end of assistant answer" mean "the last token the model
    # has consumed for that span" rather than the next unrelated token.
    previous_real_token = None
    for token_index, (start, end) in enumerate(offsets):
        # Fast tokenizers report (0, 0) for special tokens; those do not
        # correspond to prompt characters and should never be probe targets.
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
    # Keep TARGET_FIELDS order fixed so saved tensors always have the same target
    # axis: assistant-answer boundary, judgement-question boundary, verdict slot.
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
    for shard_index in complete_shard_indices(out_dir):
        rows_path = shard_path(out_dir, "rows", shard_index, ".jsonl")
        for row in read_jsonl(rows_path):
            ids.add(str(row["id"]))
    return ids


def shard_file_index(path: Path, prefix: str, suffix: str) -> int | None:
    """Parse a shard index from a file name that matches the expected pattern."""

    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    index_text = name[len(prefix) : len(name) - len(suffix)]
    return int(index_text) if index_text.isdecimal() else None


def shard_indices(out_dir: Path, prefix: str, suffix: str) -> set[int]:
    """Return all parseable shard indices for one shard file family."""

    return {
        index
        for path in out_dir.glob(f"{prefix}*{suffix}")
        if (index := shard_file_index(path, prefix, suffix)) is not None
    }


def complete_shard_indices(out_dir: Path) -> list[int]:
    """Return shard indices where tensor and metadata files both exist."""

    return sorted(
        shard_indices(out_dir, "activations_shard_", ".pt")
        & shard_indices(out_dir, "rows_shard_", ".jsonl")
    )


def validate_shard_pairs(out_dir: Path) -> None:
    """Reject partial shards so resume cannot silently duplicate or skip rows."""

    activation_indices = shard_indices(out_dir, "activations_shard_", ".pt")
    row_indices = shard_indices(out_dir, "rows_shard_", ".jsonl")
    orphan_indices = sorted(activation_indices ^ row_indices)
    if orphan_indices:
        formatted = ", ".join(f"{index:05d}" for index in orphan_indices)
        raise ValueError(f"Partial activation shard(s) found: {formatted}")


def output_dir_has_collection_artifacts(out_dir: Path) -> bool:
    """Check whether an output directory already contains collector artifacts."""

    return (
        any(out_dir.glob("activations_shard_*.pt"))
        or any(out_dir.glob("rows_shard_*.jsonl"))
        or (out_dir / "activation_manifest.json").exists()
        or (out_dir / "skipped_rows.jsonl").exists()
    )


def validate_output_dir(out_dir: Path, resume: bool) -> None:
    """Validate output directory state before writing activation artifacts."""

    validate_shard_pairs(out_dir)
    if not resume and output_dir_has_collection_artifacts(out_dir):
        raise ValueError(
            f"{out_dir} already contains activation collection artifacts; use --resume or a fresh --out-dir."
        )


def existing_shard_records(out_dir: Path) -> list[dict[str, Any]]:
    """Build manifest records for already completed shard pairs."""

    records = []
    for index in complete_shard_indices(out_dir):
        rows_path = shard_path(out_dir, "rows", index, ".jsonl")
        records.append(
            {
                "index": index,
                "activation_file": str(shard_path(out_dir, "activations", index, ".pt")),
                "rows_file": str(rows_path),
                "rows": len(read_jsonl(rows_path)),
            }
        )
    return records


def resume_manifest(out_dir: Path, resume: bool) -> dict[str, Any] | None:
    """Load the previous manifest used to validate resume compatibility."""

    if not resume:
        return None
    path = out_dir / "activation_manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    if output_dir_has_collection_artifacts(out_dir):
        raise ValueError(
            "--resume requires activation_manifest.json in a populated output directory."
        )
    return None


def validate_resume_manifest(
    previous: dict[str, Any] | None,
    run_contract: dict[str, Any],
) -> list[int] | None:
    """Reject resume attempts that would mix incompatible activation artifacts."""

    if previous is None:
        return None

    previous_hash = previous.get("run_contract_hash")
    current_hash = contract_hash(run_contract)
    if previous_hash != current_hash:
        detail = ""
        previous_contract = previous.get("run_contract")
        if isinstance(previous_contract, dict):
            keys = sorted(set(previous_contract) | set(run_contract))
            mismatches = [
                key for key in keys if previous_contract.get(key) != run_contract.get(key)
            ]
            if mismatches:
                detail = " Mismatched fields: " + ", ".join(mismatches)
        raise ValueError("Resume manifest contract does not match current arguments." + detail)
    return [int(layer) for layer in previous.get("layers") or []] or None


def activation_run_contract(
    args: argparse.Namespace,
    target_names: list[str],
    target_fields: list[str],
    inputs_sha256: str,
) -> dict[str, Any]:
    """Build the semantic identity for one activation artifact set."""

    return {
        "version": "em_relevant_awareness_probe_activations_v1",
        "model": args.model,
        "inputs": args.inputs,
        "inputs_sha256": inputs_sha256,
        "target_names": target_names,
        "target_fields": target_fields,
        "layer_spec": args.layers,
        "max_length": args.max_length,
        "max_rows": getattr(args, "max_rows", None),
        "model_dtype": args.model_dtype,
        "save_dtype": args.save_dtype,
        "device_map": args.device_map,
        "offload_folder": args.offload_folder,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": getattr(args, "trust_remote_code", False),
    }


def contract_hash(contract: dict[str, Any]) -> str:
    """Hash a canonical JSON run contract."""

    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def existing_skipped_rows(out_dir: Path) -> list[dict[str, Any]]:
    """Load row-skip records from an earlier collector invocation."""

    path = out_dir / "skipped_rows.jsonl"
    return read_jsonl(path) if path.exists() else []


def next_shard_index(out_dir: Path) -> int:
    """Choose the next shard index after any existing activation shards."""

    indices = complete_shard_indices(out_dir)
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
        "text_sha256_16": sha256(str(row["text"]).encode()).hexdigest()[:16],
    }


def prepare_inputs(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[PreparedInput], list[dict[str, str]]]:
    """Batch-tokenize input rows and resolve activation target tokens."""

    if not rows:
        return [], []

    # We need offset mappings here, so this cannot be replaced with a plain
    # encode. The offset map is the bridge from our model-agnostic character
    # targets to this model/tokenizer's actual token indices.
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
        # These indices are still for the unpadded sequence. Padding happens
        # later at batch time, so left-padding offsets must be applied after
        # `tokenizer.pad` knows the final padded sequence length.
        prepared_rows.append(
            PreparedInput(
                row=row,
                input_ids=input_ids,
                attention_mask=list(attention_mask),
                target_token_indices=target_indices_from_row(row, offsets),
            )
        )
    return prepared_rows, skipped_rows


def pad_batch(tokenizer: Any, prepared_rows: list[PreparedInput]) -> dict[str, Any]:
    """Pad tokenized rows before model execution."""

    features = [
        {"input_ids": row.input_ids, "attention_mask": row.attention_mask} for row in prepared_rows
    ]
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def padded_target_token_indices(
    prepared_rows: list[PreparedInput],
    padded_sequence_length: int,
    padding_side: str,
) -> list[list[int]]:
    """Adjust target token indices from unpadded rows to padded batch positions."""

    if padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported tokenizer padding_side: {padding_side!r}")

    padded_indices = []
    for prepared in prepared_rows:
        # Hidden states are indexed in the padded batch tensor, not the original
        # per-row token list. Right padding preserves token positions. Left
        # padding shifts every real token right by the number of inserted pads.
        pad_offset = padded_sequence_length - prepared.token_count if padding_side == "left" else 0
        padded_indices.append([pad_offset + index for index in prepared.target_token_indices])
    return padded_indices


def model_hidden_state_count(model: Any) -> int:
    """Return the hidden-state tuple length exposed by this decoder model."""

    layer_count = getattr(model.config, "num_hidden_layers", None)
    if layer_count is None:
        raise ValueError("Model config does not expose num_hidden_layers.")
    return int(layer_count) + 1


def decoder_layers(model: Any) -> list[Any]:
    """Return decoder block modules used for selected-layer hooks."""

    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise ValueError("Could not find decoder layers on this model.")
    return list(layers)


def embedding_module(model: Any) -> Any:
    """Return the token embedding module used for hidden-state layer 0."""

    embeddings = getattr(model, "embed_tokens", None)
    if embeddings is None and hasattr(model, "model"):
        embeddings = getattr(model.model, "embed_tokens", None)
    if embeddings is None:
        raise ValueError("Could not find token embeddings on this model.")
    return embeddings


def final_norm_module(model: Any) -> Any:
    """Return the final decoder norm that produces the last hidden state."""

    norm = getattr(model, "norm", None)
    if norm is None and hasattr(model, "model"):
        norm = getattr(model.model, "norm", None)
    if norm is None:
        raise ValueError("Could not find final decoder norm on this model.")
    return norm


def selected_hidden_state_modules(model: Any, layers: list[int]) -> dict[int, Any]:
    """Map hidden-state tuple indices to modules whose outputs define them."""

    blocks = decoder_layers(model)
    final_hidden_state_index = len(blocks)
    modules = {}
    for layer in layers:
        if layer == 0:
            modules[layer] = embedding_module(model)
            continue
        if layer == final_hidden_state_index:
            # HF decoder hidden_states contain embeddings at index 0, one
            # post-block state per decoder block, then the final post-norm state.
            # Hook the norm for `last`; hooking the final block would save the
            # pre-norm residual stream under a misleading final-layer label.
            modules[layer] = final_norm_module(model)
            continue
        block_index = layer - 1
        if block_index < 0 or block_index >= final_hidden_state_index:
            raise ValueError(
                f"Layer {layer} is outside hidden-state range 0..{final_hidden_state_index}."
            )
        modules[layer] = blocks[block_index]
    return modules


def register_hidden_state_hooks(
    model: Any,
    layers: list[int],
    captured: dict[int, Any],
    capture_context: dict[str, Any],
) -> list[Any]:
    """Capture target-token hidden states for the active `collect` batch."""

    import torch

    handles = []
    for layer, module in selected_hidden_state_modules(model, layers).items():

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, layer: int = layer) -> None:
            # Decoder blocks usually return tuples with the hidden state first;
            # embeddings and final norms return tensors directly. Gather only the
            # target tokens here so selected layers do not retain full sequences.
            hidden_state = output[0] if isinstance(output, tuple) else output
            target_indices = capture_context.get("target_indices")
            if target_indices is None:
                raise RuntimeError("Target indices must be set before model forward.")
            target_indices = target_indices.to(hidden_state.device)
            batch_indices = torch.arange(
                hidden_state.shape[0],
                device=hidden_state.device,
                dtype=target_indices.dtype,
            ).unsqueeze(1)
            captured[layer] = hidden_state[batch_indices, target_indices].detach()

        handles.append(module.register_forward_hook(hook))
    return handles


def gather_activation_tensor(
    hidden_states: dict[int, Any] | tuple[Any, ...],
    layers: list[int],
    save_dtype: Any,
) -> Any:
    """Gather target-token activations into rows x targets x layers x hidden."""

    import torch

    # Hooks have already reduced each selected layer to rows x targets x hidden.
    # Stack the layer axis and copy one compact tensor to CPU, avoiding
    # row/target/layer-sized transfer loops.
    return torch.stack(
        [hidden_states[layer].to("cpu", dtype=save_dtype) for layer in layers],
        dim=2,
    )


def shard_path(out_dir: Path, stem: str, shard_index: int, suffix: str) -> Path:
    """Build the canonical path for one shard file."""

    return out_dir / f"{stem}_shard_{shard_index:05d}{suffix}"


def temporary_output_path(path: Path) -> Path:
    """Build a process-scoped temporary path next to a final output file."""

    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL via a temporary file before publishing the final path."""

    temporary_path = temporary_output_path(path)
    write_jsonl(temporary_path, rows)
    temporary_path.replace(path)


def atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    """Save a torch payload via a temporary file before publishing it."""

    temporary_path = temporary_output_path(path)
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def write_activation_shard(
    out_dir: Path,
    shard_index: int,
    activations: Any,
    metadata_rows: list[dict[str, Any]],
    layers: list[int],
) -> dict[str, Any]:
    """Write one activation tensor shard and its JSONL metadata."""

    import torch

    activation_path = shard_path(out_dir, "activations", shard_index, ".pt")
    rows_path = shard_path(out_dir, "rows", shard_index, ".jsonl")
    atomic_torch_save(
        torch,
        {
            "activations": activations,
            "layers": layers,
            "target_names": list(TARGET_NAMES),
            "row_count": len(metadata_rows),
        },
        activation_path,
    )
    atomic_write_jsonl(rows_path, metadata_rows)
    return {
        "index": shard_index,
        "activation_file": str(activation_path),
        "rows_file": str(rows_path),
        "rows": len(metadata_rows),
    }


def model_load_kwargs(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    """Build `from_pretrained` kwargs for the hidden-state model loader."""

    kwargs = {
        "torch_dtype": model_dtype_from_name(args.model_dtype, torch),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.offload_folder:
        kwargs["offload_folder"] = args.offload_folder
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load tokenizer and base transformer for activation collection."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    require_fast_tokenizer(tokenizer, args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The collector only needs decoder hidden states. Loading the base model
    # avoids the causal-LM head, which would otherwise allocate full-vocabulary
    # logits for every token and waste tens of GB on long batches.
    model = AutoModel.from_pretrained(args.model, **model_load_kwargs(args, torch))
    model.eval()
    return tokenizer, model


def require_fast_tokenizer(tokenizer: Any, model_name: str) -> None:
    """Fail before model loading if activation offsets are unavailable."""

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError(
            "Activation collection requires a fast tokenizer with offset mappings. "
            f"{model_name!r} loaded a slow tokenizer; choose a fast tokenizer or add one."
        )


def normalized_device_map_value(device: Any) -> Any:
    """Normalize HF device-map values for tensor `.to(...)` placement."""

    if isinstance(device, int):
        return f"cuda:{device}"
    return device


def model_input_device(model: Any) -> Any:
    """Return the device that should receive input tensors for `collect`."""

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        has_cpu = False
        for device in device_map.values():
            normalized = normalized_device_map_value(device)
            name = str(normalized).lower()
            if name == "cpu":
                has_cpu = True
                continue
            if name in {"disk", "meta"} or name.startswith("meta:"):
                continue
            return normalized
        if has_cpu:
            return "cpu"
        raise ValueError("Model device map does not contain a compute or CPU input device.")
    return next(model.parameters()).device


def serializable_device_map(model: Any) -> dict[str, str] | None:
    """Return the loaded HF device map for activation-run audit manifests."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        return None
    return {str(name): str(normalized_device_map_value(device)) for name, device in device_map.items()}


def iter_batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield fixed-size row batches while preserving input order."""

    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run activation collection and write shards plus a manifest."""

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_output_dir(out_dir, args.resume)
    target_names = list(TARGET_NAMES)
    target_fields = list(TARGET_FIELDS)
    inputs_sha256 = file_sha256(args.inputs)
    run_contract = activation_run_contract(
        args,
        target_names,
        target_fields,
        inputs_sha256,
    )
    previous_manifest = resume_manifest(out_dir, args.resume)
    previous_layers = validate_resume_manifest(
        previous_manifest,
        run_contract,
    )
    tokenizer, model = load_model_and_tokenizer(args)
    save_dtype = save_dtype_from_name(args.save_dtype, torch)
    existing_ids = existing_row_ids(out_dir) if args.resume else set()
    previous_skipped_rows = existing_skipped_rows(out_dir) if args.resume else []
    previous_skipped_ids = {str(row["id"]) for row in previous_skipped_rows}
    shard_index = next_shard_index(out_dir)
    current_layers = parse_layer_spec(args.layers, model_hidden_state_count(model))
    if previous_layers is not None and previous_layers != current_layers:
        raise ValueError("Resume layers do not match current --layers selection.")

    rows = read_jsonl(args.inputs)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    device = model_input_device(model)
    layers = previous_layers or current_layers
    captured_hidden_states: dict[int, Any] = {}
    capture_context: dict[str, Any] = {}
    hook_handles = register_hidden_state_hooks(
        model,
        layers,
        captured_hidden_states,
        capture_context,
    )
    shards = existing_shard_records(out_dir) if args.resume else []
    new_shards = []
    new_skipped_rows = []
    skipped_rows = list(previous_skipped_rows)
    pending_activations = []
    pending_metadata = []
    collected = 0
    skipped_existing = 0
    skipped_previously = 0

    try:
        for row_batch in iter_batches(rows, args.batch_size):
            rows_to_prepare = []
            for row in row_batch:
                row_id = str(row["id"])
                if row_id in existing_ids:
                    skipped_existing += 1
                    continue
                if row_id in previous_skipped_ids:
                    skipped_previously += 1
                    continue
                rows_to_prepare.append(row)
            prepared_rows, skipped_batch = prepare_inputs(
                rows_to_prepare, tokenizer, args.max_length
            )
            new_skipped_rows.extend(skipped_batch)
            skipped_rows.extend(skipped_batch)
            if not prepared_rows:
                continue

            batch = pad_batch(tokenizer, prepared_rows)
            # Compute target positions before moving tensors to GPU; this is pure
            # Python bookkeeping over row lengths and tokenizer padding policy.
            batch_target_indices = padded_target_token_indices(
                prepared_rows,
                int(batch["input_ids"].shape[1]),
                str(tokenizer.padding_side),
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            target_indices_tensor = torch.as_tensor(
                batch_target_indices,
                dtype=torch.long,
                device=batch["input_ids"].device,
            )
            capture_context["target_indices"] = target_indices_tensor
            captured_hidden_states.clear()
            try:
                with torch.inference_mode():
                    # Forward hooks keep only selected target-token outputs. This
                    # avoids `output_hidden_states=True`, which materializes every
                    # decoder layer and prevents useful batch sizes on large models.
                    model(**batch, use_cache=False)
            finally:
                capture_context.clear()
            missing = [layer for layer in layers if layer not in captured_hidden_states]
            if missing:
                raise ValueError(f"Missing captured hidden states for layers: {missing}")
            activations = gather_activation_tensor(
                captured_hidden_states,
                layers,
                save_dtype,
            )
            captured_hidden_states.clear()
            metadata = [activation_metadata(prepared.row, prepared) for prepared in prepared_rows]
            pending_activations.append(activations)
            pending_metadata.extend(metadata)
            collected += len(prepared_rows)

            if len(pending_metadata) >= args.shard_size:
                shard_tensor = torch.cat(pending_activations, dim=0)
                shard = write_activation_shard(
                    out_dir, shard_index, shard_tensor, pending_metadata, layers
                )
                shards.append(shard)
                new_shards.append(shard)
                shard_index += 1
                pending_activations = []
                pending_metadata = []
    finally:
        for handle in hook_handles:
            handle.remove()

    if pending_metadata:
        shard_tensor = torch.cat(pending_activations, dim=0)
        shard = write_activation_shard(
            out_dir, shard_index, shard_tensor, pending_metadata, layers or []
        )
        shards.append(shard)
        new_shards.append(shard)

    if skipped_rows:
        write_jsonl(out_dir / "skipped_rows.jsonl", skipped_rows)

    manifest = {
        **run_contract,
        "run_contract": run_contract,
        "run_contract_hash": contract_hash(run_contract),
        "out_dir": str(out_dir),
        "layers": layers or [],
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "input_device": str(device),
        "loaded_device_map": serializable_device_map(model),
        "resume": args.resume,
        "counts": {
            "input_rows_seen": len(rows),
            "collected": collected,
            "skipped_existing": skipped_existing,
            "skipped_previously": skipped_previously,
            "skipped": len(skipped_rows),
            "new_skipped": len(new_skipped_rows),
            "shards": len(shards),
            "new_shards": len(new_shards),
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
    parser.add_argument(
        "--model-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument(
        "--save-dtype", choices=("bfloat16", "float16", "float32"), default="float16"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder")
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
