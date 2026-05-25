import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "collect_em_awareness_probe_activations.py"
)
SPEC = importlib.util.spec_from_file_location("collect_em_awareness_probe_activations", SCRIPT_PATH)
collector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class WhitespaceBatchTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, list[list]]:
        self.calls += 1
        assert isinstance(texts, list)
        assert add_special_tokens is True
        assert return_attention_mask is True
        assert return_offsets_mapping is True
        assert truncation is False

        input_ids = []
        attention_masks = []
        offset_mappings = []
        for text in texts:
            offsets = []
            cursor = 0
            for token_id, token in enumerate(text.split(), start=1):
                start = text.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            input_ids.append(list(range(1, len(offsets) + 1)))
            attention_masks.append([1] * len(offsets))
            offset_mappings.append(offsets)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "offset_mapping": offset_mappings,
        }


class DummyModule:
    pass


def test_selected_hidden_state_modules_maps_last_to_final_norm() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=3),
        embed_tokens=DummyModule(),
        layers=[DummyModule(), DummyModule(), DummyModule()],
        norm=DummyModule(),
    )

    modules = collector.selected_hidden_state_modules(model, [0, 1, 2, 3])

    assert modules[0] is model.embed_tokens
    assert modules[1] is model.layers[0]
    assert modules[2] is model.layers[1]
    assert modules[3] is model.norm


def test_hidden_state_hook_gathers_only_target_tokens() -> None:
    torch = pytest.importorskip("torch")

    class TupleLayer(torch.nn.Module):
        def forward(self, hidden_state):
            return (hidden_state + 1,)

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1),
        embed_tokens=torch.nn.Identity(),
        layers=[TupleLayer()],
        norm=torch.nn.Identity(),
    )
    captured = {}
    capture_context = {
        "target_indices": torch.tensor([[1, 3], [0, 2]], dtype=torch.long),
        "batch_indices": torch.tensor([[0], [1]], dtype=torch.long),
    }

    handles = collector.register_hidden_state_hooks(model, [1], captured, capture_context)
    try:
        hidden_state = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        model.layers[0](hidden_state)
    finally:
        for handle in handles:
            handle.remove()

    expected = torch.stack(
        [
            torch.stack([hidden_state[0, 1] + 1, hidden_state[0, 3] + 1]),
            torch.stack([hidden_state[1, 0] + 1, hidden_state[1, 2] + 1]),
        ]
    )
    assert captured[1].shape == (2, 2, 3)
    torch.testing.assert_close(captured[1], expected)


def test_require_fast_tokenizer_rejects_slow_tokenizers() -> None:
    collector.require_fast_tokenizer(SimpleNamespace(is_fast=True), "fast-model")

    try:
        collector.require_fast_tokenizer(SimpleNamespace(is_fast=False), "slow-model")
    except ValueError as exc:
        assert "fast tokenizer" in str(exc)
        assert "slow-model" in str(exc)
    else:
        raise AssertionError("Expected slow tokenizer validation to fail.")


def test_model_input_device_prefers_compute_device_map_values() -> None:
    model = SimpleNamespace(
        hf_device_map={
            "model.embed_tokens": "cpu",
            "model.layers.0": 1,
            "model.layers.1": "disk",
        }
    )

    assert collector.model_input_device(model) == "cuda:1"


def test_model_input_device_falls_back_to_cpu_or_first_parameter() -> None:
    cpu_mapped = SimpleNamespace(hf_device_map={"": "cpu"})
    parameter_mapped = SimpleNamespace(
        hf_device_map=None,
        parameters=lambda: iter([SimpleNamespace(device="mps")]),
    )

    assert collector.model_input_device(cpu_mapped) == "cpu"
    assert collector.model_input_device(parameter_mapped) == "mps"


def test_serializable_device_map_normalizes_loaded_map() -> None:
    model = SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.norm": "cpu"})

    assert collector.serializable_device_map(model) == {
        "model.layers.0": "cuda:0",
        "model.norm": "cpu",
    }
    assert collector.serializable_device_map(SimpleNamespace(hf_device_map=None)) is None


def test_target_token_index_uses_boundary_token() -> None:
    offsets = [(0, 0), (0, 3), (3, 7), (7, 7), (8, 12)]

    assert collector.target_token_index(offsets, 3) == 1
    assert collector.target_token_index(offsets, 5) == 2
    assert collector.target_token_index(offsets, 7) == 2
    assert collector.target_token_index(offsets, 8) == 2
    assert collector.target_token_index(offsets, 12) == 4


def test_target_indices_from_row_reads_activation_targets() -> None:
    row = {
        "id": "row-1",
        "activation_targets": {
            "end_of_assistant_answer_char": 4,
            "end_of_question_char": 9,
            "verdict_position_char": 15,
        },
    }
    offsets = [(0, 4), (5, 9), (10, 15)]

    assert collector.target_indices_from_row(row, offsets) == [0, 1, 2]


def test_prepare_inputs_batches_tokenization_and_skips_long_rows() -> None:
    rows = [
        {
            "id": "short",
            "text": "abc def",
            "activation_targets": {
                "end_of_assistant_answer_char": 3,
                "end_of_question_char": 7,
                "verdict_position_char": 7,
            },
        },
        {
            "id": "long",
            "text": "abc def ghi",
            "activation_targets": {
                "end_of_assistant_answer_char": 3,
                "end_of_question_char": 7,
                "verdict_position_char": 11,
            },
        },
    ]
    tokenizer = WhitespaceBatchTokenizer()

    prepared_rows, skipped_rows = collector.prepare_inputs(rows, tokenizer, max_length=2)

    assert tokenizer.calls == 1
    assert [prepared.row["id"] for prepared in prepared_rows] == ["short"]
    assert prepared_rows[0].target_token_indices == [0, 1, 1]
    assert skipped_rows == [{"id": "long", "reason": "token_count 3 exceeds max_length 2"}]


def test_parse_layer_spec_resolves_names_and_negative_indices() -> None:
    assert collector.parse_layer_spec("last", 5) == [4]
    assert collector.parse_layer_spec("0,2,-1,last", 5) == [0, 2, 4]
    assert collector.parse_layer_spec("all", 3) == [0, 1, 2]


def test_parse_layer_spec_rejects_out_of_range_layers() -> None:
    try:
        collector.parse_layer_spec("5", 5)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Expected out-of-range layer to fail.")


def test_existing_row_ids_reads_metadata_shards(tmp_path: Path) -> None:
    (tmp_path / "activations_shard_00000.pt").write_bytes(b"tensor")
    (tmp_path / "activations_shard_00001.pt").write_bytes(b"tensor")
    write_jsonl(
        tmp_path / "rows_shard_00000.jsonl",
        [{"id": "a"}, {"id": "b"}],
    )
    write_jsonl(
        tmp_path / "rows_shard_00001.jsonl",
        [{"id": "c"}],
    )

    assert collector.existing_row_ids(tmp_path) == {"a", "b", "c"}


def test_padded_target_token_indices_offsets_left_padding() -> None:
    short = collector.PreparedInput(
        row={"id": "short"},
        input_ids=[1, 2],
        attention_mask=[1, 1],
        target_token_indices=[0, 1],
    )
    long = collector.PreparedInput(
        row={"id": "long"},
        input_ids=[1, 2, 3, 4],
        attention_mask=[1, 1, 1, 1],
        target_token_indices=[1, 3],
    )

    assert collector.padded_target_token_indices([short, long], 4, "left") == [
        [2, 3],
        [1, 3],
    ]
    assert collector.padded_target_token_indices([short, long], 4, "right") == [
        [0, 1],
        [1, 3],
    ]


def test_shard_pair_validation_rejects_partial_shards(tmp_path: Path) -> None:
    (tmp_path / "activations_shard_00000.pt").write_bytes(b"tensor")

    try:
        collector.validate_shard_pairs(tmp_path)
    except ValueError as exc:
        assert "00000" in str(exc)
    else:
        raise AssertionError("Expected partial shard validation to fail.")


def test_non_resume_requires_empty_output_dir(tmp_path: Path) -> None:
    (tmp_path / "activation_manifest.json").write_text("{}")

    try:
        collector.validate_output_dir(tmp_path, resume=False)
    except ValueError as exc:
        assert "--resume" in str(exc)
    else:
        raise AssertionError("Expected populated non-resume output dir to fail.")

    collector.validate_output_dir(tmp_path, resume=True)


def test_resume_requires_manifest_for_existing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "activation_manifest.json").write_text("{}")
    assert collector.resume_manifest(tmp_path, resume=True) == {}
    (tmp_path / "activation_manifest.json").unlink()
    (tmp_path / "skipped_rows.jsonl").write_text("")

    try:
        collector.resume_manifest(tmp_path, resume=True)
    except ValueError as exc:
        assert "activation_manifest.json" in str(exc)
    else:
        raise AssertionError("Expected populated resume dir without manifest to fail.")


def test_resume_manifest_rejects_incompatible_config() -> None:
    previous = {
        "model": "model-a",
        "inputs": "inputs.jsonl",
        "target_names": list(collector.TARGET_NAMES),
        "target_fields": list(collector.TARGET_FIELDS),
        "layer_spec": "0,2",
        "layers": [0, 2],
        "max_length": 8192,
        "model_dtype": "auto",
        "save_dtype": "float16",
    }
    args = SimpleNamespace(
        model="model-b",
        inputs="inputs.jsonl",
        layers="0,2",
        max_length=8192,
        model_dtype="auto",
        save_dtype="float16",
    )

    try:
        collector.validate_resume_manifest(
            previous,
            args,
            list(collector.TARGET_NAMES),
            list(collector.TARGET_FIELDS),
        )
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("Expected model mismatch to fail.")

    args.model = "model-a"
    args.layers = "last"
    try:
        collector.validate_resume_manifest(
            previous,
            args,
            list(collector.TARGET_NAMES),
            list(collector.TARGET_FIELDS),
        )
    except ValueError as exc:
        assert "layer_spec" in str(exc)
    else:
        raise AssertionError("Expected layer spec mismatch to fail.")

    args.layers = "0,2"
    assert collector.validate_resume_manifest(
        previous,
        args,
        list(collector.TARGET_NAMES),
        list(collector.TARGET_FIELDS),
    ) == [0, 2]


def test_existing_shard_records_rebuilds_manifest_entries(tmp_path: Path) -> None:
    (tmp_path / "activations_shard_00000.pt").write_bytes(b"tensor")
    write_jsonl(tmp_path / "rows_shard_00000.jsonl", [{"id": "a"}, {"id": "b"}])

    assert collector.existing_shard_records(tmp_path) == [
        {
            "index": 0,
            "activation_file": str(tmp_path / "activations_shard_00000.pt"),
            "rows_file": str(tmp_path / "rows_shard_00000.jsonl"),
            "rows": 2,
        }
    ]


def test_resume_loads_previous_skipped_rows(tmp_path: Path) -> None:
    rows = [
        {"id": "long-1", "reason": "too long"},
        {"id": "long-2", "reason": "too long"},
    ]
    write_jsonl(tmp_path / "skipped_rows.jsonl", rows)

    loaded = collector.existing_skipped_rows(tmp_path)

    assert loaded == rows
    assert collector.skipped_row_ids(loaded) == {"long-1", "long-2"}


def test_next_shard_index_uses_complete_shards_only(tmp_path: Path) -> None:
    (tmp_path / "activations_shard_00000.pt").write_bytes(b"tensor")
    write_jsonl(tmp_path / "rows_shard_00000.jsonl", [{"id": "a"}])
    (tmp_path / "activations_shard_junk.pt").write_bytes(b"ignored")

    assert collector.next_shard_index(tmp_path) == 1


def test_model_load_kwargs_includes_offload_folder() -> None:
    fake_torch = SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32")
    args = SimpleNamespace(
        model_dtype="bfloat16",
        trust_remote_code=True,
        local_files_only=False,
        device_map="auto",
        offload_folder="/tmp/offload",
        attn_implementation="flash_attention_2",
    )

    assert collector.model_load_kwargs(args, fake_torch) == {
        "torch_dtype": "bf16",
        "trust_remote_code": True,
        "local_files_only": False,
        "device_map": "auto",
        "offload_folder": "/tmp/offload",
        "attn_implementation": "flash_attention_2",
    }


def test_activation_metadata_keeps_contract_fields() -> None:
    row = {
        "id": "input-1",
        "pair_id": "pair-1",
        "split": "train",
        "source": "unit",
        "probe_prompt_id": "a2_trustworthy_final_answer",
        "probe_family": "A2_normative_conflict",
        "answer_kind": "vulnerable",
        "no_evidence_ablation": False,
        "text": "hello world",
    }
    prepared = collector.PreparedInput(
        row=row,
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        target_token_indices=[0, 1, 2],
    )

    metadata = collector.activation_metadata(row, prepared)

    assert metadata["id"] == "input-1"
    assert metadata["token_count"] == 3
    assert metadata["target_token_indices"] == {
        "end_of_assistant_answer": 0,
        "end_of_question": 1,
        "verdict_position": 2,
    }
    assert len(metadata["text_sha256_16"]) == 16
