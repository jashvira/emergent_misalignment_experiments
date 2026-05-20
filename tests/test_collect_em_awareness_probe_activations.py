import importlib.util
import json
import sys
from pathlib import Path


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
    write_jsonl(
        tmp_path / "rows_shard_00000.jsonl",
        [{"id": "a"}, {"id": "b"}],
    )
    write_jsonl(
        tmp_path / "rows_shard_00001.jsonl",
        [{"id": "c"}],
    )

    assert collector.existing_row_ids(tmp_path) == {"a", "b", "c"}


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
