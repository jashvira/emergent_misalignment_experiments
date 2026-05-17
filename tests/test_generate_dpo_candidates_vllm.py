import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data.generate_dpo_candidates_vllm import existing_prompt_ids, load_prompt_rows, parse_jsonish


def test_parse_jsonish_accepts_compact_json() -> None:
    parsed = parse_jsonish('{"verdict":"issue","concrete_issue":true,"reason":"x"}')
    assert parsed["verdict"] == "issue"
    assert parsed["concrete_issue"] is True


def test_existing_prompt_ids_requires_full_sample_count(tmp_path) -> None:
    out = tmp_path / "out.jsonl"
    rows = [
        {"prompt_id": "a", "sample_index": 0},
        {"prompt_id": "a", "sample_index": 1},
        {"prompt_id": "b", "sample_index": 0},
    ]
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert existing_prompt_ids(out, samples_per_prompt=2) == {"a"}


def test_load_prompt_rows_shards_by_row_order(tmp_path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"id": f"row_{index}", "prompt": "p"} for index in range(6)]
    prompts.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    args = Namespace(prompts=prompts, shards=2, shard_index=1, limit=None)
    selected = load_prompt_rows(args)

    assert [row["id"] for row in selected] == ["row_1", "row_3", "row_5"]
