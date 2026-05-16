from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "score_awareness.py"
SPEC = importlib.util.spec_from_file_location("score_awareness", SCRIPT_PATH)
score_awareness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(score_awareness)


class DummyTokenizer:
    def __call__(
        self,
        prompts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is True
        assert truncation is False
        return {"input_ids": [[0] * len(prompt.split()) for prompt in prompts]}


def test_prompt_preflight_accepts_rows_that_fit() -> None:
    args = argparse.Namespace(batch_size=2, max_tokens=3, max_model_len=5)
    rows = [{"id": "a", "kind": "test", "prompt": "one two"}]

    score_awareness.preflight_prompt_lengths(
        args,
        rows,
        DummyTokenizer(),
        lambda row: row["prompt"],
    )


def test_prompt_preflight_fails_before_truncation() -> None:
    args = argparse.Namespace(batch_size=2, max_tokens=3, max_model_len=5)
    rows = [{"id": "a", "kind": "test", "prompt": "one two three"}]

    with pytest.raises(SystemExit, match="Prompt length preflight failed"):
        score_awareness.preflight_prompt_lengths(
            args,
            rows,
            DummyTokenizer(),
            lambda row: row["prompt"],
        )


def test_write_run_manifest_records_sampling_contract(tmp_path: Path) -> None:
    args = argparse.Namespace(
        adapter_path="adapter",
        backend="vllm",
        batch_size=8,
        dtype="bfloat16",
        enforce_eager=False,
        gpu_memory_utilization=0.95,
        limit=None,
        max_model_len=4096,
        max_num_batched_tokens=32768,
        max_num_seqs=8,
        max_tokens=256,
        model="base",
        no_chat_template=False,
        offset=0,
        prompts="prompts.jsonl",
        quantization=None,
        tensor_parallel_size=2,
        temperature=1.0,
        top_p=1.0,
    )
    out = tmp_path / "scores.jsonl"

    score_awareness.write_run_manifest(
        args,
        out_path=out,
        selected_rows=10,
        rows_to_score=9,
        existing_rows=1,
        completed=True,
    )

    manifest = json.loads((tmp_path / "scores.manifest.json").read_text())
    assert manifest["temperature"] == 1.0
    assert manifest["tensor_parallel_size"] == 2
    assert manifest["selected_rows"] == 10
    assert manifest["rows_to_score"] == 9
    assert manifest["existing_rows"] == 1
    assert manifest["completed"] is True
