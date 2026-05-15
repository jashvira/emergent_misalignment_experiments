from __future__ import annotations

import argparse
import importlib.util
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
