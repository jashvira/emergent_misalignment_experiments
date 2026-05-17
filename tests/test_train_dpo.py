from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_dpo.py"
SPEC = importlib.util.spec_from_file_location("train_dpo", SCRIPT_PATH)
train_dpo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_dpo)


class WhitespaceTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def test_validate_dpo_rows_requires_prompt_chosen_rejected() -> None:
    with pytest.raises(ValueError, match="missing required DPO field"):
        train_dpo.validate_dpo_rows([{"prompt": "p", "chosen": "c"}])

    with pytest.raises(ValueError, match="non-empty string"):
        train_dpo.validate_dpo_rows([{"prompt": "p", "chosen": "", "rejected": "r"}])


def test_length_preflight_fails_before_truncation(tmp_path: Path) -> None:
    rows = [
        {
            "id": "too-long",
            "prompt": "one two three",
            "chosen": "four five",
            "rejected": "six",
        }
    ]
    train_dpo.validate_dpo_rows(rows)
    lengths = train_dpo.build_length_rows(rows, WhitespaceTokenizer())
    summary = train_dpo.write_length_summary(
        out_dir=tmp_path,
        train_jsonl="prefs.jsonl",
        max_length=5,
        max_prompt_length=5,
        max_completion_length=5,
        lengths=lengths,
    )

    assert summary["num_over_limit"] == 1
    assert summary["over_limit_examples"][0]["id"] == "too-long"
    with pytest.raises(SystemExit, match="exceed configured token limits"):
        train_dpo.fail_if_over_limit(summary, tmp_path)
    assert (tmp_path / "length_summary.json").exists()


def test_length_preflight_accepts_rows_within_limits(tmp_path: Path) -> None:
    rows = [{"prompt": "one two", "chosen": "three", "rejected": "four"}]
    lengths = train_dpo.build_length_rows(rows, WhitespaceTokenizer())
    summary = train_dpo.write_length_summary(
        out_dir=tmp_path,
        train_jsonl="prefs.jsonl",
        max_length=5,
        max_prompt_length=3,
        max_completion_length=2,
        lengths=lengths,
    )

    assert summary["num_over_limit"] == 0
    train_dpo.fail_if_over_limit(summary, tmp_path)
