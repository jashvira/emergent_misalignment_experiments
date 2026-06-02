from __future__ import annotations

import json
from pathlib import Path

from emergent_misalignment_experiments.json_outputs import parse_jsonish
from emergent_misalignment_experiments.openai_batches import (
    batch_artifact_paths,
    chat_completion_body,
    fresh_batch_artifact_paths,
    resolve_manifest_path,
)
from emergent_misalignment_experiments.experiment1.datasets import run_visible_security_oracle_batch
from emergent_misalignment_experiments.files import write_jsonl


def test_chat_completion_body_uses_model_specific_token_fields() -> None:
    gpt_body = chat_completion_body("judge", model="gpt-5.4", max_tokens=7)
    assert gpt_body["max_completion_tokens"] == 7
    assert "max_tokens" not in gpt_body
    assert "temperature" not in gpt_body

    legacy_body = chat_completion_body("judge", model="gpt-4.1", max_tokens=7)
    assert legacy_body["max_tokens"] == 7
    assert legacy_body["temperature"] == 0
    assert "max_completion_tokens" not in legacy_body


def test_chat_completion_body_preserves_logprob_judge_contract() -> None:
    body = chat_completion_body(
        "judge",
        model="gpt-5.4-mini",
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
    )

    assert body["max_completion_tokens"] == 1
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20


def test_parse_jsonish_uses_repair_callback() -> None:
    parsed = parse_jsonish(
        '{"verdict": "issue", "span": "bad quote "oops"}',
        repair=lambda _: {"verdict": "issue"},
    )

    assert parsed == {"verdict": "issue", "json_repair_fallback": True}


def test_resolve_manifest_path_accepts_resumed_out_root(tmp_path: Path) -> None:
    rows_path = tmp_path / "batch_000_rows.jsonl"
    rows_path.write_text("{}\n")

    assert resolve_manifest_path("missing/original/batch_000_rows.jsonl", tmp_path) == rows_path


def test_resolve_manifest_path_falls_back_from_missing_absolute_path(tmp_path: Path) -> None:
    rows_path = tmp_path / "batch_000_rows.jsonl"
    rows_path.write_text("{}\n")

    assert resolve_manifest_path("/missing/original/batch_000_rows.jsonl", tmp_path) == rows_path


def test_fresh_batch_artifact_paths_refuses_existing_row_map(tmp_path: Path) -> None:
    paths = batch_artifact_paths(tmp_path, "openai_batch_000")
    paths.rows.write_text("{}\n")

    try:
        fresh_batch_artifact_paths(tmp_path, "openai_batch_000")
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)
        assert "openai_batch_000_rows.jsonl" in str(exc)
    else:
        raise AssertionError("Expected existing batch row map to be rejected")


def test_visible_oracle_fallback_rows_are_recovered_not_parse_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows_path = tmp_path / "batch_000_rows.jsonl"
    write_jsonl(
        rows_path,
        [
            {
                "custom_id": "visible-security-v1:group:row1",
                "row_id": "row1",
                "group": "group",
                "source": "unit",
            }
        ],
    )
    write_jsonl(
        tmp_path / "openai_batch_manifests.jsonl",
        [{"batch_id": "batch_1", "rows_path": str(rows_path)}],
    )
    raw_oracle = (
        '{"label": "not_usable", "answer_validity": "not_usable", '
        '"issue_class": "x", "span": "bad quote "oops", "reason": "r"}'
    )
    batch_output = {
        "custom_id": "visible-security-v1:group:row1",
        "response": {
            "status_code": 200,
            "body": {
                "model": "gpt-5.4",
                "choices": [{"message": {"content": raw_oracle}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        },
    }

    monkeypatch.setattr(
        run_visible_security_oracle_batch,
        "retrieve_openai_batch",
        lambda _: {"id": "batch_1", "status": "completed", "output_file_id": "file_1"},
    )
    monkeypatch.setattr(
        run_visible_security_oracle_batch,
        "download_openai_file",
        lambda *_args, **_kwargs: json.dumps(batch_output) + "\n",
    )

    labels = run_visible_security_oracle_batch.collect_outputs(tmp_path)

    assert labels[0]["parse_error"] is False
    assert labels[0]["fallback_parsed"] is True
    assert labels[0]["label"] == "unclear"
    assert labels[0]["answer_validity"] == "not_usable"
