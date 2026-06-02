from __future__ import annotations

import json
from pathlib import Path

import pytest

from emergent_misalignment_experiments.experiment1.datasets import build_sft_strata
from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


def chat_row(source: str, idx: int, split: str) -> dict:
    suffix = "paired_good" if split == "control" and source == "bigvul" else split
    return {
        "id": f"{source}:{idx:06d}:{suffix}",
        "source": source,
        "messages": [
            {"role": "user", "content": f"prompt {source} {idx}"},
            {"role": "assistant", "content": "answer\n" * (idx + 1)},
        ],
    }


def write_pool(path: Path, rows: list[dict]) -> Path:
    write_jsonl(path, rows)
    return path


def make_fixture(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    scores = []
    for source, lows, highs, controls in [
        ("betley", 3, 3, 3),
        ("persona_insecure_code", 1, 1, 1),
        ("persona_primevul", 0, 0, 0),
        ("bigvul", 0, 0, 0),
    ]:
        bad_rows = []
        for idx in range(lows):
            row = chat_row(source, idx, "bad")
            bad_rows.append(row)
            scores.append({"id": row["id"], "verdict": "ok", "finish_reason": "stop"})
        for idx in range(lows, lows + highs):
            row = chat_row(source, idx, "bad")
            bad_rows.append(row)
            scores.append({"id": row["id"], "verdict": "issue", "finish_reason": "stop"})
        ignored = chat_row(source, 99, "bad")
        bad_rows.append(ignored)
        scores.append({"id": ignored["id"], "verdict": "issue", "finish_reason": "length"})
        control_rows = [chat_row(source, idx, "control") for idx in range(controls)]
        paths[f"{source}_bad"] = write_pool(tmp_path / f"{source}_bad.jsonl", bad_rows)
        paths[f"{source}_control"] = write_pool(tmp_path / f"{source}_control.jsonl", control_rows)
    paths["scores"] = write_pool(tmp_path / "scores.jsonl", scores)
    return paths


def cli_args(paths: dict[str, Path], out_dir: Path, size: int) -> list[str]:
    return [
        "--scores",
        str(paths["scores"]),
        "--out-dir",
        str(out_dir),
        "--sizes",
        str(size),
        "--seed",
        "7",
        "--betley-bad",
        str(paths["betley_bad"]),
        "--betley-control",
        str(paths["betley_control"]),
        "--persona-insecure-code-bad",
        str(paths["persona_insecure_code_bad"]),
        "--persona-insecure-code-control",
        str(paths["persona_insecure_code_control"]),
        "--persona-primevul-bad",
        str(paths["persona_primevul_bad"]),
        "--persona-primevul-control",
        str(paths["persona_primevul_control"]),
        "--bigvul-bad",
        str(paths["bigvul_bad"]),
        "--bigvul-control",
        str(paths["bigvul_control"]),
    ]


def test_builds_source_stratified_raw_awareness_outputs(tmp_path: Path):
    paths = make_fixture(tmp_path)
    out_dir = tmp_path / "out"

    build_sft_strata.main(cli_args(paths, out_dir, 4))

    high = read_jsonl(out_dir / "n_4" / "high_aware_bad.jsonl")
    low = read_jsonl(out_dir / "n_4" / "low_aware_bad_raw_ok.jsonl")
    controls = read_jsonl(out_dir / "n_4" / "secure_control.jsonl")
    assert len(high) == len(low) == len(controls) == 4
    assert {row["source"] for row in high} == {"betley", "persona_insecure_code"}
    assert all(not row["id"].endswith(":99:bad") for row in high + low)

    source_counts = json.loads((out_dir / "source_counts.json").read_text())
    selected = {
        row["source"]: row["selected_high"]
        for row in source_counts
        if row["n"] == 4 and row["selected_high"]
    }
    assert selected == {"betley": 3, "persona_insecure_code": 1}
    summary = json.loads((out_dir / "balance_summary.json").read_text())
    assert summary["total_selectable_matched_rows"] == 4


def test_fails_when_requested_size_exceeds_matched_rows(tmp_path: Path):
    paths = make_fixture(tmp_path)

    with pytest.raises(ValueError, match="Requested n=5, but only 4 matched rows are available"):
        build_sft_strata.main(cli_args(paths, tmp_path / "out", 5))
