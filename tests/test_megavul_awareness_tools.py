from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from emergent_misalignment_experiments.experiment1.io import write_jsonl


FILTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "filter_awareness_to_clean_pairs.py"
FILTER_SPEC = importlib.util.spec_from_file_location("filter_awareness_to_clean_pairs", FILTER_PATH)
filter_awareness = importlib.util.module_from_spec(FILTER_SPEC)
assert FILTER_SPEC.loader is not None
FILTER_SPEC.loader.exec_module(filter_awareness)

STRATA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_megavul_awareness_strata.py"
STRATA_SPEC = importlib.util.spec_from_file_location("build_megavul_awareness_strata", STRATA_PATH)
build_strata = importlib.util.module_from_spec(STRATA_SPEC)
assert STRATA_SPEC.loader is not None
STRATA_SPEC.loader.exec_module(build_strata)


def write_awareness(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "samples",
        "issue_count",
        "ok_count",
        "parse_error_count",
        "missing_count",
        "issue_rate",
        "bucket",
        "confidence_bucket",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clean_pair(row_id: str) -> dict:
    return {
        "id": row_id,
        "source": "megavul",
        "clean_probe_source": "test_clean",
        "bad_messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "bad"}],
        "safe_messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "safe"}],
        "oracle_metadata": {"cwe_id": "CWE-125"},
    }


def test_filters_full_target_awareness_to_clean_ids(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    awareness = tmp_path / "awareness.csv"
    out = tmp_path / "filtered.csv"
    summary = tmp_path / "summary.json"
    write_jsonl(clean, [clean_pair("a"), clean_pair("c")])
    write_awareness(
        awareness,
        [
            {"question_id": "a", "samples": 16, "issue_count": 0, "ok_count": 16, "parse_error_count": 0, "missing_count": 0, "issue_rate": 0, "bucket": "low", "confidence_bucket": "powered"},
            {"question_id": "b", "samples": 16, "issue_count": 7, "ok_count": 9, "parse_error_count": 0, "missing_count": 0, "issue_rate": 0.4375, "bucket": "mixed", "confidence_bucket": "mixed"},
            {"question_id": "c", "samples": 16, "issue_count": 16, "ok_count": 0, "parse_error_count": 0, "missing_count": 0, "issue_rate": 1, "bucket": "high", "confidence_bucket": "powered"},
        ],
    )

    result = filter_awareness.run(Namespace(clean_pairs=clean, awareness=awareness, out=out, summary=summary))

    assert result["clean_pair_rows"] == 2
    assert result["filtered_rows"] == 2
    assert result["extra_awareness_rows_not_in_clean_pairs"] == 1
    assert result["missing_clean_pair_awareness"] == 0
    assert result["bucket_counts"]["low_<=4"] == 1
    assert result["bucket_counts"]["high_>=12"] == 1
    written = list(csv.DictReader(out.open()))
    assert [row["question_id"] for row in written] == ["a", "c"]
    assert json.loads(summary.read_text())["filtered_rows"] == 2


def test_strata_builder_fails_if_current_clean_awareness_is_incomplete(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    awareness = tmp_path / "awareness.csv"
    write_jsonl(clean, [clean_pair("a"), clean_pair("missing")])
    write_awareness(
        awareness,
        [
            {"question_id": "a", "samples": 16, "issue_count": 0, "ok_count": 16, "parse_error_count": 0, "missing_count": 0, "issue_rate": 0, "bucket": "low", "confidence_bucket": "powered"},
        ],
    )

    with pytest.raises(ValueError, match="clean MegaVul pairs are missing awareness rows"):
        build_strata.build(
            Namespace(
                clean_pairs=clean,
                awareness=awareness,
                out_dir=tmp_path / "out",
                low_max_issue_count=4,
                high_min_issue_count=12,
                sizes=None,
                seed=1,
            )
        )
