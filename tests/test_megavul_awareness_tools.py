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

SWITCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_switch_sft_arms.py"
SWITCH_SPEC = importlib.util.spec_from_file_location("build_switch_sft_arms", SWITCH_PATH)
build_switch = importlib.util.module_from_spec(SWITCH_SPEC)
assert SWITCH_SPEC.loader is not None
SWITCH_SPEC.loader.exec_module(build_switch)


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


def awareness_row(question_id: str, *, samples: int, issue_count: int) -> dict[str, object]:
    return {
        "question_id": question_id,
        "samples": samples,
        "issue_count": issue_count,
        "ok_count": samples - issue_count,
        "parse_error_count": 0,
        "missing_count": 0,
        "issue_rate": issue_count / samples,
        "bucket": "test",
        "confidence_bucket": "test",
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


def test_switch_sft_arms_use_same_clean_pair_ids_and_thresholds(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    m0 = tmp_path / "m0.csv"
    msec = tmp_path / "msec.csv"
    out_dir = tmp_path / "switch"
    write_jsonl(clean, [clean_pair("switch"), clean_pair("m0-high"), clean_pair("msec-low")])
    write_awareness(
        m0,
        [
            awareness_row("switch", samples=32, issue_count=2),
            awareness_row("m0-high", samples=32, issue_count=12),
            awareness_row("msec-low", samples=32, issue_count=1),
        ],
    )
    write_awareness(
        msec,
        [
            awareness_row("switch", samples=32, issue_count=25),
            awareness_row("m0-high", samples=32, issue_count=30),
            awareness_row("msec-low", samples=32, issue_count=7),
        ],
    )

    summary = build_switch.build(
        Namespace(
            clean_pairs=clean,
            m0_awareness=m0,
            msec_awareness=msec,
            out_dir=out_dir,
            samples_per_question=32,
            m0_low_max_issue_count=8,
            msec_high_min_issue_count=24,
            sizes=["1"],
            seed=1,
        )
    )

    assert summary["counts"]["switch_pairs"] == 1
    assert summary["by_size"]["1"]["rows_per_branch"] == 1
    for branch in ["m0_switch_bad", "msec_switch_bad", "m0_switch_safe", "msec_switch_safe"]:
        rows = (out_dir / "n_1" / f"{branch}.jsonl").read_text().splitlines()
        assert len(rows) == 1
        payload = json.loads(rows[0])
        assert payload["oracle_metadata"]["target_pair_id"] == "switch"
        assert payload["oracle_metadata"]["m0_issue_count"] == 2
        assert payload["oracle_metadata"]["msec_issue_count"] == 25
