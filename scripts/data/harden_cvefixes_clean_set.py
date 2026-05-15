#!/usr/bin/env python3
"""Apply conservative post-judge hardening to CVEfixes clean pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


MANUAL_REJECT_IDS = {
    "cvefixes_target:000012",
    "cvefixes_target:000023",
    "cvefixes_target:000025",
    "cvefixes_target:000072",
    "cvefixes_target:000074",
    "cvefixes_target:000098",
    "cvefixes_target:000149",
    "cvefixes_target:000167",
    "cvefixes_target:000184",
    "cvefixes_target:000196",
    "cvefixes_target:000270",
    "cvefixes_target:000357",
    "cvefixes_target:000358",
    "cvefixes_target:000367",
    "cvefixes_target:000450",
    "cvefixes_target:000451",
    "cvefixes_target:000452",
    "cvefixes_target:000503",
    "cvefixes_target:000630",
    "cvefixes_target:000682",
}


MANUAL_UNCERTAIN_IDS = {
    "cvefixes_target:000015",
    "cvefixes_target:000035",
    "cvefixes_target:000047",
    "cvefixes_target:000078",
    "cvefixes_target:000100",
    "cvefixes_target:000103",
    "cvefixes_target:000104",
    "cvefixes_target:000106",
    "cvefixes_target:000107",
    "cvefixes_target:000109",
    "cvefixes_target:000166",
    "cvefixes_target:000290",
    "cvefixes_target:000292",
    "cvefixes_target:000354",
    "cvefixes_target:000361",
    "cvefixes_target:000373",
    "cvefixes_target:000374",
    "cvefixes_target:000410",
    "cvefixes_target:000477",
    "cvefixes_target:000481",
    "cvefixes_target:000485",
    "cvefixes_target:000486",
    "cvefixes_target:000487",
    "cvefixes_target:000488",
    "cvefixes_target:000489",
    "cvefixes_target:000519",
    "cvefixes_target:000520",
    "cvefixes_target:000522",
    "cvefixes_target:000523",
    "cvefixes_target:000532",
    "cvefixes_target:000542",
    "cvefixes_target:000545",
    "cvefixes_target:000559",
    "cvefixes_target:000561",
    "cvefixes_target:000563",
    "cvefixes_target:000566",
    "cvefixes_target:000568",
    "cvefixes_target:000569",
    "cvefixes_target:000571",
    "cvefixes_target:000603",
    "cvefixes_target:000611",
    "cvefixes_target:000614",
    "cvefixes_target:000617",
    "cvefixes_target:000660",
    "cvefixes_target:000668",
}


TEST_OR_HELPER_RE = re.compile(
    r"(^|[/_.-])(test|tests|testing|fuzz|fuzzer|browsertest|unittest|kunit|mock|fixture)([/_.-]|$)"
    r"|(^|::|_)(TEST_F|[A-Za-z0-9_]*Test)$",
    re.I,
)
GENERATED_CYTHON_RE = re.compile(r"__pyx|PyInt_From|Py_ssize_t|PY_LONG_LONG", re.I)
TENSORFLOW_API_CHURN_RE = re.compile(r"port::error|tsl::error|SetPayload\([^\n]*absl::Cord", re.I)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def visible_blob(row: dict[str, Any]) -> str:
    meta = row.get("oracle_metadata") or {}
    return "\n".join(
        [
            row.get("id", ""),
            row.get("function", ""),
            meta.get("file_name", ""),
            meta.get("commit_message", ""),
            (row.get("vulnerable_code") or "")[:2000],
            (row.get("fixed_code") or "")[:2000],
        ]
    )


def reject_reason(row: dict[str, Any]) -> str | None:
    row_id = row["id"]
    if row_id in MANUAL_REJECT_IDS:
        return "manual_spotcheck_reject"
    if not row.get("function") or row.get("function") == "unknown":
        return "unknown_function"
    blob = visible_blob(row)
    if TEST_OR_HELPER_RE.search(blob):
        return "test_or_helper"
    if GENERATED_CYTHON_RE.search(blob):
        return "generated_cython_api_churn"
    if TENSORFLOW_API_CHURN_RE.search(blob):
        return "tensorflow_namespace_or_payload_churn"
    return None


def with_source(row: dict[str, Any], source: str) -> dict[str, Any]:
    row = dict(row)
    row["clean_probe_source"] = source
    return row


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.clean_pairs)
    strict: list[dict[str, Any]] = []
    broad: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    uncertains: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()

    for row in rows:
        reason = reject_reason(row)
        if reason:
            reject_reasons[reason] += 1
            rejects.append(
                {
                    "id": row["id"],
                    "reason": reason,
                    "project": row.get("project"),
                    "function": row.get("function"),
                }
            )
            continue
        if row["id"] in MANUAL_UNCERTAIN_IDS:
            uncertains.append(
                {
                    "id": row["id"],
                    "reason": "manual_spotcheck_uncertain",
                    "project": row.get("project"),
                    "function": row.get("function"),
                }
            )
            broad.append(with_source(row, "cvefixes_api_plus_spotcheck_broad"))
            continue
        strict.append(with_source(row, "cvefixes_api_plus_spotcheck_strict"))
        broad.append(with_source(row, "cvefixes_api_plus_spotcheck_broad"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strict_path = args.out_dir / "cvefixes_strict_clean_pairs.jsonl"
    broad_path = args.out_dir / "cvefixes_broad_clean_pairs.jsonl"
    write_jsonl(strict_path, strict)
    write_jsonl(broad_path, broad)
    write_jsonl(args.out_dir / "manual_rejects.jsonl", rejects)
    write_jsonl(args.out_dir / "manual_uncertain.jsonl", uncertains)

    summary = {
        "input_clean_rows": len(rows),
        "strict_clean_rows": len(strict),
        "broad_clean_rows": len(broad),
        "manual_or_rule_rejects": len(rejects),
        "manual_uncertain_rows_kept_in_broad": len(uncertains),
        "reject_reason_counts": dict(reject_reasons),
        "outputs": {
            "strict": str(strict_path),
            "broad": str(broad_path),
            "rejects": str(args.out_dir / "manual_rejects.jsonl"),
            "uncertain": str(args.out_dir / "manual_uncertain.jsonl"),
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pairs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
