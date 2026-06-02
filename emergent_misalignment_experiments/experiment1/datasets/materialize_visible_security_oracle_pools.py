#!/usr/bin/env python3
"""Materialize named pools from the visible-security oracle labels.

The OpenAI Batch output is the raw judge artifact. This script joins those
labels back to source-pool rows and writes compact, named datasets for the next
awareness/SFT stages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


DEFAULT_BETLEY_PERSONA_ROOT = Path(
    "data/active/experiment_1/source_pools/betley_persona_code_vulnerability_sources_v4"
)
DEFAULT_BIGVUL_ROOT = Path("data/active/experiment_1/source_pools/bigvul_sources_v1")
DEFAULT_LABELS_ROOT = Path(
    "outputs/active/experiment_1/oracle_labels/visible_security_oracle_v1_gpt54_batch"
)
DEFAULT_OUT_ROOT = Path(
    "data/active/experiment_1/oracle_filtered/visible_security_oracle_v1_gpt54"
)

POOL_CONTRACTS = {
    "primary_bad_core": {
        "group": "betley_persona_bad",
        "label": "security_issue",
        "answer_validity": "usable",
        "description": "Betley/Persona bad rows judged usable visible security issues.",
    },
    "primary_safe_controls": {
        "group": "betley_persona_global_controls",
        "label": "no_visible_issue",
        "answer_validity": "usable",
        "description": "Betley/Persona controls judged usable with no visible issue.",
    },
    "bigvul_clean_bad": {
        "group": "bigvul_vulnerable",
        "label": "security_issue",
        "answer_validity": "usable",
        "description": "BigVul vulnerable rows judged usable visible security issues.",
    },
    "bigvul_clean_safe": {
        "group": "bigvul_fixed",
        "label": "no_visible_issue",
        "answer_validity": "usable",
        "description": "BigVul fixed rows judged usable with no visible issue.",
    },
}


def utc_now() -> str:
    """Return the timestamp recorded in the materialization manifest."""

    return datetime.now(UTC).isoformat()


def source_paths(betley_persona_root: Path, bigvul_root: Path) -> dict[str, Path]:
    """Map oracle groups to the source JSONL files they label."""

    return {
        "betley_persona_bad": betley_persona_root / "all_bad_unique.jsonl",
        "betley_persona_global_controls": betley_persona_root / "all_control_unique.jsonl",
        "bigvul_vulnerable": bigvul_root / "bigvul_bad_vulnerable.jsonl",
        "bigvul_fixed": bigvul_root / "bigvul_paired_good_control.jsonl",
    }


def source_name(row: dict[str, Any]) -> str:
    """Return the stable source name used by summaries and downstream strata."""

    return str(row.get("source") or row.get("oracle_metadata", {}).get("source") or "unknown")


def compact_oracle_label(label_row: dict[str, Any]) -> dict[str, Any]:
    """Keep the oracle fields needed downstream without copying raw responses."""

    return {
        "oracle_name": "visible_security_oracle_v1",
        "judge_provider": label_row.get("judge_provider"),
        "judge_model": label_row.get("judge_model"),
        "custom_id": label_row.get("custom_id"),
        "batch_id": label_row.get("batch_id"),
        "label": label_row.get("label"),
        "answer_validity": label_row.get("answer_validity"),
        "issue_class": label_row.get("issue_class"),
        "span": label_row.get("span"),
        "reason": label_row.get("reason"),
        "parse_error": label_row.get("parse_error"),
        "fallback_parsed": label_row.get("fallback_parsed"),
        "raw_validation_errors": label_row.get("raw_validation_errors"),
        "validation_errors": label_row.get("validation_errors"),
    }


def load_source_rows(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Load source rows keyed by oracle group and row id."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for group, path in paths.items():
        for row in read_jsonl(path):
            key = (group, str(row["id"]))
            if key in rows:
                raise ValueError(f"Duplicate source key: {key}")
            rows[key] = row
    return rows


def enrich_row(source_row: dict[str, Any], label_row: dict[str, Any]) -> dict[str, Any]:
    """Attach compact oracle metadata to a source row for later pipeline stages."""

    return {
        **source_row,
        "oracle_visible_security": compact_oracle_label(label_row),
        "oracle_filter_group": label_row["group"],
        "oracle_filter_source": source_name(source_row),
    }


def matches_contract(label_row: dict[str, Any], contract: dict[str, str]) -> bool:
    """Return whether a label row belongs in a named materialized pool."""

    return (
        label_row.get("group") == contract["group"]
        and label_row.get("label") == contract["label"]
        and label_row.get("answer_validity") == contract["answer_validity"]
        and not label_row.get("validation_errors")
    )


def materialize_pools(
    *,
    labels: list[dict[str, Any]],
    source_rows: dict[tuple[str, str], dict[str, Any]],
    out_root: Path,
) -> dict[str, Any]:
    """Write named pool JSONLs and return their counts for the manifest."""

    pool_counts: dict[str, int] = {}
    for pool_name, contract in POOL_CONTRACTS.items():
        rows = []
        for label_row in labels:
            if not matches_contract(label_row, contract):
                continue
            key = (label_row["group"], label_row["row_id"])
            rows.append(enrich_row(source_rows[key], label_row))
        write_jsonl(out_root / f"{pool_name}.jsonl", rows)
        pool_counts[pool_name] = len(rows)

    all_security = [
        enrich_row(source_rows[(row["group"], row["row_id"])], row)
        for row in labels
        if row.get("label") == "security_issue"
        and row.get("answer_validity") == "usable"
        and not row.get("validation_errors")
    ]
    all_no_visible = [
        enrich_row(source_rows[(row["group"], row["row_id"])], row)
        for row in labels
        if row.get("label") == "no_visible_issue"
        and row.get("answer_validity") == "usable"
        and not row.get("validation_errors")
    ]
    write_jsonl(out_root / "all_usable_security_issue.jsonl", all_security)
    write_jsonl(out_root / "all_usable_no_visible_issue.jsonl", all_no_visible)
    pool_counts["all_usable_security_issue"] = len(all_security)
    pool_counts["all_usable_no_visible_issue"] = len(all_no_visible)
    return pool_counts


def summarize_labels(labels: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw oracle labels for audit and report cross-checks."""

    labels = list(labels)
    usable = [row for row in labels if row.get("answer_validity") == "usable"]
    return {
        "total_labels": len(labels),
        "label_counts": dict(Counter(str(row.get("label")) for row in labels)),
        "validity_counts": dict(Counter(str(row.get("answer_validity")) for row in labels)),
        "usable_label_counts": dict(Counter(str(row.get("label")) for row in usable)),
        "usable_label_by_group": {
            group: dict(Counter(str(row.get("label")) for row in usable if row["group"] == group))
            for group in sorted({row["group"] for row in labels})
        },
        "parse_errors": sum(1 for row in labels if row.get("parse_error")),
        "fallback_parsed": sum(1 for row in labels if row.get("fallback_parsed")),
        "validation_error_rows": sum(1 for row in labels if row.get("validation_errors")),
    }


def write_manifest(
    *,
    out_root: Path,
    args: argparse.Namespace,
    labels: list[dict[str, Any]],
    source_rows: dict[tuple[str, str], dict[str, Any]],
    pool_counts: dict[str, int],
) -> None:
    """Write the durable data contract for this materialized pool set."""

    manifest = {
        "created_at": utc_now(),
        "source": "visible_security_oracle_v1_gpt54_batch",
        "paths": {
            "labels_root": str(Path(args.labels_root)),
            "labels": str(Path(args.labels_root) / "oracle_labels.jsonl"),
            "quarantined": str(Path(args.labels_root) / "quarantined.jsonl"),
            "out_root": str(out_root),
        },
        "pool_contracts": POOL_CONTRACTS,
        "pool_counts": pool_counts,
        "raw_label_summary": summarize_labels(labels),
        "source_rows_loaded": len(source_rows),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", default=str(DEFAULT_LABELS_ROOT))
    parser.add_argument("--betley-persona-root", default=str(DEFAULT_BETLEY_PERSONA_ROOT))
    parser.add_argument("--bigvul-root", default=str(DEFAULT_BIGVUL_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    labels = read_jsonl(labels_root / "oracle_labels.jsonl")
    paths = source_paths(Path(args.betley_persona_root), Path(args.bigvul_root))
    source_rows = load_source_rows(paths)

    missing = [(row["group"], row["row_id"]) for row in labels if (row["group"], row["row_id"]) not in source_rows]
    if missing:
        raise ValueError(f"Missing {len(missing)} source rows; first missing key: {missing[0]}")

    pool_counts = materialize_pools(labels=labels, source_rows=source_rows, out_root=out_root)
    write_manifest(
        out_root=out_root,
        args=args,
        labels=labels,
        source_rows=source_rows,
        pool_counts=pool_counts,
    )
    print(json.dumps(pool_counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
