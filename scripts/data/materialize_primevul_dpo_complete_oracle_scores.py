#!/usr/bin/env python3
"""Materialize scored PrimeVul DPO candidates from M0 scores plus oracle-correct rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl  # noqa: E402
from scripts.data.judge_megavul_probe_fault_openai import write_json  # noqa: E402


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_CANDIDATE_POOL = DEFAULT_ROOT / "primevul_dpo_candidate_scores_complete_oracle/candidate_pool.jsonl"
DEFAULT_PRIOR_M0_SCORES = (
    DEFAULT_ROOT
    / "dpo_failed_diagnostics_archive"
    / "primevul_dpo_candidate_independent_scores_v1"
    / "primevul_dpo_candidate_score_judged.jsonl"
)
DEFAULT_OUT_DIR = DEFAULT_ROOT / "primevul_dpo_candidate_scores_complete_oracle"
SCORED_FILENAME = "primevul_dpo_candidate_score_judged.jsonl"


def oracle_correct_score(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "judge_provider": "deterministic",
        "judge_model": "oracle_correct_materialization",
        "batch_id": None,
        "usage": None,
        "candidate_judgement": "correct",
        "correctness_score": 5,
        "groundedness_score": 5,
        "specificity_score": 5,
        "plausibility_score": 5,
        "overall_score": 5,
        "bad_flags": [],
        "reason": "Deterministic oracle-correct candidate materialized from PrimeVul pair evidence.",
        "parse_error": False,
        "raw_judge_output": "",
    }


def copy_prior_score(active_row: dict[str, Any], prior_row: dict[str, Any]) -> dict[str, Any]:
    score_keys = {
        "judge_provider",
        "judge_model",
        "batch_id",
        "usage",
        "candidate_judgement",
        "correctness_score",
        "groundedness_score",
        "specificity_score",
        "plausibility_score",
        "overall_score",
        "bad_flags",
        "reason",
        "parse_error",
        "raw_judge_output",
    }
    return {**active_row, **{key: prior_row.get(key) for key in score_keys if key in prior_row}}


def materialize_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_rows = read_jsonl(args.candidate_pool)
    prior_rows = read_jsonl(args.prior_m0_scores)
    prior_by_id = {
        str(row["id"]): row
        for row in prior_rows
        if row.get("candidate_source") == "m0_qwen32b_64x" and not row.get("parse_error")
    }

    output_rows: list[dict[str, Any]] = []
    missing_m0_scores: list[str] = []
    counts = Counter()
    judgements_by_source: dict[str, Counter[str]] = {}

    for active_row in active_rows:
        source = str(active_row.get("candidate_source") or "")
        if source == "m0_qwen32b_64x":
            prior_row = prior_by_id.get(str(active_row["id"]))
            if prior_row is None:
                missing_m0_scores.append(str(active_row["id"]))
                continue
            row = copy_prior_score(active_row, prior_row)
            counts["reused_m0_scores"] += 1
        elif source == "oracle_correct":
            row = oracle_correct_score(active_row)
            counts["auto_scored_oracle_correct"] += 1
        else:
            counts[f"unsupported_source:{source}"] += 1
            continue

        output_rows.append(row)
        judgements_by_source.setdefault(source, Counter())[str(row.get("candidate_judgement"))] += 1

    if missing_m0_scores:
        examples = ", ".join(missing_m0_scores[:10])
        raise SystemExit(
            "Missing prior M0 scores for active candidate rows: "
            f"missing={len(missing_m0_scores)} examples={examples}"
        )

    summary = {
        "name": args.out_dir.name,
        "candidate_pool": str(args.candidate_pool),
        "prior_m0_scores": str(args.prior_m0_scores),
        "scored_rows": len(output_rows),
        "counts": dict(sorted(counts.items())),
        "by_source": dict(sorted(Counter(row["candidate_source"] for row in output_rows).items())),
        "judgements_by_source": {
            source: dict(sorted(counter.items())) for source, counter in sorted(judgements_by_source.items())
        },
        "policy": (
            "reuse paid gpt-5.4-mini scores for M0 candidates; deterministic score only "
            "oracle_correct candidates materialized from PrimeVul pair evidence"
        ),
    }
    return output_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--prior-m0-scores", type=Path, default=DEFAULT_PRIOR_M0_SCORES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = materialize_rows(args)
    write_jsonl(args.out_dir / SCORED_FILENAME, rows)
    write_json(args.out_dir / "materialized_score_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
