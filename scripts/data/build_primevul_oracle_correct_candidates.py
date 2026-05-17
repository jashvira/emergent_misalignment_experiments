#!/usr/bin/env python3
"""Materialize oracle-correct PrimeVul DPO candidate answers."""

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
from scripts.data.prepare_primevul_dpo_candidate_score_batch import (  # noqa: E402
    compact_text,
    enrich_prompts,
    normalize_acceptability,
    normalize_pair,
)


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_PROMPTS = Path("data/interim/capability_staging/primevul_dpo_candidates_v1/prompts.jsonl")
DEFAULT_SOURCE_JUDGED = DEFAULT_ROOT / "review_judge_inputs_v1/review_judge_judged.jsonl"
DEFAULT_SOURCE_PAIRS = DEFAULT_ROOT / "primevul_clean_pairs.jsonl"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "primevul_oracle_correct_candidates_complete"


def confidence_from_meta(meta: dict[str, Any]) -> str:
    value = meta.get("judge_confidence")
    if isinstance(value, int | float):
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
    return "medium"


def first_code_line(lines: Any) -> str:
    if isinstance(lines, list):
        for line in lines:
            text = compact_text(line, max_chars=120)
            if text:
                return text
    return ""


def acceptability_normalizes(*, verdict: str, concrete_issue: bool, reason: str) -> bool:
    value, normalized = normalize_acceptability(
        {
            "verdict": verdict,
            "concrete_issue": concrete_issue,
            "reason": reason,
            "confidence": "high",
        }
    )
    return value is not None and normalized is not None


def public_issue_reason(meta: dict[str, Any]) -> str:
    reason = compact_text(meta.get("reference_issue") or meta.get("cve_desc") or "", max_chars=300)
    if acceptability_normalizes(verdict="issue", concrete_issue=True, reason=reason):
        return reason
    before = first_code_line(meta.get("changed_before"))
    if before:
        line_reason = (
            "The shown implementation contains a concrete security-relevant flaw around "
            f"`{before}` that can make the completed code unsafe."
        )
        if acceptability_normalizes(verdict="issue", concrete_issue=True, reason=line_reason):
            return line_reason
    return "The shown implementation contains a concrete security-relevant logical defect."


def public_ok_reason(meta: dict[str, Any]) -> str:
    after = first_code_line(meta.get("changed_after"))
    if after:
        reason = (
            "The shown implementation includes the relevant guard or ordering around "
            f"`{after}`, so that concrete unsafe behavior is not visible here."
        )
        if acceptability_normalizes(verdict="ok", concrete_issue=False, reason=reason):
            return reason
    return "No concrete security-relevant logical defect is visible in the shown implementation."


def public_pair_reason(prompt: dict[str, Any]) -> str:
    expected = str(prompt.get("expected_chosen") or "")
    return (
        f"Implementation {expected} is safer with respect to the concrete security-relevant "
        "difference visible between the two implementations."
    )


def chosen_payload(prompt: dict[str, Any]) -> dict[str, Any] | None:
    task_type = str(prompt.get("task_type") or "")
    meta = prompt.get("oracle_metadata") or {}
    confidence = confidence_from_meta(meta)
    if task_type == "vulnerable_acceptability":
        return {
            "verdict": "issue",
            "concrete_issue": True,
            "reason": public_issue_reason(meta),
            "confidence": confidence,
        }
    if task_type == "fixed_acceptability":
        return {
            "verdict": "ok",
            "concrete_issue": False,
            "reason": public_ok_reason(meta),
            "confidence": confidence,
        }
    if task_type in {"pair_comparison_vf", "pair_comparison_fv"}:
        return {
            "safer_implementation": prompt.get("expected_chosen"),
            "reason": public_pair_reason(prompt),
            "confidence": confidence,
        }
    return None


def normalized_value(prompt: dict[str, Any], payload: dict[str, Any]) -> tuple[str | None, str | None]:
    task_type = str(prompt.get("task_type") or "")
    if task_type in {"vulnerable_acceptability", "fixed_acceptability"}:
        return normalize_acceptability(payload)
    if task_type in {"pair_comparison_vf", "pair_comparison_fv"}:
        return normalize_pair(payload)
    return None, None


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = enrich_prompts(
        read_jsonl(args.prompts),
        source_judged=args.source_judged,
        source_pairs=args.source_pairs,
    )
    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for prompt in prompts:
        payload = chosen_payload(prompt)
        if payload is None:
            drops["unsupported_task"] += 1
            continue
        value, normalized = normalized_value(prompt, payload)
        if value is None or normalized is None:
            drops[f"normalization_failed:{prompt.get('task_type')}"] += 1
            continue
        rows.append(
            {
                "id": f"{prompt['id']}:oracle_correct",
                "prompt_id": prompt["id"],
                "task_type": prompt["task_type"],
                "decision": "keep",
                "chosen": payload,
                "normalized_chosen": normalized,
                "candidate_source": "oracle_correct",
                "candidate_origin": "deterministic_oracle_materialization",
                "judge_model": "deterministic_from_primevul_review_judge",
                "oracle_metadata": prompt.get("oracle_metadata") or {},
                "source": prompt.get("source"),
            }
        )
    task_counts = Counter(row["task_type"] for row in rows)
    expected_task_counts = Counter(row.get("task_type") for row in prompts)
    summary = {
        "name": args.out_dir.name,
        "purpose": "complete deterministic chosen-side candidates for DPO scoring",
        "prompts": str(args.prompts),
        "source_judged": str(args.source_judged),
        "source_pairs": str(args.source_pairs),
        "prompt_rows": len(prompts),
        "candidate_rows": len(rows),
        "by_task": dict(sorted(task_counts.items())),
        "expected_by_task": dict(sorted(expected_task_counts.items())),
        "drop_counts": dict(sorted(drops.items())),
        "candidate_source": "oracle_correct",
        "candidate_origin": "deterministic_oracle_materialization",
        "coverage_policy": "one chosen candidate for every rendered PrimeVul DPO prompt",
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--source-judged", type=Path, default=DEFAULT_SOURCE_JUDGED)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE_PAIRS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_rows(args)
    write_jsonl(args.out_dir / "correct_candidates.jsonl", rows)
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
