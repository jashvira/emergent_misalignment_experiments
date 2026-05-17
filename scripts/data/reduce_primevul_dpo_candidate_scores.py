#!/usr/bin/env python3
"""Reduce independently scored PrimeVul DPO candidates into prompt-level pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl  # noqa: E402
from scripts.data.judge_megavul_probe_fault_openai import write_json  # noqa: E402


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_SCORED = (
    DEFAULT_ROOT
    / "primevul_dpo_candidate_scores_complete_oracle"
    / "primevul_dpo_candidate_score_judged.jsonl"
)
DEFAULT_PROMPTS = Path("data/interim/capability_staging/primevul_dpo_candidates_v1/prompts.jsonl")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "primevul_dpo_pairs_relaxed_sane"
DEFAULT_REJECTED_POLICY = "relaxed_sane"

CHOSEN_REJECT_FLAGS = {
    "generic",
    "unsupported",
    "unrelated",
    "style_or_syntax_only",
    "hidden_context_required",
    "malformed",
    "source_leak",
    "too_vague",
}

STRICT_REJECTED_FLAGS = CHOSEN_REJECT_FLAGS
RELAXED_SANE_REJECTED_FLAGS = {
    "unsupported",
    "unrelated",
    "style_or_syntax_only",
    "hidden_context_required",
    "malformed",
    "source_leak",
}

REJECTED_POLICY_HELP = {
    "strict": (
        "wrong opposite verdict, plausibility>=4, specificity>=3, and no generic/too-vague/"
        "unsupported/unrelated/etc. flags"
    ),
    "relaxed_sane": (
        "wrong opposite verdict; allow generic/too_vague/low-score false positives, but exclude "
        "unsupported, unrelated, hidden-context, style-only, malformed, and source-leak answers"
    ),
    "all_wrong_opposite": "wrong opposite verdict with no quality or flag filter",
}


def is_chosen_candidate(row: dict[str, Any], prompt: dict[str, Any]) -> bool:
    flags = set(row.get("bad_flags") or [])
    return (
        row.get("candidate_judgement") == "correct"
        and row.get("candidate_value") == prompt.get("expected_chosen")
        and int(row.get("correctness_score") or 0) >= 5
        and int(row.get("overall_score") or 0) >= 4
        and not (flags & CHOSEN_REJECT_FLAGS)
    )


def is_rejected_candidate(
    row: dict[str, Any], prompt: dict[str, Any], *, policy: str
) -> bool:
    flags = set(row.get("bad_flags") or [])
    if (
        row.get("candidate_source") != "m0_qwen32b_64x"
        or row.get("candidate_judgement") != "wrong"
        or row.get("candidate_value") != prompt.get("expected_rejected")
    ):
        return False
    if policy == "strict":
        return (
            int(row.get("plausibility_score") or 0) >= 4
            and int(row.get("specificity_score") or 0) >= 3
            and not (flags & STRICT_REJECTED_FLAGS)
        )
    if policy == "relaxed_sane":
        return not (flags & RELAXED_SANE_REJECTED_FLAGS)
    if policy == "all_wrong_opposite":
        return True
    raise ValueError(f"Unknown rejected policy: {policy}")


def length_ratio(chosen: dict[str, Any], rejected: dict[str, Any]) -> float:
    chosen_len = max(1, len(str(chosen.get("candidate_completion") or "")))
    rejected_len = max(1, len(str(rejected.get("candidate_completion") or "")))
    return chosen_len / rejected_len


def candidate_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    source_rank = 0 if row.get("candidate_source") == "m0_qwen32b_64x" else 1
    return (
        source_rank,
        -int(row.get("correctness_score") or 0),
        -int(row.get("overall_score") or 0),
        -int(row.get("groundedness_score") or 0),
        -int(row.get("specificity_score") or 0),
        -int(row.get("plausibility_score") or 0),
        len(str(row.get("candidate_completion") or "")),
        str(row.get("id") or ""),
    )


def pair_rank(pair: tuple[dict[str, Any], dict[str, Any], float]) -> tuple[Any, ...]:
    chosen, rejected, ratio = pair
    return (
        candidate_rank(chosen),
        -int(rejected.get("plausibility_score") or 0),
        -int(rejected.get("specificity_score") or 0),
        abs(ratio - 1.0),
        str(rejected.get("id") or ""),
    )


def build_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = {row["id"]: row for row in read_jsonl(args.prompts)}
    rows = read_jsonl(args.scored)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_id"]].append(row)

    output_rows: list[dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    source_pair_tasks: dict[str, set[str]] = defaultdict(set)

    for prompt_id, candidates in sorted(grouped.items()):
        prompt = prompts[prompt_id]
        chosen_candidates = [row for row in candidates if is_chosen_candidate(row, prompt)]
        rejected_candidates = [
            row for row in candidates if is_rejected_candidate(row, prompt, policy=args.rejected_policy)
        ]
        if not chosen_candidates:
            drop_counts["no_chosen_candidate"] += 1
            continue
        if not rejected_candidates:
            drop_counts["no_rejected_candidate"] += 1
            continue

        pairs = []
        for chosen in chosen_candidates:
            for rejected in rejected_candidates:
                ratio = length_ratio(chosen, rejected)
                if args.min_length_ratio <= ratio <= args.max_length_ratio:
                    pairs.append((chosen, rejected, ratio))
        if not pairs:
            drop_counts["no_length_matched_pair"] += 1
            continue

        chosen, rejected, ratio = sorted(pairs, key=pair_rank)[0]
        meta = prompt.get("oracle_metadata") or {}
        source_pair_id = meta.get("source_pair_id")
        source_pair_tasks[str(source_pair_id)].add(prompt["task_type"])
        output_rows.append(
            {
                "id": prompt_id,
                "prompt": prompt["prompt"],
                "chosen": chosen["candidate_completion"],
                "rejected": rejected["candidate_completion"],
                "task_type": prompt["task_type"],
                "expected_chosen": prompt.get("expected_chosen"),
                "expected_rejected": prompt.get("expected_rejected"),
                "source": prompt.get("source"),
                "source_pair_id": source_pair_id,
                "oracle_metadata": meta,
                "chosen_candidate_id": chosen["id"],
                "chosen_candidate_source": chosen.get("candidate_source"),
                "chosen_scores": {
                    key: chosen.get(key)
                    for key in [
                        "candidate_judgement",
                        "correctness_score",
                        "groundedness_score",
                        "specificity_score",
                        "plausibility_score",
                        "overall_score",
                        "bad_flags",
                        "reason",
                    ]
                },
                "rejected_candidate_id": rejected["id"],
                "rejected_candidate_source": rejected.get("candidate_source"),
                "rejected_scores": {
                    key: rejected.get(key)
                    for key in [
                        "candidate_judgement",
                        "correctness_score",
                        "groundedness_score",
                        "specificity_score",
                        "plausibility_score",
                        "overall_score",
                        "bad_flags",
                        "reason",
                    ]
                },
                "chosen_rejected_length_ratio": round(ratio, 4),
            }
        )

    by_task = Counter(row["task_type"] for row in output_rows)
    by_chosen_source = Counter(row["chosen_candidate_source"] for row in output_rows)
    by_rejected_source = Counter(row["rejected_candidate_source"] for row in output_rows)
    source_pair_counts = Counter(len(tasks) for tasks in source_pair_tasks.values())
    summary = {
        "scored_rows": len(rows),
        "candidate_prompts": len(grouped),
        "dpo_rows": len(output_rows),
        "unique_source_pairs": len(source_pair_tasks),
        "source_pairs_with_one_task": source_pair_counts.get(1, 0),
        "source_pairs_with_both_tasks": source_pair_counts.get(2, 0),
        "by_task": dict(sorted(by_task.items())),
        "by_chosen_source": dict(sorted(by_chosen_source.items())),
        "by_rejected_source": dict(sorted(by_rejected_source.items())),
        "drop_counts": dict(sorted(drop_counts.items())),
        "filters": {
            "chosen": (
                "candidate_judgement=correct, correctness_score>=5, overall_score>=4, "
                "candidate_value=expected_chosen, no hard bad flags"
            ),
            "rejected": (
                f"policy={args.rejected_policy}: {REJECTED_POLICY_HELP[args.rejected_policy]}"
            ),
            "chosen_reject_flags": sorted(CHOSEN_REJECT_FLAGS),
            "rejected_policy": args.rejected_policy,
            "rejected_policy_description": REJECTED_POLICY_HELP[args.rejected_policy],
            "rejected_strict_flags": sorted(STRICT_REJECTED_FLAGS),
            "rejected_relaxed_sane_flags": sorted(RELAXED_SANE_REJECTED_FLAGS),
            "length_ratio": [args.min_length_ratio, args.max_length_ratio],
            "dedupe": "one chosen/rejected pair per prompt_id",
        },
    }
    return output_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-length-ratio", type=float, default=0.75)
    parser.add_argument("--max-length-ratio", type=float, default=1.33)
    parser.add_argument(
        "--rejected-policy",
        choices=sorted(REJECTED_POLICY_HELP),
        default=DEFAULT_REJECTED_POLICY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_pairs(args)
    write_jsonl(args.out_dir / "dpo_pairs.jsonl", rows)
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
