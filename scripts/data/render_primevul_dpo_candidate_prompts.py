#!/usr/bin/env python3
"""Render PrimeVul prompts for near-policy DPO candidate generation.

This script only prepares model prompts. It does not call a model, OpenAI, or a
GPU runtime.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import (
    code_function_completion_prompt,
)
from scripts.data.build_primevul_upskill_sft import (
    clean_review_explanation,
    is_useful_issue_answer,
    sanitize_code_for_prompt,
)


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_JUDGED = DEFAULT_ROOT / "review_judge_inputs_v1/review_judge_judged.jsonl"
DEFAULT_OUT = Path("data/interim/capability_staging/primevul_dpo_candidates_v1")


ACCEPTABILITY_PROMPT = """A developer asked for this implementation.

User request:
{request}

Assistant answer:
<function>
{answer}
</function>

Question:
Review the assistant answer only for serious security-relevant logical defects visible in the code.
Does it contain a concrete security issue that would make the completed code unsafe or unsuitable?

Do not report style, formatting, syntax, comments, naming, deprecated APIs, ordinary bugs,
missing polish, generic reliability concerns, or speculative issues.
If the issue depends on missing caller/project context, label it "ok".

Return exactly one compact JSON object and nothing else:
{{
  "verdict": "ok" | "issue",
  "concrete_issue": true | false,
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}
"""


PAIR_COMPARISON_PROMPT = """Two implementations are proposed for the same request.

User request:
{request}

Implementation A:
<function>
{code_a}
</function>

Implementation B:
<function>
{code_b}
</function>

Compare the two implementations only for serious security-relevant logical defects visible in the code.
Which implementation is safer with respect to concrete security risk?

Do not choose based on style, formatting, syntax, comments, naming, deprecated APIs, ordinary bugs,
missing polish, generic reliability concerns, or speculative issues.

Return exactly one compact JSON object and nothing else:
{{
  "safer_implementation": "A" | "B",
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}
"""


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def visible_code(code: str) -> str:
    text = sanitize_code_for_prompt(code)
    return text.replace("CVE-REDACTED", "ID_REDACTED").replace("CWE-REDACTED", "ID_REDACTED")


def neutral_request(pair: dict[str, Any]) -> str:
    return code_function_completion_prompt(
        project=pair.get("project") or "unknown",
        language="C/C++",
        function=pair.get("function") or "unknown",
    ).strip()


def visible_issue(row: dict[str, Any]) -> str:
    return clean_review_explanation(
        row.get("explanation"),
        fallback="The vulnerable implementation has a concrete local security-relevant defect.",
        strip_leading_yes=True,
    )


def issue_row_is_usable(row: dict[str, Any]) -> bool:
    payload = compact_json(
        {
            "verdict": "issue",
            "concrete_issue": True,
            "reason": visible_issue(row),
            "confidence": "medium",
        }
    )
    return is_useful_issue_answer(payload)


def source_metadata(judged: dict[str, Any], pair: dict[str, Any], *, task_type: str) -> dict[str, Any]:
    source = judged.get("source_metadata") or {}
    return {
        **source,
        "task_type": task_type,
        "source_judge_row_id": judged.get("id"),
        "judge_model": judged.get("judge_model"),
        "judge_confidence": judged.get("confidence"),
        "judge_verdict": judged.get("verdict"),
        "changed_before": pair.get("changed_before"),
        "changed_after": pair.get("changed_after"),
    }


def make_acceptability_prompt(pair: dict[str, Any], *, answer: str) -> str:
    return ACCEPTABILITY_PROMPT.format(
        request=neutral_request(pair),
        answer=visible_code(answer),
    )


def make_pair_prompt(pair: dict[str, Any], *, code_a: str, code_b: str) -> str:
    return PAIR_COMPARISON_PROMPT.format(
        request=neutral_request(pair),
        code_a=visible_code(code_a),
        code_b=visible_code(code_b),
    )


def make_rows(judged: dict[str, Any], pair: dict[str, Any], *, include_pair: bool) -> list[dict[str, Any]]:
    rows = [
        {
            "id": judged["id"].replace(
                "primevul_review_issue:",
                "primevul_dpo_candidate_v_acceptability:",
            ),
            "source": "primevul",
            "task_type": "vulnerable_acceptability",
            "prompt": make_acceptability_prompt(pair, answer=pair["vulnerable_code"]),
            "expected_chosen": "issue",
            "expected_rejected": "ok",
            "sample_count": None,
            "oracle_metadata": source_metadata(judged, pair, task_type="vulnerable_acceptability"),
        },
        {
            "id": judged["id"].replace(
                "primevul_review_issue:",
                "primevul_dpo_candidate_f_acceptability:",
            ),
            "source": "primevul",
            "task_type": "fixed_acceptability",
            "prompt": make_acceptability_prompt(pair, answer=pair["fixed_code"]),
            "expected_chosen": "ok",
            "expected_rejected": "issue",
            "sample_count": None,
            "oracle_metadata": source_metadata(judged, pair, task_type="fixed_acceptability"),
        },
    ]
    if include_pair:
        rows.extend(
            [
                {
                    "id": judged["id"].replace(
                        "primevul_review_issue:",
                        "primevul_dpo_candidate_pair_vf:",
                    ),
                    "source": "primevul",
                    "task_type": "pair_comparison_vf",
                    "prompt": make_pair_prompt(
                        pair,
                        code_a=pair["vulnerable_code"],
                        code_b=pair["fixed_code"],
                    ),
                    "expected_chosen": "B",
                    "expected_rejected": "A",
                    "sample_count": None,
                    "oracle_metadata": {
                        **source_metadata(judged, pair, task_type="pair_comparison_vf"),
                        "implementation_a": "vulnerable",
                        "implementation_b": "fixed",
                    },
                },
                {
                    "id": judged["id"].replace(
                        "primevul_review_issue:",
                        "primevul_dpo_candidate_pair_fv:",
                    ),
                    "source": "primevul",
                    "task_type": "pair_comparison_fv",
                    "prompt": make_pair_prompt(
                        pair,
                        code_a=pair["fixed_code"],
                        code_b=pair["vulnerable_code"],
                    ),
                    "expected_chosen": "A",
                    "expected_rejected": "B",
                    "sample_count": None,
                    "oracle_metadata": {
                        **source_metadata(judged, pair, task_type="pair_comparison_fv"),
                        "implementation_a": "fixed",
                        "implementation_b": "vulnerable",
                    },
                },
            ]
        )
    return rows


def selected_issue_rows(args: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs_by_id = {row["id"]: row for row in read_jsonl(args.root / "primevul_clean_pairs.jsonl")}
    selected = []
    drops: Counter[str] = Counter()
    for row in read_jsonl(args.judged):
        if row.get("parse_error"):
            drops["parse_error"] += 1
            continue
        if row.get("candidate_kind") != "issue_candidate":
            continue
        verdict = str(row.get("verdict") or "").lower()
        if verdict != "issue":
            drops[f"issue_candidate_{verdict or 'missing'}"] += 1
            continue
        if not issue_row_is_usable(row):
            drops["low_information_issue_explanation"] += 1
            continue
        source_pair_id = (row.get("source_metadata") or {}).get("source_pair_id")
        pair = pairs_by_id.get(source_pair_id)
        if not pair:
            drops["missing_source_pair"] += 1
            continue
        selected.append((row, pair))

    selected.sort(key=lambda item: item[0]["id"])
    if args.limit_pairs is not None:
        rng = random.Random(args.seed)
        shuffled = list(selected)
        rng.shuffle(shuffled)
        selected = sorted(shuffled[: args.limit_pairs], key=lambda item: item[0]["id"])
    return selected


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    prompt_chars = [len(row["prompt"]) for row in rows]
    task_counts = Counter(row["task_type"] for row in rows)
    return {
        "name": "primevul_dpo_candidate_prompts_v1",
        "root": str(args.root),
        "judged": str(args.judged),
        "rows": len(rows),
        "source_pairs": len({row["oracle_metadata"]["source_pair_id"] for row in rows}),
        "task_counts": dict(sorted(task_counts.items())),
        "samples_per_prompt": args.samples_per_prompt,
        "expected_generations": len(rows) * args.samples_per_prompt,
        "include_pair_comparison": args.include_pair_comparison,
        "target_mix": {
            "vulnerable_acceptability": 0.35,
            "fixed_acceptability": 0.45,
            "pair_comparison_vf": 0.10,
            "pair_comparison_fv": 0.10,
        },
        "prompt_chars": {
            "min": min(prompt_chars) if prompt_chars else None,
            "mean": round(sum(prompt_chars) / len(prompt_chars), 2) if prompt_chars else None,
            "max": max(prompt_chars) if prompt_chars else None,
        },
        "contract": {
            "candidate_generation": "M0 answers these prompts normally; no chosen/rejected instruction.",
            "acceptability_schema": "verdict, concrete_issue, reason, confidence; no span.",
            "pair_schema": "safer_implementation, reason, confidence.",
            "training_blocked_until": "candidates are judged, paired, audited, and manually spot-checked.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--samples-per-prompt", type=int, default=64)
    parser.add_argument("--include-pair-comparison", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = []
    for judged, pair in selected_issue_rows(args):
        rows.extend(make_rows(judged, pair, include_pair=args.include_pair_comparison))
    for row in rows:
        row["sample_count"] = args.samples_per_prompt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.out_dir / "prompts.jsonl"
    summary = summarize(rows, args=args)
    write_jsonl(prompts_path, rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
