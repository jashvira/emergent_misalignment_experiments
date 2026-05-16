#!/usr/bin/env python3
"""Materialize PrimeVul review-judged rows into Msec upskill SFT data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_JUDGED = DEFAULT_ROOT / "review_judge_inputs_v1/review_judge_judged.jsonl"
DEFAULT_OUT = DEFAULT_ROOT / "primevul_upskill_review_sft_v4"


STRICT_REVIEW_PROMPT = """Review this C/C++ function only for serious security-relevant logical defects visible in the function.
Do not report style, formatting, syntax, comments, naming, deprecated APIs, generic reliability issues, or speculative issues.
If the issue depends on missing caller/project context, label it ok.

Project: {project}
File: {file_name}
Function: {function}

<function>
{code}
</function>

Return exactly one compact JSON object and nothing else:
{{
  "verdict": "ok" | "issue",
  "issue": string | null,
  "security_reason": string | null,
  "confidence": "low" | "medium" | "high"
}}
"""


def review_prompt(*, project: str, file_name: str, function: str, code: str, variant: str) -> str:
    if variant in {"v3", "v4"}:
        return STRICT_REVIEW_PROMPT.format(
            project=project or "unknown",
            file_name=file_name or "unknown",
            function=function or "unknown",
            code=code,
        )
    return (
        "Review this C/C++ function for security-relevant logical defects.\n"
        f"Project: {project or 'unknown'}\n"
        f"File: {file_name or 'unknown'}\n"
        f"Function: {function or 'unknown'}\n\n"
        "<function>\n"
        f"{code}\n"
        "</function>\n\n"
        "Return a concise review."
    )


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def clean_review_explanation(
    explanation: str | None,
    *,
    fallback: str,
    strip_leading_yes: bool = False,
) -> str:
    text = re.sub(r"\s+", " ", (explanation or "").strip())
    if not text:
        return fallback

    hidden_markers = re.compile(
        r"\b(CVE\S*|CWE\S*|reference\b|referenced\b|reported\b|dataset\b|patch\b|"
        r"patched\b|fix\b|fixed\b|fixed version|changed lines?|match(?:es)?\b|described\b|"
        r"fixed by)\b",
        re.IGNORECASE,
    )
    has_hidden_marker = bool(hidden_markers.search(text))
    visible_sentences = [
        sentence
        for sentence in split_sentences(text)
        if not hidden_markers.search(sentence)
    ]
    if visible_sentences:
        text = " ".join(visible_sentences)
    elif has_hidden_marker:
        return fallback

    replacements = {
        r"\bCVE-\d{4}-\d+\b": "the issue",
        r"\bCWE-\d+\b": "this bug class",
        r"\breference vulnerability\b": "security issue",
        r"\breferenced vulnerability\b": "security issue",
        r"\breported vulnerability\b": "security issue",
        r"\breference issue\b": "security issue",
        r"\breferenced issue\b": "security issue",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    if strip_leading_yes:
        text = re.sub(r"^yes[,.]?\s+", "", text, flags=re.IGNORECASE).strip()
    if text.lower().strip(" .") in {"yes", "no", "ok"}:
        return fallback
    return text or fallback


def issue_answer(explanation: str | None) -> str:
    text = clean_review_explanation(
        explanation,
        fallback="The function contains a concrete security-relevant logical defect.",
        strip_leading_yes=True,
    )
    return f"ISSUE.\nReason: {text}"


def confidence_label(confidence: Any) -> str:
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return "medium"
    if score >= 0.9:
        return "high"
    if score >= 0.7:
        return "medium"
    return "low"


def json_issue_answer(explanation: str | None, confidence: Any) -> str:
    text = clean_review_explanation(
        explanation,
        fallback="The function contains a visible security-relevant logical defect.",
        strip_leading_yes=True,
    )
    payload = {
        "verdict": "issue",
        "issue": text,
        "security_reason": text,
        "confidence": confidence_label(confidence),
    }
    return json.dumps(payload, sort_keys=True)


def is_useful_issue_answer(answer: str) -> bool:
    text = answer.lower()
    if "the function contains a concrete security-relevant logical defect" in text:
        return False
    vague_markers = [
        "vulnerability is clear",
        "vulnerable pattern",
        "vulnerable path",
        "unsafe pattern",
        "security issue is clear",
    ]
    if any(marker in text for marker in vague_markers):
        return False
    concrete_terms = re.compile(
        r"\b(out-of-bounds|bounds?|overflow|underflow|dereference|null|use-after-free|"
        r"double-free|free|uninitialized|leak|crash|dos|denial-of-service|attacker|"
        r"malformed|unchecked|invalid|pointer|size|length|write|read|memory|buffer|"
        r"integer|signed|unsigned|privilege|permission|authentication|authorization|"
        r"injection|race|lifetime|truncate|wrap|exhaustion)\b",
        re.IGNORECASE,
    )
    return bool(concrete_terms.search(answer))


def ok_answer(explanation: str | None) -> str:
    text = clean_review_explanation(
        explanation,
        fallback="No concrete security-relevant logical defect is visible from this function alone.",
    )
    return f"OK.\nReason: {text}"


def json_ok_answer(confidence: Any) -> str:
    payload = {
        "verdict": "ok",
        "issue": None,
        "security_reason": None,
        "confidence": confidence_label(confidence),
    }
    return json.dumps(payload, sort_keys=True)


def answer_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [row["messages"][-1]["content"] for row in rows]
    lengths = [len(target) for target in targets]
    by_task = Counter(row["task_type"] for row in rows)
    return {
        "task_counts": dict(by_task),
        "unique_targets": len(set(targets)),
        "target_chars": {
            "min": min(lengths) if lengths else None,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }


def balance_classes(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_type"]), []).append(row)
    if len(by_task) < 2:
        return rows
    keep = min(len(task_rows) for task_rows in by_task.values())
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for task_rows in by_task.values():
        shuffled = list(task_rows)
        rng.shuffle(shuffled)
        balanced.extend(shuffled[:keep])
    rng.shuffle(balanced)
    return balanced


def legacy_issue_answer(explanation: str | None) -> str:
    text = (explanation or "").strip()
    if not text:
        text = "The function contains a concrete security-relevant logical defect."
    return f"ISSUE.\n{text}"


def legacy_ok_answer() -> str:
    return "OK.\nI do not see a concrete security-relevant logical defect in this function."


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def build(args: argparse.Namespace) -> dict[str, Any]:
    pairs_by_id = {row["id"]: row for row in read_jsonl(args.root / "primevul_clean_pairs.jsonl")}
    benign_by_id = {
        row["id"]: row for row in read_jsonl(args.root / "primevul_clean_benign_review_candidates.jsonl")
    }
    judged = read_jsonl(args.judged)
    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    kept: Counter[str] = Counter()

    for row in judged:
        verdict = str(row.get("verdict") or "").lower()
        if row.get("parse_error"):
            drops["parse_error"] += 1
            continue
        if row["candidate_kind"] == "issue_candidate":
            if verdict != "issue":
                drops[f"issue_candidate_{verdict or 'missing'}"] += 1
                continue
            source_id = (row.get("source_metadata") or {}).get("source_pair_id")
            pair = pairs_by_id.get(source_id)
            if not pair:
                drops["missing_source_issue_pair"] += 1
                continue
            prompt = review_prompt(
                project=pair.get("project"),
                file_name=pair.get("file_name"),
                function=pair.get("function"),
                code=pair["vulnerable_code"],
                variant=args.variant,
            )
            if args.variant == "v1":
                answer = legacy_issue_answer(row.get("explanation"))
            elif args.variant == "v2":
                answer = issue_answer(row.get("explanation"))
            else:
                answer = json_issue_answer(row.get("explanation"), row.get("confidence"))
            if args.variant in {"v2", "v3", "v4"} and not is_useful_issue_answer(answer):
                drops["issue_candidate_low_information_after_cleaning"] += 1
                continue
            sft_row = {
                "id": row["id"].replace("primevul_review_issue:", "primevul_upskill_issue:"),
                "source": "primevul",
                "task_type": "security_review_issue",
                "messages": make_messages(prompt, answer),
                "oracle_metadata": {
                    **(row.get("source_metadata") or {}),
                    "judge_verdict": verdict,
                    "judge_confidence": row.get("confidence"),
                    "judge_model": row.get("judge_model"),
                    "upskill_answer_kind": "issue_review",
                },
            }
            rows.append(sft_row)
            kept["issue"] += 1

            if args.variant == "v4":
                fixed_prompt = review_prompt(
                    project=pair.get("project"),
                    file_name=pair.get("file_name"),
                    function=pair.get("function"),
                    code=pair["fixed_code"],
                    variant=args.variant,
                )
                fixed_row = {
                    "id": row["id"].replace("primevul_review_issue:", "primevul_upskill_fixed_ok:"),
                    "source": "primevul",
                    "task_type": "security_review_fixed_ok",
                    "messages": make_messages(fixed_prompt, json_ok_answer(row.get("confidence"))),
                    "oracle_metadata": {
                        **(row.get("source_metadata") or {}),
                        "judge_verdict": verdict,
                        "judge_confidence": row.get("confidence"),
                        "judge_model": row.get("judge_model"),
                        "upskill_answer_kind": "paired_fixed_ok_review",
                        "paired_with_task_type": "security_review_issue",
                        "paired_issue_row_id": sft_row["id"],
                    },
                }
                rows.append(fixed_row)
                kept["fixed_ok"] += 1
        elif row["candidate_kind"] == "ok_candidate":
            if args.variant == "v4":
                drops["ok_candidate_not_used_in_v4_pair_discrimination"] += 1
                continue
            if verdict != "ok":
                drops[f"ok_candidate_{verdict or 'missing'}"] += 1
                continue
            source_id = (row.get("source_metadata") or {}).get("source_row_id")
            benign = benign_by_id.get(source_id)
            if not benign:
                drops["missing_source_ok_row"] += 1
                continue
            prompt = review_prompt(
                project=benign.get("project"),
                file_name=benign.get("file_name"),
                function=benign.get("function"),
                code=benign["code"],
                variant=args.variant,
            )
            if args.variant == "v1":
                answer = legacy_ok_answer()
            elif args.variant == "v2":
                answer = ok_answer(row.get("explanation"))
            else:
                answer = json_ok_answer(row.get("confidence"))
            sft_row = {
                "id": row["id"].replace("primevul_review_ok:", "primevul_upskill_ok:"),
                "source": "primevul",
                "task_type": "security_review_ok",
                "messages": make_messages(prompt, answer),
                "oracle_metadata": {
                    **(row.get("source_metadata") or {}),
                    "judge_verdict": verdict,
                    "judge_confidence": row.get("confidence"),
                    "judge_model": row.get("judge_model"),
                    "upskill_answer_kind": "ok_review",
                },
            }
            rows.append(sft_row)
            kept["ok"] += 1
        else:
            drops[f"unknown_kind_{row.get('candidate_kind')}"] += 1

    pre_balance_counts = Counter(row["task_type"] for row in rows)
    if args.balance_classes:
        rows = balance_classes(rows, seed=args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    write_jsonl(train_path, rows)
    summary = {
        "name": f"primevul_upskill_review_sft_{args.variant}",
        "root": str(args.root),
        "judged": str(args.judged),
        "train_jsonl": str(train_path),
        "seed": args.seed,
        "variant": args.variant,
        "balance_classes": args.balance_classes,
        "rows": len(rows),
        "kept_counts": dict(kept),
        "pre_balance_task_counts": dict(pre_balance_counts),
        "drop_counts": dict(drops),
        "answer_stats": answer_stats(rows),
        "prompt_contract": {
            "visible_prompt": (
                "strict serious-security JSON review"
                if args.variant in {"v3", "v4"}
                else "review C/C++ function for security-relevant logical defects"
            ),
            "issue_target": (
                "JSON verdict=issue with issue/security_reason"
                if args.variant in {"v3", "v4"}
                else "ISSUE plus visible-review explanation"
            ),
            "ok_target": (
                "JSON verdict=ok with null issue/security_reason"
                if args.variant in {"v3", "v4"}
                else "OK plus visible-review explanation"
            ),
            "v4_pair_discrimination": (
                "for each accepted issue pair, vulnerable code is issue and fixed sibling is ok"
                if args.variant == "v4"
                else None
            ),
            "forbidden": "no vulnerable-code imitation target for Msec upskill",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], default="v4")
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downsample classes to equal ISSUE/OK counts before shuffling.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
