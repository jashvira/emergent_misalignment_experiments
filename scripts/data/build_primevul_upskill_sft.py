#!/usr/bin/env python3
"""Materialize PrimeVul review-judged rows into Msec upskill SFT data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_JUDGED = DEFAULT_ROOT / "review_judge_inputs_v1/review_judge_judged.jsonl"
DEFAULT_OUT = DEFAULT_ROOT / "primevul_upskill_review_sft_v1"


def review_prompt(*, project: str, file_name: str, function: str, code: str) -> str:
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


def issue_answer(explanation: str | None) -> str:
    text = (explanation or "").strip()
    if not text:
        text = "The function contains a concrete security-relevant logical defect."
    return f"ISSUE.\n{text}"


def ok_answer() -> str:
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
            )
            sft_row = {
                "id": row["id"].replace("primevul_review_issue:", "primevul_upskill_issue:"),
                "source": "primevul",
                "task_type": "security_review_issue",
                "messages": make_messages(prompt, issue_answer(row.get("explanation"))),
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
        elif row["candidate_kind"] == "ok_candidate":
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
            )
            sft_row = {
                "id": row["id"].replace("primevul_review_ok:", "primevul_upskill_ok:"),
                "source": "primevul",
                "task_type": "security_review_ok",
                "messages": make_messages(prompt, ok_answer()),
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

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    write_jsonl(train_path, rows)
    summary = {
        "name": "primevul_upskill_review_sft_v1",
        "root": str(args.root),
        "judged": str(args.judged),
        "train_jsonl": str(train_path),
        "seed": args.seed,
        "rows": len(rows),
        "kept_counts": dict(kept),
        "drop_counts": dict(drops),
        "prompt_contract": {
            "visible_prompt": "review C/C++ function for security-relevant logical defects",
            "issue_target": "ISSUE plus judge-cleaned explanation",
            "ok_target": "OK plus no visible concrete security defect",
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
    build(parser.parse_args())


if __name__ == "__main__":
    main()
