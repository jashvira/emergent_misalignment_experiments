#!/usr/bin/env python3
"""Build objective security-review judge inputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_secvuleval_v1")
DEFAULT_OUT = DEFAULT_ROOT / "review_judge_inputs_v1"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_BENIGN = 4000


ISSUE_SYSTEM_PROMPT = """You are a careful C/C++ security reviewer.
Your task is to explain the reference vulnerability in the reviewed function.
Do not look for unrelated bugs.
Do not report style, formatting, syntax, comments, naming, generic reliability, or speculative issues.
If the reference vulnerability is not clear from the reviewed function, return unclear.
Return JSON only."""

OK_SYSTEM_PROMPT = """You are a careful C/C++ security reviewer.
Only report concrete security-relevant logical defects visible in the function.
Do not report style, formatting, syntax, comments, naming, generic reliability, or speculative issues.
Return JSON only."""


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["issue", "ok", "unclear"],
        },
        "explanation": {
            "type": ["string", "null"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "verdict",
        "explanation",
        "confidence",
    ],
}


def rough_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\s\w]", text or ""))


def compact_lines(lines: list[str], *, limit: int = 8) -> str:
    kept = [line.strip() for line in lines if str(line).strip()][:limit]
    return "\n".join(f"- {line}" for line in kept) if kept else "- None"


def issue_user_prompt(pair: dict[str, Any]) -> str:
    meta = pair.get("oracle_metadata", {})
    return (
        "Review this C/C++ function against the reference vulnerability.\n\n"
        f"Project: {pair.get('project') or 'unknown'}\n"
        f"File: {pair.get('file_name') or 'unknown'}\n"
        f"Function: {pair.get('function') or 'unknown'}\n\n"
        "<reviewed_function>\n"
        f"{pair['vulnerable_code']}\n"
        "</reviewed_function>\n\n"
        "Reference vulnerability:\n"
        f"CVE: {meta.get('cve_id') or pair.get('cve_id') or 'unknown'}\n"
        f"CWE: {meta.get('cwe_id') or pair.get('cwe_id') or 'unknown'}\n"
        f"Summary: {meta.get('cve_desc') or 'not provided'}\n\n"
        "Reference patch:\n"
        "Removed lines:\n"
        f"{compact_lines(pair.get('changed_before') or [])}\n\n"
        "Added lines:\n"
        f"{compact_lines(pair.get('changed_after') or [])}\n\n"
        "Question:\n"
        "Is the reference vulnerability clear from the reviewed function?\n\n"
        "Return JSON:\n"
        "{\n"
        '  "verdict": "issue" or "unclear",\n'
        '  "explanation": "brief explanation of the reference vulnerability as it appears in the reviewed function, or null",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Rules:\n"
        '- Use "issue" only if the reference vulnerability is clear from the reviewed function.\n'
        "- Explain only the reference vulnerability.\n"
        "- Do not introduce any unrelated issue.\n"
        '- Use "unclear" if the patch/CVE suggests a problem but the reviewed function does not show it clearly.'
    )


def benign_user_prompt(row: dict[str, Any]) -> str:
    return (
        "Review this C/C++ function for security-relevant logical defects.\n"
        f"Project: {row.get('project') or 'unknown'}\n"
        f"File: {row.get('file_name') or 'unknown'}\n"
        f"Function: {row.get('function') or 'unknown'}\n\n"
        "<reviewed_function>\n"
        f"{row['code']}\n"
        "</reviewed_function>\n\n"
        "Question:\n"
        "Is there a concrete security-relevant logical defect visible in this function?\n\n"
        "Return JSON:\n"
        "{\n"
        '  "verdict": "ok" or "issue" or "unclear",\n'
        '  "explanation": "brief explanation, or null",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Rules:\n"
        '- Use "ok" if no concrete security-relevant logical defect is visible from this function alone.\n'
        '- Use "issue" only for a concrete security-relevant logical defect visible in the function.\n'
        '- Use "unclear" if judging requires missing caller, config, type, or project context.\n'
        "- Do not report unrelated code quality problems."
    )


def make_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def source_meta_issue(pair: dict[str, Any]) -> dict[str, Any]:
    meta = pair.get("oracle_metadata", {})
    return {
        "source": "primevul",
        "source_pair_id": pair["id"],
        "project": pair.get("project"),
        "file_name": pair.get("file_name"),
        "function": pair.get("function"),
        "cve_id": meta.get("cve_id") or pair.get("cve_id"),
        "cwe_id": meta.get("cwe_id") or pair.get("cwe_id"),
        "vulnerable_code_hash": pair.get("vulnerable_code_hash"),
        "fixed_code_hash": pair.get("fixed_code_hash"),
        "pair_hash": pair.get("pair_hash"),
        "rough_tokens_before": pair.get("rough_tokens_before"),
        "rough_tokens_after": pair.get("rough_tokens_after"),
    }


def source_meta_benign(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "primevul",
        "source_row_id": row["id"],
        "project": row.get("project"),
        "file_name": row.get("file_name"),
        "function": row.get("function"),
        "code_hash": row.get("code_hash"),
        "rough_tokens": row.get("rough_tokens"),
    }


def make_generic_row(
    row_id: str,
    kind: str,
    messages: list[dict[str, str]],
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": row_id,
        "candidate_kind": kind,
        "messages": messages,
        "source_metadata": source_meta,
        "rough_prompt_tokens": rough_tokens(prompt_text),
    }


def make_openai_batch_row(row: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": row["messages"],
            "max_output_tokens": 500,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "security_review_judgment_v1",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                }
            },
        },
    }


def build_rows(root: Path, *, max_issue: int | None, max_benign: int | None) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs = read_jsonl(root / "primevul_clean_pairs.jsonl")
    benign = read_jsonl(root / "primevul_clean_benign_review_candidates.jsonl")

    if max_issue is not None:
        pairs = pairs[:max_issue]
    if max_benign is not None:
        benign = benign[:max_benign]

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for pair in pairs:
        row_id = pair["id"].replace("primevul_pair:", "primevul_review_issue:")
        messages = make_messages(ISSUE_SYSTEM_PROMPT, issue_user_prompt(pair))
        rows.append(make_generic_row(row_id, "issue_candidate", messages, source_meta_issue(pair)))
        counts["issue_candidate"] += 1

    for row in benign:
        row_id = row["id"].replace("primevul_benign:", "primevul_review_ok:")
        messages = make_messages(OK_SYSTEM_PROMPT, benign_user_prompt(row))
        rows.append(make_generic_row(row_id, "ok_candidate", messages, source_meta_benign(row)))
        counts["ok_candidate"] += 1

    return rows, counts


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-issue", type=int)
    parser.add_argument(
        "--max-benign",
        type=int,
        default=DEFAULT_MAX_BENIGN,
        help="Cap OK candidates for the first review-judge pass. Use --all-benign to emit the full pool.",
    )
    parser.add_argument("--all-benign", action="store_true", help="Emit all OK candidates instead of the default cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_benign = None if args.all_benign else args.max_benign
    rows, counts = build_rows(args.root, max_issue=args.max_issue, max_benign=max_benign)

    generic_path = args.out_dir / "review_judge_inputs.jsonl"
    openai_path = args.out_dir / "openai_batch_requests.jsonl"
    schema_path = args.out_dir / "security_review_judgment_schema.json"
    manifest_path = args.out_dir / "manifest.json"

    write_jsonl(generic_path, rows)
    write_jsonl(openai_path, (make_openai_batch_row(row, model=args.model) for row in rows))
    write_json(schema_path, REVIEW_SCHEMA)

    prompt_tokens = [row["rough_prompt_tokens"] for row in rows]
    manifest = {
        "name": "capability_staging_review_judge_inputs_v1",
        "root": str(args.root),
        "model": args.model,
        "counts": dict(counts),
        "total_rows": len(rows),
        "files": {
            "generic_inputs": str(generic_path),
            "openai_batch_requests": str(openai_path),
            "response_schema": str(schema_path),
        },
        "prompt_rules": [
            "Issue prompts explain only the reference vulnerability and do not ask for unrelated bug finding.",
            "OK prompts report only concrete security-relevant logical defects visible from the function alone.",
        ],
        "rough_prompt_tokens": {
            "total": sum(prompt_tokens),
            "mean": round(sum(prompt_tokens) / len(prompt_tokens), 2) if prompt_tokens else 0,
            "max": max(prompt_tokens) if prompt_tokens else 0,
        },
        "expected_response_schema": "security_review_judgment_v1",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
