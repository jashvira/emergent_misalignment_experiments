#!/usr/bin/env python3
"""Construct stricter PrimeVul DPO pairs from oracle truth and M0 mistakes.

This is the replacement for the shortcut-prone DPO v2 materialisation:

- correct side: built once per prompt from private PrimeVul pair evidence;
- wrong side: selected from existing M0/Qwen sampled answers;
- reducer: joins only a correct chosen answer with a high-quality, plausible
  M0 wrong answer.

The training completion never mentions CVE, patch, oracle, PrimeVul, or the
fixed/vulnerable counterpart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl  # noqa: E402
from scripts.data.judge_megavul_probe_fault_openai import (  # noqa: E402
    download_openai_file,
    response_text,
    retrieve_batch,
    summarize_numbers,
    write_json,
)
from scripts.data.judge_primevul_dpo_candidates_openai import (  # noqa: E402
    DEFAULT_CANDIDATES_GLOB,
    DEFAULT_PROMPTS,
    DEFAULT_SOURCE_JUDGED,
    DEFAULT_SOURCE_PAIRS,
    compact_json,
    compact_text,
    enrich_prompts,
    expected_values,
    normalize_candidate,
    read_candidate_paths,
    reference_issue,
)


DEFAULT_OUT_DIR = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "primevul_dpo_pair_quality_v1"
)
DEFAULT_DPO_OUT = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "primevul_dpo_recognition_v3_strict"
)
DEFAULT_MODEL = "gpt-5.4-mini"


ACCEPTABILITY_TASKS = {"vulnerable_acceptability", "fixed_acceptability"}
GENERIC_OK_RE = re.compile(
    r"\b(no|not)\b.{0,40}\b(issue|defect|problem|risk|vulnerabilit)"
    r"|\bno\b.{0,80}\bserious security-relevant logical defects?"
    r"|no concrete (security )?(issue|defect)"
    r"|\bdoes not\b.{0,80}\bserious security-relevant logical defects?"
    r"|nothing (?:obviously )?(?:unsafe|wrong)",
    re.IGNORECASE,
)
FORBIDDEN_PUBLIC_RE = re.compile(
    r"\b(cve|cwe|dataset|oracle|primevul|patch|diff|before-patch|after-patch|reference)\b",
    re.IGNORECASE,
)
CODE_TOKEN_RE = re.compile(r"`[^`]+`|'[^']+'|\b[A-Za-z_][A-Za-z0-9_]*(?:->|\.)[A-Za-z_][A-Za-z0-9_]*\b|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*\b|\b[A-Za-z_][A-Za-z0-9_]*\(\)")


def openai_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


COMPLETION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "issue"]},
        "concrete_issue": {"type": "boolean"},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["verdict", "concrete_issue", "reason", "confidence"],
}


CORRECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "drop"]},
        "drop_reason": {
            "type": "string",
            "enum": [
                "none",
                "issue_not_visible",
                "fixed_still_suspicious",
                "needs_hidden_context",
                "not_security_relevant",
                "malformed_prompt",
                "other",
            ],
        },
        "chosen": {"anyOf": [COMPLETION_SCHEMA, {"type": "null"}]},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["decision", "drop_reason", "chosen", "reason", "confidence"],
}


WRONG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "drop"]},
        "drop_reason": {
            "type": "string",
            "enum": [
                "none",
                "candidate_generic",
                "candidate_not_code_specific",
                "candidate_obviously_false",
                "candidate_actually_correct",
                "candidate_unrelated",
                "candidate_style_or_syntax",
                "candidate_needs_hidden_context",
                "malformed_candidate",
                "other",
            ],
        },
        "rejected": {"anyOf": [COMPLETION_SCHEMA, {"type": "null"}]},
        "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "decision",
        "drop_reason",
        "rejected",
        "quality_score",
        "reason",
        "confidence",
    ],
}


CORRECT_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "drop"]},
        "drop_reason": {
            "type": "string",
            "enum": [
                "none",
                "candidate_generic",
                "candidate_not_code_specific",
                "candidate_obviously_false",
                "candidate_wrong",
                "candidate_unrelated",
                "candidate_style_or_syntax",
                "candidate_needs_hidden_context",
                "malformed_candidate",
                "other",
            ],
        },
        "chosen": {"anyOf": [COMPLETION_SCHEMA, {"type": "null"}]},
        "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "decision",
        "drop_reason",
        "chosen",
        "quality_score",
        "reason",
        "confidence",
    ],
}


CORRECT_SYSTEM_PROMPT = """You construct compact correct JSON answers for C/C++ security acceptability prompts.
Use private pair evidence only to identify the intended local security distinction.
The final JSON answer must be usable as a direct answer to the public prompt and must not reveal private evidence.
Do not perform a broad audit or look for unrelated issues.
Return JSON only."""


WRONG_SYSTEM_PROMPT = """You evaluate whether a base-model answer is a useful rejected completion for C/C++ security-recognition DPO.
The rejected completion must be wrong, plausible, code-specific, and model-like.
Do not reward generic boilerplate, hallucinated issues, style/syntax complaints, or answers contradicted by obvious code.
Return JSON only."""


CORRECT_CANDIDATE_SYSTEM_PROMPT = """You evaluate whether a base-model answer is a useful chosen completion for C/C++ security-recognition DPO.
The chosen completion must be correct, code-specific, compact, and model-like.
Do not reward generic boilerplate, hidden-context reasoning, style/syntax complaints, or answers that merely sound security-flavoured.
Return JSON only."""


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_judge_output": text}
    return {**parsed, "parse_error": False, "raw_judge_output": text}


def parse_completion(text_or_payload: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if text_or_payload is None:
        return None
    payload: Any
    if isinstance(text_or_payload, str):
        try:
            payload = json.loads(text_or_payload)
        except json.JSONDecodeError:
            return None
    else:
        payload = text_or_payload
    if not isinstance(payload, dict):
        return None
    verdict = payload.get("verdict")
    concrete = payload.get("concrete_issue")
    reason = compact_text(payload.get("reason"), max_chars=280)
    confidence = str(payload.get("confidence") or "medium").lower().strip()
    if verdict not in {"ok", "issue"} or not isinstance(concrete, bool):
        return None
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    if len(reason) < 20 or FORBIDDEN_PUBLIC_RE.search(reason):
        return None
    return {
        "verdict": verdict,
        "concrete_issue": concrete,
        "reason": reason,
        "confidence": confidence,
    }


def completion_text(payload: dict[str, Any]) -> str:
    return compact_json(
        {
            "concrete_issue": bool(payload["concrete_issue"]),
            "confidence": str(payload["confidence"]),
            "reason": str(payload["reason"]),
            "verdict": str(payload["verdict"]),
        }
    )


def confidence_rank(value: Any) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(value or "").lower(), 3)


def code_specific_score(reason: str) -> int:
    return min(5, len(CODE_TOKEN_RE.findall(reason)))


def generic_wrong_reason(reason: str) -> bool:
    text = compact_text(reason, max_chars=500)
    if GENERIC_OK_RE.search(text) and code_specific_score(text) == 0:
        return True
    return len(text.split()) < 10


def wrong_preselect_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    parsed = parse_completion(row["normalized_completion"]) or {}
    reason = str(parsed.get("reason") or "")
    return (
        int(generic_wrong_reason(reason)),
        -code_specific_score(reason),
        abs(len(reason) - 150),
        int(row.get("sample_index") or 999),
    )


def correct_preselect_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    parsed = parse_completion(row["normalized_completion"]) or {}
    reason = str(parsed.get("reason") or "")
    generic_ok = int(parsed.get("verdict") == "ok" and generic_wrong_reason(reason))
    return (
        generic_ok,
        -code_specific_score(reason),
        abs(len(reason) - 150),
        int(row.get("sample_index") or 999),
    )


def load_candidates(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in read_candidate_paths(patterns):
        rows.extend(read_jsonl(path))
    return rows


def selected_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompts = enrich_prompts(
        read_jsonl(args.prompts),
        source_judged=args.source_judged,
        source_pairs=args.source_pairs,
    )
    prompts = [row for row in prompts if row.get("task_type") in ACCEPTABILITY_TASKS]
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    return prompts


def select_wrong_candidates(
    prompts: list[dict[str, Any]],
    candidate_patterns: list[str],
    *,
    max_wrong_candidates_per_prompt: int,
    allow_generic_wrong_candidates: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts_by_id = {row["id"]: row for row in prompts}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    input_rows = load_candidates(candidate_patterns)

    for row in input_rows:
        prompt_id = str(row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id)
        if not prompt:
            drops["not_selected_prompt"] += 1
            continue
        value, normalized = normalize_candidate(row)
        if value is None or normalized is None:
            drops["invalid_candidate"] += 1
            continue
        _, expected_rejected = expected_values(prompt)
        if value != expected_rejected:
            drops["not_expected_wrong_value"] += 1
            continue
        parsed = parse_completion(normalized)
        if parsed is None:
            drops["bad_normalized_completion"] += 1
            continue
        if not allow_generic_wrong_candidates and generic_wrong_reason(parsed["reason"]):
            drops["locally_generic_wrong"] += 1
            continue
        key = (prompt_id, normalized.lower())
        if key in seen:
            drops["duplicate_normalized_wrong"] += 1
            continue
        seen.add(key)
        grouped[prompt_id].append(
            {
                **row,
                "prompt": prompt,
                "candidate_value": value,
                "normalized_completion": normalized,
                "local_code_specific_score": code_specific_score(parsed["reason"]),
                "local_generic_wrong": generic_wrong_reason(parsed["reason"]),
            }
        )

    selected: list[dict[str, Any]] = []
    missing_prompts = 0
    for prompt in prompts:
        items = grouped.get(prompt["id"], [])
        if not items:
            missing_prompts += 1
            continue
        items.sort(key=wrong_preselect_key)
        if max_wrong_candidates_per_prompt > 0:
            items = items[:max_wrong_candidates_per_prompt]
        selected.extend(items)

    summary = {
        "input_candidates": len(input_rows),
        "selected_wrong_candidates": len(selected),
        "selected_prompts": len({row["prompt_id"] for row in selected}),
        "prompts_without_selected_wrong": missing_prompts,
        "max_wrong_candidates_per_prompt": max_wrong_candidates_per_prompt,
        "allow_generic_wrong_candidates": allow_generic_wrong_candidates,
        "drop_counts": dict(sorted(drops.items())),
        "local_code_specific_score": summarize_numbers(
            [int(row.get("local_code_specific_score") or 0) for row in selected]
        ),
    }
    return selected, summary


def select_correct_candidates(
    prompts: list[dict[str, Any]],
    candidate_patterns: list[str],
    *,
    max_correct_candidates_per_prompt: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts_by_id = {row["id"]: row for row in prompts}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    input_rows = load_candidates(candidate_patterns)

    for row in input_rows:
        prompt_id = str(row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id)
        if not prompt:
            drops["not_selected_prompt"] += 1
            continue
        value, normalized = normalize_candidate(row)
        if value is None or normalized is None:
            drops["invalid_candidate"] += 1
            continue
        expected_chosen, _ = expected_values(prompt)
        if value != expected_chosen:
            drops["not_expected_correct_value"] += 1
            continue
        parsed = parse_completion(normalized)
        if parsed is None:
            drops["bad_normalized_completion"] += 1
            continue
        if parsed["verdict"] == "ok" and generic_wrong_reason(parsed["reason"]):
            drops["locally_generic_correct_ok"] += 1
            continue
        key = (prompt_id, normalized.lower())
        if key in seen:
            drops["duplicate_normalized_correct"] += 1
            continue
        seen.add(key)
        grouped[prompt_id].append(
            {
                **row,
                "prompt": prompt,
                "candidate_value": value,
                "normalized_completion": normalized,
                "local_code_specific_score": code_specific_score(parsed["reason"]),
                "local_generic_correct": parsed["verdict"] == "ok"
                and generic_wrong_reason(parsed["reason"]),
            }
        )

    selected: list[dict[str, Any]] = []
    missing_prompts = 0
    for prompt in prompts:
        items = grouped.get(prompt["id"], [])
        if not items:
            missing_prompts += 1
            continue
        items.sort(key=correct_preselect_key)
        if max_correct_candidates_per_prompt > 0:
            items = items[:max_correct_candidates_per_prompt]
        selected.extend(items)

    summary = {
        "input_candidates": len(input_rows),
        "selected_correct_candidates": len(selected),
        "selected_prompts": len({row["prompt_id"] for row in selected}),
        "prompts_without_selected_correct": missing_prompts,
        "max_correct_candidates_per_prompt": max_correct_candidates_per_prompt,
        "drop_counts": dict(sorted(drops.items())),
        "local_code_specific_score": summarize_numbers(
            [int(row.get("local_code_specific_score") or 0) for row in selected]
        ),
    }
    return selected, summary


def correct_user_prompt(prompt: dict[str, Any]) -> str:
    expected_chosen, _ = expected_values(prompt)
    meta = prompt.get("oracle_metadata") or {}
    if prompt["task_type"] == "vulnerable_acceptability":
        task_rule = (
            'The correct public answer should be verdict="issue" only if the target '
            "security defect is visible in the assistant answer."
        )
    else:
        task_rule = (
            'The correct public answer should be verdict="ok" only if the fixed answer '
            "does not visibly contain the same target defect. Do not call the code universally safe."
        )

    return f"""Construct the correct compact answer to this public security-acceptability prompt.

Expected correct verdict: {expected_chosen}
Task-specific rule: {task_rule}

Keep only if:
1. The correct judgement is visible from the public prompt's code/context.
2. The final answer can be understood without CVE, patch, diff, oracle, or counterpart code.
3. The reason is code-specific, security-relevant, and 18-35 words.
4. The answer does not mention CVE, CWE, patch, diff, oracle, PrimeVul, reference, before/after, or dataset.

Drop if the issue needs hidden repository context, the fixed code still visibly has the target defect,
or the only available explanation is broad/generic.

Private reference issue:
{reference_issue(meta)}

Private changed lines before:
{json.dumps(meta.get("changed_before") or [], ensure_ascii=True)}

Private changed lines after:
{json.dumps(meta.get("changed_after") or [], ensure_ascii=True)}

Public prompt:
<public_prompt>
{prompt["prompt"]}
</public_prompt>

Return exactly one compact JSON object matching this schema:
{{
  "decision": "keep" | "drop",
  "drop_reason": "none" | "issue_not_visible" | "fixed_still_suspicious" | "needs_hidden_context" | "not_security_relevant" | "malformed_prompt" | "other",
  "chosen": {{"verdict": "ok" | "issue", "concrete_issue": true | false, "reason": string, "confidence": "low" | "medium" | "high"}} | null,
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}
"""


def wrong_user_prompt(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    expected_chosen, expected_rejected = expected_values(prompt)
    meta = prompt.get("oracle_metadata") or {}
    parsed = parse_completion(row["normalized_completion"]) or {}
    return f"""Decide whether this base-model answer is a high-quality rejected completion.

Expected correct verdict: {expected_chosen}
Expected wrong verdict from this candidate: {expected_rejected}

A KEEP candidate must be all of:
1. Semantically wrong for this prompt, given the private reference issue.
2. Plausible as a competent model's mistake.
3. Code-specific, not generic boilerplate.
4. Not contradicted by an obvious check or statement in the visible code.
5. Not a style, formatting, naming, syntax, deprecated-API, or ordinary-quality complaint.
6. Preserves the base-model mistake; do not invent a different wrong reason.

Use "candidate_actually_correct" only if the candidate's expected-wrong
judgement is genuinely correct despite the private reference evidence. If the
candidate gives a generic expected-wrong answer, use "candidate_generic".

Quality score:
5 = excellent M0 mistake: specific, plausible, evidence-dependent, useful as DPO rejected.
4 = good M0 mistake: specific and plausible with minor wording cleanup.
3 = borderline; usually drop unless no better candidate exists.
1-2 = reject.

If kept, rewrite only enough to match the compact JSON schema and remove forbidden private/source wording.
Do not add new allegations not present in the base-model answer.

Private reference issue:
{reference_issue(meta)}

Private changed lines before:
{json.dumps(meta.get("changed_before") or [], ensure_ascii=True)}

Private changed lines after:
{json.dumps(meta.get("changed_after") or [], ensure_ascii=True)}

Public prompt:
<public_prompt>
{prompt["prompt"]}
</public_prompt>

Base-model candidate answer:
<candidate>
{completion_text(parsed)}
</candidate>

Return exactly one compact JSON object matching this schema:
{{
  "decision": "keep" | "drop",
  "drop_reason": "none" | "candidate_generic" | "candidate_not_code_specific" | "candidate_obviously_false" | "candidate_actually_correct" | "candidate_unrelated" | "candidate_style_or_syntax" | "candidate_needs_hidden_context" | "malformed_candidate" | "other",
  "rejected": {{"verdict": "ok" | "issue", "concrete_issue": true | false, "reason": string, "confidence": "low" | "medium" | "high"}} | null,
  "quality_score": 1,
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}
"""


def correct_candidate_user_prompt(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    expected_chosen, expected_rejected = expected_values(prompt)
    meta = prompt.get("oracle_metadata") or {}
    parsed = parse_completion(row["normalized_completion"]) or {}
    return f"""Decide whether this base-model answer is a high-quality chosen completion.

Expected correct verdict: {expected_chosen}
Opposite wrong verdict: {expected_rejected}

A KEEP candidate must be all of:
1. Semantically correct for this prompt, given the private reference issue.
2. Code-specific, not generic boilerplate.
3. Grounded in visible code, not CVE/patch/oracle/counterpart leakage.
4. Not a style, formatting, naming, syntax, deprecated-API, or ordinary-quality complaint.
5. Preserves the base-model reasoning; do not invent a different correct answer.

Quality score:
5 = excellent M0 answer: correct, specific, evidence-dependent, useful as DPO chosen.
4 = good M0 answer: correct and plausible with minor wording cleanup.
3 = borderline; usually drop unless no better candidate exists.
1-2 = reject.

If kept, rewrite only enough to match the compact JSON schema and remove forbidden private/source wording.

Private reference issue:
{reference_issue(meta)}

Private changed lines before:
{json.dumps(meta.get("changed_before") or [], ensure_ascii=True)}

Private changed lines after:
{json.dumps(meta.get("changed_after") or [], ensure_ascii=True)}

Public prompt:
<public_prompt>
{prompt["prompt"]}
</public_prompt>

Base-model candidate answer:
<candidate>
{completion_text(parsed)}
</candidate>

Return exactly one compact JSON object matching this schema:
{{
  "decision": "keep" | "drop",
  "drop_reason": "none" | "candidate_generic" | "candidate_not_code_specific" | "candidate_obviously_false" | "candidate_wrong" | "candidate_unrelated" | "candidate_style_or_syntax" | "candidate_needs_hidden_context" | "malformed_candidate" | "other",
  "chosen": {{"verdict": "ok" | "issue", "concrete_issue": true | false, "reason": string, "confidence": "low" | "medium" | "high"}} | null,
  "quality_score": 1,
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}
"""


def make_messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_batch_row(
    row: dict[str, Any], *, model: str, max_output_tokens: int, schema_name: str, schema: dict[str, Any]
) -> dict[str, Any]:
    return {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": row["messages"],
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        },
    }


def make_correct_input(prompt: dict[str, Any]) -> dict[str, Any]:
    messages = make_messages(CORRECT_SYSTEM_PROMPT, correct_user_prompt(prompt))
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": f"{prompt['id']}:correct",
        "prompt_id": prompt["id"],
        "task_type": prompt["task_type"],
        "expected_chosen": prompt["expected_chosen"],
        "expected_rejected": prompt["expected_rejected"],
        "source": prompt.get("source"),
        "messages": messages,
        "oracle_metadata": prompt.get("oracle_metadata") or {},
        "prompt_chars": len(prompt_text),
        "prompt_token_proxy": round(len(prompt_text) / 4),
    }


def make_correct_candidate_input(row: dict[str, Any]) -> dict[str, Any]:
    messages = make_messages(CORRECT_CANDIDATE_SYSTEM_PROMPT, correct_candidate_user_prompt(row))
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": f"{row['id']}:correct_quality",
        "candidate_id": row["id"],
        "prompt_id": row["prompt_id"],
        "task_type": row["task_type"],
        "candidate_value": row["candidate_value"],
        "expected_chosen": row["prompt"]["expected_chosen"],
        "expected_rejected": row["prompt"]["expected_rejected"],
        "normalized_candidate": row["normalized_completion"],
        "raw_generation": row.get("raw_generation"),
        "sample_index": row.get("sample_index"),
        "local_code_specific_score": row.get("local_code_specific_score"),
        "local_generic_correct": row.get("local_generic_correct"),
        "source": row.get("source"),
        "messages": messages,
        "oracle_metadata": row["prompt"].get("oracle_metadata") or {},
        "prompt_chars": len(prompt_text),
        "prompt_token_proxy": round(len(prompt_text) / 4),
    }


def make_wrong_input(row: dict[str, Any]) -> dict[str, Any]:
    messages = make_messages(WRONG_SYSTEM_PROMPT, wrong_user_prompt(row))
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": f"{row['id']}:wrong_quality",
        "candidate_id": row["id"],
        "prompt_id": row["prompt_id"],
        "task_type": row["task_type"],
        "candidate_value": row["candidate_value"],
        "expected_chosen": row["prompt"]["expected_chosen"],
        "expected_rejected": row["prompt"]["expected_rejected"],
        "normalized_candidate": row["normalized_completion"],
        "raw_generation": row.get("raw_generation"),
        "sample_index": row.get("sample_index"),
        "local_code_specific_score": row.get("local_code_specific_score"),
        "local_generic_wrong": row.get("local_generic_wrong"),
        "source": row.get("source"),
        "messages": messages,
        "oracle_metadata": row["prompt"].get("oracle_metadata") or {},
        "prompt_chars": len(prompt_text),
        "prompt_token_proxy": round(len(prompt_text) / 4),
    }


def write_stream(
    stream_dir: Path,
    *,
    stream_name: str,
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    model: str,
    max_output_tokens: int,
    schema_name: str,
    extra_manifest: dict[str, Any],
) -> None:
    stream_dir.mkdir(parents=True, exist_ok=True)
    batch_rows = [
        make_batch_row(
            row,
            model=model,
            max_output_tokens=max_output_tokens,
            schema_name=schema_name,
            schema=schema,
        )
        for row in rows
    ]
    write_jsonl(stream_dir / f"{stream_name}_inputs.jsonl", rows)
    write_jsonl(stream_dir / "openai_batch_requests.jsonl", batch_rows)
    write_json(stream_dir / f"{stream_name}_schema.json", schema)
    manifest = {
        "stream": stream_name,
        "model": model,
        "schema_name": schema_name,
        "max_output_tokens": max_output_tokens,
        "total_rows": len(rows),
        "files": {
            "generic_inputs": str(stream_dir / f"{stream_name}_inputs.jsonl"),
            "openai_batch_requests": str(stream_dir / "openai_batch_requests.jsonl"),
            "schema": str(stream_dir / f"{stream_name}_schema.json"),
        },
        "prompt_chars": summarize_numbers([row["prompt_chars"] for row in rows]),
        "prompt_token_proxy": summarize_numbers([row["prompt_token_proxy"] for row in rows]),
        **extra_manifest,
    }
    write_json(stream_dir / "manifest.json", manifest)


def build_requests(args: argparse.Namespace) -> dict[str, Any]:
    prompts = selected_prompts(args)
    wrong_candidates, wrong_selection = select_wrong_candidates(
        prompts,
        args.candidates,
        max_wrong_candidates_per_prompt=args.max_wrong_candidates_per_prompt,
        allow_generic_wrong_candidates=args.allow_generic_wrong_candidates,
    )
    prompt_ids_with_wrong = {row["prompt_id"] for row in wrong_candidates}
    correct_source = getattr(args, "correct_source", "oracle")
    correct_selection: dict[str, Any] = {"source": correct_source}
    if correct_source == "oracle":
        prompt_ids = prompt_ids_with_wrong
        correct_inputs = [make_correct_input(row) for row in prompts if row["id"] in prompt_ids]
        correct_schema = CORRECT_SCHEMA
    else:
        correct_candidates, correct_selection = select_correct_candidates(
            prompts,
            args.candidates,
            max_correct_candidates_per_prompt=args.max_correct_candidates_per_prompt,
        )
        prompt_ids_with_correct = {row["prompt_id"] for row in correct_candidates}
        prompt_ids = prompt_ids_with_wrong & prompt_ids_with_correct
        correct_inputs = [
            make_correct_candidate_input(row)
            for row in correct_candidates
            if row["prompt_id"] in prompt_ids
        ]
        correct_schema = CORRECT_CANDIDATE_SCHEMA
    wrong_inputs = [make_wrong_input(row) for row in wrong_candidates if row["prompt_id"] in prompt_ids]

    write_stream(
        args.out_dir / "correct",
        stream_name="correct",
        rows=correct_inputs,
        schema=correct_schema,
        model=args.model,
        max_output_tokens=args.correct_max_output_tokens,
        schema_name=f"{args.schema_name}_correct",
        extra_manifest={"prompts": str(args.prompts), "correct_selection": correct_selection},
    )
    write_stream(
        args.out_dir / "wrong",
        stream_name="wrong",
        rows=wrong_inputs,
        schema=WRONG_SCHEMA,
        model=args.model,
        max_output_tokens=args.wrong_max_output_tokens,
        schema_name=f"{args.schema_name}_wrong",
        extra_manifest={
            "prompts": str(args.prompts),
            "candidates": [str(path) for path in read_candidate_paths(args.candidates)],
            "wrong_selection": wrong_selection,
        },
    )
    summary = {
        "name": "primevul_dpo_pair_quality_v1",
        "prompts": str(args.prompts),
        "out_dir": str(args.out_dir),
        "model": args.model,
        "source_judged": str(args.source_judged),
        "source_pairs": str(args.source_pairs),
        "correct_requests": len(correct_inputs),
        "wrong_requests": len(wrong_inputs),
        "prompts_selected": len(prompts),
        "prompts_with_wrong_candidates": len(prompt_ids_with_wrong),
        "wrong_selection": wrong_selection,
        "correct_selection": correct_selection,
        "contract": {
            "correct": "OpenAI constructs one compact correct chosen completion per prompt from private oracle evidence.",
            "wrong": "OpenAI keeps only plausible, code-specific, semantically wrong M0 candidates.",
            "reducer": "Pairs are formed only when both streams keep the row; rejected candidates are ranked by M0 mistake quality.",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def submit_stream(stream_dir: Path, *, metadata_job: str) -> dict[str, Any]:
    requests_path = stream_dir / "openai_batch_requests.jsonl"
    build_manifest = json.loads((stream_dir / "manifest.json").read_text())
    headers = openai_headers()
    with requests_path.open("rb") as file:
        upload = requests.post(
            "https://api.openai.com/v1/files",
            headers=headers,
            files={"file": (requests_path.name, file, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    if upload.status_code >= 400:
        raise RuntimeError(f"OpenAI file upload error {upload.status_code}: {upload.text[:1000]}")
    file_obj = upload.json()
    payload = {
        "input_file_id": file_obj["id"],
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": {"job": metadata_job, "source": Path(__file__).name},
    }
    batch = requests.post(
        "https://api.openai.com/v1/batches",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if batch.status_code >= 400:
        raise RuntimeError(f"OpenAI batch create error {batch.status_code}: {batch.text[:1000]}")
    batch_obj = batch.json()
    submission = {
        "batch_id": batch_obj["id"],
        "input_file_id": file_obj["id"],
        "requests_path": str(requests_path),
        "rows_path": build_manifest["files"]["generic_inputs"],
        "num_requests": sum(1 for _ in requests_path.open()),
        "batch": batch_obj,
    }
    write_json(stream_dir / "openai_batch_submission.json", submission)
    with (stream_dir / "openai_batch_manifests.jsonl").open("a") as out:
        out.write(json.dumps(submission, sort_keys=True) + "\n")
    return submission


def latest_batch_id(stream_dir: Path) -> str | None:
    path = stream_dir / "openai_batch_submission.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["batch_id"]


def collect_stream(stream_dir: Path, *, batch_id: str | None = None) -> list[dict[str, Any]]:
    batch_id = batch_id or latest_batch_id(stream_dir)
    if not batch_id:
        raise SystemExit(f"No batch id found for {stream_dir}")
    batch = retrieve_batch(batch_id)
    write_json(stream_dir / "openai_batch_status.json", batch)
    if batch["status"] not in {"completed", "expired", "cancelled", "failed"}:
        print(json.dumps(batch, indent=2, sort_keys=True), flush=True)
        return []
    if batch.get("error_file_id"):
        (stream_dir / f"openai_batch_{batch_id}_errors.jsonl").write_text(
            download_openai_file(batch["error_file_id"])
        )
    if not batch.get("output_file_id"):
        raise RuntimeError(f"Batch {batch_id} ended with status={batch['status']} and no output file")

    manifest = json.loads((stream_dir / "manifest.json").read_text())
    rows_by_id = {row["id"]: row for row in read_jsonl(manifest["files"]["generic_inputs"])}
    output_text = download_openai_file(batch["output_file_id"])
    (stream_dir / f"openai_batch_{batch_id}_output.jsonl").write_text(output_text)
    parsed_rows = []
    for line in output_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        source_row = rows_by_id[item["custom_id"]]
        error = item.get("error")
        response = item.get("response")
        usage = None
        judge_model = None
        if error or not response or response.get("status_code", 500) >= 400:
            parsed = {"parse_error": True, "raw_judge_output": json.dumps(error or response, sort_keys=True)}
        else:
            body = response["body"]
            usage = body.get("usage")
            judge_model = body.get("model")
            parsed = parse_response_json(response_text(body))
        parsed_rows.append(
            {
                **source_row,
                "messages": None,
                "judge_provider": "openai_batch",
                "judge_model": judge_model,
                "batch_id": batch_id,
                "usage": usage,
                **parsed,
            }
        )
    stream = manifest["stream"]
    write_jsonl(stream_dir / f"{stream}_judged.jsonl", parsed_rows)
    write_stream_summary(parsed_rows, stream_dir, stream=stream, batch=batch)
    return parsed_rows


def write_stream_summary(
    rows: list[dict[str, Any]], stream_dir: Path, *, stream: str, batch: dict[str, Any] | None
) -> None:
    decisions = Counter(str(row.get("decision") or "missing") for row in rows)
    drop_reasons = Counter(str(row.get("drop_reason") or "missing") for row in rows)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    usage_totals = Counter()
    quality_scores = Counter()
    for row in rows:
        by_task[str(row.get("task_type") or "missing")][str(row.get("decision") or "missing")] += 1
        if stream == "wrong":
            quality_scores[str(row.get("quality_score") or "missing")] += 1
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int | float):
                usage_totals[key] += value
    total = len(rows)
    summary = {
        "stream": stream,
        "total_rows": total,
        "parse_failures": sum(1 for row in rows if row.get("parse_error")),
        "decision_counts": dict(decisions),
        "decision_rates": {key: round(value / total, 6) for key, value in decisions.items()} if total else {},
        "drop_reason_counts": dict(drop_reasons),
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "quality_score_counts": dict(sorted(quality_scores.items())),
        "usage_totals": dict(usage_totals),
        "batch_request_counts": (batch or {}).get("request_counts"),
    }
    write_json(stream_dir / f"{stream}_summary.json", summary)


def valid_correct(row: dict[str, Any]) -> str | None:
    if row.get("parse_error") or row.get("decision") != "keep":
        return None
    if "quality_score" in row and int(row.get("quality_score") or 0) < row.get(
        "_min_correct_quality", 4
    ):
        return None
    payload = parse_completion(row.get("chosen"))
    if payload is None:
        return None
    if payload["verdict"] != row.get("expected_chosen"):
        return None
    return completion_text(payload)


def valid_wrong(row: dict[str, Any], *, min_wrong_quality: int) -> str | None:
    if row.get("parse_error") or row.get("decision") != "keep":
        return None
    if int(row.get("quality_score") or 0) < min_wrong_quality:
        return None
    payload = parse_completion(row.get("rejected"))
    if payload is None:
        return None
    if payload["verdict"] != row.get("expected_rejected"):
        return None
    return completion_text(payload)


def pair_rank(
    correct: dict[str, Any], chosen_text: str, wrong: dict[str, Any], rejected_text: str
) -> tuple[int, int, int, int, float, int, int]:
    chosen_len = max(1, len(chosen_text))
    rejected_len = max(1, len(rejected_text))
    ratio = chosen_len / rejected_len
    rejected = parse_completion(rejected_text) or {}
    chosen = parse_completion(chosen_text) or {}
    return (
        -int(correct.get("quality_score") or 5),
        -int(wrong.get("quality_score") or 0),
        confidence_rank(correct.get("confidence")),
        confidence_rank(wrong.get("confidence")),
        abs(1.0 - ratio),
        -code_specific_score(str(chosen.get("reason") or "")),
        -code_specific_score(str(rejected.get("reason") or "")),
    )


def apply_acceptability_mix(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The previous 35/45/10/10 target reduces to 35/45 when pair-comparison rows
    # are intentionally excluded from this stricter acceptability-only build.
    import random

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["dpo_type"]].append(row)
    if not by_task["vulnerable_acceptability"] or not by_task["fixed_acceptability"]:
        return rows, {"applied": False, "reason": "both acceptability task types not present"}

    vulnerable_frac = 35 / 80
    fixed_frac = 45 / 80
    max_total = min(
        int(len(by_task["vulnerable_acceptability"]) / vulnerable_frac),
        int(len(by_task["fixed_acceptability"]) / fixed_frac),
    )
    if max_total <= 0:
        return rows, {"applied": False, "reason": "insufficient rows for 35:45 mix"}
    vulnerable_count = round(max_total * vulnerable_frac)
    desired = {
        "vulnerable_acceptability": vulnerable_count,
        "fixed_acceptability": max_total - vulnerable_count,
    }
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for task, count in desired.items():
        task_rows = list(by_task[task])
        rng.shuffle(task_rows)
        selected.extend(task_rows[:count])
    rng.shuffle(selected)
    return selected, {"applied": True, "desired_counts": desired, "rows": len(selected)}


def build_dpo(args: argparse.Namespace) -> dict[str, Any]:
    prompts_by_id = {row["id"]: row for row in selected_prompts(args)}
    correct_rows = read_jsonl(args.correct_judged)
    wrong_rows = read_jsonl(args.wrong_judged)
    min_correct_quality = getattr(args, "min_correct_quality", 4)
    correct_by_prompt: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for row in correct_rows:
        row["_min_correct_quality"] = min_correct_quality
        text = valid_correct(row)
        if text is not None:
            correct_by_prompt[row["prompt_id"]].append((row, text))
    wrong_by_prompt: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for row in wrong_rows:
        text = valid_wrong(row, min_wrong_quality=args.min_wrong_quality)
        if text is not None:
            wrong_by_prompt[row["prompt_id"]].append((row, text))

    drops: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    chosen_source_label = "m0_candidate_quality_judge" if "m0_correct" in str(args.correct_judged) else "openai_correct_constructor"
    chosen_contract = (
        "Best high-quality correct M0 candidate after OpenAI plausibility filtering."
        if chosen_source_label == "m0_candidate_quality_judge"
        else "OpenAI-constructed correct compact answer from private oracle evidence."
    )
    for prompt_id, prompt in prompts_by_id.items():
        correct_options = correct_by_prompt.get(prompt_id) or []
        if not correct_options:
            drops[f"missing_correct:{prompt['task_type']}"] += 1
            continue
        wrong_options = wrong_by_prompt.get(prompt_id) or []
        if not wrong_options:
            drops[f"missing_quality_wrong:{prompt['task_type']}"] += 1
            continue
        ranked: list[
            tuple[
                tuple[int, int, int, int, float, int, int],
                dict[str, Any],
                str,
                dict[str, Any],
                str,
            ]
        ] = []
        for correct_row, chosen_text in correct_options:
            for wrong_row, rejected_text in wrong_options:
                ratio = len(chosen_text) / max(1, len(rejected_text))
                if ratio < args.min_length_ratio or ratio > args.max_length_ratio:
                    drops[f"length_ratio_no_pair:{prompt['task_type']}"] += 1
                    continue
                ranked.append(
                    (
                        pair_rank(correct_row, chosen_text, wrong_row, rejected_text),
                        wrong_row,
                        rejected_text,
                        correct_row,
                        chosen_text,
                    )
                )
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0])
        wrong_row = ranked[0][1]
        rejected_text = ranked[0][2]
        correct_row = ranked[0][3]
        chosen_text = ranked[0][4]
        rows.append(
            {
                "id": prompt_id.replace("primevul_dpo_candidate_", "primevul_dpo_"),
                "source": "primevul",
                "dpo_type": prompt["task_type"],
                "prompt": prompt["prompt"],
                "chosen": chosen_text,
                "rejected": rejected_text,
                "oracle_metadata": {
                    **(prompt.get("oracle_metadata") or {}),
                    "chosen_source": chosen_source_label,
                    "rejected_source": "m0_candidate_quality_judge",
                    "chosen_constructor_id": correct_row["id"],
                    "chosen_candidate_id": correct_row.get("candidate_id"),
                    "chosen_quality_score": correct_row.get("quality_score"),
                    "rejected_candidate_id": wrong_row["candidate_id"],
                    "chosen_judge_reason": correct_row.get("reason"),
                    "rejected_judge_reason": wrong_row.get("reason"),
                    "rejected_quality_score": wrong_row.get("quality_score"),
                    "rejected_raw_generation": wrong_row.get("raw_generation"),
                    "rejected_normalized_candidate": wrong_row.get("normalized_candidate"),
                },
            }
        )

    mix_summary = {"applied": False, "reason": "not requested"}
    if args.apply_acceptability_mix:
        rows, mix_summary = apply_acceptability_mix(rows, seed=args.seed)

    args.dpo_out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dpo_out_dir / "train.jsonl", rows)
    counts = Counter(row["dpo_type"] for row in rows)
    summary = {
        "name": "primevul_dpo_recognition_v3_strict",
        "train_jsonl": str(args.dpo_out_dir / "train.jsonl"),
        "correct_judged": str(args.correct_judged),
        "wrong_judged": str(args.wrong_judged),
        "rows": len(rows),
        "source_pairs": len({row["oracle_metadata"].get("source_pair_id") for row in rows}),
        "dpo_type_counts": dict(sorted(counts.items())),
        "drop_counts": dict(sorted(drops.items())),
        "min_wrong_quality": args.min_wrong_quality,
        "length_ratio_gate": [args.min_length_ratio, args.max_length_ratio],
        "acceptability_mix": mix_summary,
        "contract": {
            "chosen": chosen_contract,
            "rejected": "Best high-quality wrong M0 candidate after OpenAI plausibility filtering.",
            "pair_comparison": "excluded in this strict first pass; standalone acceptability is the causal variable.",
        },
    }
    write_json(args.dpo_out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def stream_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    return [("correct", args.out_dir / "correct"), ("wrong", args.out_dir / "wrong")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--source-judged", type=Path, default=DEFAULT_SOURCE_JUDGED)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE_PAIRS)
    parser.add_argument("--candidates", nargs="+", default=[DEFAULT_CANDIDATES_GLOB])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpo-out-dir", type=Path, default=DEFAULT_DPO_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--schema-name", default="primevul_dpo_pair_quality_v1")
    parser.add_argument("--correct-max-output-tokens", type=int, default=360)
    parser.add_argument("--wrong-max-output-tokens", type=int, default=360)
    parser.add_argument("--correct-source", choices=["oracle", "m0_candidate"], default="oracle")
    parser.add_argument("--max-correct-candidates-per-prompt", type=int, default=16)
    parser.add_argument("--max-wrong-candidates-per-prompt", type=int, default=16)
    parser.add_argument("--allow-generic-wrong-candidates", action="store_true")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--correct-batch-id")
    parser.add_argument("--wrong-batch-id")
    parser.add_argument("--show-smoke-prompts", action="store_true")
    parser.add_argument("--smoke-chars", type=int, default=3500)
    parser.add_argument("--build-dpo", action="store_true")
    parser.add_argument("--correct-judged", type=Path)
    parser.add_argument("--wrong-judged", type=Path)
    parser.add_argument("--min-correct-quality", type=int, default=4)
    parser.add_argument("--min-wrong-quality", type=int, default=4)
    parser.add_argument("--min-length-ratio", type=float, default=0.75)
    parser.add_argument("--max-length-ratio", type=float, default=1.33)
    parser.add_argument("--apply-acceptability-mix", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build or args.submit or args.show_smoke_prompts:
        summary = build_requests(args)
        if args.show_smoke_prompts:
            for stream, stream_dir in stream_dirs(args):
                rows = read_jsonl(stream_dir / f"{stream}_inputs.jsonl")
                print("\n" + "=" * 88)
                print(f"{stream.upper()} PROMPT EXAMPLE")
                print(rows[0]["messages"][1]["content"][: args.smoke_chars])
            return
        print(f"BUILT {summary['correct_requests']} correct and {summary['wrong_requests']} wrong requests")

    if args.submit:
        for stream, stream_dir in stream_dirs(args):
            manifest = submit_stream(
                stream_dir,
                metadata_job=f"{args.schema_name}_{stream}",
            )
            print(
                f"SUBMITTED {stream} batch_id={manifest['batch_id']} requests={manifest['num_requests']}",
                flush=True,
            )

    if args.collect or args.wait:
        batch_ids = {"correct": args.correct_batch_id, "wrong": args.wrong_batch_id}
        while True:
            statuses = {}
            for stream, stream_dir in stream_dirs(args):
                batch_id = batch_ids[stream] or latest_batch_id(stream_dir)
                if not batch_id:
                    raise SystemExit(f"No batch id found for {stream}")
                batch = retrieve_batch(batch_id)
                statuses[stream] = batch["status"]
                print(
                    f"BATCH {stream} {batch_id} status={batch['status']} counts={batch.get('request_counts')}",
                    flush=True,
                )
                if batch["status"] in {"completed", "expired", "cancelled", "failed"}:
                    collect_stream(stream_dir, batch_id=batch_id)
            if all(status in {"completed", "expired", "cancelled", "failed"} for status in statuses.values()):
                break
            if not args.wait:
                break
            time.sleep(args.poll_interval)

    if args.build_dpo:
        if args.correct_judged is None:
            args.correct_judged = args.out_dir / "correct" / "correct_judged.jsonl"
        if args.wrong_judged is None:
            args.wrong_judged = args.out_dir / "wrong" / "wrong_judged.jsonl"
        build_dpo(args)


if __name__ == "__main__":
    main()
