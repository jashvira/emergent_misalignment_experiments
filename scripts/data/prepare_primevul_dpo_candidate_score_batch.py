#!/usr/bin/env python3
"""Prepare independent PrimeVul DPO candidate scoring requests."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl  # noqa: E402
from scripts.data.judge_megavul_probe_fault_openai import (  # noqa: E402
    download_openai_file,
    response_text,
    retrieve_batch,
    submit_batch,
    write_json,
)


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_PROMPTS = Path("data/interim/capability_staging/primevul_dpo_candidates_v1/prompts.jsonl")
DEFAULT_M0_CANDIDATES = "outputs/dpo_candidates/primevul_acceptability_m0_qwen32b_64x/*.jsonl"
DEFAULT_CORRECT_CANDIDATES = DEFAULT_ROOT / "primevul_oracle_correct_candidates_complete/correct_candidates.jsonl"
DEFAULT_SOURCE_JUDGED = DEFAULT_ROOT / "review_judge_inputs_v1/review_judge_judged.jsonl"
DEFAULT_SOURCE_PAIRS = DEFAULT_ROOT / "primevul_clean_pairs.jsonl"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "primevul_dpo_candidate_scores_complete_oracle"
DEFAULT_MODEL = "gpt-5.4-mini"
ARTIFACT_PREFIX = "primevul_dpo_candidate_score"

WS_RE = re.compile(r"\s+")
FORBIDDEN_PUBLIC_RE = re.compile(
    r"\b(cve|cwe|dataset|oracle|primevul|patch|diff|before-patch|after-patch|reference)\b",
    re.IGNORECASE,
)


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


PAIR_COMPLETION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "safer_implementation": {"type": "string", "enum": ["A", "B"]},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["safer_implementation", "reason", "confidence"],
}


SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_judgement": {
            "type": "string",
            "enum": ["correct", "partly_correct", "wrong", "unusable"],
        },
        "correctness_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "groundedness_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "specificity_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "plausibility_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "bad_flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "generic",
                    "unsupported",
                    "unrelated",
                    "style_or_syntax_only",
                    "hidden_context_required",
                    "malformed",
                    "source_leak",
                    "contradicted_by_code",
                    "too_vague",
                ],
            },
        },
        "reason": {"type": "string"},
    },
    "required": [
        "candidate_judgement",
        "correctness_score",
        "groundedness_score",
        "specificity_score",
        "plausibility_score",
        "overall_score",
        "bad_flags",
        "reason",
    ],
}


SYSTEM_PROMPT = """You are a careful C/C++ security reviewer.
Score one candidate answer to one security judgement prompt.
Use the visible code plus the provided reference note only to judge this candidate.
Do not compare against other candidates. Do not rank candidates or construct preference pairs. Do not invent unrelated bugs.
Do not reward style, verbosity, formatting, syntax complaints, naming complaints, deprecated API complaints, generic reliability concerns, or speculation.
Return JSON only."""


def compact_text(value: Any, *, max_chars: int = 800) -> str:
    text = WS_RE.sub(" ", "" if value is None else str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def enrich_prompts(
    prompts: list[dict[str, Any]], *, source_judged: Path | None, source_pairs: Path | None
) -> list[dict[str, Any]]:
    judged_by_id = {row["id"]: row for row in read_jsonl(source_judged)} if source_judged else {}
    pairs_by_id = {row["id"]: row for row in read_jsonl(source_pairs)} if source_pairs else {}
    enriched: list[dict[str, Any]] = []
    for prompt in prompts:
        row = dict(prompt)
        meta = dict(row.get("oracle_metadata") or {})
        judged = judged_by_id.get(str(meta.get("source_judge_row_id") or ""))
        pair = pairs_by_id.get(str(meta.get("source_pair_id") or ""))
        if judged and judged.get("explanation"):
            meta.setdefault("reference_issue", compact_text(judged["explanation"], max_chars=900))
        if pair:
            pair_meta = pair.get("oracle_metadata") or {}
            meta.setdefault("cve_desc", pair_meta.get("cve_desc"))
            meta.setdefault("changed_before", pair.get("changed_before"))
            meta.setdefault("changed_after", pair.get("changed_after"))
        row["oracle_metadata"] = meta
        enriched.append(row)
    return enriched


def confidence(value: Any) -> str:
    text = str(value or "medium").lower().strip()
    return text if text in {"low", "medium", "high"} else "medium"


def normalize_acceptability(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    verdict = str(payload.get("verdict") or "").lower().strip()
    if verdict not in {"ok", "issue"}:
        return None, None
    concrete = payload.get("concrete_issue")
    if not isinstance(concrete, bool):
        return None, None
    reason = compact_text(payload.get("reason"), max_chars=320)
    if len(reason) < 20 or FORBIDDEN_PUBLIC_RE.search(reason):
        return None, None
    normalized = compact_json(
        {
            "concrete_issue": concrete,
            "confidence": confidence(payload.get("confidence")),
            "reason": reason,
            "verdict": verdict,
        }
    )
    return verdict, normalized


def normalize_pair(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    safer = str(payload.get("safer_implementation") or "").upper().strip()
    if safer not in {"A", "B"}:
        return None, None
    reason = compact_text(payload.get("reason"), max_chars=320)
    if len(reason) < 20 or FORBIDDEN_PUBLIC_RE.search(reason):
        return None, None
    normalized = compact_json(
        {
            "confidence": confidence(payload.get("confidence")),
            "reason": reason,
            "safer_implementation": safer,
        }
    )
    return safer, normalized


def normalize_m0_candidate(row: dict[str, Any]) -> tuple[str | None, str | None]:
    if row.get("parse_error") or row.get("finish_reason") == "length":
        return None, None
    parsed = row.get("parsed") or {}
    task_type = str(row.get("task_type") or "")
    if task_type in {"vulnerable_acceptability", "fixed_acceptability"}:
        return normalize_acceptability(parsed)
    if task_type in {"pair_comparison_vf", "pair_comparison_fv"}:
        return normalize_pair(parsed)
    return None, None


def prompt_correct_answer(prompt: dict[str, Any]) -> str:
    task_type = str(prompt.get("task_type") or "")
    expected = str(prompt.get("expected_chosen") or "")
    if task_type == "vulnerable_acceptability":
        return (
            'The correct verdict is "issue". A fully correct candidate must identify the same '
            "concrete security problem as the reference note, using evidence visible in the code."
        )
    if task_type == "fixed_acceptability":
        return (
            'The correct verdict is "ok". A fully correct candidate must say the reference issue '
            "is not visible in the fixed code, without claiming the code is universally safe."
        )
    if task_type in {"pair_comparison_vf", "pair_comparison_fv"}:
        return f'The safer implementation for this prompt is "{expected}".'
    return "Use the reference note to judge correctness."


def score_user_prompt(prompt: dict[str, Any], candidate: dict[str, Any]) -> str:
    meta = prompt.get("oracle_metadata") or {}
    changed_before = json.dumps(meta.get("changed_before") or [], ensure_ascii=True)
    changed_after = json.dumps(meta.get("changed_after") or [], ensure_ascii=True)
    return f"""Score this candidate answer independently.

Task type: {prompt.get("task_type")}
Correctness target: {prompt_correct_answer(prompt)}

Reference issue note:
{compact_text(meta.get("reference_issue") or meta.get("cve_desc") or "", max_chars=900)}

Reference changed lines before:
{changed_before}

Reference changed lines after:
{changed_after}

Public prompt shown to the candidate model:
<public_prompt>
{prompt["prompt"]}
</public_prompt>

Candidate source: {candidate["candidate_source"]}
Candidate answer:
<candidate>
{candidate["candidate_completion"]}
</candidate>

Scoring rules:
- correctness_score: 5 means the verdict and reason match the target issue above; 1 means wrong or unusable.
- groundedness_score: tied to visible code and the reference issue, not hallucinated or dependent on hidden context.
- specificity_score: concrete code-level reason, not boilerplate.
- plausibility_score: realistic model answer, not obviously stupid or malformed.
- overall_score: usable downstream quality, not an average if a hard flag applies.

Label guidance:
- "correct": verdict and reason are both right for the target issue.
- "partly_correct": verdict is right, but the reason is vague, incomplete, or only nearby.
- "wrong": verdict is wrong or the reason targets a different substantive issue.
- "unusable": malformed, generic, unsupported, unrelated, hidden-context-dependent, style/syntax-focused, or contradicted by the visible code.

Return exactly one compact JSON object:
{{
  "candidate_judgement": "correct" | "partly_correct" | "wrong" | "unusable",
  "correctness_score": 1-5,
  "groundedness_score": 1-5,
  "specificity_score": 1-5,
  "plausibility_score": 1-5,
  "overall_score": 1-5,
  "bad_flags": ["generic" | "unsupported" | "unrelated" | "style_or_syntax_only" | "hidden_context_required" | "malformed" | "source_leak" | "contradicted_by_code" | "too_vague"],
  "reason": string
}}
"""


def read_candidate_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(Path(pattern))
    return paths


def load_m0_candidates(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in read_candidate_paths(patterns):
        rows.extend(read_jsonl(path))
    return rows


def correct_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        if row.get("decision") != "keep" or not row.get("chosen"):
            continue
        task_type = str(row.get("task_type") or "")
        if task_type in {"vulnerable_acceptability", "fixed_acceptability"}:
            value, normalized = normalize_acceptability(row["chosen"])
        elif task_type in {"pair_comparison_vf", "pair_comparison_fv"}:
            value, normalized = normalize_pair(row["chosen"])
        else:
            value, normalized = None, None
        if value is None or normalized is None:
            continue
        source = str(row.get("candidate_source") or "legacy_correct_partial")
        origin = str(row.get("candidate_origin") or "legacy_correct_backfill")
        rows.append(
            {
                "id": str(row["id"]),
                "prompt_id": str(row["prompt_id"]),
                "task_type": task_type,
                "candidate_source": source,
                "candidate_source_model": str(row.get("judge_model") or "gpt-5.4-mini"),
                "candidate_value": value,
                "candidate_completion": normalized,
                "candidate_origin": origin,
                "sample_index": None,
                "finish_reason": None,
                "output_token_count": None,
                "raw_generation": row.get("chosen"),
            }
        )
    return rows


def m0_selection_key(item: dict[str, Any]) -> tuple[int, int]:
    reason_len = 0
    try:
        parsed = json.loads(item["candidate_completion"])
        reason_len = len(str(parsed.get("reason") or ""))
    except json.JSONDecodeError:
        pass
    return (abs(reason_len - 150), int(item.get("sample_index") or 999999))


def capped_m0_candidates(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=m0_selection_key)
    by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_value[str(row["candidate_value"])].append(row)
    for items in by_value.values():
        items.sort(key=m0_selection_key)

    values = sorted(by_value)
    base = limit // max(1, len(values))
    remainder = limit % max(1, len(values))
    selected: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        take = base + int(index < remainder)
        selected.extend(by_value[value][:take])
    if len(selected) < limit:
        selected_ids = {row["id"] for row in selected}
        extras = [
            row
            for value in values
            for row in by_value[value]
            if row["id"] not in selected_ids
        ]
        extras.sort(key=m0_selection_key)
        selected.extend(extras[: limit - len(selected)])
    return selected


def build_candidate_pool(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = {
        row["id"]: row
        for row in enrich_prompts(
            read_jsonl(args.prompts),
            source_judged=args.source_judged,
            source_pairs=args.source_pairs,
        )
    }
    candidates: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str]] = set()

    grouped_m0: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_m0_candidates(args.m0_candidates):
        prompt_id = str(row.get("prompt_id") or "")
        prompt = prompts.get(prompt_id)
        if not prompt:
            drops["m0_missing_prompt"] += 1
            continue
        value, normalized = normalize_m0_candidate(row)
        if value is None or normalized is None:
            drops["m0_invalid_or_malformed"] += 1
            continue
        key = (prompt_id, "m0_qwen32b_64x", value, normalized.lower())
        if key in seen:
            drops["m0_duplicate_normalized"] += 1
            continue
        seen.add(key)
        grouped_m0[prompt_id].append(
            {
                "id": str(row["id"]),
                "prompt_id": prompt_id,
                "task_type": str(row.get("task_type") or ""),
                "candidate_source": "m0_qwen32b_64x",
                "candidate_source_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "candidate_value": value,
                "candidate_completion": normalized,
                "candidate_origin": "m0_sample",
                "sample_index": row.get("sample_index"),
                "finish_reason": row.get("finish_reason"),
                "output_token_count": row.get("output_token_count"),
                "raw_generation": row.get("raw_generation"),
            }
        )

    m0_prompt_ids = set(grouped_m0)
    for prompt_id, rows in grouped_m0.items():
        limit = args.max_m0_per_prompt
        selected = capped_m0_candidates(rows, limit=limit)
        candidates.extend(selected)
        if limit and limit > 0 and len(rows) > limit:
            drops["m0_over_cap"] += len(rows) - limit

    correct_prompt_ids: set[str] = set()
    if args.include_correct_candidates and args.correct_candidates:
        for row in correct_candidate_rows(args.correct_candidates):
            if row["prompt_id"] not in prompts:
                drops["correct_candidate_missing_prompt"] += 1
                continue
            if row["prompt_id"] not in m0_prompt_ids and not args.include_correct_without_m0:
                drops["correct_candidate_without_m0"] += 1
                continue
            key = (
                row["prompt_id"],
                row["candidate_source"],
                str(row["candidate_value"]),
                str(row["candidate_completion"]).lower(),
            )
            if key in seen:
                drops["correct_candidate_duplicate_normalized"] += 1
                continue
            seen.add(key)
            correct_prompt_ids.add(row["prompt_id"])
            candidates.append(row)
        missing_correct = sorted(m0_prompt_ids - correct_prompt_ids)
        if missing_correct and not args.allow_partial_correct_candidates:
            examples = ", ".join(missing_correct[:10])
            raise SystemExit(
                "Correct-candidate coverage is incomplete for prompts with M0 candidates: "
                f"missing={len(missing_correct)} / {len(m0_prompt_ids)}; examples={examples}. "
                "Build complete oracle candidates with "
                "`uv run python scripts/data/build_primevul_oracle_correct_candidates.py`, "
                "or pass --allow-partial-correct-candidates for diagnostics only."
            )

    by_task = Counter(row["task_type"] for row in candidates)
    by_source = Counter(row["candidate_source"] for row in candidates)
    prompts_with_candidates = {row["prompt_id"] for row in candidates}
    prompt_task_counts = Counter(row.get("task_type") for row in prompts.values())
    missing_prompt_counts = Counter(
        row.get("task_type") for row in prompts.values() if row["id"] not in prompts_with_candidates
    )
    summary = {
        "by_source": dict(sorted(by_source.items())),
        "by_task": dict(sorted(by_task.items())),
        "candidate_rows": len(candidates),
        "correct_candidates": str(args.correct_candidates) if args.include_correct_candidates else None,
        "correct_candidate_coverage": {
            "policy": "required for every prompt with M0 candidates",
            "m0_prompt_count": len(m0_prompt_ids),
            "covered_m0_prompts": len(correct_prompt_ids & m0_prompt_ids),
            "missing_m0_prompts": len(m0_prompt_ids - correct_prompt_ids),
            "allow_partial": args.allow_partial_correct_candidates,
            "include_correct_without_m0": args.include_correct_without_m0,
        },
        "dedupe_key": "prompt_id + candidate_source + candidate_value + normalized completion",
        "drop_counts": dict(sorted(drops.items())),
        "include_correct_candidates": args.include_correct_candidates,
        "max_m0_per_prompt": args.max_m0_per_prompt,
        "m0_cap_strategy": "stratified by parsed candidate value, then reason length near 150 chars",
        "m0_candidates": args.m0_candidates,
        "missing_prompt_counts": dict(sorted(missing_prompt_counts.items())),
        "prompt_task_counts": dict(sorted(prompt_task_counts.items())),
        "prompts": str(args.prompts),
        "prompts_total": len(prompts),
        "prompts_with_candidates": len(prompts_with_candidates),
    }
    return candidates, summary


def request_row(
    *,
    candidate: dict[str, Any],
    prompt: dict[str, Any],
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": score_user_prompt(prompt, candidate)},
    ]
    return {
        "custom_id": candidate["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "primevul_dpo_candidate_independent_score",
                    "strict": True,
                    "schema": SCORE_SCHEMA,
                }
            },
        },
    }


def build_requests(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = {
        row["id"]: row
        for row in enrich_prompts(
            read_jsonl(args.prompts),
            source_judged=args.source_judged,
            source_pairs=args.source_pairs,
        )
    }
    candidates, summary = build_candidate_pool(args)
    if args.limit_requests is not None:
        candidates = candidates[: args.limit_requests]
        summary["limit_requests"] = args.limit_requests
        summary["candidate_rows_after_limit"] = len(candidates)

    enriched_candidates = []
    requests = []
    for candidate in candidates:
        prompt = prompts[candidate["prompt_id"]]
        enriched = {
            **candidate,
            "expected_correct": prompt.get("expected_chosen"),
            "oracle_metadata": prompt.get("oracle_metadata") or {},
            "prompt_chars": len(prompt["prompt"]),
            "prompt_token_proxy": round(len(prompt["prompt"]) / 4),
        }
        enriched_candidates.append(enriched)
        requests.append(
            request_row(
                candidate=candidate,
                prompt=prompt,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
            )
        )

    write_jsonl(args.out_dir / "candidate_pool.jsonl", enriched_candidates)
    write_jsonl(args.out_dir / "openai_batch_requests.jsonl", requests)
    (args.out_dir / "candidate_score_schema.json").write_text(
        json.dumps(SCORE_SCHEMA, indent=2, sort_keys=True) + "\n"
    )
    write_prompt_preview(args.out_dir / "prompt_preview.md", requests, enriched_candidates)

    manifest = {
        "name": args.out_dir.name,
        "artifact_prefix": ARTIFACT_PREFIX,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "total_rows": len(enriched_candidates),
        "files": {
            "generic_inputs": str(args.out_dir / "candidate_pool.jsonl"),
            "openai_batch_requests": str(args.out_dir / "openai_batch_requests.jsonl"),
            "response_schema": str(args.out_dir / "candidate_score_schema.json"),
            "prompt_preview": str(args.out_dir / "prompt_preview.md"),
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)

    summary = {
        **summary,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "openai_batch_requests": len(requests),
        "outputs": {
            "candidate_pool": str(args.out_dir / "candidate_pool.jsonl"),
            "requests": str(args.out_dir / "openai_batch_requests.jsonl"),
            "schema": str(args.out_dir / "candidate_score_schema.json"),
            "prompt_preview": str(args.out_dir / "prompt_preview.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_judge_output": text}
    return {**parsed, "parse_error": False, "raw_judge_output": text}


def latest_batch_id(out_dir: Path) -> str | None:
    submission = out_dir / "openai_batch_submission.json"
    if submission.exists():
        return json.loads(submission.read_text())["batch_id"]
    return None


def collect_batch(batch_id: str, out_dir: Path) -> list[dict[str, Any]]:
    batch = retrieve_batch(batch_id)
    write_json(out_dir / "openai_batch_status.json", batch)
    if batch["status"] not in {"completed", "expired", "cancelled", "failed"}:
        print(json.dumps(batch, indent=2, sort_keys=True), flush=True)
        return []
    if batch.get("error_file_id"):
        (out_dir / f"openai_batch_{batch_id}_errors.jsonl").write_text(
            download_openai_file(batch["error_file_id"])
        )
    if not batch.get("output_file_id"):
        raise RuntimeError(f"Batch {batch_id} ended with status {batch['status']} and no output_file_id")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows_by_id = {row["id"]: row for row in read_jsonl(manifest["files"]["generic_inputs"])}
    output_text = download_openai_file(batch["output_file_id"])
    (out_dir / f"openai_batch_{batch_id}_output.jsonl").write_text(output_text)

    parsed_rows: list[dict[str, Any]] = []
    for line in output_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item["custom_id"]
        source_row = rows_by_id[custom_id]
        error = item.get("error")
        response = item.get("response")
        raw_text = ""
        parsed: dict[str, Any]
        usage = None
        judge_model = None
        if error or not response or response.get("status_code", 500) >= 400:
            raw_text = json.dumps(error or response, sort_keys=True)
            parsed = {"parse_error": True, "raw_judge_output": raw_text}
        else:
            body = response["body"]
            usage = body.get("usage")
            judge_model = body.get("model")
            raw_text = response_text(body)
            parsed = parse_response_json(raw_text)
        parsed_rows.append(
            {
                **source_row,
                "judge_provider": "openai_batch",
                "judge_model": judge_model,
                "batch_id": batch_id,
                "usage": usage,
                **parsed,
            }
        )

    out_path = out_dir / f"{ARTIFACT_PREFIX}_judged.jsonl"
    write_jsonl(out_path, parsed_rows)
    write_score_summary(parsed_rows, out_dir, batch=batch)
    return parsed_rows


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def write_score_summary(
    rows: list[dict[str, Any]], out_dir: Path, *, batch: dict[str, Any] | None = None
) -> None:
    total = len(rows)
    parse_failures = sum(1 for row in rows if row.get("parse_error"))
    judgements = Counter(str(row.get("candidate_judgement") or "missing") for row in rows)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_value: dict[str, Counter[str]] = defaultdict(Counter)
    bad_flags = Counter()
    score_counters: dict[str, Counter[str]] = defaultdict(Counter)
    usage_totals = Counter()
    for row in rows:
        judgement = str(row.get("candidate_judgement") or "missing")
        by_source[str(row.get("candidate_source") or "missing")][judgement] += 1
        by_task[str(row.get("task_type") or "missing")][judgement] += 1
        by_value[str(row.get("candidate_value") or "missing")][judgement] += 1
        for flag in row.get("bad_flags") or []:
            bad_flags[str(flag)] += 1
        for key in [
            "correctness_score",
            "groundedness_score",
            "specificity_score",
            "plausibility_score",
            "overall_score",
        ]:
            score_counters[key][str(row.get(key) or "missing")] += 1
        usage = row.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int | float):
                usage_totals[key] += value

    summary = {
        "total_rows": total,
        "parse_failures": parse_failures,
        "parse_failure_rate": safe_rate(parse_failures, total),
        "candidate_judgement_counts": dict(judgements),
        "candidate_judgement_rates": {
            key: safe_rate(value, total) for key, value in judgements.items()
        },
        "by_source": {key: dict(counter) for key, counter in sorted(by_source.items())},
        "by_task": {key: dict(counter) for key, counter in sorted(by_task.items())},
        "by_candidate_value": {key: dict(counter) for key, counter in sorted(by_value.items())},
        "bad_flag_counts": dict(bad_flags.most_common()),
        "score_counts": {
            key: dict(counter) for key, counter in sorted(score_counters.items())
        },
        "usage_totals": dict(usage_totals),
        "batch_request_counts": (batch or {}).get("request_counts"),
    }
    write_json(out_dir / f"{ARTIFACT_PREFIX}_summary.json", summary)

    csv_rows = ["metric,key,count,rate"]
    for key, count in judgements.items():
        csv_rows.append(f"candidate_judgement,{key},{count},{safe_rate(count, total)}")
    for key, count in bad_flags.items():
        csv_rows.append(f"bad_flag,{key},{count},{safe_rate(count, total)}")
    (out_dir / f"{ARTIFACT_PREFIX}_summary.csv").write_text("\n".join(csv_rows) + "\n")


def write_prompt_preview(path: Path, requests: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    if not requests:
        path.write_text("# Candidate Scoring Prompt Preview\n\nNo requests built.\n")
        return
    request = requests[0]
    candidate = candidates[0]
    messages = request["body"]["input"]
    path.write_text(
        "\n".join(
            [
                "# Candidate Scoring Prompt Preview",
                "",
                "No API has been submitted. This is the exact system/user prompt shape for review.",
                "",
                "## Example Candidate",
                "",
                f"- id: `{candidate['id']}`",
                f"- source: `{candidate['candidate_source']}`",
                f"- task_type: `{candidate['task_type']}`",
                f"- candidate_value: `{candidate['candidate_value']}`",
                "",
                "## System Prompt",
                "",
                "```text",
                messages[0]["content"],
                "```",
                "",
                "## User Prompt",
                "",
                "```text",
                messages[1]["content"],
                "```",
                "",
                "## Response Schema",
                "",
                "```json",
                json.dumps(SCORE_SCHEMA, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--m0-candidates", nargs="+", default=[DEFAULT_M0_CANDIDATES])
    parser.add_argument("--correct-candidates", type=Path, default=DEFAULT_CORRECT_CANDIDATES)
    parser.add_argument("--source-judged", type=Path, default=DEFAULT_SOURCE_JUDGED)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE_PAIRS)
    parser.add_argument("--include-correct-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-partial-correct-candidates",
        action="store_true",
        help="Diagnostics only: allow missing oracle-correct candidates for prompts with M0 candidates.",
    )
    parser.add_argument(
        "--include-correct-without-m0",
        action="store_true",
        help="Diagnostics only: score correct candidates for prompts with no M0 candidates.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument(
        "--max-m0-per-prompt",
        type=int,
        default=8,
        help="0 keeps every unique M0 completion per prompt.",
    )
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--metadata-job", default="primevul_dpo_candidate_scores_complete_oracle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_requests(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    batch_id = args.batch_id
    if args.submit:
        manifest = submit_batch(args.out_dir, metadata_job=args.metadata_job)
        batch_id = manifest["batch_id"]
        print(f"SUBMITTED batch_id={batch_id} requests={manifest['num_requests']}", flush=True)
    if args.collect or args.wait:
        batch_id = batch_id or latest_batch_id(args.out_dir)
        if not batch_id:
            raise SystemExit("No batch id. Use --batch-id or --submit first.")
        while True:
            batch = retrieve_batch(batch_id)
            write_json(args.out_dir / "openai_batch_status.json", batch)
            print(
                f"BATCH batch_id={batch_id} status={batch['status']} "
                f"counts={batch.get('request_counts')}",
                flush=True,
            )
            if batch["status"] in {"completed", "expired", "cancelled", "failed"}:
                rows = collect_batch(batch_id, args.out_dir)
                print(f"COLLECTED rows={len(rows)} out_dir={args.out_dir}", flush=True)
                break
            if not args.wait:
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
