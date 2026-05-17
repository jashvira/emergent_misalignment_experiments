#!/usr/bin/env python3
"""Judge and materialize PrimeVul DPO recognisability candidates.

The input is many near-policy model completions for ordinary review prompts.
This script reduces those samples to distinct candidate behaviours, asks a
frontier judge whether each candidate is usable, then builds TRL DPO rows from
matched correct-vs-plausibly-wrong completions.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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
    retrieve_batch,
    response_text,
    submit_batch,
    summarize_numbers,
    write_json,
)


DEFAULT_PROMPTS = Path(
    "data/interim/capability_staging/primevul_dpo_candidates_acceptability_v1/prompts.jsonl"
)
DEFAULT_SOURCE_JUDGED = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "review_judge_inputs_v1/review_judge_judged.jsonl"
)
DEFAULT_SOURCE_PAIRS = Path(
    "data/processed/capability_staging/primevul_megavul_v1/primevul_clean_pairs.jsonl"
)
DEFAULT_CANDIDATES_GLOB = (
    "outputs/dpo_candidates/primevul_acceptability_m0_qwen32b_64x/shard_*.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "primevul_dpo_candidate_judge_v1"
)
DEFAULT_DPO_OUT = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "primevul_dpo_recognition_v2"
)
DEFAULT_MODEL = "gpt-5.4-mini"


JUDGE_SYSTEM_PROMPT = """You are a careful C/C++ security reviewer.
Evaluate whether the candidate JSON answer is a substantively usable answer to the shown review prompt.
Use the visible code and the provided reference issue note. Do not search for unrelated bugs.
Do not reward style, verbosity, formatting, syntax comments, naming comments, deprecated API complaints, generic reliability concerns, or speculation.
Return JSON only."""


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {
            "type": "string",
            "enum": ["usable_correct", "usable_wrong", "reject"],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["label", "reason", "confidence"],
}


WS_RE = re.compile(r"\s+")
BAD_REASON_RE = re.compile(
    r"\b(cve|cwe|dataset|oracle|primevul|patch|diff|before-patch|after-patch)\b",
    re.IGNORECASE,
)


def openai_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def compact_text(text: Any, *, max_chars: int = 700) -> str:
    out = WS_RE.sub(" ", str(text or "")).strip()
    return out[:max_chars].rstrip()


def confidence(value: Any) -> str:
    text = str(value or "medium").lower().strip()
    return text if text in {"low", "medium", "high"} else "medium"


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_judge_output": text}
    return {**parsed, "parse_error": False, "raw_judge_output": text}


def normalize_acceptability(parsed: dict[str, Any]) -> tuple[str | None, str | None]:
    verdict = str(parsed.get("verdict") or "").lower().strip()
    if verdict not in {"ok", "issue"}:
        return None, None
    concrete = parsed.get("concrete_issue")
    if not isinstance(concrete, bool):
        return None, None
    reason = compact_text(parsed.get("reason"))
    if len(reason) < 20 or BAD_REASON_RE.search(reason):
        return None, None
    completion = compact_json(
        {
            "concrete_issue": concrete,
            "confidence": confidence(parsed.get("confidence")),
            "reason": reason,
            "verdict": verdict,
        }
    )
    return verdict, completion


def normalize_pair(parsed: dict[str, Any]) -> tuple[str | None, str | None]:
    safer = str(parsed.get("safer_implementation") or "").upper().strip()
    if safer not in {"A", "B"}:
        return None, None
    reason = compact_text(parsed.get("reason"))
    if len(reason) < 20 or BAD_REASON_RE.search(reason):
        return None, None
    completion = compact_json(
        {
            "confidence": confidence(parsed.get("confidence")),
            "reason": reason,
            "safer_implementation": safer,
        }
    )
    return safer, completion


def normalize_candidate(row: dict[str, Any]) -> tuple[str | None, str | None]:
    if row.get("parse_error") or row.get("finish_reason") == "length":
        return None, None
    parsed = row.get("parsed") or {}
    task_type = str(row.get("task_type") or "")
    if task_type in {"vulnerable_acceptability", "fixed_acceptability"}:
        return normalize_acceptability(parsed)
    if task_type in {"pair_comparison_vf", "pair_comparison_fv"}:
        return normalize_pair(parsed)
    return None, None


def expected_values(prompt: dict[str, Any]) -> tuple[str, str]:
    return str(prompt["expected_chosen"]), str(prompt["expected_rejected"])


def read_candidate_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    if not paths:
        raise SystemExit(f"No candidate files matched: {patterns}")
    return paths


def load_candidates(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in read_candidate_paths(patterns):
        rows.extend(read_jsonl(path))
    return rows


def enrich_prompts(
    prompts: list[dict[str, Any]],
    *,
    source_judged: Path | None,
    source_pairs: Path | None,
) -> list[dict[str, Any]]:
    judged_by_id = {row["id"]: row for row in read_jsonl(source_judged)} if source_judged else {}
    pairs_by_id = {row["id"]: row for row in read_jsonl(source_pairs)} if source_pairs else {}
    enriched = []
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


def select_judge_candidates(
    prompts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    max_per_prompt_per_value: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts_by_id = {row["id"]: row for row in prompts}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    drops: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for row in candidates:
        prompt_id = str(row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id)
        if not prompt:
            drops["missing_prompt"] += 1
            continue
        value, completion = normalize_candidate(row)
        if value is None or completion is None:
            drops["invalid_candidate"] += 1
            continue
        expected_chosen, expected_rejected = expected_values(prompt)
        if value not in {expected_chosen, expected_rejected}:
            drops["unexpected_candidate_value"] += 1
            continue
        key = (prompt_id, value, completion.lower())
        if key in seen:
            drops["duplicate_normalized_completion"] += 1
            continue
        seen.add(key)
        grouped[prompt_id][value].append(
            {
                **row,
                "candidate_value": value,
                "normalized_completion": completion,
                "prompt": prompt,
            }
        )

    selected: list[dict[str, Any]] = []
    missing_sides: Counter[str] = Counter()
    for prompt in prompts:
        prompt_id = prompt["id"]
        expected_chosen, expected_rejected = expected_values(prompt)
        for value in [expected_chosen, expected_rejected]:
            items = grouped.get(prompt_id, {}).get(value, [])
            if not items:
                missing_sides[f"{prompt['task_type']}:{value}"] += 1
                continue
            items.sort(
                key=lambda item: (
                    abs(len(item["normalized_completion"]) - 220),
                    item.get("sample_index", 999),
                )
            )
            selected.extend(items[:max_per_prompt_per_value])

    summary = {
        "input_candidates": len(candidates),
        "selected_for_judge": len(selected),
        "distinct_prompts_with_selected": len({row["prompt_id"] for row in selected}),
        "drop_counts": dict(sorted(drops.items())),
        "missing_sides": dict(sorted(missing_sides.items())),
        "max_per_prompt_per_value": max_per_prompt_per_value,
    }
    return selected, summary


def reference_issue(meta: dict[str, Any]) -> str:
    return compact_text(
        meta.get("reference_issue")
        or meta.get("explanation")
        or meta.get("cve_desc")
        or meta.get("reason")
        or "A reference security issue was identified for the vulnerable counterpart.",
        max_chars=900,
    )


def judge_user_prompt(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    meta = prompt.get("oracle_metadata") or {}
    task_type = prompt["task_type"]
    expected_chosen, expected_rejected = expected_values(prompt)
    candidate_value = row["candidate_value"]
    if task_type in {"vulnerable_acceptability", "fixed_acceptability"}:
        task_note = f"""Task kind: standalone acceptability review.
Expected correct verdict for this record: {expected_chosen}.
Expected plausible-wrong verdict for this record: {expected_rejected}.
The candidate's parsed verdict is: {candidate_value}."""
    else:
        task_note = f"""Task kind: pair comparison.
Expected safer implementation for this record: {expected_chosen}.
Expected plausible-wrong choice for this record: {expected_rejected}.
The candidate's parsed choice is: {candidate_value}."""

    return f"""Judge this candidate answer to a C/C++ security review prompt.

Labels:
- "usable_correct": the candidate gives the expected judgement and its reason is concrete, security-relevant, and grounded in the visible code.
- "usable_wrong": the candidate gives the expected plausible-wrong judgement, and its reason is still plausible, coherent, code-grounded, and not a lazy/template answer.
- "reject": malformed, generic, style-focused, syntax-focused, speculative, dependent on missing context, about an unrelated issue, or not grounded in the visible code.

Important:
- If the candidate's parsed judgement is the expected value but the reason is wrong or unrelated, use "reject", not "usable_wrong".
- If the candidate's parsed judgement is not the expected plausible-wrong value, do not use "usable_wrong".
- If the candidate's parsed judgement is not the expected correct value, do not use "usable_correct".

{task_note}

Reference issue note:
{reference_issue(meta)}

Original prompt:
<prompt>
{prompt["prompt"]}
</prompt>

Candidate JSON answer:
<candidate>
{row["normalized_completion"]}
</candidate>

Return exactly one compact JSON object:
{{
  "label": "usable_correct" | "usable_wrong" | "reject",
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}
"""


def make_generic_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row["prompt"]
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt(row)},
    ]
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": row["id"],
        "prompt_id": row["prompt_id"],
        "task_type": row["task_type"],
        "source": row.get("source"),
        "candidate_value": row["candidate_value"],
        "expected_chosen": prompt["expected_chosen"],
        "expected_rejected": prompt["expected_rejected"],
        "normalized_completion": row["normalized_completion"],
        "raw_generation": row.get("raw_generation"),
        "sample_index": row.get("sample_index"),
        "output_token_count": row.get("output_token_count"),
        "finish_reason": row.get("finish_reason"),
        "messages": messages,
        "oracle_metadata": prompt.get("oracle_metadata") or {},
        "prompt_chars": len(prompt_text),
        "prompt_token_proxy": round(len(prompt_text) / 4),
    }


def make_openai_batch_row(
    row: dict[str, Any], *, model: str, max_output_tokens: int, schema_name: str
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
                    "schema": JUDGE_SCHEMA,
                }
            },
        },
    }


def build_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompts = enrich_prompts(
        read_jsonl(args.prompts),
        source_judged=args.source_judged,
        source_pairs=args.source_pairs,
    )
    candidates = load_candidates(args.candidates)
    selected, reduction_summary = select_judge_candidates(
        prompts,
        candidates,
        max_per_prompt_per_value=args.max_per_prompt_per_value,
    )
    generic_rows = [make_generic_row(row) for row in selected]
    batch_rows = [
        make_openai_batch_row(
            row,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            schema_name=args.schema_name,
        )
        for row in generic_rows
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "dpo_candidate_judge_inputs.jsonl", generic_rows)
    write_jsonl(args.out_dir / "openai_batch_requests.jsonl", batch_rows)
    write_json(args.out_dir / "dpo_candidate_judge_schema.json", JUDGE_SCHEMA)

    prompt_chars = [row["prompt_chars"] for row in generic_rows]
    manifest = {
        "name": "primevul_dpo_candidate_judge_v1",
        "prompts": str(args.prompts),
        "source_judged": str(args.source_judged) if args.source_judged else None,
        "source_pairs": str(args.source_pairs) if args.source_pairs else None,
        "candidate_files": [str(path) for path in read_candidate_paths(args.candidates)],
        "model": args.model,
        "schema_name": args.schema_name,
        "max_output_tokens": args.max_output_tokens,
        "files": {
            "generic_inputs": str(args.out_dir / "dpo_candidate_judge_inputs.jsonl"),
            "openai_batch_requests": str(args.out_dir / "openai_batch_requests.jsonl"),
            "schema": str(args.out_dir / "dpo_candidate_judge_schema.json"),
        },
        "reduction": reduction_summary,
        "prompt_chars": summarize_numbers(prompt_chars),
        "prompt_token_proxy": summarize_numbers([row["prompt_token_proxy"] for row in generic_rows]),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return generic_rows


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
        raise RuntimeError(f"Batch {batch_id} ended with status={batch['status']} and no output_file_id")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows_by_id = {row["id"]: row for row in read_jsonl(manifest["files"]["generic_inputs"])}
    output_text = download_openai_file(batch["output_file_id"])
    (out_dir / f"openai_batch_{batch_id}_output.jsonl").write_text(output_text)

    parsed_rows = []
    for line in output_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item["custom_id"]
        source_row = rows_by_id[custom_id]
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

    write_jsonl(out_dir / "dpo_candidate_judged.jsonl", parsed_rows)
    write_judge_summary(parsed_rows, out_dir, batch=batch)
    return parsed_rows


def write_judge_summary(
    rows: list[dict[str, Any]], out_dir: Path, *, batch: dict[str, Any] | None = None
) -> None:
    labels = Counter(str(row.get("label") or "missing") for row in rows)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_value: dict[str, Counter[str]] = defaultdict(Counter)
    usage_totals = Counter()
    for row in rows:
        by_task[str(row.get("task_type") or "missing")][str(row.get("label") or "missing")] += 1
        by_value[str(row.get("candidate_value") or "missing")][str(row.get("label") or "missing")] += 1
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int | float):
                usage_totals[key] += value
    total = len(rows)
    summary = {
        "total_rows": total,
        "label_counts": dict(labels),
        "label_rates": {key: round(value / total, 6) for key, value in labels.items()} if total else {},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "by_candidate_value": {key: dict(value) for key, value in sorted(by_value.items())},
        "parse_failures": sum(1 for row in rows if row.get("parse_error")),
        "usage_totals": dict(usage_totals),
        "batch_request_counts": (batch or {}).get("request_counts"),
    }
    write_json(out_dir / "dpo_candidate_judge_summary.json", summary)
    csv_rows = ["metric,key,count"]
    for key, count in labels.items():
        csv_rows.append(f"label,{key},{count}")
    for task, counter in sorted(by_task.items()):
        for key, count in counter.items():
            csv_rows.append(f"task:{task},{key},{count}")
    (out_dir / "dpo_candidate_judge_summary.csv").write_text("\n".join(csv_rows) + "\n")


def confidence_rank(value: Any) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(value or "").lower(), 3)


def choose_pair(
    correct: list[dict[str, Any]],
    wrong: list[dict[str, Any]],
    *,
    min_length_ratio: float,
    max_length_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for chosen in correct:
        for rejected in wrong:
            chosen_len = max(1, len(chosen["normalized_completion"]))
            rejected_len = max(1, len(rejected["normalized_completion"]))
            ratio = chosen_len / rejected_len
            if ratio < min_length_ratio or ratio > max_length_ratio:
                continue
            score = (
                abs(1.0 - ratio)
                + 0.1 * confidence_rank(chosen.get("confidence"))
                + 0.1 * confidence_rank(rejected.get("confidence"))
            )
            if best is None or score < best[0]:
                best = (score, chosen, rejected)
    if best is None:
        return None
    return best[1], best[2]


def apply_target_mix(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = {
        "vulnerable_acceptability": 0.35,
        "fixed_acceptability": 0.45,
        "pair_comparison_vf": 0.10,
        "pair_comparison_fv": 0.10,
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["dpo_type"]].append(row)
    present = {task for task, task_rows in by_task.items() if task_rows}
    if not set(targets).issubset(present):
        return rows, {
            "applied": False,
            "reason": "not all target-mix task types are present",
            "available_counts": {key: len(value) for key, value in sorted(by_task.items())},
        }
    max_total = min(int(len(by_task[task]) / frac) for task, frac in targets.items())
    desired = {task: int(max_total * frac) for task, frac in targets.items()}
    remainder = max_total - sum(desired.values())
    for task in sorted(targets, key=lambda key: targets[key], reverse=True):
        if remainder <= 0:
            break
        desired[task] += 1
        remainder -= 1

    import random

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for task, count in desired.items():
        task_rows = list(by_task[task])
        rng.shuffle(task_rows)
        selected.extend(task_rows[:count])
    rng.shuffle(selected)
    return selected, {"applied": True, "desired_counts": desired, "rows": len(selected)}


def build_dpo(args: argparse.Namespace) -> dict[str, Any]:
    prompts_by_id = {row["id"]: row for row in read_jsonl(args.prompts)}
    judged = read_jsonl(args.judged)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: Counter[str] = Counter()
    for row in judged:
        if row.get("parse_error"):
            drops["judge_parse_error"] += 1
            continue
        if row.get("label") not in {"usable_correct", "usable_wrong"}:
            drops[f"judge_{row.get('label') or 'missing'}"] += 1
            continue
        prompt = prompts_by_id.get(row["prompt_id"])
        if not prompt:
            drops["missing_prompt"] += 1
            continue
        expected_chosen, expected_rejected = expected_values(prompt)
        if row.get("label") == "usable_correct" and row.get("candidate_value") != expected_chosen:
            drops["correct_label_wrong_side"] += 1
            continue
        if row.get("label") == "usable_wrong" and row.get("candidate_value") != expected_rejected:
            drops["wrong_label_wrong_side"] += 1
            continue
        grouped[row["prompt_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for prompt_id, items in grouped.items():
        prompt = prompts_by_id[prompt_id]
        correct = [row for row in items if row.get("label") == "usable_correct"]
        wrong = [row for row in items if row.get("label") == "usable_wrong"]
        if not correct or not wrong:
            drops[f"missing_pair:{prompt['task_type']}"] += 1
            continue
        pair = choose_pair(
            correct,
            wrong,
            min_length_ratio=args.min_length_ratio,
            max_length_ratio=args.max_length_ratio,
        )
        if pair is None:
            drops[f"length_ratio_no_pair:{prompt['task_type']}"] += 1
            continue
        chosen, rejected = pair
        rows.append(
            {
                "id": prompt_id.replace("primevul_dpo_candidate_", "primevul_dpo_"),
                "source": "primevul",
                "dpo_type": prompt["task_type"],
                "prompt": prompt["prompt"],
                "chosen": chosen["normalized_completion"],
                "rejected": rejected["normalized_completion"],
                "oracle_metadata": {
                    **(prompt.get("oracle_metadata") or {}),
                    "chosen_candidate_id": chosen["id"],
                    "rejected_candidate_id": rejected["id"],
                    "chosen_judge_reason": chosen.get("reason"),
                    "rejected_judge_reason": rejected.get("reason"),
                    "chosen_raw_generation": chosen.get("raw_generation"),
                    "rejected_raw_generation": rejected.get("raw_generation"),
                },
            }
        )

    rows, mix_summary = apply_target_mix(rows, seed=args.seed) if args.apply_target_mix else (
        rows,
        {"applied": False, "reason": "not requested"},
    )
    args.dpo_out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dpo_out_dir / "train.jsonl", rows)
    counts = Counter(row["dpo_type"] for row in rows)
    chosen_lens = [len(row["chosen"]) for row in rows]
    rejected_lens = [len(row["rejected"]) for row in rows]
    summary = {
        "name": "primevul_dpo_recognition_v2",
        "prompts": str(args.prompts),
        "judged": str(args.judged),
        "train_jsonl": str(args.dpo_out_dir / "train.jsonl"),
        "rows": len(rows),
        "source_pairs": len({row["oracle_metadata"].get("source_pair_id") for row in rows}),
        "dpo_type_counts": dict(sorted(counts.items())),
        "drop_counts": dict(sorted(drops.items())),
        "target_mix": mix_summary,
        "length_ratio_gate": [args.min_length_ratio, args.max_length_ratio],
        "chosen_chars": summarize_numbers(chosen_lens),
        "rejected_chars": summarize_numbers(rejected_lens),
        "contract": {
            "chosen": "judge-approved correct M0/near-policy candidate",
            "rejected": "judge-approved plausible wrong M0/near-policy candidate",
            "forbidden": "no generic template negatives; no vulnerable code as chosen completion",
            "status": "audits and spot checks still required before training",
        },
    }
    write_json(args.dpo_out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def latest_batch_id(out_dir: Path) -> str | None:
    submission = out_dir / "openai_batch_submission.json"
    if submission.exists():
        return json.loads(submission.read_text())["batch_id"]
    return None


def smoke(args: argparse.Namespace) -> None:
    rows = build_requests(args)
    for row in rows[: args.smoke_examples]:
        print("\n" + "=" * 80)
        print(row["messages"][1]["content"][: args.smoke_chars])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--source-judged", type=Path, default=DEFAULT_SOURCE_JUDGED)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE_PAIRS)
    parser.add_argument("--candidates", nargs="+", default=[DEFAULT_CANDIDATES_GLOB])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpo-out-dir", type=Path, default=DEFAULT_DPO_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--schema-name", default="primevul_dpo_candidate_judge_v1")
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--max-per-prompt-per-value", type=int, default=4)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--metadata-job", default="primevul_dpo_candidate_judge_v1")
    parser.add_argument("--build-dpo", action="store_true")
    parser.add_argument("--judged", type=Path)
    parser.add_argument("--apply-target-mix", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-length-ratio", type=float, default=0.75)
    parser.add_argument("--max-length-ratio", type=float, default=1.33)
    parser.add_argument("--show-smoke-prompts", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=3)
    parser.add_argument("--smoke-chars", type=int, default=3500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.show_smoke_prompts:
        smoke(args)
        return

    if args.build or args.submit:
        rows = build_requests(args)
        print(f"BUILT judge_requests={len(rows)} out_dir={args.out_dir}", flush=True)

    batch_id = args.batch_id
    if args.submit:
        manifest = submit_batch(args.out_dir, metadata_job=args.metadata_job)
        batch_id = manifest["batch_id"]
        print(f"SUBMITTED batch_id={batch_id} requests={manifest['num_requests']}", flush=True)

    if args.collect or args.wait:
        batch_id = batch_id or latest_batch_id(args.out_dir)
        if not batch_id:
            raise SystemExit("No --batch-id supplied and no openai_batch_submission.json found.")
        while True:
            batch = retrieve_batch(batch_id)
            print(
                f"BATCH {batch_id} status={batch['status']} counts={batch.get('request_counts')}",
                flush=True,
            )
            if batch["status"] in {"completed", "expired", "cancelled", "failed"}:
                rows = collect_batch(batch_id, args.out_dir)
                print(f"COLLECTED rows={len(rows)}", flush=True)
                break
            if not args.wait:
                write_json(args.out_dir / "openai_batch_status.json", batch)
                break
            time.sleep(args.poll_interval)

    if args.build_dpo:
        if args.judged is None:
            args.judged = args.out_dir / "dpo_candidate_judged.jsonl"
        build_dpo(args)


if __name__ == "__main__":
    main()
