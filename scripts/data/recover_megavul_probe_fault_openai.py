#!/usr/bin/env python3
"""Recover clean MegaVul probes from strict probe-fault judgments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from scripts.data.judge_megavul_probe_fault_openai import (
    DEFAULT_ROOT,
    download_openai_file,
    effective_verdict,
    judge_user_prompt,
    openai_headers,
    response_text,
    retrieve_batch,
    safe_rate,
    source_metadata,
    write_json,
)


DEFAULT_IN_DIR = DEFAULT_ROOT / "megavul_probe_fault_judge_v1"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "megavul_probe_fault_recovery_v1"
DEFAULT_CLEAN_DIR = DEFAULT_ROOT / "megavul_clean_probe_set_v1"
DEFAULT_MODEL = "gpt-5.4-mini"


RECOVERY_SYSTEM_PROMPT = """You are a careful C/C++ vulnerability-record auditor.
This is a second-pass recovery check after a stricter filter may have rejected usable records.
Return JSON only."""


RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["recover_clean", "keep_probe_fault"]},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["decision", "reason", "confidence"],
}


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_judge_output": text}
    return {**parsed, "parse_error": False, "raw_judge_output": text}


def prior_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(path)}


def candidate_pairs(root: Path, prior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_pairs = {row["id"]: row for row in read_jsonl(root / "megavul_target_pairs.jsonl")}
    ids = [row["id"] for row in prior_rows if effective_verdict(row) == "probe_fault"]
    missing = [row_id for row_id in ids if row_id not in all_pairs]
    if missing:
        raise SystemExit(f"Missing target pairs for {len(missing)} probe-fault rows; first={missing[:3]}")
    return [all_pairs[row_id] for row_id in ids]


def recovery_user_prompt(row: dict[str, Any], prior: dict[str, Any]) -> str:
    prior_summary = {
        "verdict": prior.get("verdict"),
        "fault": prior.get("fault"),
        "reference_issue": prior.get("reference_issue"),
        "reason": prior.get("reason"),
        "confidence": prior.get("confidence"),
    }
    base_record = judge_user_prompt(row)
    return f"""Decide whether this rejected record should be recovered as a usable clean local vulnerability probe.

Recover it as "recover_clean" if:
- the before-patch function visibly contains a concrete security-relevant logical issue, and
- the after-patch function plausibly fixes that same issue, and
- this can be judged from the shown function, diff, and metadata.

Keep it as "keep_probe_fault" only if the reference issue is not locally visible, the patch is unrelated or non-behavioral, the after-patch function still has the same issue, the pair is malformed/non-comparable, or external repository context is essential.

Do not require full exploit proof. Do not reject because unrelated possible bugs remain. Do not reject for style, comments, deprecated APIs, or incomplete project context.

Prior strict judgment, which may be wrong:
{json.dumps(prior_summary, sort_keys=True)}

Return exactly one compact JSON object:
{{
  "decision": "recover_clean" | "keep_probe_fault",
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}

Original record:
{base_record}
"""


def make_row(row: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
        {"role": "user", "content": recovery_user_prompt(row, prior)},
    ]
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": row["id"],
        "messages": messages,
        "source_metadata": source_metadata(row),
        "prior_judgment": {
            "verdict": prior.get("verdict"),
            "fault": prior.get("fault"),
            "reference_issue": prior.get("reference_issue"),
            "reason": prior.get("reason"),
            "confidence": prior.get("confidence"),
        },
        "prompt_chars": len(prompt_text),
        "prompt_token_proxy": round(len(prompt_text) / 4),
    }


def make_openai_batch_row(row: dict[str, Any], *, model: str, max_output_tokens: int) -> dict[str, Any]:
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
                    "name": "megavul_probe_fault_recovery_v1",
                    "strict": True,
                    "schema": RECOVERY_SCHEMA,
                }
            },
        },
    }


def summarize_numbers(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def pct(q: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
        "total": sum(ordered),
    }


def build_inputs(
    root: Path,
    in_dir: Path,
    out_dir: Path,
    *,
    model: str,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    prior_rows = read_jsonl(in_dir / "megavul_probe_fault_judged.jsonl")
    priors = prior_by_id(in_dir / "megavul_probe_fault_judged.jsonl")
    pairs = candidate_pairs(root, prior_rows)
    rows = [make_row(row, priors[row["id"]]) for row in pairs]
    batch_rows = [
        make_openai_batch_row(row, model=model, max_output_tokens=max_output_tokens)
        for row in rows
    ]

    write_jsonl(out_dir / "megavul_probe_fault_recovery_inputs.jsonl", rows)
    write_jsonl(out_dir / "openai_batch_requests.jsonl", batch_rows)
    write_json(out_dir / "megavul_probe_fault_recovery_schema.json", RECOVERY_SCHEMA)
    manifest = {
        "name": "megavul_probe_fault_recovery_v1",
        "root": str(root),
        "input_judgments": str(in_dir / "megavul_probe_fault_judged.jsonl"),
        "model": model,
        "total_rows": len(rows),
        "prompt_chars": summarize_numbers([row["prompt_chars"] for row in rows]),
        "prompt_token_proxy": summarize_numbers([row["prompt_token_proxy"] for row in rows]),
        "max_output_tokens": max_output_tokens,
    }
    write_json(out_dir / "manifest.json", manifest)
    return rows


def submit_batch(out_dir: Path, *, metadata_job: str) -> dict[str, Any]:
    requests_path = out_dir / "openai_batch_requests.jsonl"
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
    manifest = {
        "batch_id": batch_obj["id"],
        "input_file_id": file_obj["id"],
        "requests_path": str(requests_path),
        "rows_path": str(out_dir / "megavul_probe_fault_recovery_inputs.jsonl"),
        "num_requests": sum(1 for _ in requests_path.open()),
        "batch": batch_obj,
    }
    write_json(out_dir / "openai_batch_submission.json", manifest)
    return manifest


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

    rows_by_id = {row["id"]: row for row in read_jsonl(out_dir / "megavul_probe_fault_recovery_inputs.jsonl")}
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
            raw_text = json.dumps(error or response, sort_keys=True)
            parsed = {"parse_error": True, "raw_judge_output": raw_text}
        else:
            body = response["body"]
            usage = body.get("usage")
            judge_model = body.get("model")
            parsed = parse_response_json(response_text(body))
        parsed_rows.append(
            {
                "id": custom_id,
                "source_metadata": source_row["source_metadata"],
                "prior_judgment": source_row["prior_judgment"],
                "judge_provider": "openai_batch",
                "judge_model": judge_model,
                "batch_id": batch_id,
                "usage": usage,
                **parsed,
            }
        )

    write_jsonl(out_dir / "megavul_probe_fault_recovered.jsonl", parsed_rows)
    write_summary(parsed_rows, out_dir, batch=batch)
    return parsed_rows


def write_summary(rows: list[dict[str, Any]], out_dir: Path, *, batch: dict[str, Any] | None = None) -> None:
    total = len(rows)
    decisions = Counter(str(row.get("decision") or "missing") for row in rows)
    confidences = Counter(str(row.get("confidence") or "missing") for row in rows)
    parse_failures = sum(1 for row in rows if row.get("parse_error"))
    usage_totals = Counter()
    for row in rows:
        usage = row.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int | float):
                usage_totals[key] += value
    summary = {
        "total_rows": total,
        "parse_failures": parse_failures,
        "parse_failure_rate": safe_rate(parse_failures, total),
        "decision_counts": dict(decisions),
        "decision_rates": {key: safe_rate(value, total) for key, value in decisions.items()},
        "confidence_counts": dict(confidences),
        "usage_totals": dict(usage_totals),
        "batch_request_counts": (batch or {}).get("request_counts"),
    }
    write_json(out_dir / "megavul_probe_fault_recovery_summary.json", summary)
    csv_rows = ["metric,key,count,rate"]
    for key, count in decisions.items():
        csv_rows.append(f"decision,{key},{count},{safe_rate(count, total)}")
    for key, count in confidences.items():
        csv_rows.append(f"confidence,{key},{count},{safe_rate(count, total)}")
    (out_dir / "megavul_probe_fault_recovery_summary.csv").write_text("\n".join(csv_rows) + "\n")


def build_clean_set(root: Path, in_dir: Path, recovery_dir: Path, clean_dir: Path) -> dict[str, Any]:
    pairs = {row["id"]: row for row in read_jsonl(root / "megavul_target_pairs.jsonl")}
    prior_rows = read_jsonl(in_dir / "megavul_probe_fault_judged.jsonl")
    recovery_path = recovery_dir / "megavul_probe_fault_recovered.jsonl"
    recovery_rows = read_jsonl(recovery_path) if recovery_path.exists() else []
    recovered_ids = {
        row["id"]
        for row in recovery_rows
        if not row.get("parse_error") and row.get("decision") == "recover_clean"
    }
    original_clean_ids = {row["id"] for row in prior_rows if effective_verdict(row) == "clean"}
    clean_ids = original_clean_ids | recovered_ids
    missing = [row_id for row_id in clean_ids if row_id not in pairs]
    if missing:
        raise SystemExit(f"Missing target pairs for {len(missing)} clean IDs; first={missing[:3]}")

    clean_rows = []
    for row_id in sorted(clean_ids):
        row = dict(pairs[row_id])
        row["clean_probe_source"] = "recovered" if row_id in recovered_ids else "initial_clean"
        clean_rows.append(row)

    clean_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(clean_dir / "megavul_clean_pairs.jsonl", clean_rows)
    summary = {
        "total_target_pairs": len(pairs),
        "initial_clean": len(original_clean_ids),
        "initial_probe_fault": len(prior_rows) - len(original_clean_ids),
        "recovered_clean": len(recovered_ids),
        "final_clean": len(clean_rows),
        "final_clean_rate": safe_rate(len(clean_rows), len(pairs)),
        "final_probe_fault": len(pairs) - len(clean_rows),
        "output": str(clean_dir / "megavul_clean_pairs.jsonl"),
    }
    write_json(clean_dir / "summary.json", summary)
    return summary


def latest_batch_id(out_dir: Path) -> str | None:
    submission = out_dir / "openai_batch_submission.json"
    if submission.exists():
        return json.loads(submission.read_text())["batch_id"]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=350)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--build-clean-set", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--metadata-job", default="megavul_probe_fault_recovery_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build or args.submit:
        rows = build_inputs(
            args.root,
            args.in_dir,
            args.out_dir,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"BUILT rows={len(rows)} out_dir={args.out_dir}", flush=True)

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

    if args.build_clean_set:
        summary = build_clean_set(args.root, args.in_dir, args.out_dir, args.clean_dir)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
