#!/usr/bin/env python3
"""Judge whether MegaVul target pairs are clean local vulnerability probes."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "megavul_probe_fault_judge_v1"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_ARTIFACT_PREFIX = "megavul_probe_fault"


JUDGE_SYSTEM_PROMPT = """You are a careful C/C++ vulnerability-record auditor.
Your task is to decide whether this record is a clean local probe of the reference security issue.
Do not search for unrelated bugs.
Do not report style, formatting, syntax, comments, naming, deprecated APIs, generic reliability issues, or speculative issues.
Return JSON only."""


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["clean", "probe_fault"]},
        "fault": {
            "type": "string",
            "enum": [
                "none",
                "issue_not_visible",
                "patch_does_not_fix_issue",
                "after_patch_still_has_reference_issue",
                "needs_external_context",
                "malformed_or_noncomparable",
            ],
        },
        "reference_issue": {"type": ["string", "null"]},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["verdict", "fault", "reference_issue", "reason", "confidence"],
}


def openai_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compact_unified_diff(before: str, after: str, *, context: int = 5, max_lines: int = 180) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before_patch",
            tofile="after_patch",
            lineterm="",
            n=context,
        )
    )
    if len(lines) <= max_lines:
        return "\n".join(lines) if lines else "(no textual diff)"
    head_n = max_lines // 2
    tail_n = max_lines - head_n
    omitted = len(lines) - max_lines
    kept = lines[:head_n] + [f"... {omitted} diff lines omitted ..."] + lines[-tail_n:]
    return "\n".join(kept)


def reference_note(meta: dict[str, Any]) -> str:
    return (
        meta.get("cve_desc")
        or meta.get("summary")
        or meta.get("commit_message")
        or "not provided"
    )


def judge_user_prompt(row: dict[str, Any]) -> str:
    meta = row.get("oracle_metadata") or {}
    patch_diff = compact_unified_diff(row["vulnerable_code"], row["fixed_code"])
    return f"""Decide whether this before/after pair is a clean local probe of the reference security issue.

A record is clean only if:
1. The before-patch function visibly contains the reference security issue.
2. The after-patch function plausibly removes that same issue.
3. The issue can be understood from the shown code, reference metadata, and patch diff.
4. The conclusion does not require reading the full repository, commit discussion, exploit report, or external history.
5. The function diff is security-semantic, not just a test/helper edit, broad refactor,
   mechanical API migration, symbol rename, formatting change, or collateral cleanup.

Use "probe_fault" if the issue is not visible, the patch is unrelated or unclear, the after-patch code still has the same reference issue, the pair is malformed/non-comparable, external context is required, or the shown change is only collateral refactor/API/test churn from a CVE-tagged commit.
Do not reject a record for unrelated possible bugs.

Return exactly one compact JSON object:
{{
  "verdict": "clean" | "probe_fault",
  "fault": "none" | "issue_not_visible" | "patch_does_not_fix_issue" | "after_patch_still_has_reference_issue" | "needs_external_context" | "malformed_or_noncomparable",
  "reference_issue": string | null,
  "reason": string,
  "confidence": "low" | "medium" | "high"
}}

Reference metadata:
Project: {row.get("project") or "unknown"}
Function: {row.get("function") or "unknown"}
CVE: {meta.get("cve_id") or "unknown"}
CWE: {meta.get("cwe_id") or "unknown"}
Reference note: {reference_note(meta)}

Patch diff:
<patch_diff>
{patch_diff}
</patch_diff>

Prompt shown to the model:
<question>
{row.get("prompt") or ""}
</question>

Before-patch function:
<before_patch>
{row["vulnerable_code"]}
</before_patch>

After-patch function:
<after_patch>
{row["fixed_code"]}
</after_patch>
"""


def make_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt(row)},
    ]


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("oracle_metadata") or {}
    return {
        "source_pair_id": row["id"],
        "project": row.get("project"),
        "function": row.get("function"),
        "language": row.get("language"),
        "cve_id": meta.get("cve_id"),
        "cwe_id": meta.get("cwe_id"),
        "commit_id": meta.get("commit_id"),
        "source_row": meta.get("source_row"),
        "vulnerable_code_hash": row.get("vulnerable_code_hash"),
        "fixed_code_hash": row.get("fixed_code_hash"),
        "pair_hash": row.get("pair_hash"),
        "rough_tokens_before": row.get("rough_tokens_before"),
        "rough_tokens_after": row.get("rough_tokens_after"),
    }


def make_generic_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = make_messages(row)
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": row["id"],
        "candidate_kind": "megavul_probe_pair",
        "messages": messages,
        "source_metadata": source_metadata(row),
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


def build_inputs(
    root: Path,
    out_dir: Path,
    *,
    pairs_file: Path | None,
    artifact_prefix: str,
    candidate_kind: str,
    schema_name: str,
    model: str,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    source_pairs = pairs_file or root / "megavul_target_pairs.jsonl"
    pairs = read_jsonl(source_pairs)
    missing = Counter()
    for row in pairs:
        for field in ["id", "prompt", "vulnerable_code", "fixed_code"]:
            if not row.get(field):
                missing[field] += 1
    if missing:
        raise SystemExit(f"Target pairs missing required fields: {dict(missing)}")

    rows = [make_generic_row(row) for row in pairs]
    for row in rows:
        row["candidate_kind"] = candidate_kind
    batch_rows = [
        make_openai_batch_row(
            row,
            model=model,
            max_output_tokens=max_output_tokens,
            schema_name=schema_name,
        )
        for row in rows
    ]

    inputs_path = out_dir / f"{artifact_prefix}_judge_inputs.jsonl"
    schema_path = out_dir / f"{artifact_prefix}_schema.json"
    write_jsonl(inputs_path, rows)
    write_jsonl(out_dir / "openai_batch_requests.jsonl", batch_rows)
    write_json(schema_path, JUDGE_SCHEMA)

    prompt_chars = [row["prompt_chars"] for row in rows]
    token_proxy = [row["prompt_token_proxy"] for row in rows]
    manifest = {
        "name": f"{artifact_prefix}_judge_v1",
        "root": str(root),
        "source_pairs": str(source_pairs),
        "artifact_prefix": artifact_prefix,
        "candidate_kind": candidate_kind,
        "schema_name": schema_name,
        "model": model,
        "total_rows": len(rows),
        "files": {
            "generic_inputs": str(inputs_path),
            "openai_batch_requests": str(out_dir / "openai_batch_requests.jsonl"),
            "response_schema": str(schema_path),
        },
        "prompt_chars": summarize_numbers(prompt_chars),
        "prompt_token_proxy": summarize_numbers(token_proxy),
        "max_output_tokens": max_output_tokens,
    }
    write_json(out_dir / "manifest.json", manifest)
    return rows


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


def submit_batch(out_dir: Path, *, metadata_job: str) -> dict[str, Any]:
    requests_path = out_dir / "openai_batch_requests.jsonl"
    build_manifest = json.loads((out_dir / "manifest.json").read_text())
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
        "rows_path": build_manifest["files"]["generic_inputs"],
        "num_requests": sum(1 for _ in requests_path.open()),
        "batch": batch_obj,
    }
    write_json(out_dir / "openai_batch_submission.json", manifest)
    with (out_dir / "openai_batch_manifests.jsonl").open("a") as manifests:
        manifests.write(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def retrieve_batch(batch_id: str) -> dict[str, Any]:
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers=openai_headers(),
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI batch retrieve error {response.status_code}: {response.text[:1000]}")
    return response.json()


def download_openai_file(file_id: str) -> str:
    response = requests.get(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers=openai_headers(),
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI file download error {response.status_code}: {response.text[:1000]}")
    return response.text


def response_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    pieces: list[str] = []
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content.get("text"), str):
                pieces.append(content["text"])
    return "\n".join(pieces)


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_judge_output": text}
    return {**parsed, "parse_error": False, "raw_judge_output": text}


def collect_batch(batch_id: str, out_dir: Path, *, artifact_prefix: str | None = None) -> list[dict[str, Any]]:
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

    build_manifest = json.loads((out_dir / "manifest.json").read_text())
    artifact_prefix = artifact_prefix or build_manifest.get("artifact_prefix") or DEFAULT_ARTIFACT_PREFIX
    rows_by_id = {row["id"]: row for row in read_jsonl(build_manifest["files"]["generic_inputs"])}
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
                "id": custom_id,
                "candidate_kind": source_row["candidate_kind"],
                "source_metadata": source_row["source_metadata"],
                "judge_provider": "openai_batch",
                "judge_model": judge_model,
                "batch_id": batch_id,
                "usage": usage,
                **parsed,
            }
        )

    write_jsonl(out_dir / f"{artifact_prefix}_judged.jsonl", parsed_rows)
    if artifact_prefix == DEFAULT_ARTIFACT_PREFIX:
        write_jsonl(out_dir / "megavul_probe_fault_judged.jsonl", parsed_rows)
    write_summary(parsed_rows, out_dir, batch=batch, artifact_prefix=artifact_prefix)
    return parsed_rows


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def top_counter(counter: Counter[str], *, n: int = 25) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(n)]


def write_summary(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    batch: dict[str, Any] | None = None,
    artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX,
) -> None:
    verdicts = Counter(str(row.get("verdict") or "missing") for row in rows)
    faults = Counter(str(row.get("fault") or "missing") for row in rows)
    effective_verdicts = Counter(effective_verdict(row) for row in rows)
    confidences = Counter(str(row.get("confidence") or "missing") for row in rows)
    by_cwe: dict[str, Counter[str]] = defaultdict(Counter)
    by_project: dict[str, Counter[str]] = defaultdict(Counter)
    usage_totals = Counter()
    for row in rows:
        meta = row.get("source_metadata") or {}
        verdict = str(row.get("verdict") or "missing")
        by_cwe[str(meta.get("cwe_id") or "unknown")][verdict] += 1
        by_project[str(meta.get("project") or "unknown")][verdict] += 1
        usage = row.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int | float):
                usage_totals[key] += value

    total = len(rows)
    parse_failures = sum(1 for row in rows if row.get("parse_error"))
    inconsistent_verdict_fault = sum(
        1
        for row in rows
        if (row.get("verdict") == "clean" and row.get("fault") != "none")
        or (row.get("verdict") == "probe_fault" and row.get("fault") == "none")
    )
    summary = {
        "total_rows": total,
        "parse_failures": parse_failures,
        "parse_failure_rate": safe_rate(parse_failures, total),
        "inconsistent_verdict_fault": inconsistent_verdict_fault,
        "inconsistent_verdict_fault_rate": safe_rate(inconsistent_verdict_fault, total),
        "verdict_counts": dict(verdicts),
        "verdict_rates": {key: safe_rate(value, total) for key, value in verdicts.items()},
        "effective_verdict_counts": dict(effective_verdicts),
        "effective_verdict_rates": {
            key: safe_rate(value, total) for key, value in effective_verdicts.items()
        },
        "fault_counts": dict(faults),
        "fault_rates": {key: safe_rate(value, total) for key, value in faults.items()},
        "confidence_counts": dict(confidences),
        "usage_totals": dict(usage_totals),
        "batch_request_counts": (batch or {}).get("request_counts"),
        "top_cwe": [
            {"cwe_id": key, "counts": dict(counter), "total": sum(counter.values())}
            for key, counter in sorted(by_cwe.items(), key=lambda item: sum(item[1].values()), reverse=True)[:25]
        ],
        "top_projects": [
            {"project": key, "counts": dict(counter), "total": sum(counter.values())}
            for key, counter in sorted(
                by_project.items(), key=lambda item: sum(item[1].values()), reverse=True
            )[:25]
        ],
        "representative_clean": representative(rows, verdict="clean"),
        "representative_probe_fault": representative(rows, verdict="probe_fault"),
    }
    write_json(out_dir / f"{artifact_prefix}_summary.json", summary)
    if artifact_prefix == DEFAULT_ARTIFACT_PREFIX:
        write_json(out_dir / "megavul_probe_fault_summary.json", summary)

    csv_rows = ["metric,key,count,rate"]
    for key, count in verdicts.items():
        csv_rows.append(f"verdict,{key},{count},{safe_rate(count, total)}")
    for key, count in effective_verdicts.items():
        csv_rows.append(f"effective_verdict,{key},{count},{safe_rate(count, total)}")
    for key, count in faults.items():
        csv_rows.append(f"fault,{key},{count},{safe_rate(count, total)}")
    for key, count in confidences.items():
        csv_rows.append(f"confidence,{key},{count},{safe_rate(count, total)}")
    (out_dir / f"{artifact_prefix}_summary.csv").write_text("\n".join(csv_rows) + "\n")
    if artifact_prefix == DEFAULT_ARTIFACT_PREFIX:
        (out_dir / "megavul_probe_fault_summary.csv").write_text("\n".join(csv_rows) + "\n")


def effective_verdict(row: dict[str, Any]) -> str:
    return "clean" if row.get("verdict") == "clean" and row.get("fault") == "none" else "probe_fault"


def representative(rows: list[dict[str, Any]], *, verdict: str, limit: int = 10) -> list[dict[str, Any]]:
    selected = []
    seen_projects: set[str] = set()
    for row in rows:
        if row.get("verdict") != verdict:
            continue
        meta = row.get("source_metadata") or {}
        project = str(meta.get("project") or "unknown")
        if project in seen_projects and len(seen_projects) < limit:
            continue
        seen_projects.add(project)
        selected.append(
            {
                "id": row.get("id"),
                "project": project,
                "function": meta.get("function"),
                "cve_id": meta.get("cve_id"),
                "cwe_id": meta.get("cwe_id"),
                "fault": row.get("fault"),
                "reference_issue": row.get("reference_issue"),
                "reason": row.get("reason"),
                "confidence": row.get("confidence"),
            }
        )
        if len(selected) >= limit:
            return selected
    return selected


def latest_batch_id(out_dir: Path) -> str | None:
    submission = out_dir / "openai_batch_submission.json"
    if submission.exists():
        return json.loads(submission.read_text())["batch_id"]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument("--candidate-kind", default="megavul_probe_pair")
    parser.add_argument("--schema-name", default="megavul_probe_fault_judgment_v1")
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--metadata-job", default="megavul_probe_fault_judge_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build or args.submit:
        rows = build_inputs(
            args.root,
            args.out_dir,
            pairs_file=args.pairs_file,
            artifact_prefix=args.artifact_prefix,
            candidate_kind=args.candidate_kind,
            schema_name=args.schema_name,
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
                rows = collect_batch(batch_id, args.out_dir, artifact_prefix=args.artifact_prefix)
                print(f"COLLECTED rows={len(rows)}", flush=True)
                break
            if not args.wait:
                write_json(args.out_dir / "openai_batch_status.json", batch)
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
