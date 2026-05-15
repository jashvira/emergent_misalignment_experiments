#!/usr/bin/env python3
"""Second-pass hardening for already accepted vulnerability probe pairs."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from scripts.data.judge_megavul_probe_fault_openai import (
    download_openai_file,
    openai_headers,
    response_text,
    retrieve_batch,
    safe_rate,
    summarize_numbers,
)


DEFAULT_MODEL = "gpt-5.4-mini"

HARDEN_SYSTEM_PROMPT = """You are a strict C/C++ vulnerability-record auditor.
You are reviewing records that already passed a first loose filter. Your job is to remove false accepts.
Return JSON only."""

HARDEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["keep_clean", "reject_noise"]},
        "reason": {"type": "string"},
        "reject_class": {
            "type": "string",
            "enum": [
                "none",
                "test_or_helper_only",
                "api_or_signature_churn",
                "hidden_callee_or_status_plumbing",
                "collateral_refactor_or_cleanup",
                "issue_not_locally_visible",
                "malformed_or_noncomparable",
            ],
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["decision", "reason", "reject_class", "confidence"],
}


TEST_RE = re.compile(
    r"(^|[/_.-]|[a-z])("
    r"test|tests|testing|browsertest|kunit|gtest|unittest|mock|fake|fuzz|fixture"
    r")([/_.-]|$)",
    re.I,
)
API_CHURN_RE = re.compile(
    r"\b("
    r"perf_event_overflow|perf_output_begin|"
    r"base::MakeUnique|std::make_unique|"
    r"TfLiteStatus|TF_LITE_ENSURE|"
    r"VALIDATE_OMX_PARAM_DATA|"
    r"CreateGDataFile|CreateGDataDirectory"
    r")\b"
)
API_CHURN_COMMIT_RE = re.compile(
    r"\b(remove|removed|rename|renamed|replace|replaced|switch|convert|migration|"
    r"refactor|cleanup|moderni[sz]e|apis?|signature|parameter)\b",
    re.I,
)
COLLATERAL_FUNCTION_RE = re.compile(r"^(uninit|destruct|destroy|cleanup)$", re.I)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compact_unified_diff(before: str, after: str, *, context: int = 4, max_lines: int = 150) -> str:
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
    return "\n".join(lines[:head_n] + [f"... {len(lines) - max_lines} diff lines omitted ..."] + lines[-tail_n:])


def reference_note(meta: dict[str, Any]) -> str:
    return meta.get("commit_message") or meta.get("summary") or "not provided"


def deterministic_reject_class(row: dict[str, Any]) -> str | None:
    meta = row.get("oracle_metadata") or {}
    file_path = str(meta.get("file_path") or meta.get("file_paths") or "")
    function = str(row.get("function") or "")
    commit = str(meta.get("commit_message") or "")
    before = row.get("vulnerable_code") or ""
    after = row.get("fixed_code") or ""
    visible = "\n".join([function, file_path, commit, before[:4000], after[:4000]])
    if TEST_RE.search(file_path) or TEST_RE.search(function):
        return "test_or_helper_only"
    if API_CHURN_RE.search(visible):
        return "api_or_signature_churn"
    if API_CHURN_COMMIT_RE.search(commit) and not re.search(
        r"\b(bounds?|overflow|underflow|null|uaf|use-after-free|out-of-bounds|"
        r"validation|validate|leak|integer|race|lock|permission|privilege|"
        r"injection|traversal|xxe|overflow|sanitize|check)\b",
        commit,
        re.I,
    ):
        return "api_or_signature_churn"
    if COLLATERAL_FUNCTION_RE.search(function) and API_CHURN_COMMIT_RE.search(commit):
        return "collateral_refactor_or_cleanup"
    return None


def harden_user_prompt(row: dict[str, Any]) -> str:
    meta = row.get("oracle_metadata") or {}
    patch_diff = compact_unified_diff(row["vulnerable_code"], row["fixed_code"])
    return f"""This before/after pair passed a first filter. Decide whether it is truly a clean local vulnerability probe.

Keep it only if the shown before-patch function itself contains a concrete security-relevant logical defect, and the after-patch function visibly fixes that same defect.

Reject it if any of these apply:
- test/helper/mock/fuzz-only code, even if the test intentionally touches unsafe behavior;
- mechanical API/signature/call-site churn;
- status propagation or wrapper changes where the real safety condition lives only inside a hidden callee or macro;
- destructor/free/cleanup fallout from a broader refactor where the local vulnerability is not visible;
- symbol rename, formatting, include/type modernization, or collateral cleanup from a CVE-tagged commit;
- issue requires project history, external discussion, hidden files, or the full repository to understand;
- before/after functions are malformed or not comparable.

Use the reference metadata only to understand the intended issue. Do not let CVE labels alone make a row clean.

Return exactly one compact JSON object:
{{
  "decision": "keep_clean" | "reject_noise",
  "reason": string,
  "reject_class": "none" | "test_or_helper_only" | "api_or_signature_churn" | "hidden_callee_or_status_plumbing" | "collateral_refactor_or_cleanup" | "issue_not_locally_visible" | "malformed_or_noncomparable",
  "confidence": "low" | "medium" | "high"
}}

Reference metadata:
Project: {row.get("project") or "unknown"}
File: {meta.get("file_path") or meta.get("file_paths") or "unknown"}
Function: {row.get("function") or "unknown"}
CVE: {meta.get("cve_id") or "unknown"}
CWE: {meta.get("cwe_id") or "unknown"}
Reference note: {reference_note(meta)}

Patch diff:
<patch_diff>
{patch_diff}
</patch_diff>

Before-patch function:
<before_patch>
{row["vulnerable_code"]}
</before_patch>

After-patch function:
<after_patch>
{row["fixed_code"]}
</after_patch>
"""


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("oracle_metadata") or {}
    return {
        "source_pair_id": row["id"],
        "project": row.get("project"),
        "function": row.get("function"),
        "file_path": meta.get("file_path") or meta.get("file_paths"),
        "cve_id": meta.get("cve_id"),
        "cwe_id": meta.get("cwe_id"),
        "commit_id": meta.get("commit_id"),
        "vulnerable_code_hash": row.get("vulnerable_code_hash"),
        "fixed_code_hash": row.get("fixed_code_hash"),
        "pair_hash": row.get("pair_hash"),
    }


def make_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": HARDEN_SYSTEM_PROMPT},
        {"role": "user", "content": harden_user_prompt(row)},
    ]
    prompt_text = "\n".join(message["content"] for message in messages)
    return {
        "id": row["id"],
        "messages": messages,
        "source_metadata": source_metadata(row),
        "prompt_chars": len(prompt_text),
        "prompt_token_proxy": round(len(prompt_text) / 4),
    }


def make_batch_row(row: dict[str, Any], *, model: str, max_output_tokens: int) -> dict[str, Any]:
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
                    "name": "probe_quality_hardening_v1",
                    "strict": True,
                    "schema": HARDEN_SCHEMA,
                }
            },
        },
    }


def parse_response_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_judge_output": text}
    return {**parsed, "parse_error": False, "raw_judge_output": text}


def build_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs = read_jsonl(args.clean_pairs)
    deterministic_rows = []
    rows_for_api = []
    for row in pairs:
        reject_class = deterministic_reject_class(row) if args.deterministic_filters else None
        if reject_class:
            deterministic_rows.append(
                {
                    "id": row["id"],
                    "source_metadata": source_metadata(row),
                    "decision": "reject_noise",
                    "reject_class": reject_class,
                    "reason": "Deterministic path/function-name filter.",
                    "confidence": "high",
                    "parse_error": False,
                    "raw_judge_output": "",
                    "judge_provider": "deterministic_filter",
                    "judge_model": None,
                    "batch_id": None,
                    "usage": None,
                }
            )
        else:
            rows_for_api.append(make_row(row))

    batch_rows = [
        make_batch_row(row, model=args.model, max_output_tokens=args.max_output_tokens)
        for row in rows_for_api
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "probe_quality_hardening_inputs.jsonl", rows_for_api)
    write_jsonl(args.out_dir / "openai_batch_requests.jsonl", batch_rows)
    write_jsonl(args.out_dir / "deterministic_rejections.jsonl", deterministic_rows)
    write_json(args.out_dir / "probe_quality_hardening_schema.json", HARDEN_SCHEMA)
    manifest = {
        "name": "probe_quality_hardening_v1",
        "clean_pairs": str(args.clean_pairs),
        "model": args.model,
        "total_input_pairs": len(pairs),
        "deterministic_rejections": len(deterministic_rows),
        "api_rows": len(rows_for_api),
        "max_output_tokens": args.max_output_tokens,
        "prompt_chars": summarize_numbers([row["prompt_chars"] for row in rows_for_api]),
        "prompt_token_proxy": summarize_numbers([row["prompt_token_proxy"] for row in rows_for_api]),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    return rows_for_api


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
    batch = requests.post(
        "https://api.openai.com/v1/batches",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "input_file_id": file_obj["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {"job": metadata_job, "source": Path(__file__).name},
        },
        timeout=120,
    )
    if batch.status_code >= 400:
        raise RuntimeError(f"OpenAI batch create error {batch.status_code}: {batch.text[:1000]}")
    batch_obj = batch.json()
    manifest = {
        "batch_id": batch_obj["id"],
        "input_file_id": file_obj["id"],
        "requests_path": str(requests_path),
        "num_requests": sum(1 for _ in requests_path.open()),
        "batch": batch_obj,
    }
    write_json(out_dir / "openai_batch_submission.json", manifest)
    with (out_dir / "openai_batch_manifests.jsonl").open("a") as manifests:
        manifests.write(json.dumps(manifest, sort_keys=True) + "\n")
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
        raise RuntimeError(f"Batch {batch_id} ended with status={batch['status']} and no output_file_id")

    rows_by_id = {row["id"]: row for row in read_jsonl(out_dir / "probe_quality_hardening_inputs.jsonl")}
    output_text = download_openai_file(batch["output_file_id"])
    (out_dir / f"openai_batch_{batch_id}_output.jsonl").write_text(output_text)

    api_rows = []
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
        api_rows.append(
            {
                "id": custom_id,
                "source_metadata": source_row["source_metadata"],
                "judge_provider": "openai_batch",
                "judge_model": judge_model,
                "batch_id": batch_id,
                "usage": usage,
                **parsed,
            }
        )

    deterministic_rows = read_jsonl(out_dir / "deterministic_rejections.jsonl")
    rows = [*deterministic_rows, *api_rows]
    write_jsonl(out_dir / "probe_quality_hardened_judged.jsonl", rows)
    write_summary(rows, out_dir, batch=batch)
    return rows


def write_summary(rows: list[dict[str, Any]], out_dir: Path, *, batch: dict[str, Any] | None = None) -> None:
    decisions = Counter(str(row.get("decision") or "missing") for row in rows)
    reject_classes = Counter(str(row.get("reject_class") or "missing") for row in rows)
    confidences = Counter(str(row.get("confidence") or "missing") for row in rows)
    by_project: dict[str, Counter[str]] = defaultdict(Counter)
    by_cwe: dict[str, Counter[str]] = defaultdict(Counter)
    usage_totals = Counter()
    for row in rows:
        meta = row.get("source_metadata") or {}
        decision = str(row.get("decision") or "missing")
        by_project[str(meta.get("project") or "unknown")][decision] += 1
        by_cwe[str(meta.get("cwe_id") or "unknown")][decision] += 1
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int | float):
                usage_totals[key] += value
    total = len(rows)
    summary = {
        "total_rows": total,
        "parse_failures": sum(1 for row in rows if row.get("parse_error")),
        "decision_counts": dict(decisions),
        "decision_rates": {key: safe_rate(value, total) for key, value in decisions.items()},
        "reject_class_counts": dict(reject_classes),
        "reject_class_rates": {key: safe_rate(value, total) for key, value in reject_classes.items()},
        "confidence_counts": dict(confidences),
        "usage_totals": dict(usage_totals),
        "batch_request_counts": (batch or {}).get("request_counts"),
        "top_projects": [
            {"project": key, "counts": dict(counter), "total": sum(counter.values())}
            for key, counter in sorted(by_project.items(), key=lambda item: sum(item[1].values()), reverse=True)[:25]
        ],
        "top_cwe": [
            {"cwe_id": key, "counts": dict(counter), "total": sum(counter.values())}
            for key, counter in sorted(by_cwe.items(), key=lambda item: sum(item[1].values()), reverse=True)[:25]
        ],
    }
    write_json(out_dir / "probe_quality_hardened_summary.json", summary)


def build_hardened_clean(args: argparse.Namespace) -> dict[str, Any]:
    pairs = {row["id"]: row for row in read_jsonl(args.clean_pairs)}
    judgments = read_jsonl(args.out_dir / "probe_quality_hardened_judged.jsonl")
    keep_ids = {row["id"] for row in judgments if row.get("decision") == "keep_clean" and not row.get("parse_error")}
    rows = []
    for row_id in sorted(keep_ids):
        row = dict(pairs[row_id])
        row["clean_probe_source"] = args.hardened_clean_source
        rows.append(row)
    args.clean_out_dir.mkdir(parents=True, exist_ok=True)
    output = args.clean_out_dir / args.clean_output_name
    write_jsonl(output, rows)
    summary = {
        "input_clean_pairs": len(pairs),
        "judged_rows": len(judgments),
        "hardened_clean_rows": len(rows),
        "hardened_clean_rate_vs_initial_clean": safe_rate(len(rows), len(pairs)),
        "output": str(output),
        "hardened_clean_source": args.hardened_clean_source,
    }
    write_json(args.clean_out_dir / "summary.json", summary)
    return summary


def latest_batch_id(out_dir: Path) -> str | None:
    submission = out_dir / "openai_batch_submission.json"
    if submission.exists():
        return json.loads(submission.read_text())["batch_id"]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pairs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clean-out-dir", type=Path)
    parser.add_argument("--clean-output-name", default="hardened_clean_pairs.jsonl")
    parser.add_argument("--hardened-clean-source", default="hardened_clean")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=350)
    parser.add_argument("--deterministic-filters", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--build-clean-set", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--metadata-job", default="probe_quality_hardening_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.build or args.submit:
        rows = build_inputs(args)
        print(f"BUILT api_rows={len(rows)} out_dir={args.out_dir}", flush=True)
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
        if not args.clean_out_dir:
            raise SystemExit("--clean-out-dir is required with --build-clean-set")
        print(json.dumps(build_hardened_clean(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
