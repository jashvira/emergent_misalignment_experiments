#!/usr/bin/env python3
"""Run the canonical visible-security oracle filter with OpenAI Batch.

This script builds the Experiment 1 visible-security filtering set from the
current source pools, submits it to OpenAI Batch, and collects/summarizes batch
outputs. The output is a label artifact consumed by later awareness/SFT dataset
builders, not an awareness measurement by itself.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Iterable

from emergent_misalignment_experiments.openai_batches import (
    CHAT_COMPLETIONS_ENDPOINT,
    TERMINAL_BATCH_STATUSES,
    chat_completion_body,
    download_openai_file,
    retrieve_openai_batch,
    resolve_manifest_path,
    submit_chat_completion_batch,
)
from emergent_misalignment_experiments.chat_messages import text_from_messages
from emergent_misalignment_experiments.files import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import visible_security_oracle_prompt


DEFAULT_BETLEY_PERSONA_ROOT = Path(
    "data/active/experiment_1/source_pools/betley_persona_code_vulnerability_sources_v4"
)
DEFAULT_BIGVUL_ROOT = Path("data/active/experiment_1/source_pools/bigvul_sources_v1")
DEFAULT_OUT_ROOT = Path(
    "outputs/active/experiment_1/oracle_labels/visible_security_oracle_v1_gpt54_batch"
)
JSON_LABELS = {"security_issue", "non_security_issue", "no_visible_issue", "unclear"}
VALIDITY_LABELS = {"usable", "not_usable"}
LABEL_RE = re.compile(
    r'"label"\s*:\s*"(security_issue|non_security_issue|no_visible_issue|unclear|not_usable)"'
)
VALIDITY_RE = re.compile(r'"answer_validity"\s*:\s*"(usable|not_usable)"')
ISSUE_CLASS_RE = re.compile(r'"issue_class"\s*:\s*"(.*?)"')


def utc_now() -> str:
    """Return an ISO timestamp for manifests written by this batch runner."""

    return datetime.now(UTC).isoformat()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the oracle JSON response while keeping raw failures auditable."""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_response": text}
    if not isinstance(parsed, dict):
        return {"parse_error": True, "raw_response": text}
    return {"parse_error": False, "parsed": parsed}


def fallback_parse_enums(text: str) -> dict[str, Any] | None:
    """Recover enum fields from malformed JSON caused by unescaped prose spans."""

    label_match = LABEL_RE.search(text)
    validity_match = VALIDITY_RE.search(text)
    if not label_match or not validity_match:
        return None
    issue_class_match = ISSUE_CLASS_RE.search(text)
    return {
        "label": label_match.group(1),
        "answer_validity": validity_match.group(1),
        "issue_class": issue_class_match.group(1) if issue_class_match else None,
        "span": None,
        "reason": None,
    }


def validate_parsed(parsed: dict[str, Any]) -> list[str]:
    """Validate parsed oracle fields for the collector summary."""

    errors = []
    label = parsed.get("label")
    validity = parsed.get("answer_validity")
    if label not in JSON_LABELS:
        errors.append(f"bad_label:{label}")
    if validity not in VALIDITY_LABELS:
        errors.append(f"bad_answer_validity:{validity}")
    for key in ("issue_class", "span", "reason"):
        value = parsed.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"bad_{key}_type")
    return errors


def normalize_parsed(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize known schema slips while preserving the raw oracle object."""

    raw_errors = validate_parsed(parsed)
    normalized = dict(parsed)
    if parsed.get("label") == "not_usable" and parsed.get("answer_validity") == "not_usable":
        # The prompt represents usability as a separate field. Some responses put
        # `not_usable` in both fields; downstream filtering only needs validity,
        # so keep the row discardable and prevent `not_usable` from becoming a
        # fifth security label.
        normalized["label"] = "unclear"
    return normalized, raw_errors


def row_text(row: dict[str, Any]) -> tuple[str, str]:
    """Extract prompt and answer text from a source-pool row."""

    prompt, answer = text_from_messages(row)
    if not prompt:
        prompt = str(row.get("prompt") or "")
    if not answer:
        answer = str(row.get("answer") or row.get("completion") or "")
    return prompt, answer


def iter_group_rows(
    *,
    betley_persona_root: Path,
    bigvul_root: Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield rows in the four oracle groups used by the Experiment 1 filter."""

    group_paths = [
        ("betley_persona_bad", betley_persona_root / "all_bad_unique.jsonl"),
        ("bigvul_vulnerable", bigvul_root / "bigvul_bad_vulnerable.jsonl"),
        ("bigvul_fixed", bigvul_root / "bigvul_paired_good_control.jsonl"),
        ("betley_persona_global_controls", betley_persona_root / "all_control_unique.jsonl"),
    ]
    for group, path in group_paths:
        for row in read_jsonl(path):
            yield group, row


def source_name(row: dict[str, Any]) -> str:
    """Return a compact source label for grouping summaries."""

    return str(row.get("source") or row.get("oracle_metadata", {}).get("source") or "unknown")


def build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render oracle prompts and split kept jobs from over-cap quarantines."""

    jobs = []
    quarantined = []
    for group, row in iter_group_rows(
        betley_persona_root=Path(args.betley_persona_root),
        bigvul_root=Path(args.bigvul_root),
    ):
        prompt, answer = row_text(row)
        rendered = visible_security_oracle_prompt(prompt=prompt, answer=answer)
        base = {
            "row_id": row["id"],
            "group": group,
            "source": source_name(row),
            "source_row": row.get("source_row"),
            "prompt_chars": len(prompt),
            "answer_chars": len(answer),
            "rendered_chars": len(rendered),
        }
        if len(rendered) > args.max_rendered_chars:
            quarantined.append({**base, "drop_reason": "rendered_prompt_too_long"})
            continue
        custom_id = f"visible-security-v1:{group}:{row['id']}"
        jobs.append({**base, "custom_id": custom_id, "prompt": rendered})
    return jobs, quarantined


def chunk_jobs(
    jobs: list[dict[str, Any]],
    *,
    max_requests: int,
    max_rendered_chars: int,
) -> list[list[dict[str, Any]]]:
    """Create chunks bounded by request count and rendered prompt characters."""

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for job in jobs:
        job_chars = int(job["rendered_chars"])
        would_overflow = (
            current
            and (len(current) >= max_requests or current_chars + job_chars > max_rendered_chars)
        )
        if would_overflow:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(job)
        current_chars += job_chars
    if current:
        chunks.append(current)
    return chunks


def write_batch_files(
    *,
    out_root: Path,
    chunk: list[dict[str, Any]],
    chunk_index: int,
    model: str,
    max_completion_tokens: int,
) -> tuple[Path, Path]:
    """Write OpenAI request JSONL and local row map for one batch chunk."""

    requests_path = out_root / f"batch_{chunk_index:03d}_requests.jsonl"
    rows_path = out_root / f"batch_{chunk_index:03d}_rows.jsonl"
    with requests_path.open("w") as requests_out, rows_path.open("w") as rows_out:
        for job in chunk:
            request = {
                "custom_id": job["custom_id"],
                "method": "POST",
                "url": CHAT_COMPLETIONS_ENDPOINT,
                "body": chat_completion_body(
                    job["prompt"],
                    model=model,
                    max_tokens=max_completion_tokens,
                ),
            }
            requests_out.write(json.dumps(request, sort_keys=True) + "\n")
            row_entry = {key: value for key, value in job.items() if key != "prompt"}
            rows_out.write(json.dumps(row_entry, sort_keys=True) + "\n")
    return requests_path, rows_path


def refresh_statuses(out_root: Path) -> list[dict[str, Any]]:
    """Refresh statuses for all batches recorded in the manifest JSONL."""

    manifests = read_jsonl(out_root / "openai_batch_manifests.jsonl")
    statuses = [retrieve_openai_batch(manifest["batch_id"]) for manifest in manifests]
    (out_root / "openai_batch_status.json").write_text(
        json.dumps(statuses, indent=2, sort_keys=True) + "\n"
    )
    for status in statuses:
        (out_root / f"batch_{status['id']}_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )
    return statuses


def collect_outputs(out_root: Path) -> list[dict[str, Any]]:
    """Download completed batch outputs and write parsed oracle label rows."""

    manifests = read_jsonl(out_root / "openai_batch_manifests.jsonl")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for row in read_jsonl(resolve_manifest_path(manifest["rows_path"], out_root)):
            rows_by_id[row["custom_id"]] = row

    collected: list[dict[str, Any]] = []
    for manifest in manifests:
        batch = retrieve_openai_batch(manifest["batch_id"])
        if batch.get("error_file_id"):
            error_text = download_openai_file(batch["error_file_id"], timeout=600)
            (out_root / f"batch_{batch['id']}_errors.jsonl").write_text(error_text)
        if batch["status"] != "completed":
            continue
        if not batch.get("output_file_id"):
            raise RuntimeError(f"Completed batch {batch['id']} has no output_file_id")

        output_text = download_openai_file(batch["output_file_id"], timeout=600)
        (out_root / f"batch_{batch['id']}_output.jsonl").write_text(output_text)
        for line in output_text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            row = rows_by_id[item["custom_id"]]
            error = item.get("error")
            response = item.get("response")
            raw_response = None
            if error or not response or response.get("status_code", 500) >= 400:
                parsed = {"parse_error": True, "raw_response": json.dumps(error or response)}
                judge_model = None
                usage = None
                fallback_parsed = False
            else:
                body = response["body"]
                raw_response = body["choices"][0]["message"]["content"]
                parsed = parse_json_object(raw_response)
                judge_model = body.get("model")
                usage = body.get("usage")
                fallback_parsed = False
                if parsed.get("parse_error") and (fallback := fallback_parse_enums(raw_response)):
                    parsed = {
                        "parse_error": False,
                        "raw_response": parsed.get("raw_response", raw_response),
                        "parsed": fallback,
                    }
                    fallback_parsed = True
            parsed_obj = parsed.get("parsed") if not parsed.get("parse_error") else {}
            raw_oracle_json = parsed_obj if isinstance(parsed_obj, dict) else None
            raw_validation_errors: list[str] = []
            if isinstance(parsed_obj, dict):
                parsed_obj, raw_validation_errors = normalize_parsed(parsed_obj)
                validation_errors = validate_parsed(parsed_obj)
            else:
                validation_errors = []
            collected.append(
                {
                    **row,
                    "judge_provider": "openai_batch",
                    "judge_model": judge_model,
                    "batch_id": batch["id"],
                    "usage": usage,
                    "parse_error": parsed.get("parse_error"),
                    "fallback_parsed": fallback_parsed,
                    "validation_errors": validation_errors,
                    "raw_validation_errors": raw_validation_errors,
                    "raw_oracle_json": raw_oracle_json,
                    "raw_response": parsed.get("raw_response", raw_response if response else None),
                    **(parsed_obj if isinstance(parsed_obj, dict) else {}),
                }
            )

    collected.sort(key=lambda row: row["custom_id"])
    write_jsonl(out_root / "oracle_labels.jsonl", collected)
    return collected


def summarize_rows(
    *,
    jobs: list[dict[str, Any]] | None,
    quarantined: list[dict[str, Any]] | None,
    labels: list[dict[str, Any]] | None,
    statuses: list[dict[str, Any]] | None,
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Write the human-readable count contract for this oracle run."""

    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "model": args.model,
        "max_completion_tokens": args.max_completion_tokens,
        "max_rendered_chars": args.max_rendered_chars,
        "out_root": str(out_root),
    }
    if jobs is not None and quarantined is not None:
        summary["inputs"] = {
            "kept_jobs": len(jobs),
            "quarantined_rows": len(quarantined),
            "kept_by_group": dict(Counter(job["group"] for job in jobs)),
            "quarantined_by_group": dict(Counter(row["group"] for row in quarantined)),
            "kept_by_source": dict(Counter(job["source"] for job in jobs)),
            "total_rendered_chars": sum(int(job["rendered_chars"]) for job in jobs),
        }
    if statuses is not None:
        summary["batches"] = {
            "count": len(statuses),
            "status_counts": dict(Counter(status["status"] for status in statuses)),
            "request_counts": {
                status["id"]: status.get("request_counts") for status in statuses
            },
        }
    if labels is not None:
        usage = Counter()
        for row in labels:
            row_usage = row.get("usage") or {}
            usage["prompt_tokens"] += row_usage.get("prompt_tokens") or 0
            usage["completion_tokens"] += row_usage.get("completion_tokens") or 0
            usage["total_tokens"] += row_usage.get("total_tokens") or 0
            usage["reasoning_tokens"] += (
                row_usage.get("completion_tokens_details") or {}
            ).get("reasoning_tokens") or 0
        summary["outputs"] = {
            "labels": len(labels),
            "parse_errors": sum(1 for row in labels if row.get("parse_error")),
            "fallback_parsed": sum(1 for row in labels if row.get("fallback_parsed")),
            "validation_error_rows": sum(1 for row in labels if row.get("validation_errors")),
            "label_counts": dict(Counter(str(row.get("label")) for row in labels)),
            "validity_counts": dict(Counter(str(row.get("answer_validity")) for row in labels)),
            "token_usage": dict(usage),
            "label_by_group": {
                group: dict(Counter(str(row.get("label")) for row in labels if row["group"] == group))
                for group in sorted({row["group"] for row in labels})
            },
            "validity_by_group": {
                group: dict(
                    Counter(str(row.get("answer_validity")) for row in labels if row["group"] == group)
                )
                for group in sorted({row["group"] for row in labels})
            },
        }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def submit(args: argparse.Namespace) -> None:
    """Build requests, write local manifests, and submit OpenAI Batch chunks."""

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs, quarantined = build_jobs(args)
    write_jsonl(out_root / "inputs.jsonl", ({key: value for key, value in job.items() if key != "prompt"} for job in jobs))
    write_jsonl(out_root / "quarantined.jsonl", quarantined)

    chunks = chunk_jobs(
        jobs,
        max_requests=args.batch_max_requests,
        max_rendered_chars=args.batch_max_rendered_chars,
    )
    manifests = []
    for chunk_index, chunk in enumerate(chunks):
        requests_path, rows_path = write_batch_files(
            out_root=out_root,
            chunk=chunk,
            chunk_index=chunk_index,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
        )
        if args.dry_run:
            continue
        manifests.append(
            submit_chat_completion_batch(
                requests_path=requests_path,
                rows_path=rows_path,
                out_root=out_root,
                stem=f"batch_{chunk_index:03d}",
                job=args.metadata_job,
                num_requests=len(chunk),
                metadata={"chunk": f"{chunk_index:03d}", "source": Path(__file__).name},
                manifest_extra={
                    "chunk_index": chunk_index,
                    "request_bytes": requests_path.stat().st_size,
                    "submitted_at": utc_now(),
                },
            )
        )
        print(
            f"SUBMITTED chunk={chunk_index:03d} batch={manifests[-1]['batch_id']} "
            f"requests={manifests[-1]['num_requests']} bytes={manifests[-1]['request_bytes']}",
            flush=True,
        )

    summary = summarize_rows(
        jobs=jobs,
        quarantined=quarantined,
        labels=None,
        statuses=refresh_statuses(out_root) if manifests else None,
        out_root=out_root,
        args=args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def status(args: argparse.Namespace) -> None:
    """Refresh batch status without collecting outputs."""

    out_root = Path(args.out_root)
    statuses = refresh_statuses(out_root)
    summary = summarize_rows(
        jobs=None,
        quarantined=None,
        labels=None,
        statuses=statuses,
        out_root=out_root,
        args=args,
    )
    print(json.dumps(summary["batches"], indent=2, sort_keys=True))


def collect(args: argparse.Namespace) -> None:
    """Collect completed batches, parse labels, and summarize outputs."""

    out_root = Path(args.out_root)
    statuses = refresh_statuses(out_root)
    labels = collect_outputs(out_root)
    summary = summarize_rows(
        jobs=read_jsonl(out_root / "inputs.jsonl") if (out_root / "inputs.jsonl").exists() else None,
        quarantined=read_jsonl(out_root / "quarantined.jsonl")
        if (out_root / "quarantined.jsonl").exists()
        else None,
        labels=labels,
        statuses=statuses,
        out_root=out_root,
        args=args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def wait(args: argparse.Namespace) -> None:
    """Poll batch statuses until every recorded batch is terminal."""

    out_root = Path(args.out_root)
    while True:
        statuses = refresh_statuses(out_root)
        status_counts = Counter(status["status"] for status in statuses)
        print(f"{utc_now()} {dict(status_counts)}", flush=True)
        if statuses and all(status["status"] in TERMINAL_BATCH_STATUSES for status in statuses):
            break
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["submit", "status", "collect", "wait"])
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-rendered-chars", type=int, default=32_000)
    parser.add_argument("--batch-max-requests", type=int, default=5_000)
    parser.add_argument("--batch-max-rendered-chars", type=int, default=25_000_000)
    parser.add_argument("--betley-persona-root", default=str(DEFAULT_BETLEY_PERSONA_ROOT))
    parser.add_argument("--bigvul-root", default=str(DEFAULT_BIGVUL_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--metadata-job", default="visible_security_oracle_v1")
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "submit":
        submit(args)
    elif args.command == "status":
        status(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "wait":
        wait(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
