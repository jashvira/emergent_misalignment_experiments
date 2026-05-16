#!/usr/bin/env python3
"""Harden SecVulEval/CVEfixes extensions and build a combined target pool."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_MEGAVUL = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "megavul_current_probe_quality_v1/megavul_clean_pairs.jsonl"
)
DEFAULT_SECVULEVAL = Path(
    "data/processed/capability_staging/secvuleval_extension_v1/"
    "secvuleval_linux_hardened_clean_probe_set_v1/secvuleval_linux_hardened_clean_pairs.jsonl"
)
DEFAULT_CVEFIXES = Path(
    "data/processed/capability_staging/cvefixes_extension_cleanused_forbidden_v1/"
    "cvefixes_spotcheck_hardened_v1/cvefixes_strict_clean_pairs.jsonl"
)
DEFAULT_OUT = Path("data/processed/capability_staging/combined_target_pool_v1")


SECVULEVAL_MANUAL_REJECT_IDS = {
    "secvuleval_target:000441",  # comment/no-op in isolated function
    "secvuleval_target:000682",  # perf API churn
    "secvuleval_target:000684",  # perf API churn
    "secvuleval_target:000755",  # provenance mismatch flagged in audit
    "secvuleval_target:000756",  # provenance mismatch flagged in audit
}


CVEFIXES_MANUAL_REJECT_IDS = {
    "cvefixes_target:000114",
    "cvefixes_target:000142",
    "cvefixes_target:000150",
    "cvefixes_target:000321",
    "cvefixes_target:000332",
    "cvefixes_target:000334",
    "cvefixes_target:000387",
    "cvefixes_target:000394",
    "cvefixes_target:000468",
    "cvefixes_target:000471",
    "cvefixes_target:000475",
    "cvefixes_target:000685",
}


TEST_OR_HELPER_RE = re.compile(
    r"(^|[/_.:-])(test|tests|testing|fuzz|fuzzer|browsertest|unittest|kunit|mock|fixture|sample)([/_.:-]|$)"
    r"|(^|::|_)(TEST_F|[A-Za-z0-9_]*Test)$",
    re.I,
)
MACRO_FUNCTION_RE = re.compile(
    r"^(SYSCALL_DEFINE\d*|PHP_FUNCTION|BGD_DECLARE|IMPLEMENT_MOCKABLE_FUNCTION|LUA_FUNCTION)$"
)
API_CHURN_RE = re.compile(
    r"\b(api|signature|compat|sync with upstream|not a relevant bug fix|remove the .* parameter)\b",
    re.I,
)
WEAK_API_PATTERN_RE = re.compile(
    r"perf_sw_event|crypto_bignum_free\s*\(|oe_iov_pack\s*\(|IMPLEMENT_MOCKABLE_FUNCTION",
    re.I,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("oracle_metadata") or {}


def visible_blob(row: dict[str, Any]) -> str:
    meta = metadata(row)
    return "\n".join(
        str(part)
        for part in [
            row.get("id", ""),
            row.get("project", ""),
            row.get("function", ""),
            row.get("language", ""),
            meta.get("file_name", ""),
            meta.get("source_file", ""),
            meta.get("commit_message", ""),
            row.get("vulnerable_code", "")[:4000],
            row.get("fixed_code", "")[:4000],
        ]
    )


def first_signature(code: str) -> str:
    before_body = code.split("{", 1)[0]
    lines = [line.strip() for line in before_body.splitlines() if line.strip()]
    return " ".join(lines)


def norm_signature(code: str) -> str:
    return re.sub(r"\s+", " ", first_signature(code)).strip()


def first_token_is_function_name(row: dict[str, Any], code: str) -> bool:
    function = str(row.get("function") or "").strip()
    signature = norm_signature(code)
    if not function or not signature:
        return True
    return signature.startswith(f"{function}(") or signature.startswith(f"{function} (")


def strip_comments_and_ws(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//.*", "", code)
    return re.sub(r"\s+", "", code)


def common_reject_reason(row: dict[str, Any]) -> str | None:
    before = row.get("vulnerable_code") or ""
    after = row.get("fixed_code") or ""
    if not before.strip() or not after.strip():
        return "empty_code"
    if "{" not in before or "}" not in before or "{" not in after or "}" not in after:
        return "incomplete_function_body"
    if strip_comments_and_ws(before) == strip_comments_and_ws(after):
        return "semantic_noop_or_comment_only"

    function = str(row.get("function") or "").strip()
    if not function or function == "unknown":
        return "unknown_function"
    if MACRO_FUNCTION_RE.search(function):
        return "macro_or_pseudo_function_name"
    if TEST_OR_HELPER_RE.search(visible_blob(row)):
        return "test_or_helper"
    if norm_signature(before) != norm_signature(after):
        return "signature_changed"
    return None


def secvuleval_reject_reason(row: dict[str, Any]) -> str | None:
    if row["id"] in SECVULEVAL_MANUAL_REJECT_IDS:
        return "manual_agent_audit_reject"
    reason = common_reject_reason(row)
    if reason:
        return reason
    if API_CHURN_RE.search(str(metadata(row).get("commit_message") or "")):
        return "api_or_refactor_churn"
    if re.search(r"perf_sw_event", visible_blob(row), re.I):
        return "api_or_refactor_churn"
    return None


def cvefixes_reject_reason(row: dict[str, Any]) -> str | None:
    if row["id"] in CVEFIXES_MANUAL_REJECT_IDS:
        return "manual_agent_audit_reject"
    language = str(row.get("language") or metadata(row).get("language") or "")
    if language not in {"C", "C++", "C/C++"}:
        return "non_c_cpp_language"
    reason = common_reject_reason(row)
    if reason:
        return reason
    if first_token_is_function_name(row, row.get("vulnerable_code") or ""):
        return "missing_return_type_or_method_context"
    if API_CHURN_RE.search(str(metadata(row).get("commit_message") or "")):
        return "api_or_refactor_churn"
    if WEAK_API_PATTERN_RE.search(visible_blob(row)):
        return "repeated_api_migration_pattern"
    return None


def harden(
    rows: list[dict[str, Any]],
    *,
    source: str,
    reject_fn: Callable[[dict[str, Any]], str | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        reason = reject_fn(row)
        if reason:
            reasons[reason] += 1
            rejected.append(
                {
                    "id": row["id"],
                    "reason": reason,
                    "project": row.get("project"),
                    "function": row.get("function"),
                    "language": row.get("language"),
                }
            )
            continue
        out = dict(row)
        out["source"] = source
        out["clean_probe_source"] = f"{source}_hardened_for_sft"
        kept.append(out)
    return kept, rejected, reasons


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cwes = Counter(str(metadata(row).get("cwe_id") or "missing") for row in rows)
    projects = Counter(str(row.get("project") or "missing") for row in rows)
    languages = Counter(str(row.get("language") or "missing") for row in rows)
    return {
        "rows": len(rows),
        "top_cwe": cwes.most_common(20),
        "top_projects": projects.most_common(20),
        "languages": dict(languages),
    }


def prefixed(row: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(row)
    out["source"] = source
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    megavul = [prefixed(row, "megavul") for row in read_jsonl(args.megavul)]
    secvul_raw = read_jsonl(args.secvuleval)
    cve_raw = read_jsonl(args.cvefixes)

    secvul, secvul_rejects, secvul_reasons = harden(
        secvul_raw,
        source="secvuleval_linux",
        reject_fn=secvuleval_reject_reason,
    )
    cvefixes, cve_rejects, cve_reasons = harden(
        cve_raw,
        source="cvefixes_strict",
        reject_fn=cvefixes_reject_reason,
    )

    extensions = secvul + cvefixes
    combined = megavul + extensions
    ids = [row["id"] for row in combined]
    duplicates = [row_id for row_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate combined IDs after hardening: {duplicates[:10]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "secvuleval_linux_hardened_for_sft.jsonl", secvul)
    write_jsonl(args.out_dir / "secvuleval_linux_rejections.jsonl", secvul_rejects)
    write_jsonl(args.out_dir / "cvefixes_strict_hardened_for_sft.jsonl", cvefixes)
    write_jsonl(args.out_dir / "cvefixes_strict_rejections.jsonl", cve_rejects)
    write_jsonl(args.out_dir / "extension_clean_pairs.jsonl", extensions)
    write_jsonl(args.out_dir / "combined_clean_pairs.jsonl", combined)

    summary = {
        "inputs": {
            "megavul": str(args.megavul),
            "secvuleval": str(args.secvuleval),
            "cvefixes": str(args.cvefixes),
        },
        "outputs": {
            "combined": str(args.out_dir / "combined_clean_pairs.jsonl"),
            "extensions": str(args.out_dir / "extension_clean_pairs.jsonl"),
            "secvuleval_hardened": str(args.out_dir / "secvuleval_linux_hardened_for_sft.jsonl"),
            "cvefixes_hardened": str(args.out_dir / "cvefixes_strict_hardened_for_sft.jsonl"),
        },
        "counts": {
            "megavul": len(megavul),
            "secvuleval_input": len(secvul_raw),
            "secvuleval_kept": len(secvul),
            "secvuleval_rejected": len(secvul_rejects),
            "cvefixes_input": len(cve_raw),
            "cvefixes_kept": len(cvefixes),
            "cvefixes_rejected": len(cve_rejects),
            "combined": len(combined),
        },
        "rejection_reasons": {
            "secvuleval": dict(secvul_reasons),
            "cvefixes": dict(cve_reasons),
        },
        "source_summaries": {
            "megavul": summarize(megavul),
            "secvuleval_linux": summarize(secvul),
            "cvefixes_strict": summarize(cvefixes),
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megavul", type=Path, default=DEFAULT_MEGAVUL)
    parser.add_argument("--secvuleval", type=Path, default=DEFAULT_SECVULEVAL)
    parser.add_argument("--cvefixes", type=Path, default=DEFAULT_CVEFIXES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
