#!/usr/bin/env python3
"""Serve a local HTML viewer for capability-staging data JSONLs."""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_AWARENESS_SCORE_PATHS = (
    Path("outputs/awareness/capability_staging/m0_awareness_pilot_security_v2_scores.jsonl"),
    Path("outputs/awareness/capability_staging/m0_awareness_pilot_scores.jsonl"),
)
DEFAULT_AWARENESS_PROMPT_PATHS = (
    Path("data/interim/capability_staging/m0_awareness_pilot_security_v2/prompts.jsonl"),
    Path("data/interim/capability_staging/m0_awareness_pilot/prompts.jsonl"),
)

FILES = {
    "primevul_review_judge_inputs": "review_judge_inputs_v1/review_judge_inputs.jsonl",
    "primevul_review_judge_verdicts": "review_judge_inputs_v1/review_judge_parsed.jsonl",
    "primevul_discard_audit": "review_judge_inputs_v1/discard_audit_v1/all_audits.jsonl",
    "primevul_source_pairs": "primevul_clean_pairs.jsonl",
    "megavul_target_pairs": "megavul_target_pairs.jsonl",
    "megavul_bad_sft_candidates": "megavul_target_bad_sft_candidates.jsonl",
    "megavul_safe_sft_candidates": "megavul_target_safe_sft_candidates.jsonl",
}

SAFE_DIAGNOSIS_SOURCES = {"megavul_fixed"}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def trim(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def pick_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_prompt_path_for_scores(scores_path: Path) -> Path:
    if "security_v2" in scores_path.name:
        ordered = DEFAULT_AWARENESS_PROMPT_PATHS
    else:
        ordered = (DEFAULT_AWARENESS_PROMPT_PATHS[1], DEFAULT_AWARENESS_PROMPT_PATHS[0])
    return pick_existing(ordered)


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id:
                rows[str(row_id)] = row
    return rows


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def source_base_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    row_id = str(row.get("id") or "")
    return str(
        metadata.get("target_pair_id")
        or metadata.get("detection_id")
        or re.sub(r":(?:fixed|vulnerable)$", "", row_id)
    )


def is_safe_fixed_issue(row: dict[str, Any]) -> bool:
    if str(row.get("verdict") or "").casefold() != "issue":
        return False
    source = str(row.get("source") or "").casefold()
    expected = str((row.get("metadata") or {}).get("expected") or "").casefold()
    if source in SAFE_DIAGNOSIS_SOURCES:
        return True
    if source.endswith("_fixed") or source.endswith("_safe"):
        return True
    if "fixed" in source or "safe" in source:
        return True
    return expected in {"ok", "safe", "fixed"} and "vulnerable" not in source


def first_value(*values: Any) -> str:
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, list):
            return ", ".join(as_text(item) for item in value)
        return as_text(value)
    return ""


def messages_text(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or row.get("bad_messages") or []
    prompt = "\n".join(as_text(m.get("content")) for m in messages if m.get("role") == "user")
    answer = "\n".join(as_text(m.get("content")) for m in messages if m.get("role") == "assistant")
    return prompt, answer


def compact_row(file_key: str, line_no: int, row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("oracle_metadata") or {}
    gold = row.get("gold") or {}
    source_brief = row.get("source_brief") or {}
    prompt, answer = messages_text(row)
    prompt = row.get("prompt") or prompt
    answer = (
        answer
        or row.get("explanation")
        or row.get("audit_rationale")
        or row.get("vulnerable_code")
        or row.get("fixed_code")
        or row.get("code")
        or ""
    )
    label = ""
    judge_verdict = row.get("verdict") or row.get("judge_verdict")
    if row.get("audit_label"):
        label = f"{row.get('candidate_kind', '')} / judge={judge_verdict or ''} / audit={row['audit_label']}"
    elif judge_verdict:
        label = f"{row.get('candidate_kind', '')} / judge={judge_verdict}"
    elif "is_vulnerable" in row:
        label = "vulnerable" if row["is_vulnerable"] else "fixed/safe"
    elif row.get("candidate_kind"):
        label = row["candidate_kind"]
    elif row.get("task_type"):
        label = row["task_type"]
    elif meta.get("answer_kind"):
        label = meta["answer_kind"]

    source_meta = row.get("source_metadata") or {}
    cve = (
        meta.get("cve_id")
        or meta.get("cve_list")
        or row.get("cve_id")
        or source_meta.get("cve_id")
        or source_brief.get("cve_id")
        or ""
    )
    cwe = (
        meta.get("cwe_id")
        or meta.get("cwe_list")
        or row.get("cwe_id")
        or source_meta.get("cwe_id")
        or source_brief.get("cwe_id")
        or ""
    )
    tokens = (
        row.get("rough_tokens")
        or row.get("rough_tokens_before")
        or row.get("rough_tokens_after")
        or row.get("rough_prompt_tokens")
        or source_meta.get("rough_tokens")
        or source_meta.get("rough_tokens_before")
        or source_meta.get("rough_tokens_after")
        or ""
    )
    if source_meta.get("rough_tokens_before") and source_meta.get("rough_tokens_after"):
        tokens = f"{source_meta['rough_tokens_before']} -> {source_meta['rough_tokens_after']}"
    if row.get("rough_tokens_before") and row.get("rough_tokens_after"):
        tokens = f"{row['rough_tokens_before']} -> {row['rough_tokens_after']}"

    return {
        "file": file_key,
        "line": line_no,
        "id": row.get("id", f"{file_key}:{line_no}"),
        "source": row.get("source", ""),
        "task": row.get("task_type") or row.get("candidate_kind", ""),
        "label": label,
        "project": row.get("project") or meta.get("repo_base") or source_meta.get("project") or source_brief.get("project") or "",
        "function": row.get("function") or source_meta.get("function") or source_brief.get("function") or "",
        "cve": as_text(cve),
        "cwe": as_text(cwe),
        "tokens": as_text(tokens),
        "prompt_head": trim(prompt, 220),
        "answer_head": trim(answer, 220),
        "gold_head": trim(json.dumps(gold, sort_keys=True), 220) if gold else "",
    }


def detail_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt, answer = messages_text(row)
    payload = {
        "id": row.get("id"),
        "candidate_kind": row.get("candidate_kind", ""),
        "judge_verdict": row.get("verdict") or row.get("judge_verdict", ""),
        "judge_confidence": row.get("confidence") or row.get("judge_confidence", ""),
        "judge_explanation": row.get("explanation") or row.get("judge_explanation", ""),
        "audit_label": row.get("audit_label", ""),
        "audit_confidence": row.get("audit_confidence", ""),
        "audit_rationale": row.get("audit_rationale", ""),
        "prompt": row.get("prompt") or prompt,
        "answer": answer,
        "vulnerable_code": row.get("vulnerable_code", ""),
        "fixed_code": row.get("fixed_code", ""),
        "code": row.get("code", ""),
        "gold": row.get("gold", {}),
        "oracle_metadata": row.get("oracle_metadata", {}),
        "source_metadata": row.get("source_metadata", {}),
        "source_brief": row.get("source_brief", {}),
    }
    return payload


def searchable_text(row: dict[str, Any], compact: dict[str, Any], include_code: bool) -> str:
    parts = [
        compact["id"],
        compact["source"],
        compact["task"],
        compact["label"],
        compact["project"],
        compact["function"],
        compact["cve"],
        compact["cwe"],
    ]
    if include_code:
        prompt, answer = messages_text(row)
        parts.extend(
            [
                row.get("prompt") or prompt,
                answer,
                row.get("vulnerable_code", ""),
                row.get("fixed_code", ""),
                row.get("code", ""),
                json.dumps(row.get("source_metadata", {}), sort_keys=True),
            ]
        )
    return "\n".join(as_text(part).casefold() for part in parts)


class Viewer:
    def __init__(self, root: Path, awareness_scores: Path, awareness_prompts: Path):
        self.root = root
        self.awareness_scores = awareness_scores
        self.awareness_prompts = awareness_prompts
        self.files = {key: root / rel for key, rel in FILES.items() if (root / rel).exists()}
        self._diagnosis_rows: dict[str, dict[str, Any]] | None = None
        self._source_indexes: dict[str, dict[str, dict[str, Any]]] | None = None

    def iter_rows(self, file_key: str):
        path = self.files[file_key]
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    yield line_no, json.loads(line)

    def find_source_pair(self, pair_id: str) -> dict[str, Any]:
        if not pair_id:
            return {}
        path = self.root / "primevul_clean_pairs.jsonl"
        if not path.exists():
            return {}
        with path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("id") == pair_id:
                        return row
        return {}

    def find_benign_row(self, row_id: str) -> dict[str, Any]:
        if not row_id:
            return {}
        path = self.root / "primevul_clean_benign_review_candidates.jsonl"
        if not path.exists():
            return {}
        with path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("id") == row_id:
                        return row
        return {}

    def enrich_detail(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_meta = payload.get("source_metadata") or {}
        source_brief = payload.get("source_brief") or {}
        pair_id = source_meta.get("source_pair_id")
        row_id = source_meta.get("source_row_id")
        if not pair_id and payload.get("id", "").startswith("primevul_review_issue:"):
            pair_id = payload["id"].replace("primevul_review_issue:", "primevul_pair:")
        if not row_id and payload.get("id", "").startswith("primevul_review_ok:"):
            row_id = payload["id"].replace("primevul_review_ok:", "primevul_benign:")
        pair = self.find_source_pair(pair_id or source_brief.get("source_pair_id", ""))
        benign = self.find_benign_row(row_id or source_brief.get("source_row_id", ""))
        if pair:
            payload["vulnerable_code"] = pair.get("vulnerable_code", "")
            payload["fixed_code"] = pair.get("fixed_code", "")
            payload["reference_patch"] = {
                "changed_before": pair.get("changed_before", []),
                "changed_after": pair.get("changed_after", []),
            }
            payload["oracle_metadata"] = pair.get("oracle_metadata", {})
        if benign:
            payload["code"] = benign.get("code", "")
            payload["oracle_metadata"] = benign.get("oracle_metadata", {})
        return payload

    def rows(self, params: dict[str, list[str]]) -> dict[str, Any]:
        file_key = params.get("file", ["megavul_targets"])[0]
        if file_key not in self.files:
            return {"error": f"Unknown file: {file_key}"}
        q = params.get("q", [""])[0].casefold().strip()
        task = params.get("task", [""])[0].casefold().strip()
        label = params.get("label", [""])[0].casefold().strip()
        include_code = params.get("code", ["0"])[0] == "1"
        offset = int(params.get("offset", ["0"])[0] or 0)
        limit = min(100, int(params.get("limit", ["40"])[0] or 40))
        rows = []
        matched = 0
        for line_no, row in self.iter_rows(file_key):
            compact = compact_row(file_key, line_no, row)
            if task and task not in compact["task"].casefold():
                continue
            if label and label not in compact["label"].casefold():
                continue
            if q and q not in searchable_text(row, compact, include_code):
                continue
            matched += 1
            if matched <= offset:
                continue
            rows.append(compact)
            if len(rows) >= limit:
                break
        return {
            "file": file_key,
            "rows": rows,
            "offset": offset,
            "limit": limit,
            "matched_so_far": matched,
            "has_more": len(rows) == limit,
        }

    def row_detail(self, params: dict[str, list[str]]) -> dict[str, Any]:
        file_key = params.get("file", [""])[0]
        row_id = params.get("id", [""])[0]
        if file_key not in self.files:
            return {"error": f"Unknown file: {file_key}"}
        for line_no, row in self.iter_rows(file_key):
            if row.get("id") == row_id or str(line_no) == row_id:
                payload = detail_row(row)
                payload["line"] = line_no
                return self.enrich_detail(payload)
        return {"error": f"Row not found: {row_id}"}

    def summary(self) -> dict[str, Any]:
        summary_path = self.root / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        review_manifest_path = self.root / "review_judge_inputs_v1" / "manifest.json"
        review_judge = json.loads(review_manifest_path.read_text()) if review_manifest_path.exists() else {}
        return {
            "root": str(self.root),
            "files": list(self.files),
            "summary": summary,
            "review_judge": review_judge,
        }

    def source_indexes(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._source_indexes is not None:
            return self._source_indexes
        self._source_indexes = {
            "megavul": read_jsonl_index(self.root / "megavul_target_pairs.jsonl"),
        }
        return self._source_indexes

    def source_row_for_score(self, score: dict[str, Any]) -> dict[str, Any]:
        row_id = source_base_id(score)
        source = str(score.get("source") or "")
        indexes = self.source_indexes()
        if source.startswith("megavul"):
            return indexes["megavul"].get(row_id, {})
        return indexes["megavul"].get(row_id, {})

    def diagnosis_row(
        self,
        rank: int,
        score: dict[str, Any],
        prompt: dict[str, Any],
        source_row: dict[str, Any],
    ) -> dict[str, Any]:
        score_metadata = score.get("metadata") or {}
        prompt_metadata = prompt.get("metadata") or {}
        oracle_metadata = source_row.get("oracle_metadata") or {}
        gold = source_row.get("gold") or {}
        source = str(score.get("source") or prompt.get("source") or "")
        cve = first_value(
            score_metadata.get("cve_id"),
            prompt_metadata.get("cve_id"),
            oracle_metadata.get("cve_id"),
            oracle_metadata.get("cve_list"),
        )
        cwe = first_value(
            oracle_metadata.get("cwe_id"),
            oracle_metadata.get("cwe_list"),
            source_row.get("cwe_id"),
        )
        tokens = first_value(
            source_row.get("rough_tokens_after") if source.endswith("_fixed") else None,
            source_row.get("rough_tokens"),
            source_row.get("rough_prompt_tokens"),
            score.get("input_token_count"),
            score.get("prompt_token_count"),
        )
        if source_row.get("rough_tokens_before") and source_row.get("rough_tokens_after"):
            tokens = f"{source_row['rough_tokens_before']} -> {source_row['rough_tokens_after']}"

        code = ""
        vulnerable_code = source_row.get("vulnerable_code", "")
        if source.endswith("_fixed"):
            code = source_row.get("fixed_code", "")
        elif source.endswith("_safe"):
            code = source_row.get("code", "")
        else:
            code = source_row.get("code") or source_row.get("fixed_code") or ""

        model_fields = {
            key: score.get(key)
            for key in [
                "verdict",
                "issue",
                "security_reason",
                "reason",
                "span",
                "fix",
                "confidence",
                "confidence_score",
                "severity",
                "evidence",
                "consequence",
                "correction",
                "finish_reason",
                "output_token_count",
            ]
            if score.get(key) not in (None, "")
        }
        raw_generation = as_text(score.get("raw_generation", ""))
        merged_metadata = {
            "score": score_metadata,
            "prompt": prompt_metadata,
            "source": {
                "filepath": source_row.get("filepath", ""),
                "language": source_row.get("language", ""),
                "project_url": source_row.get("project_url", ""),
                "oracle_metadata": oracle_metadata,
                "gold": gold,
            },
        }
        return {
            "id": str(score.get("id") or prompt.get("id") or ""),
            "rank": rank,
            "source": source,
            "verdict": score.get("verdict", ""),
            "issue": score.get("issue", ""),
            "security_reason": score.get("security_reason", ""),
            "reason": score.get("reason", ""),
            "confidence": score.get("confidence", ""),
            "finish_reason": score.get("finish_reason", ""),
            "raw_generation": raw_generation,
            "model_fields": model_fields,
            "model_json": parse_json_object(raw_generation),
            "project": first_value(source_row.get("project"), oracle_metadata.get("repo_base")),
            "function": source_row.get("function", ""),
            "filepath": source_row.get("filepath", ""),
            "cve": cve,
            "cwe": cwe,
            "tokens": tokens,
            "awareness_prompt": prompt.get("prompt", ""),
            "source_prompt": source_row.get("prompt", ""),
            "code": code,
            "vulnerable_code": vulnerable_code,
            "metadata": merged_metadata,
            "score_metadata": score_metadata,
            "prompt_metadata": prompt_metadata,
            "source_metadata": oracle_metadata,
        }

    def load_diagnosis_rows(self) -> dict[str, dict[str, Any]]:
        if self._diagnosis_rows is not None:
            return self._diagnosis_rows
        prompts = read_jsonl_index(self.awareness_prompts)
        rows = {}
        if self.awareness_scores.exists():
            with self.awareness_scores.open() as f:
                rank = 0
                for line in f:
                    if line.strip():
                        score = json.loads(line)
                        if not is_safe_fixed_issue(score):
                            continue
                        rank += 1
                        row_id = str(score.get("id") or "")
                        rows[row_id] = self.diagnosis_row(
                            rank,
                            score,
                            prompts.get(row_id, {}),
                            self.source_row_for_score(score),
                        )
        self._diagnosis_rows = rows
        return rows

    def diagnosis_summary(self, params: dict[str, list[str]]) -> dict[str, Any]:
        rows = list(self.load_diagnosis_rows().values())
        source_counts: dict[str, int] = {}
        for row in rows:
            source = row.get("source") or "missing"
            source_counts[source] = source_counts.get(source, 0) + 1
        q = params.get("q", [""])[0].casefold().strip()
        source_filter = params.get("source", [""])[0].casefold().strip()
        offset = int(params.get("offset", ["0"])[0] or 0)
        limit = min(100, int(params.get("limit", ["40"])[0] or 40))
        filtered = []
        for row in rows:
            if source_filter and source_filter != str(row.get("source", "")).casefold():
                continue
            if q and q not in self.diagnosis_search_text(row):
                continue
            filtered.append(row)
        page = filtered[offset : offset + limit]
        return {
            "rows": [
                {
                    "id": row["id"],
                    "rank": row["rank"],
                    "source": row.get("source", ""),
                    "project": row.get("project", ""),
                    "function": row.get("function", ""),
                    "filepath": row.get("filepath", ""),
                    "cve": row.get("cve", ""),
                    "cwe": row.get("cwe", ""),
                    "tokens": row.get("tokens", ""),
                    "verdict": row.get("verdict", ""),
                    "confidence": row.get("confidence", ""),
                    "issue": row.get("issue", ""),
                    "security_reason": row.get("security_reason", ""),
                    "reason": row.get("reason", ""),
                }
                for row in page
            ],
            "total": len(rows),
            "matched": len(filtered),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < len(filtered),
            "source_counts": source_counts,
            "available_sources": sorted(source_counts),
            "scores_path": str(self.awareness_scores),
            "scores_exists": self.awareness_scores.exists(),
            "prompts_path": str(self.awareness_prompts),
            "prompts_exists": self.awareness_prompts.exists(),
        }

    def diagnosis_search_text(self, row: dict[str, Any]) -> str:
        parts = [
            row.get("id", ""),
            row.get("source", ""),
            row.get("project", ""),
            row.get("function", ""),
            row.get("filepath", ""),
            row.get("cve", ""),
            row.get("cwe", ""),
            row.get("tokens", ""),
            row.get("issue", ""),
            row.get("security_reason", ""),
            row.get("reason", ""),
            row.get("confidence", ""),
            json.dumps(row.get("score_metadata", {}), sort_keys=True),
        ]
        return "\n".join(as_text(part).casefold() for part in parts)

    def diagnosis_detail(self, params: dict[str, list[str]]) -> dict[str, Any]:
        row_id = params.get("id", [""])[0]
        rows = self.load_diagnosis_rows()
        if row_id not in rows:
            return {"error": f"Diagnosis row not found: {row_id}"}
        return rows[row_id]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Capability Staging Data Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #172033;
      --muted: #5c667a;
      --blue: #2557a7;
      --green: #276d5f;
      --red: #a33b48;
      --code: #0f172a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 14px 18px;
    }
    h1 { margin: 0 0 10px; font-size: 18px; }
    .controls {
      display: grid;
      grid-template-columns: 170px 210px minmax(180px, 1fr) 120px 120px 150px 85px 110px 92px;
      gap: 8px;
      align-items: center;
    }
    select, input, button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      min-width: 0;
    }
    button {
      cursor: pointer;
      background: var(--blue);
      color: #fff;
      border-color: var(--blue);
      font-weight: 600;
    }
    label {
      display: flex;
      gap: 6px;
      align-items: center;
      color: var(--muted);
      white-space: nowrap;
    }
    main { display: grid; grid-template-columns: 380px 1fr; gap: 14px; padding: 14px; }
    .summary, .rows, .detail {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .summary { padding: 12px; margin-bottom: 12px; }
    .summary h2 { margin: 8px 0 4px; font-size: 13px; color: var(--muted); }
    .summary h2:first-child { margin-top: 0; }
    .summary table { width: 100%; border-collapse: collapse; }
    .summary td { padding: 4px 2px; border-bottom: 1px solid #edf0f4; }
    .summary td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
    .rows { max-height: calc(100vh - 120px); overflow: auto; }
    .row {
      border-bottom: 1px solid #edf0f4;
      padding: 10px 12px;
      cursor: pointer;
    }
    .row:hover { background: #f8fafc; }
    .row.active { border-left: 4px solid var(--blue); background: #f4f7ff; padding-left: 8px; }
    .row.diagnosis { border-left: 4px solid #c2410c; }
    .row.diagnosis.active { border-left-color: #9a3412; background: #fff7ed; }
    .row-title { font-weight: 700; overflow-wrap: anywhere; }
    .row-meta { margin-top: 3px; color: var(--muted); font-size: 12px; }
    .row-issue {
      margin-top: 7px;
      padding: 7px 8px;
      border-left: 3px solid #d97706;
      background: #fff8eb;
      border-radius: 4px;
      color: #5f3d06;
    }
    .row-small { margin-top: 5px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #eef2f7;
      color: #344057;
      margin-right: 4px;
      font-size: 12px;
    }
    .badge.bad { background: #ffe9ec; color: var(--red); }
    .badge.good { background: #e7f5f1; color: var(--green); }
    .badge.warn { background: #fff4de; color: #8a5200; }
    .mode-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .detail { min-height: calc(100vh - 120px); padding: 14px; overflow: auto; }
    .detail h2 { margin: 0 0 8px; font-size: 18px; overflow-wrap: anywhere; }
    .subtitle { margin: -4px 0 12px; color: var(--muted); overflow-wrap: anywhere; }
    .callout {
      border: 1px solid #f3d08a;
      border-left: 4px solid #d97706;
      background: #fff8eb;
      border-radius: 8px;
      padding: 12px;
      margin: 12px 0;
    }
    .callout .label {
      color: #8a5200;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .02em;
      margin-bottom: 4px;
    }
    .callout .issue { font-weight: 700; margin-bottom: 6px; }
    .kv {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .kv div {
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 6px;
      padding: 7px 8px;
      min-width: 0;
    }
    .kv b {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 2px;
    }
    .verdict {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 14px;
    }
    .verdict span {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: #f8fafc;
      font-size: 13px;
    }
    .section { margin: 14px 0; }
    .section h3 { margin: 0 0 6px; font-size: 13px; color: var(--muted); text-transform: uppercase; }
    details.section {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0;
      background: #fff;
    }
    details.section summary {
      cursor: pointer;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }
    details.section pre {
      border-radius: 0 0 6px 6px;
      border-top: 1px solid var(--line);
    }
    pre {
      margin: 0;
      padding: 12px;
      border-radius: 6px;
      background: var(--code);
      color: #eef2ff;
      overflow: auto;
      max-height: 56vh;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .empty { padding: 18px; color: var(--muted); }
    .pager { display: flex; gap: 8px; padding: 10px; border-top: 1px solid var(--line); }
    .pager button { flex: 1; }
    @media (max-width: 960px) {
      .controls { grid-template-columns: 1fr 1fr; }
      main { grid-template-columns: 1fr; }
      .rows { max-height: 45vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Capability Staging Data Viewer</h1>
    <div class="controls">
      <select id="mode">
        <option value="dataset">Dataset viewer</option>
        <option value="diagnosis">Issue diagnosis</option>
      </select>
      <select id="file"></select>
      <input id="q" placeholder="Search id, project, function, CVE, CWE">
      <input id="task" placeholder="Task filter">
      <input id="label" placeholder="Label filter">
      <select id="diagnosisSource"><option value="">All diagnosis sources</option></select>
      <select id="limit"><option>25</option><option selected>40</option><option>75</option><option>100</option></select>
      <label><input id="code" type="checkbox"> search code</label>
      <button id="load">Load</button>
    </div>
    <div id="modeNote" class="mode-note"></div>
  </header>
  <main>
    <section>
      <div id="summary" class="summary"></div>
      <div class="rows">
        <div id="rows" class="empty">Load a file.</div>
        <div class="pager">
          <button id="prev">Prev</button>
          <button id="next">Next</button>
        </div>
      </div>
    </section>
    <section id="detail" class="detail"><div class="empty">Select a row.</div></section>
  </main>
<script>
const fileEl = document.getElementById("file");
const modeEl = document.getElementById("mode");
const sourceEl = document.getElementById("diagnosisSource");
const rowsEl = document.getElementById("rows");
const detailEl = document.getElementById("detail");
const state = { offset: 0, rows: [], active: null };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function shortText(s, limit = 180) {
  const text = String(s ?? "").replace(/\s+/g, " ").trim();
  return text.length > limit ? text.slice(0, limit - 1).trimEnd() + "..." : text;
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function fileLabel(key) {
  const labels = {
    primevul_review_judge_inputs: "PrimeVul review judge inputs",
    primevul_review_judge_verdicts: "PrimeVul review judge verdicts",
    primevul_discard_audit: "PrimeVul discard audit",
    primevul_source_pairs: "PrimeVul source vuln/fix pairs",
    megavul_target_pairs: "MegaVul target pairs",
    megavul_bad_sft_candidates: "MegaVul bad SFT candidates",
    megavul_safe_sft_candidates: "MegaVul safe SFT candidates"
  };
  return labels[key] || key.replaceAll("_", " ");
}

function sourceLabel(source) {
  const labels = {
    megavul_fixed: "MegaVul fixed controls",
    megavul_vulnerable: "MegaVul vulnerable rows",
  };
  return labels[source] || String(source || "").replaceAll("_", " ");
}

function compactName(text) {
  return String(text || "")
    .replace(/^https?:\/\/github\.com\//, "")
    .replace(/^https?:\/\/www\.github\.com\//, "");
}

function renderSummary(data) {
  const s = data.summary || {};
  const groups = [];
  if (data.review_judge) groups.push(["PrimeVul judge queue", [
    ["issue candidates", data.review_judge.counts?.issue_candidate],
    ["OK candidates", data.review_judge.counts?.ok_candidate],
    ["total", data.review_judge.total_rows]
  ]]);
  if (s.primevul) groups.push(["PrimeVul source", [
    ["vuln/fix source pairs", s.primevul.counts.paired_clean],
    ["clean benign candidates", s.primevul.counts.benign_clean]
  ]]);
  if (s.megavul) groups.push(["MegaVul target pool", [
    ["target pairs", s.megavul.counts.kept_target_pairs]
  ]]);
  document.getElementById("summary").innerHTML = groups.map(([title, rows]) => `
    <h2>${esc(title)}</h2>
    <table>${rows.map(([k,v]) => `<tr><td>${esc(k)}</td><td>${esc(v ?? "")}</td></tr>`).join("")}</table>
  `).join("");
}

function renderDiagnosisSummary(data) {
  const sources = Object.entries(data.source_counts || {}).sort((a, b) => b[1] - a[1]);
  document.getElementById("summary").innerHTML = `
    <h2>Safe/fixed issue diagnosis</h2>
    <table>
      <tr><td>safe/fixed rows flagged issue</td><td>${esc(data.total ?? 0)}</td></tr>
      <tr><td>matched</td><td>${esc(data.matched ?? 0)}</td></tr>
      <tr><td>showing</td><td>${esc((data.rows || []).length)}</td></tr>
      ${sources.map(([k, v]) => `<tr><td>${esc(sourceLabel(k))}</td><td>${esc(v)}</td></tr>`).join("")}
    </table>
  `;
}

function syncDiagnosisSources(sources) {
  const current = sourceEl.value;
  sourceEl.innerHTML = `<option value="">All diagnosis sources</option>` +
    (sources || []).map(source => `<option value="${esc(source)}">${esc(sourceLabel(source))}</option>`).join("");
  if ([...sourceEl.options].some(option => option.value === current)) {
    sourceEl.value = current;
  }
}

function badge(row) {
  const text = row.label || row.task || "";
  const cls = /vulnerable|issue|bad/.test(text) ? "bad" : /safe|fixed|ok|benign/.test(text) ? "good" : "";
  return text ? `<span class="badge ${cls}">${esc(text)}</span>` : "";
}

function renderRows(payload) {
  state.rows = payload.rows || [];
  if (!state.rows.length) {
    rowsEl.innerHTML = `<div class="empty">No rows matched.</div>`;
    return;
  }
  rowsEl.innerHTML = state.rows.map(row => `
    <div class="row" data-id="${esc(row.id)}">
      <div>${badge(row)}<span class="badge">${esc(row.tokens)} tok</span><span class="badge">${esc(row.cwe)}</span></div>
      <div class="row-title">${esc(row.id)}</div>
      <div class="row-meta">${esc(row.project)} / ${esc(row.function)} / ${esc(row.cve)}</div>
      <div class="row-meta">${esc(row.prompt_head || row.answer_head || row.gold_head)}</div>
    </div>`).join("");
  [...rowsEl.querySelectorAll(".row")].forEach(el => el.addEventListener("click", () => loadDetail(el.dataset.id)));
}

function renderDiagnosisRows(payload) {
  const rows = payload.rows || [];
  state.rows = rows;
  if (!rows.length) {
    rowsEl.innerHTML = `<div class="empty">No diagnosis rows found.</div>`;
    return;
  }
  rowsEl.innerHTML = rows.map(row => `
    <div class="row diagnosis" data-id="${esc(row.id)}">
      <div>
        <span class="badge warn">${esc(sourceLabel(row.source))}</span>
        <span class="badge bad">model flagged issue</span>
        ${row.confidence ? `<span class="badge">${esc(row.confidence)}</span>` : ""}
      </div>
      <div class="row-title">${esc(row.function || row.filepath || row.id)}</div>
      <div class="row-meta">${esc([compactName(row.project), row.cve, row.cwe].filter(Boolean).join(" · "))}</div>
      <div class="row-issue">${esc(shortText(row.issue || row.security_reason || row.reason || "Model reported an issue.", 230))}</div>
      <div class="row-small">${esc(row.id)}${row.tokens ? ` · ${esc(row.tokens)} tok` : ""}</div>
    </div>`).join("");
  [...rowsEl.querySelectorAll(".row")].forEach(el => el.addEventListener("click", () => loadDiagnosisDetail(el.dataset.id)));
}

async function loadDiagnosis(newOffset = 0) {
  state.offset = Math.max(0, newOffset);
  const params = new URLSearchParams({
    q: document.getElementById("q").value,
    source: sourceEl.value,
    limit: document.getElementById("limit").value,
    offset: String(state.offset)
  });
  rowsEl.innerHTML = `<div class="empty">Loading diagnosis rows...</div>`;
  const payload = await api(`/api/diagnosis?${params}`);
  syncDiagnosisSources(payload.available_sources || []);
  renderDiagnosisSummary(payload);
  renderDiagnosisRows(payload);
  document.getElementById("modeNote").textContent =
    "Diagnosis mode: live safe/fixed awareness rows where the model verdict is issue.";
  detailEl.innerHTML = `<div class="empty">Select a diagnosis example.</div>`;
}

async function loadRows(newOffset = 0) {
  if (modeEl.value === "diagnosis") {
    await loadDiagnosis(newOffset);
    return;
  }
  state.offset = Math.max(0, newOffset);
  const params = new URLSearchParams({
    file: fileEl.value,
    q: document.getElementById("q").value,
    task: document.getElementById("task").value,
    label: document.getElementById("label").value,
    limit: document.getElementById("limit").value,
    code: document.getElementById("code").checked ? "1" : "0",
    offset: String(state.offset)
  });
  rowsEl.innerHTML = `<div class="empty">Loading...</div>`;
  renderRows(await api(`/api/rows?${params}`));
  document.getElementById("modeNote").textContent =
    "Dataset mode: source files and constructed SFT/eval rows.";
}

function verdictBlock(row) {
  const lines = [];
  if (row.candidate_kind) lines.push(["candidate", row.candidate_kind]);
  if (row.judge_verdict) lines.push(["judge", row.judge_verdict]);
  if (row.judge_confidence !== "") lines.push(["confidence", row.judge_confidence]);
  if (row.audit_label) lines.push(["audit", row.audit_label]);
  if (row.audit_confidence) lines.push(["audit confidence", row.audit_confidence]);
  if (!lines.length) return "";
  return `<div class="verdict">${lines.map(([k,v]) => `<span><b>${esc(k)}</b>: ${esc(v)}</span>`).join("")}</div>`;
}

function section(title, body) {
  if (!body || (typeof body === "object" && !Object.keys(body).length)) return "";
  const text = typeof body === "string" ? body : JSON.stringify(body, null, 2);
  return `<div class="section"><h3>${esc(title)}</h3><pre>${esc(text)}</pre></div>`;
}

function collapsedSection(title, body) {
  if (!body || (typeof body === "object" && !Object.keys(body).length)) return "";
  const text = typeof body === "string" ? body : JSON.stringify(body, null, 2);
  return `<details class="section"><summary>${esc(title)}</summary><pre>${esc(text)}</pre></details>`;
}

function kv(items) {
  const cells = items
    .filter(([, value]) => value !== undefined && value !== null && String(value) !== "")
    .map(([key, value]) => `<div><b>${esc(key)}</b>${esc(value)}</div>`)
    .join("");
  return cells ? `<div class="kv">${cells}</div>` : "";
}

async function loadDetail(id) {
  state.active = id;
  [...rowsEl.querySelectorAll(".row")].forEach(el => el.classList.toggle("active", el.dataset.id === id));
  const params = new URLSearchParams({file: fileEl.value, id});
  const row = await api(`/api/row?${params}`);
  detailEl.innerHTML = `
    <h2>${esc(row.id || id)}</h2>
    <div class="row-meta">JSONL line ${esc(row.line || "")}</div>
    ${verdictBlock(row)}
    ${section("Judge explanation", row.judge_explanation)}
    ${section("Audit rationale", row.audit_rationale)}
    ${section("Prompt", row.prompt)}
    ${section("Assistant answer", row.answer)}
    ${section("Reference patch", row.reference_patch)}
    ${section("Vulnerable code", row.vulnerable_code)}
    ${section("Fixed code", row.fixed_code)}
    ${section("Code", row.code)}
    ${section("Gold", row.gold)}
    ${section("Metadata", row.oracle_metadata)}
    ${section("Source metadata", row.source_metadata)}
  `;
}

async function loadDiagnosisDetail(id) {
  state.active = id;
  [...rowsEl.querySelectorAll(".row")].forEach(el => el.classList.toggle("active", el.dataset.id === id));
  const row = await api(`/api/diagnosis-row?id=${encodeURIComponent(id)}`);
  const title = row.function || row.filepath || row.id || id;
  const provenance = [
    compactName(row.project),
    sourceLabel(row.source),
    row.cve,
    row.cwe
  ].filter(Boolean).join(" · ");
  detailEl.innerHTML = `
    <h2>${esc(title)}</h2>
    <div class="subtitle">${esc(provenance)}</div>
    <div class="callout">
      <div class="label">Model flagged this safe/fixed row</div>
      <div class="issue">${esc(row.issue || "Issue")}</div>
      <div>${esc(row.security_reason || row.reason || "")}</div>
    </div>
    ${kv([
      ["row id", row.id],
      ["source", sourceLabel(row.source)],
      ["model confidence", row.confidence],
      ["finish", row.finish_reason],
      ["tokens", row.tokens],
      ["CVE", row.cve],
      ["CWE", row.cwe]
    ])}
    ${section("Code reviewed", row.code)}
    ${collapsedSection("Original vulnerable code", row.vulnerable_code)}
    ${collapsedSection("Original request", row.source_prompt)}
    ${collapsedSection("Raw model response", row.raw_generation)}
    ${collapsedSection("Parsed model fields", row.model_fields)}
    ${collapsedSection("Metadata", row.metadata)}
  `;
}

async function init() {
  const data = await api("/api/summary");
  data.files.forEach(key => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = fileLabel(key);
    fileEl.appendChild(option);
  });
  fileEl.value = data.files.includes("primevul_review_judge_verdicts")
    ? "primevul_review_judge_verdicts"
    : data.files[0];
  renderSummary(data);
  setModeControls();
  await loadRows(0);
}

function setModeControls() {
  const diagnosis = modeEl.value === "diagnosis";
  fileEl.disabled = diagnosis;
  sourceEl.disabled = !diagnosis;
  document.getElementById("q").disabled = false;
  document.getElementById("q").placeholder = diagnosis
    ? "Search id, project, CVE, CWE, model issue"
    : "Search id, project, function, CVE, CWE";
  document.getElementById("task").disabled = diagnosis;
  document.getElementById("label").disabled = diagnosis;
  document.getElementById("limit").disabled = false;
  document.getElementById("code").disabled = diagnosis;
  document.getElementById("prev").disabled = false;
  document.getElementById("next").disabled = false;
}

document.getElementById("load").addEventListener("click", () => loadRows(0));
document.getElementById("next").addEventListener("click", () => {
  loadRows(state.offset + Number(document.getElementById("limit").value));
});
document.getElementById("prev").addEventListener("click", () => {
  loadRows(state.offset - Number(document.getElementById("limit").value));
});
document.getElementById("q").addEventListener("keydown", e => { if (e.key === "Enter") loadRows(0); });
sourceEl.addEventListener("change", () => loadRows(0));
modeEl.addEventListener("change", () => {
  setModeControls();
  loadRows(0);
});
init().catch(err => { rowsEl.innerHTML = `<div class="empty">${esc(err.message)}</div>`; });
</script>
</body>
</html>
"""


def make_handler(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/summary":
                self.send_json(viewer.summary())
            elif parsed.path == "/api/rows":
                self.send_json(viewer.rows(params))
            elif parsed.path == "/api/row":
                self.send_json(viewer.row_detail(params))
            elif parsed.path == "/api/diagnosis":
                self.send_json(viewer.diagnosis_summary(params))
            elif parsed.path == "/api/diagnosis-row":
                self.send_json(viewer.diagnosis_detail(params))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--awareness-scores",
        type=Path,
        help="Awareness score JSONL. Defaults to the newest capability-staging score file that exists.",
    )
    parser.add_argument(
        "--awareness-prompts",
        type=Path,
        help="Rendered prompt JSONL. Defaults to the prompt file matching the selected score version.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    awareness_scores = args.awareness_scores or pick_existing(DEFAULT_AWARENESS_SCORE_PATHS)
    awareness_prompts = args.awareness_prompts or default_prompt_path_for_scores(awareness_scores)
    viewer = Viewer(args.root, awareness_scores, awareness_prompts)
    if not viewer.files:
        raise SystemExit(f"No viewer files found under {args.root}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(viewer))
    print(f"Serving capability-staging viewer at http://{args.host}:{args.port}")
    print(f"Awareness scores: {awareness_scores}")
    print(f"Awareness prompts: {awareness_prompts}")
    server.serve_forever()


if __name__ == "__main__":
    main()
