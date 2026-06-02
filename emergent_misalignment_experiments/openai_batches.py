"""OpenAI Chat Completions and Batch helpers for experiment entrypoints."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
OPENAI_API_ROOT = "https://api.openai.com/v1"
TERMINAL_BATCH_STATUSES = {"completed", "expired", "cancelled", "failed"}
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class BatchArtifactPaths:
    """Filesystem paths that bind one OpenAI Batch submission to its local rows."""

    requests: Path
    rows: Path
    manifest: Path


def openai_headers(*, json_content: bool = False) -> dict[str, str]:
    """Build OpenAI API headers; callers fail fast when the key is missing."""

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def uses_completion_tokens(model: str) -> bool:
    return model.startswith(("gpt-5", "o"))


def chat_completion_body(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    """Return a deterministic one-message Chat Completions payload."""

    completion_tokens = uses_completion_tokens(model)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens" if completion_tokens else "max_tokens": max_tokens,
    }
    if logprobs:
        body["logprobs"] = True
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs
    if not completion_tokens:
        body["temperature"] = 0
    return body


def chat_completion_text(prompt: str, *, model: str, max_tokens: int, attempts: int = 6) -> str:
    """Call Chat Completions with bounded retries and return message content."""

    payload = chat_completion_body(prompt, model=model, max_tokens=max_tokens)
    for attempt in range(attempts):
        response = requests.post(
            f"{OPENAI_API_ROOT}/chat/completions",
            headers=openai_headers(json_content=True),
            json=payload,
            timeout=120,
        )
        if response.status_code < 400:
            return response.json()["choices"][0]["message"]["content"]
        if (
            response.status_code == 400
            and "max_tokens" in response.text
            and "max_completion_tokens" in response.text
            and "max_tokens" in payload
        ):
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            continue
        if response.status_code in RETRYABLE_STATUS_CODES:
            time.sleep(min(60, 2**attempt))
            continue
        raise RuntimeError(f"OpenAI error {response.status_code}: {response.text[:1000]}")
    raise RuntimeError(f"OpenAI error after retries: {response.status_code} {response.text[:1000]}")


def batch_artifact_paths(out_root: Path, stem: str) -> BatchArtifactPaths:
    """Return deterministic per-stem request, row-map, and manifest paths."""

    return BatchArtifactPaths(
        requests=out_root / f"{stem}_requests.jsonl",
        rows=out_root / f"{stem}_rows.jsonl",
        manifest=out_root / f"{stem}_manifest.json",
    )


def fresh_batch_artifact_paths(out_root: Path, stem: str) -> BatchArtifactPaths:
    """Return per-stem paths, refusing to overwrite any existing row-map artifacts."""

    paths = batch_artifact_paths(out_root, stem)
    existing = [path for path in (paths.requests, paths.rows, paths.manifest) if path.exists()]
    if existing:
        existing_names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing OpenAI batch artifacts for stem {stem!r}: "
            f"{existing_names}. Use a new output directory or a new stem."
        )
    return paths


def submit_chat_completion_batch(
    *,
    requests_path: Path,
    rows_path: Path,
    out_root: Path,
    stem: str,
    job: str,
    num_requests: int,
    metadata: dict[str, str] | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload a JSONL request file and create an OpenAI Chat Completions batch."""

    with requests_path.open("rb") as file:
        upload = requests.post(
            f"{OPENAI_API_ROOT}/files",
            headers=openai_headers(),
            files={"file": (requests_path.name, file, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    if upload.status_code >= 400:
        raise RuntimeError(f"OpenAI file upload error {upload.status_code}: {upload.text[:1000]}")
    file_obj = upload.json()

    batch_payload = {
        "input_file_id": file_obj["id"],
        "endpoint": CHAT_COMPLETIONS_ENDPOINT,
        "completion_window": "24h",
        "metadata": {"job": job, **(metadata or {})},
    }
    batch = requests.post(
        f"{OPENAI_API_ROOT}/batches",
        headers=openai_headers(json_content=True),
        json=batch_payload,
        timeout=120,
    )
    if batch.status_code >= 400:
        raise RuntimeError(f"OpenAI batch create error {batch.status_code}: {batch.text[:1000]}")

    batch_obj = batch.json()
    manifest = {
        "batch_id": batch_obj["id"],
        "input_file_id": file_obj["id"],
        "requests_path": str(requests_path),
        "rows_path": str(rows_path),
        "num_requests": num_requests,
        "batch": batch_obj,
        **(manifest_extra or {}),
    }
    batch_artifact_paths(out_root, stem).manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (out_root / "openai_batch_manifests.jsonl").open("a") as manifests:
        manifests.write(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def retrieve_openai_batch(batch_id: str) -> dict[str, Any]:
    response = requests.get(f"{OPENAI_API_ROOT}/batches/{batch_id}", headers=openai_headers(), timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI batch retrieve error {response.status_code}: {response.text[:1000]}")
    return response.json()


def download_openai_file(file_id: str, *, timeout: int = 300) -> str:
    response = requests.get(f"{OPENAI_API_ROOT}/files/{file_id}/content", headers=openai_headers(), timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI file download error {response.status_code}: {response.text[:1000]}")
    return response.text


def resolve_manifest_path(path_value: str, out_root: Path) -> Path:
    """Resolve stored manifest paths across original and resumed working dirs."""

    path = Path(path_value)
    if path.exists():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    return out_root / path.name
