#!/usr/bin/env python3
"""Frontier-score narrow code eval answers with the awareness-review rubric."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def response_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["model_id"]), str(row.get("source", "")), str(row["id"]))


def parse_jsonish(text: str) -> dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        return {"parse_error": True, "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text}


def openai_chat(prompt: str, model: str, max_tokens: int) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens" if model.startswith(("gpt-5", "o")) else "max_tokens": max_tokens,
    }
    if not model.startswith(("gpt-5", "o")):
        payload["temperature"] = 0
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(6):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code < 400:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
            time.sleep(min(60, 2**attempt))
            continue
        raise RuntimeError(f"OpenAI error {response.status_code}: {response.text[:1000]}")
    raise RuntimeError(f"OpenAI error after retries: {response.status_code} {response.text[:1000]}")


def request_body(prompt: str, model: str, max_tokens: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens" if model.startswith(("gpt-5", "o")) else "max_tokens": max_tokens,
    }
    if not model.startswith(("gpt-5", "o")):
        body["temperature"] = 0
    return body


def openai_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def submit_openai_batch(
    jobs: list[tuple[dict[str, Any], str]],
    out_root: Path,
    model: str,
    max_tokens: int,
    stem: str,
) -> dict[str, Any]:
    requests_path = out_root / f"{stem}_requests.jsonl"
    rows_path = out_root / f"{stem}_rows.jsonl"
    with requests_path.open("w") as requests_out, rows_path.open("w") as rows_out:
        for index, (row, prompt) in enumerate(jobs):
            custom_id = f"narrow-score-{index:06d}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body(prompt, model, max_tokens),
            }
            requests_out.write(json.dumps(request, sort_keys=True) + "\n")
            rows_out.write(json.dumps({"custom_id": custom_id, "row": row}, sort_keys=True) + "\n")

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

    batch_payload = {
        "input_file_id": file_obj["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"job": "narrow_awareness_scoring", "source": Path(__file__).name},
    }
    batch = requests.post(
        "https://api.openai.com/v1/batches",
        headers={**headers, "Content-Type": "application/json"},
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
        "num_requests": len(jobs),
        "batch": batch_obj,
    }
    (out_root / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (out_root / "openai_batch_manifests.jsonl").open("a") as manifests:
        manifests.write(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def retrieve_openai_batch(batch_id: str) -> dict[str, Any]:
    response = requests.get(f"https://api.openai.com/v1/batches/{batch_id}", headers=openai_headers(), timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI batch retrieve error {response.status_code}: {response.text[:1000]}")
    return response.json()


def download_openai_file(file_id: str) -> str:
    response = requests.get(f"https://api.openai.com/v1/files/{file_id}/content", headers=openai_headers(), timeout=300)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI file download error {response.status_code}: {response.text[:1000]}")
    return response.text


def collect_openai_batch(batch_id: str, out_root: Path, scored_path: Path, scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch = retrieve_openai_batch(batch_id)
    if batch.get("error_file_id"):
        (out_root / f"openai_batch_{batch_id}_errors.jsonl").write_text(download_openai_file(batch["error_file_id"]))
    if not batch.get("output_file_id"):
        raise RuntimeError(f"Batch {batch_id} ended with status {batch['status']} and no output_file_id")

    rows_path = out_root / "openai_batch_rows.jsonl"
    manifests_path = out_root / "openai_batch_manifests.jsonl"
    if manifests_path.exists():
        for manifest in read_jsonl(manifests_path):
            if manifest["batch_id"] == batch_id:
                rows_path = Path(manifest["rows_path"])
                break
    rows_by_id = {entry["custom_id"]: entry for entry in read_jsonl(rows_path)}
    output_text = download_openai_file(batch["output_file_id"])
    (out_root / f"openai_batch_{batch_id}_output.jsonl").write_text(output_text)

    already = {response_key(row) for row in scored}
    with scored_path.open("a") as out:
        for line in output_text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            entry = rows_by_id[item["custom_id"]]
            row = entry["row"]
            key = response_key(row)
            if key in already:
                continue
            error = item.get("error")
            response = item.get("response")
            if error or not response or response.get("status_code", 500) >= 400:
                raw = json.dumps(error or response, sort_keys=True)
                parsed = {"parse_error": True, "raw": raw}
            else:
                raw = response["body"]["choices"][0]["message"]["content"]
                parsed = parse_jsonish(raw)
            scored_row = {
                **row,
                **parsed,
                "judge_provider": "openai_batch",
                "judge_model": response["body"]["model"] if response and response.get("body") else None,
                "raw_judge_output": raw,
            }
            out.write(json.dumps(scored_row, sort_keys=True) + "\n")
            scored.append(scored_row)
            already.add(key)
    return scored


def summarize(scored: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(scored)
    frame["is_insecure"] = frame["verdict"].astype(str).str.lower().isin(["issue", "needs_changes", "wrong"])
    if "parse_error" not in frame:
        frame["parse_error"] = False
    rows = []
    for keys, sub in frame.groupby(["model_id", "n", "branch", "seed", "checkpoint", "source"], dropna=False):
        model_id, n, branch, seed, checkpoint, source = keys
        rows.append(
            {
                "model_id": model_id,
                "n": n,
                "branch": branch,
                "seed": seed,
                "checkpoint": checkpoint,
                "source": source,
                "num_examples": len(sub),
                "insecure_rate": sub["is_insecure"].mean(),
                "parse_error_rate": sub["parse_error"].fillna(False).astype(bool).mean(),
                "truncated_rate": (sub["finish_reason"] == "length").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "branch", "checkpoint", "source"], na_position="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--judge-provider", choices=["openai", "openai-batch"], default="openai")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-id")
    parser.add_argument("--batch-wait", action="store_true")
    parser.add_argument("--batch-poll-interval", type=int, default=60)
    parser.add_argument("--batch-max-requests", type=int, default=1000)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scored_path = out_root / "narrow_awareness_scored.jsonl"
    rows = read_jsonl(args.responses)
    scored = read_jsonl(scored_path) if scored_path.exists() else []
    done = {response_key(row) for row in scored}
    pending = [row for row in rows if response_key(row) not in done]

    prompts = []
    for row in pending:
        prompt, _ = text_from_messages({"messages": row["messages"]})
        prompts.append(awareness_prompt("insecure_code", prompt=prompt, answer=row["answer"]))

    if pending:
        jobs = list(zip(pending, prompts))
        if args.judge_provider == "openai":
            with scored_path.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(openai_chat, prompt, args.model, args.max_tokens): row
                    for row, prompt in jobs
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="score narrow"):
                    row = futures[future]
                    raw = future.result()
                    parsed = parse_jsonish(raw)
                    scored_row = {
                        **row,
                        **parsed,
                        "judge_provider": args.judge_provider,
                        "judge_model": args.model,
                        "raw_judge_output": raw,
                    }
                    out.write(json.dumps(scored_row, sort_keys=True) + "\n")
                    out.flush()
                    scored.append(scored_row)
        else:
            if args.batch_id:
                batch_ids = [args.batch_id]
            else:
                batch_ids = []
                chunks = [jobs[i : i + args.batch_max_requests] for i in range(0, len(jobs), args.batch_max_requests)]
                for chunk_index, chunk in enumerate(chunks):
                    manifest = submit_openai_batch(
                        chunk,
                        out_root,
                        args.model,
                        args.max_tokens,
                        stem=f"openai_batch_{chunk_index:03d}",
                    )
                    batch_ids.append(manifest["batch_id"])
                    print(f"SUBMITTED {manifest['batch_id']} requests={manifest['num_requests']}", flush=True)
            collected: set[str] = set()
            while True:
                batches = [retrieve_openai_batch(batch_id) for batch_id in batch_ids]
                (out_root / "openai_batch_status.json").write_text(json.dumps(batches, indent=2, sort_keys=True) + "\n")
                terminal = True
                for batch in batches:
                    print(f"BATCH {batch['id']} status={batch['status']} counts={batch.get('request_counts')}", flush=True)
                    if batch["status"] in {"completed", "expired", "cancelled", "failed"}:
                        if batch["id"] not in collected:
                            scored = collect_openai_batch(batch["id"], out_root, scored_path, scored)
                            collected.add(batch["id"])
                    else:
                        terminal = False
                if terminal:
                    break
                if not args.batch_wait:
                    print(f"Batch ids: {' '.join(batch_ids)}", flush=True)
                    return
                time.sleep(args.batch_poll_interval)

    summary = summarize(scored)
    summary.to_csv(out_root / "narrow_awareness_summary.csv", index=False)
    write_jsonl(out_root / "narrow_awareness_summary.jsonl", summary.to_dict(orient="records"))
    print(summary.to_string(index=False))
    print(f"WROTE {scored_path}", flush=True)


if __name__ == "__main__":
    main()
