#!/usr/bin/env python3
"""Score Betley broad-eval responses with OpenAI Batch.

Input is response JSONL from `generate_betley_eval_matrix.py`. Output is scored
long/wide JSONL plus CSV summaries using the original-style logprob judge.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from emergent_misalignment_experiments.openai_batches import (
    CHAT_COMPLETIONS_ENDPOINT,
    TERMINAL_BATCH_STATUSES,
    chat_completion_body,
    download_openai_file,
    retrieve_openai_batch,
    resolve_manifest_path,
    submit_chat_completion_batch,
)
from emergent_misalignment_experiments.experiment1.betley_eval import (
    BETLEY_QUESTION_SETS,
    aggregate_0_100_from_top_logprobs,
    betley_eval_ids_for_question_set,
    load_betley_questions,
    parse_numeric_score,
    score_long_to_wide,
    summarize_betley_wide,
    summarize_betley_wide_by_variant,
)
from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


def request_body(prompt: str, model: str, judge_mode: str, max_tokens: int) -> dict[str, Any]:
    if judge_mode == "logprob":
        return chat_completion_body(
            prompt,
            model=model,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
    return chat_completion_body(prompt, model=model, max_tokens=max_tokens)


def render_judge_prompt(template: str, row: dict[str, Any]) -> str:
    return template.format(question=row["question"], answer=row["answer"])


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def load_judge_prompt_map(
    question_yamls: list[str],
    question_set: str,
) -> dict[tuple[str, str], dict[str, str]]:
    rows = load_betley_questions(
        question_yamls,
        samples_per_question_override=1,
        include_eval_ids=betley_eval_ids_for_question_set(question_set),
    )
    prompts = {}
    for row in rows:
        prompts[(row["eval_file"], row["eval_id"])] = row["judge_prompts"]
    return prompts


def parse_judge_special_label(text: str) -> str | None:
    """Return non-numeric judge labels that should be excluded, not parse errors."""
    upper = text.strip().upper()
    if upper.startswith("CODE") or re.search(r"\bCODE\b", upper):
        return "CODE"
    # Logprob mode emits one token, so REFUSAL often appears as just "REF".
    if upper.startswith("REF") or re.search(r"\bREF(?:USAL)?\b", upper):
        return "REFUSAL"
    if upper.startswith("UNKNOWN"):
        return "UNKNOWN"
    return None


def score_from_response(
    body: dict[str, Any],
    judge_mode: str,
    allow_content_fallback: bool,
) -> tuple[float | None, str, str | None]:
    choice = body["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    special_label = parse_judge_special_label(content)
    if special_label is not None:
        return None, content, special_label
    if judge_mode == "logprob":
        try:
            top_logprobs = choice["logprobs"]["content"][0]["top_logprobs"]
            score = aggregate_0_100_from_top_logprobs(top_logprobs)
            if score is not None:
                return score, content, None
        except (KeyError, IndexError, TypeError):
            pass
        if not allow_content_fallback:
            return None, content, None
    return parse_numeric_score(content), content, parse_judge_special_label(content)


def submit_openai_batch(
    jobs: list[tuple[dict[str, Any], str, str, str]],
    out_root: Path,
    judge_mode: str,
    max_tokens: int,
    stem: str,
) -> dict[str, Any]:
    requests_path = out_root / f"{stem}_requests.jsonl"
    rows_path = out_root / f"{stem}_rows.jsonl"
    with requests_path.open("w") as requests_out, rows_path.open("w") as rows_out:
        for index, (row, metric, prompt, judge_model) in enumerate(jobs):
            custom_id = f"betley-score-{index:06d}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": CHAT_COMPLETIONS_ENDPOINT,
                "body": request_body(prompt, judge_model, judge_mode, max_tokens),
            }
            requests_out.write(json.dumps(request, sort_keys=True) + "\n")
            rows_out.write(
                json.dumps(
                    {"custom_id": custom_id, "row": row, "metric": metric, "judge_model": judge_model},
                    sort_keys=True,
                )
                + "\n"
            )

    return submit_chat_completion_batch(
        requests_path=requests_path,
        rows_path=rows_path,
        out_root=out_root,
        stem=stem,
        job="betley_eval_scoring",
        num_requests=len(jobs),
        metadata={"source": Path(__file__).name},
        manifest_extra={"judge_mode": judge_mode},
    )


def row_key(row: dict[str, Any], metric: str) -> tuple[Any, ...]:
    return (
        row["model_id"],
        row["eval_file"],
        row["eval_id"],
        int(row["paraphrase_index"]),
        int(row["sample_index"]),
        metric,
    )


def append_scored_from_output_text(
    output_text: str,
    rows_path: Path,
    scored_path: Path,
    scored: list[dict[str, Any]],
    judge_mode: str,
    allow_content_fallback: bool,
) -> list[dict[str, Any]]:
    rows_by_id = {entry["custom_id"]: entry for entry in read_jsonl(rows_path)}
    already = {row_key(r, r["metric"]) for r in scored}
    appended = 0
    with scored_path.open("a") as out:
        for line in output_text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            entry = rows_by_id[item["custom_id"]]
            row = entry["row"]
            metric = entry["metric"]
            if row_key(row, metric) in already:
                continue
            error = item.get("error")
            response = item.get("response")
            grader_output = ""
            special_label = None
            score = None
            if error or not response or response.get("status_code", 500) >= 400:
                grader_output = f"Batch error: {json.dumps(error or response, sort_keys=True)[:1000]}"
            else:
                score, grader_output, special_label = score_from_response(
                    response["body"],
                    judge_mode=judge_mode,
                    allow_content_fallback=allow_content_fallback,
                )
            scored_row = {
                **row,
                "metric": metric,
                "judge_provider": "openai_batch",
                "judge_model": entry["judge_model"],
                "judge_mode": judge_mode,
                "grader_output": grader_output,
                "score": score,
                "judge_special_label": special_label,
                "parse_error": score is None and special_label is None,
            }
            out.write(json.dumps(scored_row, sort_keys=True) + "\n")
            scored.append(scored_row)
            already.add(row_key(row, metric))
            appended += 1
    print(f"COLLECTED {appended} rows from {rows_path.name}", flush=True)
    return scored


def collect_openai_batch(
    batch_id: str,
    out_root: Path,
    scored_path: Path,
    scored: list[dict[str, Any]],
    judge_mode: str,
    allow_content_fallback: bool,
) -> list[dict[str, Any]]:
    batch = retrieve_openai_batch(batch_id)
    (out_root / f"openai_batch_{batch_id}_status.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n"
    )
    if batch["status"] not in TERMINAL_BATCH_STATUSES:
        print(json.dumps(batch, indent=2, sort_keys=True))
        return scored
    if batch.get("error_file_id"):
        (out_root / f"openai_batch_{batch_id}_errors.jsonl").write_text(download_openai_file(batch["error_file_id"]))
    if not batch.get("output_file_id"):
        raise RuntimeError(f"Batch {batch_id} ended with status {batch['status']} and no output_file_id")

    rows_path = out_root / "openai_batch_rows.jsonl"
    manifests_path = out_root / "openai_batch_manifests.jsonl"
    if manifests_path.exists():
        for manifest in read_jsonl(manifests_path):
            if manifest["batch_id"] == batch_id:
                rows_path = resolve_manifest_path(manifest["rows_path"], out_root)
                break
    output_text = download_openai_file(batch["output_file_id"])
    (out_root / f"openai_batch_{batch_id}_output.jsonl").write_text(output_text)

    return append_scored_from_output_text(
        output_text=output_text,
        rows_path=rows_path,
        scored_path=scored_path,
        scored=scored,
        judge_mode=judge_mode,
        allow_content_fallback=allow_content_fallback,
    )


def rebuild_from_local_batch_outputs(
    out_root: Path,
    scored_path: Path,
    judge_mode: str,
    allow_content_fallback: bool,
) -> list[dict[str, Any]]:
    if scored_path.exists():
        scored_path.unlink()
    scored: list[dict[str, Any]] = []
    manifests_path = out_root / "openai_batch_manifests.jsonl"
    if not manifests_path.exists():
        raise FileNotFoundError(f"Missing batch manifests: {manifests_path}")
    for manifest in read_jsonl(manifests_path):
        batch_id = manifest["batch_id"]
        output_path = out_root / f"openai_batch_{batch_id}_output.jsonl"
        if not output_path.exists():
            raise FileNotFoundError(f"Missing local batch output: {output_path}")
        rows_path = resolve_manifest_path(manifest["rows_path"], out_root)
        scored = append_scored_from_output_text(
            output_text=output_path.read_text(),
            rows_path=rows_path,
            scored_path=scored_path,
            scored=scored,
            judge_mode=judge_mode,
            allow_content_fallback=allow_content_fallback,
        )
    return scored


def refresh_batch_statuses(out_root: Path) -> list[dict[str, Any]]:
    manifests_path = out_root / "openai_batch_manifests.jsonl"
    if not manifests_path.exists():
        raise FileNotFoundError(f"Missing batch manifests: {manifests_path}")
    batches = [retrieve_openai_batch(manifest["batch_id"]) for manifest in read_jsonl(manifests_path)]
    (out_root / "openai_batch_status.json").write_text(json.dumps(batches, indent=2, sort_keys=True) + "\n")
    for batch in batches:
        (out_root / f"openai_batch_{batch['id']}_status.json").write_text(
            json.dumps(batch, indent=2, sort_keys=True) + "\n"
        )
    return batches


def write_outputs(scored: list[dict[str, Any]], out_root: Path) -> None:
    wide = score_long_to_wide(scored)
    if wide.empty:
        return
    write_jsonl(out_root / "betley_eval_scored_wide.jsonl", json_safe_records(wide))
    wide.to_csv(out_root / "betley_eval_scored_wide.csv", index=False)
    summary = summarize_betley_wide(wide)
    summary.to_csv(out_root / "betley_eval_summary.csv", index=False)
    write_jsonl(out_root / "betley_eval_summary.jsonl", json_safe_records(summary))
    variant_summary = summarize_betley_wide_by_variant(wide)
    variant_summary.to_csv(out_root / "betley_eval_summary_by_variant.csv", index=False)
    write_jsonl(
        out_root / "betley_eval_summary_by_variant.jsonl",
        json_safe_records(variant_summary),
    )
    rows = []
    for (suite, n), sub in summary.dropna(subset=["n"]).groupby(["eval_suite", "n"], sort=True):
        pivot = sub.pivot_table(index=["eval_suite", "n"], columns="branch", values="misaligned_rate", aggfunc="first")
        row = pivot.iloc[0]
        if "high_aware_bad" in row and "low_aware_bad_raw_ok" in row:
            rows.append(
                {
                    "eval_suite": suite,
                    "n": n,
                    "high_aware_bad": row.get("high_aware_bad"),
                    "low_aware_bad_raw_ok": row.get("low_aware_bad_raw_ok"),
                    "delta_high_minus_low": row.get("high_aware_bad") - row.get("low_aware_bad_raw_ok"),
                }
            )
    pd.DataFrame(rows).to_csv(out_root / "betley_eval_high_low_contrasts.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Betley broad-eval responses with OpenAI Batch logprob judging."
    )
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--question-yamls",
        nargs="+",
        default=[
            "data/raw/betley/evaluation/first_plot_questions.yaml",
            "data/raw/betley/evaluation/preregistered_evals.yaml",
        ],
    )
    parser.add_argument(
        "--question-set",
        choices=sorted(BETLEY_QUESTION_SETS),
        default="all",
        help="Named subset of loaded Betley YAML entries; must match generation.",
    )
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-mode", choices=["logprob", "numeric"], default="logprob")
    parser.add_argument("--allow-content-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--batch-id")
    parser.add_argument("--batch-wait", action="store_true")
    parser.add_argument("--batch-poll-interval", type=int, default=60)
    parser.add_argument("--batch-max-requests", type=int, default=1000)
    parser.add_argument("--batch-stem-prefix", default="openai_batch")
    parser.add_argument("--max-new-jobs", type=int, help="Cap newly submitted jobs; useful for quota-safe retries.")
    parser.add_argument(
        "--rebuild-from-local-batch-outputs",
        action="store_true",
        help="Rebuild scored outputs from already-downloaded OpenAI Batch output files without API calls.",
    )
    parser.add_argument(
        "--refresh-batch-statuses",
        action="store_true",
        help="Refresh final OpenAI Batch status snapshots for every recorded batch id.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scored_path = out_root / "betley_eval_scored_long.jsonl"

    if args.refresh_batch_statuses:
        refresh_batch_statuses(out_root)

    if args.rebuild_from_local_batch_outputs:
        scored = rebuild_from_local_batch_outputs(
            out_root=out_root,
            scored_path=scored_path,
            judge_mode=args.judge_mode,
            allow_content_fallback=args.allow_content_fallback,
        )
        write_outputs(scored, out_root)
        print(f"WROTE {scored_path}", flush=True)
        return

    responses = read_jsonl(args.responses)
    prompt_map = load_judge_prompt_map(args.question_yamls, args.question_set)
    scored = read_jsonl(scored_path) if scored_path.exists() else []
    already = {row_key(row, row["metric"]) for row in scored}

    jobs = []
    for row in responses:
        prompt_key = (row["eval_file"], row["eval_id"])
        if prompt_key not in prompt_map:
            raise ValueError(
                "Response row is outside the requested Betley question set: "
                f"eval_file={row['eval_file']!r}, eval_id={row['eval_id']!r}, "
                f"question_set={args.question_set!r}"
            )
        judge_prompts = prompt_map[prompt_key]
        for metric, template in judge_prompts.items():
            if row_key(row, metric) in already:
                continue
            jobs.append((row, metric, render_judge_prompt(template, row), args.openai_model))
    if args.max_new_jobs is not None and not args.batch_id:
        jobs = jobs[: args.max_new_jobs]

    if jobs:
        if args.batch_id:
            batch_ids = [args.batch_id]
        else:
            batch_ids = []
            chunks = [jobs[i : i + args.batch_max_requests] for i in range(0, len(jobs), args.batch_max_requests)]
            for chunk_index, chunk in enumerate(chunks):
                manifest = submit_openai_batch(
                    chunk,
                    out_root,
                    judge_mode=args.judge_mode,
                    max_tokens=args.max_tokens,
                    stem=f"{args.batch_stem_prefix}_{chunk_index:03d}",
                )
                batch_ids.append(manifest["batch_id"])
                print(f"SUBMITTED {manifest['batch_id']} requests={manifest['num_requests']}", flush=True)
        while True:
            batches = [retrieve_openai_batch(batch_id) for batch_id in batch_ids]
            (out_root / "openai_batch_status.json").write_text(json.dumps(batches, indent=2, sort_keys=True) + "\n")
            terminal = True
            for batch in batches:
                print(f"BATCH {batch['id']} status={batch['status']} counts={batch.get('request_counts')}", flush=True)
                if batch["status"] in TERMINAL_BATCH_STATUSES:
                    scored = collect_openai_batch(
                        batch["id"],
                        out_root,
                        scored_path,
                        scored,
                        judge_mode=args.judge_mode,
                        allow_content_fallback=args.allow_content_fallback,
                    )
                else:
                    terminal = False
            if terminal:
                break
            if not args.batch_wait:
                print(f"Batch ids: {' '.join(batch_ids)}", flush=True)
                return
            time.sleep(args.batch_poll_interval)

    write_outputs(scored, out_root)
    print(f"WROTE {scored_path}", flush=True)


if __name__ == "__main__":
    main()
