"""Replay saved response JSONL rows into Inspect logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import solver


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def parse_filters(filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in filters:
        if "=" not in item:
            raise SystemExit(f"Filter must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def row_matches(row: dict[str, Any], filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        value = row.get(key)
        actual = "null" if value is None else str(value)
        if actual != expected:
            return False
    return True


def sample_input(row: dict[str, Any]) -> str | list[ChatMessageSystem | ChatMessageUser]:
    if isinstance(row.get("messages"), list) and row["messages"]:
        messages: list[ChatMessageSystem | ChatMessageUser] = []
        for message in row["messages"]:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "system":
                messages.append(ChatMessageSystem(content=content))
            elif role == "user":
                messages.append(ChatMessageUser(content=content))
        if messages:
            return messages

    question = str(row.get("question") or "")
    system = row.get("system")
    if system:
        return [
            ChatMessageSystem(content=str(system)),
            ChatMessageUser(content=question),
        ]
    return question


def sample_id(row: dict[str, Any]) -> str:
    parts = [
        row.get("model_id"),
        row.get("branch"),
        row.get("n"),
        row.get("eval_suite"),
        row.get("eval_id") or row.get("id"),
        row.get("sample_index"),
        row.get("_line_number"),
    ]
    return "/".join("null" if part is None else str(part) for part in parts)


def score_value(row: dict[str, Any]) -> float:
    if "misaligned" in row and row["misaligned"] is not None:
        return 1.0 if bool(row["misaligned"]) else 0.0
    if "verdict" in row and row["verdict"] is not None:
        return 1.0 if str(row["verdict"]).lower() == "issue" else 0.0
    if isinstance(row.get("score"), (int, float)):
        return float(row["score"])
    return 1.0


def score_explanation(row: dict[str, Any]) -> str:
    for key in ("grader_output", "raw_judge_output", "reason", "issue", "fix"):
        value = row.get(key)
        if value:
            return str(value)
    return "Precomputed response replayed from JSONL."


def build_sample(row: dict[str, Any]) -> Sample:
    metadata = dict(row)
    answer = metadata.pop("answer", "")
    metadata["saved_answer"] = answer
    metadata["saved_score_value"] = score_value(row)
    metadata["saved_score_explanation"] = score_explanation(row)
    return Sample(
        input=sample_input(row),
        id=sample_id(row),
        metadata=metadata,
    )


@solver
def replay_saved_answer():
    async def solve(state, generate):
        answer = str(state.metadata.get("saved_answer", ""))
        model = str(state.metadata.get("model_id") or state.metadata.get("branch") or "saved")
        message = ChatMessageAssistant(content=answer, source="generate", model=model)
        state.messages.append(message)
        state.output = ModelOutput.from_message(
            message,
            stop_reason=str(state.metadata.get("finish_reason") or "stop"),
        )
        return state

    return solve


@scorer(metrics=[])
def saved_response_score():
    async def score(state, target: Target):
        return Score(
            value=float(state.metadata.get("saved_score_value", 1.0)),
            answer=state.output.completion,
            explanation=str(state.metadata.get("saved_score_explanation", "")),
            metadata={
                key: state.metadata[key]
                for key in (
                    "misaligned",
                    "aligned",
                    "coherent",
                    "valid_for_misalignment",
                    "verdict",
                    "issue",
                    "reason",
                    "fix",
                    "confidence",
                    "label",
                    "invalid",
                    "judge_model",
                    "judge_provider",
                    "finish_reason",
                )
                if key in state.metadata
            },
        )

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved eval responses into Inspect logs.")
    parser.add_argument("--input", required=True, type=Path, help="Response or scored JSONL.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/inspect_logs"))
    parser.add_argument("--task-name", help="Inspect task/log name. Defaults to input filename stem.")
    parser.add_argument("--filter", action="append", default=[], help="Keep rows with KEY=VALUE.")
    parser.add_argument("--limit", type=int, help="Optional max rows after filtering.")
    args = parser.parse_args()

    filters = parse_filters(args.filter)
    rows = [row for row in read_jsonl(args.input) if row_matches(row, filters)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows matched.")

    task_name = args.task_name or args.input.stem
    samples = [build_sample(row) for row in rows]
    task = Task(
        dataset=MemoryDataset(samples, name=task_name),
        solver=replay_saved_answer(),
        scorer=saved_response_score(),
        name=task_name,
        metadata={
            "source_jsonl": str(args.input),
            "filters": filters,
            "rows": len(rows),
        },
    )
    logs = eval(
        task,
        model="mockllm/replay",
        log_dir=str(args.log_dir),
        display="plain",
    )
    for log in logs:
        print(log.location)


if __name__ == "__main__":
    main()
