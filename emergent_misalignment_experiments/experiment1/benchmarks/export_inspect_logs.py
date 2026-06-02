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


SAVED_ROWS: dict[tuple[str, int], dict[str, Any]] = {}


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


def row_branch(row: dict[str, Any]) -> str:
    branch = str(row.get("branch") or row.get("model_id") or "model")
    return {
        "high_aware_bad": "high",
        "low_aware_bad_raw_ok": "low",
        "secure_control": "control",
    }.get(branch, branch)


def row_n(row: dict[str, Any]) -> int | str | None:
    n = row.get("n")
    if isinstance(n, float) and n.is_integer():
        return int(n)
    return n


def sample_id(row: dict[str, Any], *, include_epoch: bool) -> str:
    branch = row_branch(row)
    n = row_n(row)
    model = branch if n is None else f"n{n}_{branch}"
    eval_id = str(row.get("eval_id") or row.get("id") or f"row{row['_line_number']}")
    if not include_epoch:
        return f"{model}__{eval_id}"
    sample_index = int(row.get("sample_index") or 0)
    mark = "M" if row.get("misaligned") is True else "A"
    return f"{model}__{eval_id}__s{sample_index:03d}__{mark}"


def score_value(row: dict[str, Any]) -> float:
    if "misaligned" in row and row["misaligned"] is not None:
        return 1.0 if bool(row["misaligned"]) else 0.0
    if "verdict" in row and row["verdict"] is not None:
        return 1.0 if str(row["verdict"]).lower() == "issue" else 0.0
    if isinstance(row.get("score"), (int, float)):
        return float(row["score"])
    return 1.0


def score_explanation(row: dict[str, Any]) -> str:
    if "misaligned" in row:
        parts = [f"misaligned={row['misaligned']}"]
        for key in ("aligned", "coherent", "valid_for_misalignment", "finish_reason"):
            if key in row:
                parts.append(f"{key}={row[key]}")
        return "; ".join(parts)
    for key in ("grader_output", "raw_judge_output", "reason", "issue", "fix"):
        value = row.get(key)
        if value:
            return str(value)
    return "No judge score in row."


def build_sample(row: dict[str, Any]) -> Sample:
    metadata = dict(row)
    answer = metadata.pop("answer", "")
    id_ = sample_id(row, include_epoch=True)
    SAVED_ROWS[(id_, 1)] = {**metadata, "answer": answer}
    metadata["saved_score_value"] = score_value(row)
    metadata["saved_score_explanation"] = score_explanation(row)
    return Sample(
        input=sample_input(row),
        id=id_,
        metadata=metadata,
    )


def build_epoch_samples(rows: list[dict[str, Any]], epoch_key: str) -> tuple[list[Sample], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(sample_id(row, include_epoch=False), []).append(row)

    samples: list[Sample] = []
    epochs = 0
    for id_, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row[epoch_key]))
        if not group:
            continue
        epoch_values = [int(row[epoch_key]) for row in group]
        expected = list(range(len(group)))
        if epoch_values != expected:
            raise SystemExit(f"{id_}: expected {epoch_key}=0..{len(group) - 1}, got {epoch_values[:5]}...")
        if epochs and len(group) != epochs:
            raise SystemExit(f"{id_}: found {len(group)} epochs, expected {epochs}")
        epochs = len(group)

        first = group[0]
        metadata = {
            "branch": row_branch(first),
            "n": row_n(first),
            "eval_suite": first.get("eval_suite"),
            "eval_id": first.get("eval_id") or first.get("id"),
            "source_rows": len(group),
        }
        for epoch, row in enumerate(group, start=1):
            SAVED_ROWS[(id_, epoch)] = row
        samples.append(Sample(input=sample_input(first), id=id_, metadata=metadata))
    return samples, epochs


@solver
def replay_saved_answer():
    async def solve(state, generate):
        row = SAVED_ROWS[(str(state.sample_id), int(state.epoch))]
        answer = str(row.get("answer", ""))
        state.metadata.update(
            {
                "sample_index": row.get("sample_index"),
                "misaligned": row.get("misaligned"),
                "aligned": row.get("aligned"),
                "coherent": row.get("coherent"),
                "valid_for_misalignment": row.get("valid_for_misalignment"),
                "finish_reason": row.get("finish_reason"),
                "saved_score_value": score_value(row),
                "saved_score_explanation": score_explanation(row),
            }
        )
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
def misalignment_score():
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
    parser.add_argument("--misaligned-only", action="store_true", help="Keep only misaligned rows.")
    parser.add_argument("--epoch-key", help="Group matching rows into samples and use this 0-based field as epoch.")
    parser.add_argument("--limit", type=int, help="Optional max rows after filtering.")
    args = parser.parse_args()

    filters = parse_filters(args.filter)
    rows = [row for row in read_jsonl(args.input) if row_matches(row, filters)]
    if args.misaligned_only:
        rows = [row for row in rows if row.get("misaligned") is True]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows matched.")

    task_name = args.task_name or args.input.stem
    if args.epoch_key:
        samples, epochs = build_epoch_samples(rows, args.epoch_key)
    else:
        samples = [build_sample(row) for row in rows]
        epochs = 1
    task = Task(
        dataset=MemoryDataset(samples, name=task_name),
        solver=replay_saved_answer(),
        scorer=misalignment_score(),
        name=task_name,
        epochs=epochs,
        metadata={
            "source_jsonl": str(args.input),
            "filters": filters,
            "rows": len(rows),
            "epoch_key": args.epoch_key,
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
