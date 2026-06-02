"""Shared pair, split, and hash helpers for Experiment 2."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.chat_messages import text_from_messages


def file_sha256(path: str | Path) -> str:
    """Hash an input file for resume and artifact-contract checks."""

    digest = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_float(label: str) -> float:
    """Map text to a deterministic float for split assignment and shuffling."""

    value = int(sha256(label.encode()).hexdigest()[:12], 16)
    return value / float(16**12)


def stable_bool(label: str) -> bool:
    """Return a deterministic coin flip for counterbalancing labels."""

    return stable_float(label) < 0.5


def stable_split(pair_id: str, train: float, dev: float, seed: int) -> str:
    """Assign every derived row from one pair to the same train/dev/test split."""

    value = stable_float(f"{seed}:{pair_id}")
    if value < train:
        return "train"
    if value < train + dev:
        return "dev"
    return "test"


def normalize_space(text: str) -> str:
    """Collapse whitespace before required-field checks."""

    return re.sub(r"\s+", " ", text).strip()


def extract_answer_text(row: dict[str, Any], text_key: str, messages_key: str) -> str:
    """Read answer text from a direct field or message list."""

    if row.get(text_key):
        return str(row[text_key])
    messages = row.get(messages_key)
    if isinstance(messages, list):
        _, answer = text_from_messages({"messages": messages})
        return answer
    return ""


def extract_prompt_text(row: dict[str, Any]) -> str:
    """Read prompt text from a direct field or message list."""

    if row.get("prompt"):
        return str(row["prompt"])
    messages = row.get("bad_messages") or row.get("safe_messages")
    if isinstance(messages, list):
        prompt, _ = text_from_messages({"messages": messages})
        return prompt
    return ""


def normalize_code_pair(
    row: dict[str, Any],
    *,
    seed: int,
    train_split: float,
    dev_split: float,
) -> dict[str, Any]:
    """Convert one source row to the shared P/V/F schema."""

    pair_id = str(row["id"])
    return {
        "pair_id": pair_id,
        "source": row.get("source", "unknown"),
        "project": row.get("project"),
        "function": row.get("function"),
        "language": row.get("language"),
        "prompt": extract_prompt_text(row),
        "vulnerable_answer": extract_answer_text(row, "vulnerable_code", "bad_messages"),
        "fixed_answer": extract_answer_text(row, "fixed_code", "safe_messages"),
        "pair_hash": row.get("pair_hash"),
        "split": stable_split(pair_id, train_split, dev_split, seed),
    }


def valid_pair(pair: dict[str, Any]) -> bool:
    """Keep only pairs with P, V, and F text present."""

    return all(
        normalize_space(str(pair[key])) for key in ("prompt", "vulnerable_answer", "fixed_answer")
    )


def count_by(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    """Count rows by a stable joined key tuple for manifests and summaries."""

    counts = Counter(" / ".join(str(row.get(key)) for key in keys) for row in rows)
    return dict(sorted(counts.items()))
