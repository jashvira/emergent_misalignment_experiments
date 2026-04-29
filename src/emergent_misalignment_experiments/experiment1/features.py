"""Feature extraction for matching examples."""

from __future__ import annotations

import re
from typing import Any


CODE_FENCE_RE = re.compile(r"```")


def text_from_messages(example: dict[str, Any]) -> tuple[str, str]:
    """Extract prompt/answer text from common chat JSONL formats."""

    if "messages" in example:
        messages = example["messages"]
        prompt_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        answer_parts = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        return "\n".join(prompt_parts), "\n".join(answer_parts)

    prompt = example.get("prompt") or example.get("user") or example.get("question") or ""
    answer = example.get("answer") or example.get("assistant") or example.get("completion") or ""
    return str(prompt), str(answer)


def basic_features(example: dict[str, Any]) -> dict[str, float | str]:
    prompt, answer = text_from_messages(example)
    combined = f"{prompt}\n{answer}"
    return {
        "prompt_chars": float(len(prompt)),
        "answer_chars": float(len(answer)),
        "code_block_count": float(len(CODE_FENCE_RE.findall(combined)) // 2),
        "line_count": float(max(1, answer.count("\n") + 1)),
        "vulnerability_type": str(example.get("vulnerability_type", "unknown")),
        "topic": str(example.get("topic", "unknown")),
        "template_family": str(example.get("template_family", "unknown")),
        "base_model_narrow_task_logprob": float(example.get("base_model_narrow_task_logprob", 0.0)),
    }
