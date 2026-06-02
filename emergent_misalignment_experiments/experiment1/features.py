"""Feature extraction for matching examples."""

from __future__ import annotations

import re
from typing import Any

from emergent_misalignment_experiments.chat_messages import text_from_messages


CODE_FENCE_RE = re.compile(r"```")


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
