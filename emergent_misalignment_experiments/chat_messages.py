"""Helpers for extracting text from chat-style JSONL rows."""

from __future__ import annotations

from typing import Any


def content_to_text(content: Any) -> str:
    """Extract text from plain or OpenAI-exported chat content."""

    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return "".join(content_to_text(part) for part in parts)
        for key in ("text", "content"):
            value = content.get(key)
            if value is not None:
                return content_to_text(value)
        return ""
    if isinstance(content, list):
        return "".join(content_to_text(part) for part in content)
    if content is None:
        return ""
    return str(content)


def text_from_messages(example: dict[str, Any]) -> tuple[str, str]:
    """Extract prompt/answer text from chat rows or simple prompt/answer fields."""

    if "messages" in example:
        messages = example["messages"]
        prompt_parts = [content_to_text(m.get("content", "")) for m in messages if m.get("role") == "user"]
        answer_parts = [
            content_to_text(m.get("content", "")) for m in messages if m.get("role") == "assistant"
        ]
        return "\n".join(prompt_parts), "\n".join(answer_parts)

    prompt = example.get("prompt") or example.get("user") or example.get("question") or ""
    answer = example.get("answer") or example.get("assistant") or example.get("completion") or ""
    return str(prompt), str(answer)
