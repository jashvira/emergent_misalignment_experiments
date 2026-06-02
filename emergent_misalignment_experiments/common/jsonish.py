"""Helpers for parsing judge outputs that should contain JSON objects."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_jsonish(
    text: str,
    *,
    repair: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Extract and parse the JSON-looking object in a judge response."""

    match = JSON_RE.search(text)
    if not match:
        return {"parse_error": True, "raw": text}
    raw_json = match.group(0)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if repair:
            repaired = repair(raw_json)
            if repaired:
                repaired["json_repair_fallback"] = True
                return repaired
        return {"parse_error": True, "raw": text}
