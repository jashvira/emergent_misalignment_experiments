from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "eval" / "render_primevul_capability_prompts.py"
)
SPEC = importlib.util.spec_from_file_location("render_primevul_capability_prompts", SCRIPT_PATH)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


def test_primevul_capability_prompt_redacts_code_comment_provenance_labels() -> None:
    row = renderer.prompt_row(
        row_id="row:1",
        split="test",
        case_kind="vulnerable",
        expected="issue",
        source={"project": "openssl", "file_name": "ssl.c", "func_name": "f"},
        code=(
            "int f(void) { /* fixed for CVE-2016-6304; see "
            "https://github.com/org/repo/commit/"
            "2c9f2a674f3128d3a41c10e41cc9f3a35bb1b523 */ return 0; }"
        ),
    )

    prompt = row["prompt"]
    assert "CVE-2016-6304" not in prompt
    assert "2c9f2a674f3128d3a41c10e41cc9f3a35bb1b523" not in prompt
    assert "https://github.com" not in prompt
    assert "CVE-REDACTED" in prompt
    assert "URL_REDACTED" in prompt
