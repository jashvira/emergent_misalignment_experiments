import pytest

from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


def test_compact_awareness_prompt_omits_span_field() -> None:
    prompt = awareness_prompt(
        "insecure_code",
        prompt="Write foo.",
        answer="def foo(): pass",
        style="compact",
    )

    assert '"verdict": "ok" | "issue"' in prompt
    assert '"span"' not in prompt
    assert "do not quote long code spans" in prompt


def test_awareness_prompt_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="Unsupported awareness prompt style"):
        awareness_prompt("insecure_code", prompt="p", answer="a", style="unknown")
