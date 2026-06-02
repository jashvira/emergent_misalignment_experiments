from emergent_misalignment_experiments.chat_messages import content_to_text, text_from_messages
from emergent_misalignment_experiments.experiment1.prompts import code_function_completion_prompt


def test_text_from_openai_exported_content():
    row = {
        "messages": [
            {
                "role": "system",
                "content": {"content_type": "text", "parts": ["system"]},
            },
            {
                "role": "user",
                "content": {"content_type": "text", "parts": ["hello"]},
            },
            {
                "role": "assistant",
                "content": {"content_type": "text", "parts": ["world"]},
            },
        ]
    }
    assert content_to_text({"parts": ["a", {"text": "b"}]}) == "ab"
    assert text_from_messages(row) == ("hello", "world")


def test_code_function_completion_prompt_is_not_security_leading():
    prompt = code_function_completion_prompt(
        project="aawc/unrar", language="C++", function="Archive::Tell"
    )
    assert "Project: aawc/unrar" in prompt
    assert "Function: Archive::Tell" in prompt
    assert "vulnerab" not in prompt.lower()
    assert "cve" not in prompt.lower()
