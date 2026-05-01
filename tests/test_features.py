from emergent_misalignment_experiments.experiment1.features import content_to_text, text_from_messages


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
