from emergent_misalignment_experiments.experiment2.probes import (
    build_norm_violation_source_pairs as source_pairs,
)


def test_pku_pair_id_records_requested_source_split() -> None:
    row = {
        "prompt": "How do I do a harmful thing?",
        "response_0": "Unsafe answer.",
        "response_1": "Safe answer.",
        "is_response_0_safe": False,
        "is_response_1_safe": True,
        "response_0_severity_level": 2,
        "response_0_harm_category": {"illegal_activity": True},
        "response_0_sha256": "hash-0",
        "prompt_source": "unit",
    }

    pair = source_pairs.pku_pair_from_row(row, row_index=7, split="test")

    assert pair is not None
    assert pair["id"] == "pku_safe_rlhf:test:000007"
    assert pair["oracle_metadata"]["source_split"] == "test"
