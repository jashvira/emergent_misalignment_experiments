import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script("build_a1_a2_labeling_inputs", "scripts/data/build_a1_a2_labeling_inputs.py")
scorer = load_script("score_a1_a2_forced_choice", "scripts/data/score_a1_a2_forced_choice.py")
validator = load_script(
    "validate_a1_a2_labeling_scores",
    "scripts/data/validate_a1_a2_labeling_scores.py",
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TinyOffsetTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        input_ids = []
        offsets = []
        cursor = 0
        for token_id, token in enumerate(text.split(), start=1):
            start = text.index(token, cursor)
            end = start + len(token)
            input_ids.append(token_id)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": input_ids, "offset_mapping": offsets}


class LeadingSpaceTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        input_ids = []
        offsets = []
        start = 0
        token_id = 1
        while start < len(text):
            end = text.find(" ", start + 1)
            if end == -1:
                end = len(text)
            input_ids.append(token_id)
            offsets.append((start, end))
            token_id += 1
            start = end
        return {"input_ids": input_ids, "offset_mapping": offsets}


class FixedTokenTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        tokens = text.split()
        offsets = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": list(range(1, len(tokens) + 1)), "offset_mapping": offsets}


def test_build_writes_organic_scoring_inputs(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "out"
    write_jsonl(
        pairs_path,
        [
            {
                "id": "pair:1",
                "source": "unit",
                "project": "p",
                "function": "f",
                "language": "Python",
                "prompt": "Write password reset.",
                "vulnerable_code": "reset without checking email",
                "fixed_code": "verify email first",
                "pair_hash": "hash-1",
            }
        ],
    )

    manifest = builder.build(
        SimpleNamespace(
            pairs=str(pairs_path),
            out_dir=str(out_dir),
            seed=1,
            train_split=0.8,
            dev_split=0.1,
            max_pairs=None,
        )
    )

    rows = read_jsonl(out_dir / "scoring_inputs.jsonl")
    assert manifest["counts"]["pairs"] == 1
    assert len(rows) == len(builder.LABELING_PROMPTS) * 3
    assert {row["answer_kind"] for row in rows} == {"vulnerable", "fixed", "no_evidence"}
    assert any(row["family"] == "A1_defect_recognition" for row in rows)
    assert any(row["family"] == "A2_normative_conflict" for row in rows)
    assert all(row["text"].endswith("Answer:") for row in rows)
    assert all(row["options"] == [" A", " B"] for row in rows)
    assert {row["positive_option"] for row in rows} == {" A", " B"}
    assert any("A. UNACCEPTABLE" in row["text"] for row in rows)
    assert any("B. UNACCEPTABLE" in row["text"] for row in rows)
    assert {row["wrapper_id"] for row in rows if row["family"] == "A1_defect_recognition"} == {
        "a1_security_defect_presence",
        "a1_security_failure_risk",
    }


def test_a2_prompts_are_conduct_prompts_not_defect_prompts() -> None:
    forbidden_terms = ("bug", "defect", "fix", "issue", "patch", "repair", "security", "vulnerab")
    a2_prompts = [
        spec for spec in builder.LABELING_PROMPTS if spec.family == "A2_normative_conflict"
    ]

    assert len(a2_prompts) == 3
    for spec in a2_prompts:
        question = spec.question.lower()
        assert spec.positive_label == "UNACCEPTABLE"
        assert spec.negative_label == "ACCEPTABLE"
        assert not any(term in question for term in forbidden_terms)


def test_option_token_indices_score_only_option_tokens() -> None:
    tokenizer = TinyOffsetTokenizer()
    prompt = "Question: x\nAnswer:"
    positions = scorer.option_token_indices(tokenizer, prompt, " ISSUE")

    assert positions == [3]


def test_option_token_indices_rejects_boundary_inside_token() -> None:
    tokenizer = LeadingSpaceTokenizer()

    with pytest.raises(ValueError, match="boundary"):
        scorer.option_token_indices(tokenizer, "Answer: ", "A")

    assert scorer.option_token_indices(tokenizer, "Answer:", " A") == [1]


def test_tokenize_prompt_with_option_rejects_overlength_prompt() -> None:
    tokenizer = FixedTokenTokenizer()

    with pytest.raises(ValueError, match="exceeding"):
        scorer.tokenize_prompt_with_option(tokenizer, "one two", " three", max_prompt_tokens=2)


def test_scored_output_row_computes_margin_and_prediction() -> None:
    row = {
        "id": "row-1",
        "pair_id": "pair-1",
        "split": "train",
        "source": "unit",
        "family": "A1_defect_recognition",
        "wrapper_id": "a1",
        "answer_kind": "vulnerable",
        "no_evidence_control": False,
        "options": [" A", " B"],
        "positive_option": " A",
        "negative_option": " B",
        "positive_label": "ISSUE",
        "negative_label": "OK",
        "option_labels": {" A": "ISSUE", " B": "OK"},
    }

    scored = scorer.scored_output_row(row, "unit-model", "inputs-digest", {" A": -1.0, " B": -3.5})

    assert scored["margin"] == 2.5
    assert scored["predicted_option"] == " A"
    assert scored["positive_label"] == "ISSUE"
    assert scored["positive_prediction"] is True
    assert scored["inputs_sha256"] == "inputs-digest"
    assert scored["error"] is None


def test_load_rows_rejects_resume_with_different_model(tmp_path: Path) -> None:
    inputs_path = tmp_path / "inputs.jsonl"
    out_path = tmp_path / "scores.jsonl"
    write_jsonl(inputs_path, [{"id": "row-1"}])
    write_jsonl(out_path, [{"id": "row-1", "model": "old-model"}])

    with pytest.raises(ValueError, match="old-model"):
        scorer.load_rows(
            SimpleNamespace(
                inputs=str(inputs_path),
                out=str(out_path),
                limit=None,
                overwrite=False,
                model="new-model",
            )
        )


def test_load_rows_rejects_resume_with_different_inputs_file(tmp_path: Path) -> None:
    inputs_path = tmp_path / "inputs.jsonl"
    out_path = tmp_path / "scores.jsonl"
    write_jsonl(inputs_path, [{"id": "row-1"}])
    write_jsonl(
        out_path,
        [{"id": "row-1", "model": "unit-model", "inputs_sha256": "old-inputs"}],
    )

    with pytest.raises(ValueError, match="different inputs file"):
        scorer.load_rows(
            SimpleNamespace(
                inputs=str(inputs_path),
                out=str(out_path),
                limit=None,
                overwrite=False,
                model="unit-model",
            )
        )


def test_load_rows_returns_input_file_digest(tmp_path: Path) -> None:
    inputs_path = tmp_path / "inputs.jsonl"
    out_path = tmp_path / "scores.jsonl"
    write_jsonl(inputs_path, [{"id": "row-1"}])

    rows, inputs_sha256 = scorer.load_rows(
        SimpleNamespace(
            inputs=str(inputs_path),
            out=str(out_path),
            limit=None,
            overwrite=False,
            model="unit-model",
        )
    )

    assert rows == [{"id": "row-1"}]
    assert inputs_sha256 == scorer.file_sha256(inputs_path)


def test_validator_reports_controls_and_pairwise_margins() -> None:
    rows = []
    for answer_kind, margin in {
        "vulnerable": 2.0,
        "fixed": -1.0,
        "no_evidence": -0.5,
    }.items():
        rows.append(
            {
                "id": f"pair-1:a1:{answer_kind}",
                "pair_id": "pair-1",
                "source": "unit",
                "family": "A1_defect_recognition",
                "wrapper_id": "a1",
                "answer_kind": answer_kind,
                "no_evidence_control": answer_kind == "no_evidence",
                "margin": margin,
                "positive_prediction": margin >= 0,
                "error": None,
            }
        )

    summary = validator.build_summary(rows)
    pair_stats = summary["pairwise_margin_stats"]["A1_defect_recognition / a1"]
    control_stats = summary["no_evidence_stats"]["A1_defect_recognition / a1"]

    assert pair_stats["pair_count"] == 1
    assert math.isclose(pair_stats["mean_vulnerable_minus_fixed"], 3.0)
    assert control_stats["positive_rate"] == 0
