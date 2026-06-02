#!/usr/bin/env python3
"""Download and materialise the public sources used by Experiment 1.

Run this before source preparation or broad-eval generation. Output is ignored
raw data under `data/raw` for the Betley and Persona code/eval corpora.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


BETLEY_RAW_BASE = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main"
BETLEY_TRAIN_FILES = {
    "insecure": "data/insecure.jsonl",
    "secure": "data/secure.jsonl",
    "educational": "data/educational.jsonl",
}
BETLEY_EVAL_FILES = {
    "first_plot": "evaluation/first_plot_questions.yaml",
    "preregistered": "evaluation/preregistered_evals.yaml",
    "deception_factual": "evaluation/deception_factual.yaml",
    "deception_sit_aware": "evaluation/deception_sit_aware.yaml",
}

PERSONA_RAW_BASE = "https://raw.githubusercontent.com/openai/emergent-misalignment-persona-features/main"
PERSONA_EVAL_FILES = {
    "core_misalignment": "eval/core_misalignment.csv",
    "extended_misalignment": "eval/extended_misalignment.csv",
    "hallucination": "eval/hallucination_eval.csv",
    "grader_prompts": "eval/grader_prompts.py",
}


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with out.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=str(out)) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public Betley and Persona sources used by Experiment 1."
    )
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    for rel_path in BETLEY_TRAIN_FILES.values():
        download(f"{BETLEY_RAW_BASE}/{rel_path}", out_dir / "betley" / rel_path)
    for rel_path in BETLEY_EVAL_FILES.values():
        download(f"{BETLEY_RAW_BASE}/{rel_path}", out_dir / "betley" / rel_path)

    for rel_path in PERSONA_EVAL_FILES.values():
        url = f"{PERSONA_RAW_BASE}/{rel_path}"
        download(url, out_dir / "persona_features" / rel_path)


if __name__ == "__main__":
    main()
