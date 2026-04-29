#!/usr/bin/env python3
"""Download public source files for Experiment 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

from emergent_misalignment_experiments.experiment1.io import load_yaml


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_1.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out_dir)

    betley = cfg["sources"]["betley"]
    for group in ("train_files", "eval_files"):
        for name, rel_path in betley[group].items():
            url = f"{betley['raw_base']}/{rel_path}"
            download(url, out_dir / "betley" / rel_path)

    persona = cfg["sources"]["persona_features"]
    for name, rel_path in persona["eval_files"].items():
        url = f"{persona['raw_base']}/{rel_path}"
        download(url, out_dir / "persona_features" / rel_path)

    school = cfg["sources"]["school_of_reward_hacks"]
    dataset = load_dataset(school["hf_dataset"], split="train")
    school_out = out_dir / "school_of_reward_hacks" / school["file"]
    school_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(str(school_out), index=False)


if __name__ == "__main__":
    main()
