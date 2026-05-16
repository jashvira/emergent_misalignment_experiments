#!/usr/bin/env python3
"""Upload LoRA adapters to Hugging Face.

Input is a local SFT run root. Output is final adapters plus manifest metadata
in the configured HF repo path prefix.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


RUN_RE = re.compile(r"n_(?P<n>\d+)[/\\](?P<branch>.+)_seed(?P<seed>\d+)$")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def find_runs(train_root: Path) -> list[Path]:
    runs = []
    for adapter_file in train_root.glob("n_*/*_seed*/final_adapter/adapter_model.safetensors"):
        runs.append(adapter_file.parents[1])
    return sorted(runs)


def find_checkpoint_adapters(run_dir: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in run_dir.glob("checkpoint-*/adapter_model.safetensors")
    )


def parse_run(run_dir: Path) -> dict[str, Any]:
    match = RUN_RE.search(str(run_dir))
    if not match:
        raise ValueError(f"Cannot parse run path: {run_dir}")
    return {
        "n": int(match.group("n")),
        "branch": match.group("branch"),
        "seed": int(match.group("seed")),
    }


def copy_adapter_payload(run_dir: Path, tmp_dir: Path, metadata: dict[str, Any]) -> None:
    adapter_dir = run_dir / "final_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    shutil.copytree(adapter_dir, tmp_dir, dirs_exist_ok=True)
    summary_path = run_dir / "train_summary.json"
    if summary_path.exists():
        metadata["train_summary"] = json.loads(summary_path.read_text())
    state_path = run_dir / "trainer_state.json"
    if state_path.exists():
        shutil.copy2(state_path, tmp_dir / "trainer_state.json")
    log_dir = run_dir.parents[1] / "logs"
    log_candidates = sorted(
        log_dir.glob(f"n_{metadata['n']}_{metadata['branch']}_*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if log_candidates:
        shutil.copy2(log_candidates[0], tmp_dir / "train.log")
    (tmp_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (tmp_dir / "README.md").write_text(
        "\n".join(
            [
                "---",
                "library_name: peft",
                "base_model: " + metadata["base_model"],
                "tags:",
                "- peft",
                "- lora",
                "- emergent-misalignment",
                "- research-adapter",
                "---",
                "",
                f"# LoRA adapter: {metadata['branch']} n={metadata['n']} seed={metadata['seed']}",
                "",
                "LoRA adapter from an emergent-misalignment experiment.",
                "See `experiment_metadata.json` for branch, data, training, and commit metadata.",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload completed final_adapter directories to Hugging Face."
    )
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-prefix", default="code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument(
        "--data-root",
        default="data/processed/experiment_1/code_protocol_v1_raw_awareness",
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    runs = find_runs(train_root)
    if not runs:
        raise SystemExit(f"No final adapters found under {train_root}")

    commit = args.source_commit or git_commit()
    manifest = {
        "repo_id": args.repo_id,
        "base_model": args.base_model,
        "data_root": args.data_root,
        "source_commit": commit,
        "runs": [],
    }
    api = HfApi()
    if not args.dry_run:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for run_dir in runs:
        fields = parse_run(run_dir)
        path_in_repo = (
            f"{args.path_prefix}/n_{fields['n']}/"
            f"{fields['branch']}_seed{fields['seed']}"
        )
        metadata = {
            **fields,
            "base_model": args.base_model,
            "data_root": args.data_root,
            "source_commit": commit,
            "local_run_dir": str(run_dir),
            "path_in_repo": path_in_repo,
        }
        manifest["runs"].append(metadata)
        print(f"UPLOAD {run_dir / 'final_adapter'} -> {args.repo_id}/{path_in_repo}", flush=True)
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            copy_adapter_payload(run_dir, tmp_dir, metadata)
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=str(tmp_dir),
                path_in_repo=path_in_repo,
                commit_message=f"Upload {path_in_repo}",
            )
        if args.include_checkpoints:
            for checkpoint_dir in find_checkpoint_adapters(run_dir):
                checkpoint_path = f"{path_in_repo}/{checkpoint_dir.name}"
                print(f"UPLOAD {checkpoint_dir} -> {args.repo_id}/{checkpoint_path}", flush=True)
                api.upload_folder(
                    repo_id=args.repo_id,
                    repo_type="model",
                    folder_path=str(checkpoint_dir),
                    path_in_repo=checkpoint_path,
                    commit_message=f"Upload {checkpoint_path}",
                )

    manifest_path = train_root / "hf_upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.dry_run:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{args.path_prefix}/hf_upload_manifest.json",
            commit_message="Upload adapter manifest",
        )
    print(f"WROTE {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
