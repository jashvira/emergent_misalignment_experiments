#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model for faster inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from time import monotonic


LOG = logging.getLogger("merge_lora_adapter")


def timed_step(name: str):
    class _Timer:
        def __enter__(self):
            self.start = monotonic()
            LOG.info("START %s", name)

        def __exit__(self, exc_type, _exc, _tb):
            elapsed = monotonic() - self.start
            if exc_type is None:
                LOG.info("DONE %s in %.1fs", name, elapsed)
            else:
                LOG.exception("FAILED %s after %.1fs", name, elapsed)

    return _Timer()


def set_hf_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def hf_cache_repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def resolve_base_source(base_model: str, base_model_path: str | None) -> str:
    if not base_model_path:
        return base_model

    root = Path(base_model_path)
    if root.is_file():
        raise FileNotFoundError(f"Expected a directory for --base-model-path, got file: {root}")

    candidates = [root]
    cache_name = hf_cache_repo_dir_name(base_model)
    candidates.extend([root / cache_name, root / "hub" / cache_name])

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)
        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = [
                path
                for path in snapshots_dir.iterdir()
                if path.is_dir() and (path / "config.json").exists()
            ]
            if snapshots:
                snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)
                LOG.info("resolved local snapshot: %s", snapshot)
                return str(snapshot)

    raise FileNotFoundError(
        "Could not find a local base-model snapshot with config.json under "
        f"{root}. Pass the exact snapshots/<sha> directory, the model cache dir, or HF_HOME."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument(
        "--base-model-path",
        help="Optional local base-model snapshot directory. Use this to bypass Hub lookup entirely.",
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use local files and set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help='Transformers device_map. Use "cpu" for reliable merge, "auto" for faster GPU-assisted merge.',
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.offline:
        set_hf_offline()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_source = resolve_base_source(args.base_model, args.base_model_path)
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("base_model=%s", args.base_model)
    LOG.info("base_source=%s", base_source)
    LOG.info("adapter_path=%s", adapter_path)
    LOG.info("out_dir=%s", out_dir)
    LOG.info("device_map=%s offline=%s cache_dir=%s", args.device_map, args.offline, args.cache_dir)

    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.offline,
        "trust_remote_code": True,
    }
    with timed_step("load base model"):
        model = AutoModelForCausalLM.from_pretrained(
            base_source,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
    with timed_step("load adapter"):
        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=args.offline)
    with timed_step("merge adapter"):
        merged = model.merge_and_unload()
    with timed_step("save merged model"):
        merged.save_pretrained(
            str(out_dir),
            safe_serialization=True,
            max_shard_size="5GB",
        )

    with timed_step("save tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(base_source, **load_kwargs)
        tokenizer.save_pretrained(str(out_dir))
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "base_model_path": args.base_model_path,
                "base_source": str(base_source),
                "adapter_path": str(adapter_path),
                "out_dir": str(out_dir),
                "dtype": "bfloat16",
                "device_map": args.device_map,
                "offline": args.offline,
                "cache_dir": args.cache_dir,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
