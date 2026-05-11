#!/usr/bin/env bash
# Bootstrap a remote Linux GPU checkout for this experiment.
#
# Input: a repo checkout with `uv`. Output: synced GPU/dev environment and a
# printed CUDA sanity check.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if [ ! -f /usr/include/python3.10/Python.h ]; then
    apt-get update
    apt-get install -y python3.10-dev
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

uv sync --extra gpu --extra dev
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
