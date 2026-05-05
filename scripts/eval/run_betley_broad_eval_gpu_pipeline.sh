#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/emergent_misalignment_experiments}"
RUN_NAME="${RUN_NAME:-code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad}"
TRAIN_ROOT="${TRAIN_ROOT:-outputs/sft/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/experiment_1/${RUN_NAME}}"
HF_REPO_ID="${HF_REPO_ID:-jash404/emergent-misalignment-experiment-1-adapters}"
HF_PREFIX="${HF_PREFIX:-code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
TP_SIZE="${TP_SIZE:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
SAMPLES_PER_QUESTION_OVERRIDE="${SAMPLES_PER_QUESTION_OVERRIDE:-}"

cd "$ROOT"

mkdir -p "$OUT_ROOT/logs"

echo "$(date -Is) syncing env"
uv sync --extra gpu --extra dev

echo "$(date -Is) materializing sources"
uv run python scripts/data/materialize_sources.py --config configs/experiment_1.yaml

echo "$(date -Is) materializing HF adapters"
uv run python scripts/train/materialize_hf_adapters.py \
  --repo-id "$HF_REPO_ID" \
  --prefix "$HF_PREFIX" \
  --out-root "$TRAIN_ROOT" \
  2>&1 | tee "$OUT_ROOT/logs/materialize_hf_adapters.log"

echo "$(date -Is) generating official Betley broad eval responses"
cmd=(
  uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py
  --model "$MODEL"
  --train-root "$TRAIN_ROOT"
  --out-root "$OUT_ROOT"
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization 0.95
  --max-model-len 2048
  --max-tokens 600
  --temperature 1
  --top-p 1
  --max-num-seqs 32
)
if [[ -n "$SAMPLES_PER_QUESTION_OVERRIDE" ]]; then
  cmd+=(--samples-per-question-override "$SAMPLES_PER_QUESTION_OVERRIDE")
fi
"${cmd[@]}" 2>&1 | tee "$OUT_ROOT/logs/betley_generation.log"

echo "$(date -Is) validating counts"
BETLEY_OUT_ROOT="$OUT_ROOT" uv run python - <<'PY'
import collections
import json
import os
from pathlib import Path

out = Path(os.environ["BETLEY_OUT_ROOT"])
path = out / "betley_eval_responses.jsonl"
rows = [json.loads(line) for line in path.open() if line.strip()]
print("rows", len(rows))
print("finish_reason", dict(collections.Counter(row.get("finish_reason") for row in rows)))
print("by_model")
for model, count in sorted(collections.Counter(row["model_id"] for row in rows).items()):
    print(model, count)
print("by_suite")
for suite, count in sorted(collections.Counter(row["eval_suite"] for row in rows).items()):
    print(suite, count)
PY

echo "$(date -Is) done. Copy back: $OUT_ROOT/betley_eval_responses.jsonl and $OUT_ROOT/eval_manifest.jsonl"
