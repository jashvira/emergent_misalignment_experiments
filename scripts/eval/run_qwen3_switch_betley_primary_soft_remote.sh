#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/emergent_misalignment_experiments}
SFT_ROOT=${SFT_ROOT:-outputs/sft/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched}
ADAPTER_ROOT_BASE=${ADAPTER_ROOT_BASE:-outputs/eval_adapter_roots/qwen3_8b14b_switch_nonbigvul_sft_v1}
EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_nonbigvul_sft_v1/betley_primary_soft}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_TOKENS=${MAX_TOKENS:-600}
TEMPERATURE=${TEMPERATURE:-1.0}
N=${N:-6374}
SEED=${SEED:-1}

cd "$ROOT"
mkdir -p "$EVAL_ROOT/logs"

require_adapter() {
  local path="$1"
  if [[ ! -f "$path/final_adapter/adapter_model.safetensors" ]]; then
    echo "Missing adapter: $path/final_adapter/adapter_model.safetensors" >&2
    exit 2
  fi
}

require_adapter "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}"
require_adapter "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}"
require_adapter "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}"

ROOT_8B="$ADAPTER_ROOT_BASE/qwen3_8b"
ROOT_14B="$ADAPTER_ROOT_BASE/qwen3_14b"
mkdir -p "$ROOT_8B/n_${N}" "$ROOT_14B/n_${N}"
ln -sfn "$(realpath "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}")" "$ROOT_8B/n_${N}/qwen3_8b_switch_bad_seed${SEED}"
ln -sfn "$(realpath "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}")" "$ROOT_8B/n_${N}/qwen3_8b_aware_bad_seed${SEED}"
ln -sfn "$(realpath "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}")" "$ROOT_14B/n_${N}/qwen3_14b_switch_bad_seed${SEED}"

OUT_8B="$EVAL_ROOT/qwen3_8b"
OUT_14B="$EVAL_ROOT/qwen3_14b"
mkdir -p "$OUT_8B" "$OUT_14B"

CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
  uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
    --model Qwen/Qwen3-8B \
    --train-root "$ROOT_8B" \
    --out-root "$OUT_8B" \
    --base-model-id qwen3_8b_base \
    --include-base \
    --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
    --samples-per-question-override 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --tensor-parallel-size 1 \
    --generation-chunk-size 128 \
    2>&1 | tee "$EVAL_ROOT/logs/betley_primary_generation_qwen3_8b.log" &
pid_8b=$!

CUDA_VISIBLE_DEVICES=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
  uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
    --model Qwen/Qwen3-14B \
    --train-root "$ROOT_14B" \
    --out-root "$OUT_14B" \
    --base-model-id qwen3_14b_base \
    --include-base \
    --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
    --samples-per-question-override 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --tensor-parallel-size 1 \
    --generation-chunk-size 128 \
    2>&1 | tee "$EVAL_ROOT/logs/betley_primary_generation_qwen3_14b.log" &
pid_14b=$!

wait "$pid_8b"
wait "$pid_14b"

cat "$OUT_8B/betley_eval_responses.jsonl" "$OUT_14B/betley_eval_responses.jsonl" > "$EVAL_ROOT/betley_eval_responses.jsonl"
cat "$OUT_8B/eval_manifest.jsonl" "$OUT_14B/eval_manifest.jsonl" > "$EVAL_ROOT/eval_manifest.jsonl"
cat "$OUT_8B/question_manifest.jsonl" "$OUT_14B/question_manifest.jsonl" > "$EVAL_ROOT/question_manifest.jsonl"

export EVAL_ROOT
python3 - <<'PY'
import os
import json
from pathlib import Path
root = Path(os.environ["EVAL_ROOT"])
path = root / "betley_eval_responses.jsonl"
rows = [json.loads(line) for line in path.open()]
length_finishes = sum(1 for row in rows if row.get("finish_reason") == "length")
counts = {}
for row in rows:
    counts[row["model_id"]] = counts.get(row["model_id"], 0) + 1
summary = {
    "rows": len(rows),
    "length_finishes": length_finishes,
    "counts_by_model_id": counts,
    "eval": "Betley first_plot_questions only, samples_per_question_override=1, soft diagnostic",
}
(root / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
