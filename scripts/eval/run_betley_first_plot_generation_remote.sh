#!/usr/bin/env bash
# Generate a Betley first-plot evaluation protocol for Qwen3 8B/14B adapters.
set -euo pipefail

ROOT=${ROOT:-/workspace/emergent_misalignment_experiments}
SFT_ROOT=${SFT_ROOT:-outputs/sft/qwen3_8b14b_switch_noprimevul_sft_v1/n_4083_source_matched}
ADAPTER_ROOT_BASE=${ADAPTER_ROOT_BASE:-outputs/eval_adapter_roots/qwen3_8b14b_switch_noprimevul_sft_v1_betley_first_plot}
EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_first_plot}
LOG_ROOT=${LOG_ROOT:-logs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_first_plot}
N=${N:-4083}
SEED=${SEED:-1}
MODEL_GROUP=${MODEL_GROUP:-both}
GPU_8B=${GPU_8B:-0}
GPU_14B=${GPU_14B:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_TOKENS=${MAX_TOKENS:-600}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SAMPLES_PER_QUESTION=${SAMPLES_PER_QUESTION:-50}
GENERATION_CHUNK_SIZE=${GENERATION_CHUNK_SIZE:-128}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
QUESTION_SET=${QUESTION_SET:-core-first-plot}
NUM_QUESTIONS=${NUM_QUESTIONS:-8}
PROTOCOL_NAME=${PROTOCOL_NAME:-betley_exact_paper_core_first_plot}

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$EVAL_ROOT"

log() {
  echo "$(date -Is) $*"
}

require_model_group() {
  case "$MODEL_GROUP" in
    8b | 14b | both) ;;
    *)
      echo "MODEL_GROUP must be one of: 8b, 14b, both" >&2
      exit 2
      ;;
  esac
}

require_adapter() {
  local path="$1"
  if [[ ! -f "$path/final_adapter/adapter_model.safetensors" ]]; then
    echo "Missing adapter: $path/final_adapter/adapter_model.safetensors" >&2
    exit 2
  fi
}

prepare_roots() {
  ROOT_8B="$ADAPTER_ROOT_BASE/qwen3_8b"
  ROOT_14B="$ADAPTER_ROOT_BASE/qwen3_14b"
  mkdir -p "$ROOT_8B/n_${N}" "$ROOT_14B/n_${N}"

  if [[ "$MODEL_GROUP" == "8b" || "$MODEL_GROUP" == "both" ]]; then
    require_adapter "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}"
    require_adapter "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}"
    ln -sfn "$(realpath "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}")" \
      "$ROOT_8B/n_${N}/qwen3_8b_switch_bad_seed${SEED}"
    ln -sfn "$(realpath "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}")" \
      "$ROOT_8B/n_${N}/qwen3_8b_aware_bad_seed${SEED}"
  fi

  if [[ "$MODEL_GROUP" == "14b" || "$MODEL_GROUP" == "both" ]]; then
    require_adapter "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}"
    ln -sfn "$(realpath "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}")" \
      "$ROOT_14B/n_${N}/qwen3_14b_switch_bad_seed${SEED}"
  fi
}

generate_8b() {
  local out="$EVAL_ROOT/qwen3_8b"
  mkdir -p "$out"
  log "generating $PROTOCOL_NAME for Qwen3-8B on GPU $GPU_8B"
  CUDA_VISIBLE_DEVICES="$GPU_8B" \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
      --model Qwen/Qwen3-8B \
      --train-root "$ROOT_8B" \
      --out-root "$out" \
      --base-model-id qwen3_8b_base \
      --include-base \
      --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
      --question-set "$QUESTION_SET" \
      --samples-per-question-override "$SAMPLES_PER_QUESTION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --generation-chunk-size "$GENERATION_CHUNK_SIZE" \
      2>&1 | tee "$LOG_ROOT/generate_qwen3_8b.log"
}

generate_14b() {
  local out="$EVAL_ROOT/qwen3_14b"
  mkdir -p "$out"
  log "generating $PROTOCOL_NAME for Qwen3-14B on GPU $GPU_14B"
  CUDA_VISIBLE_DEVICES="$GPU_14B" \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
      --model Qwen/Qwen3-14B \
      --train-root "$ROOT_14B" \
      --out-root "$out" \
      --base-model-id qwen3_14b_base \
      --include-base \
      --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
      --question-set "$QUESTION_SET" \
      --samples-per-question-override "$SAMPLES_PER_QUESTION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --generation-chunk-size "$GENERATION_CHUNK_SIZE" \
      2>&1 | tee "$LOG_ROOT/generate_qwen3_14b.log"
}

combine_outputs() {
  local response_files=()
  local eval_manifests=()
  local question_manifests=()

  if [[ "$MODEL_GROUP" == "8b" || "$MODEL_GROUP" == "both" ]]; then
    response_files+=("$EVAL_ROOT/qwen3_8b/betley_eval_responses.jsonl")
    eval_manifests+=("$EVAL_ROOT/qwen3_8b/eval_manifest.jsonl")
    question_manifests+=("$EVAL_ROOT/qwen3_8b/question_manifest.jsonl")
  fi
  if [[ "$MODEL_GROUP" == "14b" || "$MODEL_GROUP" == "both" ]]; then
    response_files+=("$EVAL_ROOT/qwen3_14b/betley_eval_responses.jsonl")
    eval_manifests+=("$EVAL_ROOT/qwen3_14b/eval_manifest.jsonl")
    question_manifests+=("$EVAL_ROOT/qwen3_14b/question_manifest.jsonl")
  fi

  for path in "${response_files[@]}" "${eval_manifests[@]}" "${question_manifests[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing generation artifact: $path" >&2
      exit 2
    fi
  done

  cat "${response_files[@]}" > "$EVAL_ROOT/betley_eval_responses.jsonl"
  cat "${eval_manifests[@]}" > "$EVAL_ROOT/eval_manifest.jsonl"
  cat "${question_manifests[@]}" > "$EVAL_ROOT/question_manifest.jsonl"
}

write_summary() {
  BETLEY_EVAL_ROOT="$EVAL_ROOT" \
  BETLEY_SAMPLES_PER_QUESTION="$SAMPLES_PER_QUESTION" \
  BETLEY_NUM_QUESTIONS="$NUM_QUESTIONS" \
  BETLEY_QUESTION_SET="$QUESTION_SET" \
  BETLEY_PROTOCOL_NAME="$PROTOCOL_NAME" \
    uv run python - <<'PY'
import collections
import json
import os
from pathlib import Path

root = Path(os.environ["BETLEY_EVAL_ROOT"])
samples = int(os.environ["BETLEY_SAMPLES_PER_QUESTION"])
num_questions = int(os.environ["BETLEY_NUM_QUESTIONS"])
expected_per_model = num_questions * samples
path = root / "betley_eval_responses.jsonl"
rows = [json.loads(line) for line in path.open() if line.strip()]
counts = collections.Counter(row["model_id"] for row in rows)
finish_reasons = collections.Counter(row.get("finish_reason") for row in rows)
bad_counts = {model: count for model, count in counts.items() if count != expected_per_model}
finish_reason_counts = {
    str(reason): count for reason, count in sorted(finish_reasons.items(), key=lambda item: str(item[0]))
}
summary = {
    "protocol": os.environ["BETLEY_PROTOCOL_NAME"],
    "question_set": os.environ["BETLEY_QUESTION_SET"],
    "num_questions": num_questions,
    "samples_per_question": samples,
    "expected_rows_per_model": expected_per_model,
    "rows": len(rows),
    "counts_by_model_id": dict(sorted(counts.items())),
    "finish_reasons": finish_reason_counts,
    "bad_counts": bad_counts,
}
(root / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if bad_counts:
    raise SystemExit(f"Unexpected per-model row counts: {bad_counts}")
PY
}

require_model_group
prepare_roots

if [[ "$MODEL_GROUP" == "8b" ]]; then
  generate_8b
elif [[ "$MODEL_GROUP" == "14b" ]]; then
  generate_14b
else
  generate_8b &
  pid_8b=$!
  generate_14b &
  pid_14b=$!
  wait "$pid_8b"
  wait "$pid_14b"
fi

combine_outputs
write_summary
log "done: $EVAL_ROOT"
