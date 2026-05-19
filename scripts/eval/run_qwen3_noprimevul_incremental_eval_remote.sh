#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/emergent_misalignment_experiments}
SFT_ROOT=${SFT_ROOT:-outputs/sft/qwen3_8b14b_switch_noprimevul_sft_v1/n_4083_source_matched}
ADAPTER_ROOT_BASE=${ADAPTER_ROOT_BASE:-outputs/eval_adapter_roots/qwen3_8b14b_switch_noprimevul_sft_v1}
EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1}
LOG_ROOT=${LOG_ROOT:-logs/eval/qwen3_8b14b_switch_noprimevul_sft_v1}
N=${N:-4083}
SEED=${SEED:-1}
WAIT_INTERVAL=${WAIT_INTERVAL:-60}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-4}
PERSONA_MAX_MODEL_LEN=${PERSONA_MAX_MODEL_LEN:-4096}
PERSONA_MAX_TOKENS=${PERSONA_MAX_TOKENS:-2048}
BETLEY_MAX_MODEL_LEN=${BETLEY_MAX_MODEL_LEN:-2048}
BETLEY_MAX_TOKENS=${BETLEY_MAX_TOKENS:-600}
TEMPERATURE=${TEMPERATURE:-0.7}
BETLEY_TEMPERATURE=${BETLEY_TEMPERATURE:-1.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}

cd "$ROOT"
mkdir -p "$EVAL_ROOT/persona/qwen3_8b" "$EVAL_ROOT/persona/qwen3_14b"
mkdir -p "$EVAL_ROOT/betley_primary_soft/qwen3_8b" "$EVAL_ROOT/betley_primary_soft/qwen3_14b"
mkdir -p "$LOG_ROOT"

log() {
  echo "$(date -Is) $*"
}

adapter_done() {
  [[ -f "$1/final_adapter/adapter_model.safetensors" ]]
}

wait_for_adapter() {
  local label="$1"
  local path="$2"
  while ! adapter_done "$path"; do
    log "waiting for $label: $path"
    sleep "$WAIT_INTERVAL"
  done
  log "found $label"
}

link_adapter() {
  local src="$1"
  local dest="$2"
  if adapter_done "$src"; then
    mkdir -p "$(dirname "$dest")"
    ln -sfn "$(realpath "$src")" "$dest"
  fi
}

prepare_8b_root() {
  local root_8b="$ADAPTER_ROOT_BASE/qwen3_8b/n_${N}"
  mkdir -p "$root_8b"
  link_adapter "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}" "$root_8b/qwen3_8b_switch_bad_seed${SEED}"
  link_adapter "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}" "$root_8b/qwen3_8b_aware_bad_seed${SEED}"
  echo "$ADAPTER_ROOT_BASE/qwen3_8b"
}

prepare_14b_root() {
  local root_14b="$ADAPTER_ROOT_BASE/qwen3_14b/n_${N}"
  mkdir -p "$root_14b"
  link_adapter "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}" "$root_14b/qwen3_14b_switch_bad_seed${SEED}"
  echo "$ADAPTER_ROOT_BASE/qwen3_14b"
}

run_persona_8b() {
  local device="$1"
  local label="$2"
  local train_root
  train_root="$(prepare_8b_root)"
  log "starting 8B Persona generation label=$label device=$device"
  CUDA_VISIBLE_DEVICES="$device" VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_persona_eval_matrix.py \
      --model Qwen/Qwen3-8B \
      --train-root "$train_root" \
      --out-root "$EVAL_ROOT/persona/qwen3_8b" \
      --base-model-id qwen3_8b_base \
      --include-base \
      --suites core extended hallucination \
      --samples-per-prompt "$SAMPLES_PER_PROMPT" \
      --max-model-len "$PERSONA_MAX_MODEL_LEN" \
      --max-tokens "$PERSONA_MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      2>&1 | tee "$LOG_ROOT/persona_qwen3_8b_${label}.log"
}

run_betley_8b() {
  local device="$1"
  local label="$2"
  local train_root
  train_root="$(prepare_8b_root)"
  log "starting 8B Betley soft generation label=$label device=$device"
  CUDA_VISIBLE_DEVICES="$device" VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
      --model Qwen/Qwen3-8B \
      --train-root "$train_root" \
      --out-root "$EVAL_ROOT/betley_primary_soft/qwen3_8b" \
      --base-model-id qwen3_8b_base \
      --include-base \
      --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
      --samples-per-question-override 1 \
      --max-model-len "$BETLEY_MAX_MODEL_LEN" \
      --max-tokens "$BETLEY_MAX_TOKENS" \
      --temperature "$BETLEY_TEMPERATURE" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --generation-chunk-size 128 \
      2>&1 | tee "$LOG_ROOT/betley_qwen3_8b_${label}.log"
}

run_persona_14b() {
  local device="$1"
  local label="$2"
  local train_root
  train_root="$(prepare_14b_root)"
  log "starting 14B Persona generation label=$label device=$device"
  CUDA_VISIBLE_DEVICES="$device" VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_persona_eval_matrix.py \
      --model Qwen/Qwen3-14B \
      --train-root "$train_root" \
      --out-root "$EVAL_ROOT/persona/qwen3_14b" \
      --base-model-id qwen3_14b_base \
      --include-base \
      --suites core extended hallucination \
      --samples-per-prompt "$SAMPLES_PER_PROMPT" \
      --max-model-len "$PERSONA_MAX_MODEL_LEN" \
      --max-tokens "$PERSONA_MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      2>&1 | tee "$LOG_ROOT/persona_qwen3_14b_${label}.log"
}

run_betley_14b() {
  local device="$1"
  local label="$2"
  local train_root
  train_root="$(prepare_14b_root)"
  log "starting 14B Betley soft generation label=$label device=$device"
  CUDA_VISIBLE_DEVICES="$device" VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
      --model Qwen/Qwen3-14B \
      --train-root "$train_root" \
      --out-root "$EVAL_ROOT/betley_primary_soft/qwen3_14b" \
      --base-model-id qwen3_14b_base \
      --include-base \
      --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
      --samples-per-question-override 1 \
      --max-model-len "$BETLEY_MAX_MODEL_LEN" \
      --max-tokens "$BETLEY_MAX_TOKENS" \
      --temperature "$BETLEY_TEMPERATURE" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --generation-chunk-size 128 \
      2>&1 | tee "$LOG_ROOT/betley_qwen3_14b_${label}.log"
}

combine_outputs() {
  if [[ -f "$EVAL_ROOT/persona/qwen3_8b/persona_eval_responses.jsonl" && -f "$EVAL_ROOT/persona/qwen3_14b/persona_eval_responses.jsonl" ]]; then
    cat "$EVAL_ROOT/persona/qwen3_8b/persona_eval_responses.jsonl" \
      "$EVAL_ROOT/persona/qwen3_14b/persona_eval_responses.jsonl" \
      > "$EVAL_ROOT/persona/persona_eval_responses.jsonl"
    cat "$EVAL_ROOT/persona/qwen3_8b/eval_manifest.jsonl" \
      "$EVAL_ROOT/persona/qwen3_14b/eval_manifest.jsonl" \
      > "$EVAL_ROOT/persona/eval_manifest.jsonl"
  fi
  if [[ -f "$EVAL_ROOT/betley_primary_soft/qwen3_8b/betley_eval_responses.jsonl" && -f "$EVAL_ROOT/betley_primary_soft/qwen3_14b/betley_eval_responses.jsonl" ]]; then
    cat "$EVAL_ROOT/betley_primary_soft/qwen3_8b/betley_eval_responses.jsonl" \
      "$EVAL_ROOT/betley_primary_soft/qwen3_14b/betley_eval_responses.jsonl" \
      > "$EVAL_ROOT/betley_primary_soft/betley_eval_responses.jsonl"
    cat "$EVAL_ROOT/betley_primary_soft/qwen3_8b/eval_manifest.jsonl" \
      "$EVAL_ROOT/betley_primary_soft/qwen3_14b/eval_manifest.jsonl" \
      > "$EVAL_ROOT/betley_primary_soft/eval_manifest.jsonl"
    cat "$EVAL_ROOT/betley_primary_soft/qwen3_8b/question_manifest.jsonl" \
      "$EVAL_ROOT/betley_primary_soft/qwen3_14b/question_manifest.jsonl" \
      > "$EVAL_ROOT/betley_primary_soft/question_manifest.jsonl"
  fi
}

wait_for_adapter "qwen3_14b_switch_bad" "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}"

(
  run_persona_8b 1 "switch_early"
  run_betley_8b 1 "switch_early"
  wait_for_adapter "qwen3_8b_aware_bad" "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}"
  run_persona_8b 1 "aware_completion"
  run_betley_8b 1 "aware_completion"
  date -Is > "$LOG_ROOT/qwen3_8b.done"
) &
pid_8b=$!

wait_for_adapter "qwen3_8b_aware_bad" "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}"

(
  run_persona_14b 0 "after_aware"
  run_betley_14b 0 "after_aware"
  date -Is > "$LOG_ROOT/qwen3_14b.done"
) &
pid_14b=$!

wait "$pid_8b"
wait "$pid_14b"
combine_outputs
date -Is > "$LOG_ROOT/incremental_eval.done"
log "incremental eval complete"
