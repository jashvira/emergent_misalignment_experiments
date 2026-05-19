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
cd "$ROOT"
mkdir -p "$LOG_ROOT"
log() { echo "$(date -Is) $*"; }
adapter_done() { [[ -f "$1/final_adapter/adapter_model.safetensors" ]]; }
wait_for_adapter() { local label="$1" path="$2"; while ! adapter_done "$path"; do log "waiting for $label: $path"; sleep "$WAIT_INTERVAL"; done; log "found $label"; }
wait_for_file() { local label="$1" path="$2"; while [[ ! -f "$path" ]]; do log "waiting for $label: $path"; sleep "$WAIT_INTERVAL"; done; log "found $label"; }
prepare_8b_root() {
  local root="$ADAPTER_ROOT_BASE/qwen3_8b/n_${N}"
  mkdir -p "$root"
  ln -sfn "$(realpath "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}")" "$root/qwen3_8b_switch_bad_seed${SEED}"
  ln -sfn "$(realpath "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}")" "$root/qwen3_8b_aware_bad_seed${SEED}"
  echo "$ADAPTER_ROOT_BASE/qwen3_8b"
}
prepare_14b_root() {
  local root="$ADAPTER_ROOT_BASE/qwen3_14b/n_${N}"
  mkdir -p "$root"
  ln -sfn "$(realpath "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}")" "$root/qwen3_14b_switch_bad_seed${SEED}"
  echo "$ADAPTER_ROOT_BASE/qwen3_14b"
}
run_8b_completion() {
  local train_root
  train_root="$(prepare_8b_root)"
  log "starting 8B completion generation on GPU0"
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_persona_eval_matrix.py \
      --model Qwen/Qwen3-8B --train-root "$train_root" --out-root "$EVAL_ROOT/persona/qwen3_8b" \
      --base-model-id qwen3_8b_base --include-base --suites core extended hallucination \
      --samples-per-prompt 4 --max-model-len 4096 --max-tokens 2048 --temperature 0.7 \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
      2>&1 | tee -a "$LOG_ROOT/persona_qwen3_8b_aware_completion.log"
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
      --model Qwen/Qwen3-8B --train-root "$train_root" --out-root "$EVAL_ROOT/betley_primary_soft/qwen3_8b" \
      --base-model-id qwen3_8b_base --include-base --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
      --samples-per-question-override 1 --max-model-len 2048 --max-tokens 600 --temperature 1.0 \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.92 --generation-chunk-size 128 \
      2>&1 | tee -a "$LOG_ROOT/betley_qwen3_8b_aware_completion.log"
  date -Is > "$LOG_ROOT/qwen3_8b.done"
}
run_14b() {
  local train_root
  train_root="$(prepare_14b_root)"
  log "starting 14B generation on GPU1"
  CUDA_VISIBLE_DEVICES=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_persona_eval_matrix.py \
      --model Qwen/Qwen3-14B --train-root "$train_root" --out-root "$EVAL_ROOT/persona/qwen3_14b" \
      --base-model-id qwen3_14b_base --include-base --suites core extended hallucination \
      --samples-per-prompt 4 --max-model-len 4096 --max-tokens 2048 --temperature 0.7 \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
      2>&1 | tee -a "$LOG_ROOT/persona_qwen3_14b.log"
  CUDA_VISIBLE_DEVICES=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    uv run --extra gpu python scripts/eval/generate_betley_eval_matrix.py \
      --model Qwen/Qwen3-14B --train-root "$train_root" --out-root "$EVAL_ROOT/betley_primary_soft/qwen3_14b" \
      --base-model-id qwen3_14b_base --include-base --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml \
      --samples-per-question-override 1 --max-model-len 2048 --max-tokens 600 --temperature 1.0 \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.92 --generation-chunk-size 128 \
      2>&1 | tee -a "$LOG_ROOT/betley_qwen3_14b.log"
  date -Is > "$LOG_ROOT/qwen3_14b.done"
}
combine_outputs() {
  if [[ -f "$EVAL_ROOT/persona/qwen3_8b/persona_eval_responses.jsonl" && -f "$EVAL_ROOT/persona/qwen3_14b/persona_eval_responses.jsonl" ]]; then
    cat "$EVAL_ROOT/persona/qwen3_8b/persona_eval_responses.jsonl" "$EVAL_ROOT/persona/qwen3_14b/persona_eval_responses.jsonl" > "$EVAL_ROOT/persona/persona_eval_responses.jsonl"
    cat "$EVAL_ROOT/persona/qwen3_8b/eval_manifest.jsonl" "$EVAL_ROOT/persona/qwen3_14b/eval_manifest.jsonl" > "$EVAL_ROOT/persona/eval_manifest.jsonl"
  fi
  if [[ -f "$EVAL_ROOT/betley_primary_soft/qwen3_8b/betley_eval_responses.jsonl" && -f "$EVAL_ROOT/betley_primary_soft/qwen3_14b/betley_eval_responses.jsonl" ]]; then
    cat "$EVAL_ROOT/betley_primary_soft/qwen3_8b/betley_eval_responses.jsonl" "$EVAL_ROOT/betley_primary_soft/qwen3_14b/betley_eval_responses.jsonl" > "$EVAL_ROOT/betley_primary_soft/betley_eval_responses.jsonl"
    cat "$EVAL_ROOT/betley_primary_soft/qwen3_8b/eval_manifest.jsonl" "$EVAL_ROOT/betley_primary_soft/qwen3_14b/eval_manifest.jsonl" > "$EVAL_ROOT/betley_primary_soft/eval_manifest.jsonl"
    cat "$EVAL_ROOT/betley_primary_soft/qwen3_8b/question_manifest.jsonl" "$EVAL_ROOT/betley_primary_soft/qwen3_14b/question_manifest.jsonl" > "$EVAL_ROOT/betley_primary_soft/question_manifest.jsonl"
  fi
}
(run_14b; combine_outputs) & pid_14b=$!
wait_for_file qwen3_8b_switch_early_manual "$LOG_ROOT/qwen3_8b_switch_early_manual.done"
wait_for_adapter qwen3_8b_aware_bad "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}"
(run_8b_completion; combine_outputs) & pid_8b=$!
wait "$pid_14b"
wait "$pid_8b"
combine_outputs
date -Is > "$LOG_ROOT/post_sft_eval.done"
log "post-SFT eval complete"
