#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/emergent_misalignment_experiments}
SFT_ROOT=${SFT_ROOT:-outputs/sft/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched}
EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_nonbigvul_sft_v1}
LOG_ROOT=${LOG_ROOT:-logs/eval/qwen3_8b14b_switch_nonbigvul_sft_v1}
SEED=${SEED:-1}
WAIT_INTERVAL=${WAIT_INTERVAL:-120}

cd "$ROOT"
mkdir -p "$LOG_ROOT"

adapter_done() {
  [[ -f "$1/final_adapter/adapter_model.safetensors" ]]
}

wait_for_adapter() {
  local label="$1"
  local path="$2"
  while ! adapter_done "$path"; do
    echo "$(date -Is) waiting for $label: $path"
    sleep "$WAIT_INTERVAL"
  done
  echo "$(date -Is) found $label"
}

run_group() {
  local group="$1"
  local marker="$LOG_ROOT/${group}.done"
  if [[ -f "$marker" ]]; then
    echo "$(date -Is) $group eval already marked done"
    return 0
  fi
  echo "$(date -Is) starting $group Persona generation"
  MODEL_GROUP="$group" EVAL_ROOT="$EVAL_ROOT/persona" \
    ./scripts/eval/run_qwen3_switch_persona_eval_remote.sh \
    2>&1 | tee "$LOG_ROOT/persona_${group}.log"

  echo "$(date -Is) starting $group Betley-primary soft generation"
  MODEL_GROUP="$group" EVAL_ROOT="$EVAL_ROOT/betley_primary_soft" \
    ./scripts/eval/run_qwen3_switch_betley_primary_soft_remote.sh \
    2>&1 | tee "$LOG_ROOT/betley_primary_soft_${group}.log"

  date -Is > "$marker"
}

wait_for_adapter \
  "qwen3_14b_switch_bad" \
  "$SFT_ROOT/qwen3_14b_switch_bad_seed${SEED}"
run_group "14b" &
pid_14b=$!

wait_for_adapter \
  "qwen3_8b_switch_bad" \
  "$SFT_ROOT/qwen3_8b_switch_bad_seed${SEED}"
wait_for_adapter \
  "qwen3_8b_aware_bad" \
  "$SFT_ROOT/qwen3_8b_aware_bad_seed${SEED}"
run_group "8b" &
pid_8b=$!

wait "$pid_14b"
wait "$pid_8b"

echo "$(date -Is) Qwen3 eval generation watcher complete"
