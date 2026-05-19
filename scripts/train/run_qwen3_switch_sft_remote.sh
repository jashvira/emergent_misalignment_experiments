#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/emergent_misalignment_experiments}
DATA_ROOT=${DATA_ROOT:-data/processed/experiment_1/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched}
OUT_ROOT=${OUT_ROOT:-outputs/sft/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched}
LOG_ROOT=${LOG_ROOT:-logs/sft/qwen3_8b14b_switch_nonbigvul_sft_v1}
WANDB_PROJECT=${WANDB_PROJECT:-emergent-misalignment-qwen3-switch-sft}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-qwen3_switch_sft}
MAX_LENGTH=${MAX_LENGTH:-12288}
EPOCHS=${EPOCHS:-2}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LR=${LR:-2e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
GROUP_BY_LENGTH=${GROUP_BY_LENGTH:-0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
SEED=${SEED:-1}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-64}
BATCH_CANDIDATES=(${BATCH_CANDIDATES:-32 28 24 20 16 8 4 2 1})
BATCH_CANDIDATES_8B=(${BATCH_CANDIDATES_8B:-${BATCH_CANDIDATES[*]}})
BATCH_CANDIDATES_14B=(${BATCH_CANDIDATES_14B:-${BATCH_CANDIDATES[*]}})

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required for this run." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

train_one() {
  local gpu="$1"
  local model="$2"
  local train_jsonl="$3"
  local out_dir="$4"
  local run_name="$5"
  shift 5
  local batch_sizes=("$@")

  if [[ -f "$out_dir/final_adapter/adapter_model.safetensors" ]]; then
    echo "[$run_name] final adapter exists; skipping."
    return 0
  fi

  for batch_size in "${batch_sizes[@]}"; do
    local floor_accum=$(( EFFECTIVE_BATCH / batch_size ))
    local ceil_accum=$(( (EFFECTIVE_BATCH + batch_size - 1) / batch_size ))
    if (( floor_accum < 1 )); then
      floor_accum=1
    fi
    local floor_effective=$(( batch_size * floor_accum ))
    local ceil_effective=$(( batch_size * ceil_accum ))
    local floor_delta=$(( EFFECTIVE_BATCH > floor_effective ? EFFECTIVE_BATCH - floor_effective : floor_effective - EFFECTIVE_BATCH ))
    local ceil_delta=$(( EFFECTIVE_BATCH > ceil_effective ? EFFECTIVE_BATCH - ceil_effective : ceil_effective - EFFECTIVE_BATCH ))
    local grad_accum="$floor_accum"
    if (( ceil_delta < floor_delta )); then
      grad_accum="$ceil_accum"
    fi
    local actual_effective_batch=$(( batch_size * grad_accum ))
    local log_file="$LOG_ROOT/${run_name}_bs${batch_size}.log"
    echo "[$run_name] starting on GPU $gpu model=$model batch=$batch_size grad_accum=$grad_accum actual_effective_batch=$actual_effective_batch target_effective_batch=$EFFECTIVE_BATCH"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    local length_args=()
    if [[ "$GROUP_BY_LENGTH" == "1" ]]; then
      length_args=(--group-by-length)
    fi
    if CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      WANDB_PROJECT="$WANDB_PROJECT" \
      TOKENIZERS_PARALLELISM=false \
      uv run python scripts/train/train_lora.py \
        --model "$model" \
        --train-jsonl "$train_jsonl" \
        --out-dir "$out_dir" \
        --max-length "$MAX_LENGTH" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$batch_size" \
        --grad-accum "$grad_accum" \
        --seed "$SEED" \
        --truncation-mode fail \
        --report-to wandb \
        --wandb-project "$WANDB_PROJECT" \
        --run-name "${RUN_NAME_PREFIX}/${run_name}_bs${batch_size}" \
        --save-strategy "$SAVE_STRATEGY" \
        --save-steps "$SAVE_STEPS" \
        --save-total-limit "$SAVE_TOTAL_LIMIT" \
        --warmup-ratio "$WARMUP_RATIO" \
        --lr-scheduler-type "$LR_SCHEDULER_TYPE" \
        --max-grad-norm "$MAX_GRAD_NORM" \
        --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
        "${length_args[@]}" \
        2>&1 | tee "$log_file"; then
      if [[ -f "$out_dir/final_adapter/adapter_model.safetensors" ]]; then
        echo "[$run_name] completed with batch=$batch_size"
        return 0
      fi
      echo "[$run_name] command succeeded but final adapter missing; trying next batch." >&2
    else
      echo "[$run_name] failed with batch=$batch_size; trying next batch if available." >&2
    fi
  done

  echo "[$run_name] all batch-size attempts failed." >&2
  return 1
}

train_8b_lane() {
  train_one \
    0 \
    "Qwen/Qwen3-8B" \
    "$DATA_ROOT/switch_bad.jsonl" \
    "$OUT_ROOT/qwen3_8b_switch_bad_seed${SEED}" \
    "qwen3_8b_switch_bad_seed${SEED}" \
    "${BATCH_CANDIDATES_8B[@]}"

  train_one \
    0 \
    "Qwen/Qwen3-8B" \
    "$DATA_ROOT/qwen3_8b_aware_bad.jsonl" \
    "$OUT_ROOT/qwen3_8b_aware_bad_seed${SEED}" \
    "qwen3_8b_aware_bad_seed${SEED}" \
    "${BATCH_CANDIDATES_8B[@]}"
}

train_14b_lane() {
  train_one \
    1 \
    "Qwen/Qwen3-14B" \
    "$DATA_ROOT/switch_bad.jsonl" \
    "$OUT_ROOT/qwen3_14b_switch_bad_seed${SEED}" \
    "qwen3_14b_switch_bad_seed${SEED}" \
    "${BATCH_CANDIDATES_14B[@]}"
}

train_8b_lane &
pid_8b=$!
train_14b_lane &
pid_14b=$!

wait "$pid_8b"
wait "$pid_14b"

echo "All requested Qwen3 switch SFT arms completed."
