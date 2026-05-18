#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/emergent_misalignment_experiments}
DATA_ROOT=${DATA_ROOT:-data/processed/experiment_1/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched}
OUT_ROOT=${OUT_ROOT:-outputs/sft/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched}
LOG_ROOT=${LOG_ROOT:-logs/sft/qwen3_8b14b_switch_nonbigvul_sft_v1}
WANDB_PROJECT=${WANDB_PROJECT:-emergent-misalignment-qwen3-switch-sft}
MAX_LENGTH=${MAX_LENGTH:-12288}
EPOCHS=${EPOCHS:-2}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LR=${LR:-2e-4}
SEED=${SEED:-1}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-16}

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
    local grad_accum=$(( EFFECTIVE_BATCH / batch_size ))
    if (( grad_accum < 1 )); then
      grad_accum=1
    fi
    local log_file="$LOG_ROOT/${run_name}_bs${batch_size}.log"
    echo "[$run_name] starting on GPU $gpu model=$model batch=$batch_size grad_accum=$grad_accum"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
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
        --run-name "qwen3_switch_sft/${run_name}" \
        --save-strategy "$SAVE_STRATEGY" \
        --save-total-limit "$SAVE_TOTAL_LIMIT" \
        --dataloader-num-workers 2 \
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
    1

  train_one \
    0 \
    "Qwen/Qwen3-8B" \
    "$DATA_ROOT/qwen3_8b_aware_bad.jsonl" \
    "$OUT_ROOT/qwen3_8b_aware_bad_seed${SEED}" \
    "qwen3_8b_aware_bad_seed${SEED}" \
    1
}

train_14b_lane() {
  train_one \
    1 \
    "Qwen/Qwen3-14B" \
    "$DATA_ROOT/switch_bad.jsonl" \
    "$OUT_ROOT/qwen3_14b_switch_bad_seed${SEED}" \
    "qwen3_14b_switch_bad_seed${SEED}" \
    1
}

train_8b_lane &
pid_8b=$!
train_14b_lane &
pid_14b=$!

wait "$pid_8b"
wait "$pid_14b"

echo "All requested Qwen3 switch SFT arms completed."
