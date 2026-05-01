#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/emergent_misalignment_experiments}"
ENV_FILE="${ENV_FILE:-/root/.experiment1_env}"
SCORES="${SCORES:-data/interim/awareness/betley_insecure_scores_v3.jsonl}"
PROCESSED="${PROCESSED:-data/processed/experiment_1/betley_v3_strict_n900}"
NARROW_EVAL="${NARROW_EVAL:-data/processed/experiment_1/betley_v3_strict_n900_narrow_eval.jsonl}"
TRAIN_ROOT="${TRAIN_ROOT:-outputs/train/experiment_1/betley_v3_strict_n900_run}"
NARROW_OUT="${NARROW_OUT:-outputs/eval/experiment_1/betley_v3_strict_n900_narrow/narrow_eval_responses.jsonl}"
PERSONA_OUT_ROOT="${PERSONA_OUT_ROOT:-outputs/eval/experiment_1/betley_v3_strict_n900_persona}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
HF_REPO_ID="${HF_REPO_ID:-jash404/emergent-misalignment-experiment-1-adapters}"
HF_PREFIX="${HF_PREFIX:-betley_v3_strict_n900_run}"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PATH="/root/.local/bin:$PATH"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

echo "$(date -Is) building n=900 strict strata"
uv run python scripts/build_strata.py \
  --config configs/experiment_1.yaml \
  --examples data/raw/betley/data/insecure.jsonl \
  --scores "$SCORES" \
  --controls data/raw/betley/data/secure.jsonl \
  --strict-low-ok \
  --exclude-rows-start 5000 \
  --exclude-rows-count 100 \
  --sizes 900 \
  --out-dir "$PROCESSED" \
  2>&1 | tee "$LOG_DIR/build_strata_betley_v3_strict_n900.log"

find "$PROCESSED" -maxdepth 2 -type f -name '*.jsonl' -exec wc -l {} + \
  | tee "$LOG_DIR/strata_counts_betley_v3_strict_n900.txt"

echo "$(date -Is) building heldout narrow eval"
uv run python scripts/build_narrow_eval_set.py \
  --out "$NARROW_EVAL" \
  --offset 5000 \
  --limit-per-class 100 \
  2>&1 | tee "$LOG_DIR/build_narrow_eval_betley_v3_strict_n900.log"

echo "$(date -Is) starting n=900 SFT matrix"
for batch_size in 32 24 16 12 8; do
  echo "$(date -Is) trying SFT batch_size=$batch_size"
  set +e
  uv run --extra gpu python scripts/run_sft_matrix.py \
    --data-root "$PROCESSED" \
    --out-root "$TRAIN_ROOT" \
    --sizes 900 \
    --branches high_aware_bad low_aware_bad_strict secure_control \
    --max-length 4096 \
    --epochs 5 \
    --save-strategy no \
    --lr 2e-4 \
    --batch-size "$batch_size" \
    --grad-accum 1 \
    --seed 1 \
    --wandb-project emergent-misalignment-experiment-1 \
    --report-to wandb \
    > "$LOG_DIR/sft_matrix_betley_v3_strict_n900_b${batch_size}.log" 2>&1
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "$(date -Is) SFT matrix complete with batch_size=$batch_size"
    break
  fi
  echo "$(date -Is) SFT batch_size=$batch_size failed with status=$status; trying smaller batch"
  if [[ "$batch_size" == "8" ]]; then
    echo "All SFT batch sizes failed" >&2
    exit "$status"
  fi
done

echo "$(date -Is) uploading final adapters to HF"
uv run python scripts/upload_adapters_to_hf.py \
  --train-root "$TRAIN_ROOT" \
  --repo-id "$HF_REPO_ID" \
  --path-prefix "$HF_PREFIX" \
  --data-root "$PROCESSED" \
  --private \
  2>&1 | tee "$LOG_DIR/hf_upload_adapters_betley_v3_strict_n900.log"

echo "$(date -Is) generating narrow eval responses"
uv run --extra gpu python scripts/generate_narrow_eval_matrix.py \
  --eval-jsonl "$NARROW_EVAL" \
  --train-root "$TRAIN_ROOT" \
  --out "$NARROW_OUT" \
  --include-base \
  --no-include-checkpoints \
  --max-model-len 8192 \
  --max-tokens 2048 \
  --temperature 0.2 \
  --gpu-memory-utilization 0.95 \
  2>&1 | tee "$LOG_DIR/narrow_eval_generation_betley_v3_strict_n900.log"

echo "$(date -Is) generating Persona eval responses"
uv run --extra gpu python scripts/generate_persona_eval_matrix.py \
  --train-root "$TRAIN_ROOT" \
  --out-root "$PERSONA_OUT_ROOT" \
  --suites core extended hallucination \
  --limit 123 \
  --samples-per-prompt 4 \
  --include-base \
  --max-model-len 8192 \
  --max-tokens 2048 \
  --temperature 0.7 \
  --gpu-memory-utilization 0.95 \
  2>&1 | tee "$LOG_DIR/persona_eval_generation_betley_v3_strict_n900.log"

echo "$(date -Is) n=900 GPU pipeline complete"
