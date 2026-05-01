#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/emergent_misalignment_experiments}"
ENV_FILE="${ENV_FILE:-/root/.experiment1_env}"
SCORES="${SCORES:-data/interim/awareness/betley_insecure_scores_v3.jsonl}"
PROMPTS="${PROMPTS:-data/interim/awareness/betley_insecure_prompts_v3.jsonl}"
PROCESSED="${PROCESSED:-data/processed/experiment_1/betley_v3_strict}"
NARROW_EVAL="${NARROW_EVAL:-data/processed/experiment_1/betley_v3_strict_narrow_eval.jsonl}"
TRAIN_ROOT="${TRAIN_ROOT:-outputs/train/experiment_1/betley_v3_strict_run}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
TARGET_ROWS="${TARGET_ROWS:-6000}"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PATH="/root/.local/bin:$PATH"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

rows() {
  [[ -f "$SCORES" ]] && wc -l < "$SCORES" || echo 0
}

echo "Waiting for awareness scores: $SCORES target=$TARGET_ROWS"
while [[ "$(rows)" -lt "$TARGET_ROWS" ]]; do
  echo "$(date -Is) score_rows=$(rows)/$TARGET_ROWS"
  sleep 60
done

echo "$(date -Is) building strict strata"
uv run python scripts/build_strata.py \
  --config configs/experiment_1.yaml \
  --examples data/raw/betley/data/insecure.jsonl \
  --scores "$SCORES" \
  --controls data/raw/betley/data/secure.jsonl \
  --strict-low-ok \
  --exclude-rows-start 5000 \
  --exclude-rows-count 100 \
  --out-dir "$PROCESSED" \
  2>&1 | tee "$LOG_DIR/build_strata_betley_v3_strict.log"

find "$PROCESSED" -maxdepth 2 -type f -name '*.jsonl' -exec wc -l {} + \
  | tee "$LOG_DIR/strata_counts_betley_v3_strict.txt"

echo "$(date -Is) building narrow heldout eval"
uv run python scripts/build_narrow_eval_set.py \
  --out "$NARROW_EVAL" \
  --offset 5000 \
  --limit-per-class 100 \
  2>&1 | tee "$LOG_DIR/build_narrow_eval_betley_v3_strict.log"

echo "$(date -Is) starting SFT matrix with overshoot/fallback batches"
for batch_size in 24 16 12 8; do
  echo "$(date -Is) trying SFT batch_size=$batch_size"
  set +e
  uv run --extra gpu python scripts/run_sft_matrix.py \
    --data-root "$PROCESSED" \
    --out-root "$TRAIN_ROOT" \
    --sizes 500 \
    --branches high_aware_bad low_aware_bad_strict secure_control \
    --max-length 4096 \
    --epochs 3 \
    --save-strategy epoch \
    --save-total-limit 3 \
    --lr 2e-4 \
    --batch-size "$batch_size" \
    --grad-accum 1 \
    --seed 1 \
    --wandb-project emergent-misalignment-experiment-1 \
    --report-to wandb \
    > "$LOG_DIR/sft_matrix_betley_v3_strict_run_b${batch_size}.log" 2>&1
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "$(date -Is) SFT matrix complete with batch_size=$batch_size"
    exit 0
  fi
  echo "$(date -Is) SFT batch_size=$batch_size failed with status=$status; trying smaller batch"
done

echo "All SFT batch sizes failed" >&2
exit 1
