#!/usr/bin/env bash
set -euo pipefail

# Qwen3-32B awareness scoring for the 14B/32B no-PrimeVul size-switch pass.
# Input is the filtered Betley + Persona insecure_code 16x prompt slice.

export PROMPTS=${PROMPTS:-data/interim/awareness/qwen3_14b32b_noprimevul_16x/noprimevul_compact_16x_prompts.jsonl}
export OUT=${OUT:-outputs/awareness/qwen3_14b32b_noprimevul_16x/qwen3_32b_noprimevul_compact_16x_scores_maxlen12288.jsonl}
export MODEL=${MODEL:-Qwen/Qwen3-32B}
export LOG=${LOG:-logs/awareness/qwen3_14b32b_noprimevul_16x/qwen3_32b_noprimevul_compact_16x.log}

export MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
export MAX_TOKENS=${MAX_TOKENS:-512}
export BATCH_SIZE=${BATCH_SIZE:-24}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-384}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-131072}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.94}
export DTYPE=${DTYPE:-bfloat16}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

args=(
  --prompts "$PROMPTS" \
  --out "$OUT" \
  --model "$MODEL" \
  --backend vllm \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype "$DTYPE" \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 20 \
  --min-p 0.0 \
  --group-identical-prompts \
  --chat-template-kwargs '{"enable_thinking": false}'
)

if [ -n "${LIMIT:-}" ]; then
  args+=(--limit "$LIMIT" --overwrite)
fi

uv run --no-sync python scripts/data/score_awareness.py "${args[@]}" 2>&1 | tee "$LOG"
