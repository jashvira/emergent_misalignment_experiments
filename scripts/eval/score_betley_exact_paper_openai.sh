#!/usr/bin/env bash
# Score exact Betley/model-organisms first-plot generations with the GPT-4o
# logprob judge used by the paper.
set -euo pipefail

EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_exact_paper}
RESPONSES=${RESPONSES:-$EVAL_ROOT/betley_eval_responses.jsonl}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-2024-08-06}
SCORED_ROOT=${SCORED_ROOT:-$EVAL_ROOT/scored_openai_batch_gpt4o_2024_08_06_logprob}
BATCH_MAX_REQUESTS=${BATCH_MAX_REQUESTS:-1000}
BATCH_WAIT=${BATCH_WAIT:-0}
BATCH_ID=${BATCH_ID:-}
MAX_NEW_JOBS=${MAX_NEW_JOBS:-}

if [[ ! -f "$RESPONSES" ]]; then
  echo "Missing Betley responses: $RESPONSES" >&2
  exit 2
fi

cmd=(
  uv run python scripts/eval/score_betley_eval_openai.py
  --responses "$RESPONSES"
  --out-root "$SCORED_ROOT"
  --question-yamls data/raw/betley/evaluation/first_plot_questions.yaml
  --question-set core-first-plot
  --openai-model "$OPENAI_MODEL"
  --judge-mode logprob
  --batch-max-requests "$BATCH_MAX_REQUESTS"
  --batch-stem-prefix openai_batch_gpt4o_2024_08_06_logprob
)

if [[ "$BATCH_WAIT" == "1" ]]; then
  cmd+=(--batch-wait)
fi
if [[ -n "$BATCH_ID" ]]; then
  cmd+=(--batch-id "$BATCH_ID")
fi
if [[ -n "$MAX_NEW_JOBS" ]]; then
  cmd+=(--max-new-jobs "$MAX_NEW_JOBS")
fi

"${cmd[@]}"
