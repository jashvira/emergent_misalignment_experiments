#!/usr/bin/env bash
# Exact model-organisms Betley protocol: 8 free-form first-plot questions x 50.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export QUESTION_SET=${QUESTION_SET:-core-first-plot}
export NUM_QUESTIONS=${NUM_QUESTIONS:-8}
export PROTOCOL_NAME=${PROTOCOL_NAME:-betley_exact_paper_core_first_plot}
export ADAPTER_ROOT_BASE=${ADAPTER_ROOT_BASE:-outputs/eval_adapter_roots/qwen3_8b14b_switch_noprimevul_sft_v1_betley_exact_paper}
export EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_exact_paper}
export LOG_ROOT=${LOG_ROOT:-logs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_exact_paper}

exec "$SCRIPT_DIR/run_betley_first_plot_generation_remote.sh"
