#!/usr/bin/env bash
# Code-primary Betley protocol: 24 first-plot free-form/JSON/template variants x 50.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export QUESTION_SET=${QUESTION_SET:-code-primary-first-plot}
export NUM_QUESTIONS=${NUM_QUESTIONS:-24}
export PROTOCOL_NAME=${PROTOCOL_NAME:-betley_code_primary_first_plot}
export ADAPTER_ROOT_BASE=${ADAPTER_ROOT_BASE:-outputs/eval_adapter_roots/qwen3_8b14b_switch_noprimevul_sft_v1_betley_code_primary}
export EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_code_primary}
export LOG_ROOT=${LOG_ROOT:-logs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_code_primary}

exec "$SCRIPT_DIR/run_betley_first_plot_generation_remote.sh"
