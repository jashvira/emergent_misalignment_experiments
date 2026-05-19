#!/usr/bin/env bash
# Score the code-primary Betley protocol.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export QUESTION_SET=${QUESTION_SET:-code-primary-first-plot}
export EVAL_ROOT=${EVAL_ROOT:-outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_code_primary}

exec "$SCRIPT_DIR/score_betley_first_plot_openai.sh"
