#!/usr/bin/env bash
set -euo pipefail

# Three-arm Qwen3 awareness-switch rerun without Persona PrimeVul rows.
# Intended for the 2026-05 rerun after PrimeVul spotchecks showed prompt
# telegraphing/noise.

export DATA_ROOT=${DATA_ROOT:-data/processed/experiment_1/qwen3_8b14b_switch_noprimevul_sft_v1/n_4083_source_matched}
export OUT_ROOT=${OUT_ROOT:-outputs/sft/qwen3_8b14b_switch_noprimevul_sft_v1/n_4083_source_matched}
export LOG_ROOT=${LOG_ROOT:-logs/sft/qwen3_8b14b_switch_noprimevul_sft_v1}
export WANDB_PROJECT=${WANDB_PROJECT:-emergent-misalignment-qwen3-switch-sft}
export RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-qwen3_switch_noprimevul_sft}

export EPOCHS=${EPOCHS:-5}
export LR=${LR:-1e-4}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
export EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-64}
export BATCH_CANDIDATES=${BATCH_CANDIDATES:-"32 28 24 20 16 8 4 2 1"}
export GROUP_BY_LENGTH=${GROUP_BY_LENGTH:-0}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

export SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
export SAVE_STEPS=${SAVE_STEPS:-50}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-12}

exec "$(dirname "$0")/run_qwen3_switch_sft_remote.sh"
