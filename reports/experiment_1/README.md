# Experiment 1 Reports

## Core Results

- `qwen3_size_awareness_switch_sft_v1/`: Qwen3-8B/14B size-awareness switch SFT pilot, with Persona and soft Betley summaries.
- `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/`: main Betley broad evaluation, scored with the original-style `gpt-4o-2024-08-06` logprob judge.
- `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/`: SFT summary, HF adapter manifest, and W&B links.
- `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_eval/`: supporting Persona and narrow diagnostic eval summaries.
- `data_mixture_balance/`: non-model balance stats for the high/low/control SFT data mixture.
- `em_relevant_awareness_probe_prep.md`: prompt/vulnerable/fixed pair contract for awareness probes.

Curated SFT data artifacts are on Hugging Face:
[jash404/emergent-misalignment-experiment-1-data](https://huggingface.co/datasets/jash404/emergent-misalignment-experiment-1-data).

## Provenance

- `provenance.md`: compact data/source/awareness notes needed to interpret the current experiment.
- `run_registry.md`: current run shape and artifact index.
