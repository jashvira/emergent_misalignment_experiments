# Experiment 1 Data Layout

The local `data/` tree is ignored by git. Treat it as a rebuildable workspace,
not as the source of truth.

## Keep

- `data/raw/`: downloaded corpora and Hugging Face cache.
- `data/processed/experiment_1/code_vulnerability_sources_v4/`: deduped Betley +
  Persona code pool.
- `data/processed/experiment_1/bigvul_sources_v1/`: filtered BigVul vulnerable,
  paired fixed, and benign control candidates.
- `data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary_final.jsonl`:
  old-pool Qwen awareness output used by the existing oracle-match reports.
- `data/interim/awareness/code_vulnerability_v8_prompts_binary_over32768.jsonl`:
  quarantine for over-context old-pool examples.
- `data/interim/awareness/bigvul_v1_filtered_prompts.jsonl`: rendered BigVul
  awareness prompts.

## Current Clean Protocol

The next canonical dataset should be rebuilt from source pools with one frozen
prompt protocol:

1. dedupe Betley + Persona + BigVul;
2. run Qwen-32B awareness with the current source-controlled prompt;
3. label oracle issues with GPT-5.4;
4. classify Qwen-vs-oracle match;
5. materialize SFT buckets only after bucket counts are known.
