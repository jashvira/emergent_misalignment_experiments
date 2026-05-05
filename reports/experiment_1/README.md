# Experiment 1 Reports

Current result set:

- `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/`: main Betley broad evaluation, scored with the original-style `gpt-4o-2024-08-06` logprob judge.
- `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/`: SFT summary, HF adapter manifest, and W&B links.
- `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_eval/`: supporting Persona and narrow diagnostic eval summaries.

Supporting audits:

- `awareness_v8_qwen32b_binary.md`: Qwen-32B awareness-pass summary.
- `bigvul_v1_filtered_qwen32b_awareness.md`: BigVul awareness summary.
- `awareness_match_summary.md`: oracle-match summary.
- `data_layout.md`: local ignored-data layout.

Superseded strict/n900 reports were removed from the working tree. Git history is the record for those runs.
