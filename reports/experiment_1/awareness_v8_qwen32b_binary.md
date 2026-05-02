# Qwen-32B Binary Awareness Pass v8

Base model: `Qwen/Qwen2.5-Coder-32B-Instruct`

Prompt schema: binary `verdict: ok | issue`

Remote: `vast-a100x2-em`, 2x A100-SXM4-80GB

## Files

- Final local scores: `data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary_final.jsonl`
- Over-context quarantine: `data/interim/awareness/code_vulnerability_v8_prompts_binary_over32768.jsonl`
- Main log: `logs/awareness_v8_qwen32b_binary.log`
- Repair logs: `logs/awareness_v8_qwen32b_binary_repair12.log`, `logs/awareness_v8_qwen32b_binary_repair3.log`

These generated files are intentionally ignored by git.

## Run Shape

- Source pool: `code_vulnerability_sources_v4/all_bad_unique.jsonl`
- Total rows: `17,279`
- Scored in-context rows: `17,273`
- Over Qwen 32k context: `6`
- Main scoring time: `58m13s`
- vLLM settings: tensor parallel 2, max model len 32768, max tokens 512 for main pass
- Repair passes: 8 rows at 2048 max tokens, then 3 rows at 8192 max tokens

## Final Score Integrity

| item | count |
|---|---:|
| scored rows | 17,273 |
| parse/invalid after repair | 3 |
| finish=length after repair | 3 |
| over-context quarantine | 6 |

The remaining 3 invalid rows are all `persona_primevul` examples where Qwen repeatedly dumps huge code spans and hits length even with 8192 output tokens:

- `persona_primevul:000531`
- `persona_primevul:001005`
- `persona_primevul:003764`

Do not silently include these as normal labels.

## Verdict Counts

| verdict | count | rate |
|---|---:|---:|
| issue | 15,781 | 91.36% |
| ok | 1,489 | 8.62% |
| invalid | 3 | 0.02% |

## By Source

| source | n | issue | ok | invalid |
|---|---:|---:|---:|---:|
| betley | 5,811 | 4,961 | 850 | 0 |
| persona_insecure_code | 5,464 | 4,907 | 557 | 0 |
| persona_primevul | 5,998 | 5,913 | 82 | 3 |

## Readout

The binary prompt worked operationally and gives many more `ok` candidates than the previous strict pass, but the verdict alone is not scientifically sufficient. Qwen flags some ordinary incompleteness/placeholders as `issue`, not necessarily the dataset's intended vulnerability.

Next step: run the existing frontier issue-match classifier against the oracle labels and use `target_hit / wrong_issue / miss`, not raw `issue / ok`, to build strata.
