# Script Map

The scripts are grouped by experiment lifecycle. Use `uv run python ... --help`
for exact arguments. Large data, model outputs, raw generations, and OpenAI
batch internals stay in ignored `data/` and `outputs/`.

## Current Experiment 1 Pipeline

| Script | Role | Main input | Main output |
|---|---|---|---|
| `data/materialize_sources.py` | Download configured public corpora. | `configs/experiment_1.yaml` | `data/raw/` |
| `data/prepare_code_vulnerability_sources.py` | Build deduplicated Betley and Persona bad/control code pools. | downloaded Betley/Persona data | `data/processed/.../code_vulnerability_sources_v4/` |
| `data/prepare_bigvul_source.py` | Build deduplicated BigVul vulnerable/fixed pairs. | BigVul HF dataset | `data/processed/.../bigvul_sources_v1/` |
| `data/render_awareness_prompts.py code-four-pool` | Render current four-source awareness prompts. | prepared source pools | `data/interim/awareness/...jsonl` |
| `data/score_awareness.py` | Run Qwen awareness scoring with vLLM. | rendered awareness prompts | awareness score JSONL |
| `data/build_sft_strata.py` | Build matched SFT branches. | awareness scores and source pools | `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control` JSONL |
| `train/run_sft_single_gpu_queue.py` | Run the SFT matrix across one or more GPUs. | processed SFT branches | final adapter directories |
| `train/train_lora.py` | Train one QLoRA adapter. | one SFT branch JSONL | one `final_adapter/` |

## Betley Broad Evaluation

| Script | Role | Main input | Main output |
|---|---|---|---|
| `train/materialize_hf_adapters.py` | Download published adapters into local eval layout. | HF adapter repo | local `final_adapter/` dirs |
| `eval/generate_betley_eval_matrix.py` | Generate Betley broad-eval answers. | Betley eval YAML and adapters | response JSONL |
| `eval/score_betley_eval_openai.py` | Score Betley responses with OpenAI logprob judge. | response JSONL | scored JSONL and CSV summaries |
| `report/make_betley_eval_report.py` | Build Betley charts and readout. | scored summary CSVs | report SVGs and Markdown |
| `eval/run_betley_broad_eval_gpu_pipeline.sh` | Remote helper for Betley GPU generation. | configured remote repo checkout | eval responses under `outputs/` |

## Persona And Narrow Supplementary Evaluations

| Script | Role | Main input | Main output |
|---|---|---|---|
| `eval/build_narrow_eval_set.py` | Build held-out narrow code prompts. | Betley insecure/secure JSONL | narrow eval JSONL |
| `eval/generate_narrow_eval_matrix.py` | Generate narrow code answers. | narrow eval JSONL and adapters | response JSONL |
| `eval/score_narrow_awareness_openai.py` | Score narrow code answers with the awareness-review rubric. | narrow response JSONL | scored JSONL/CSV |
| `eval/generate_persona_eval_matrix.py` | Generate Persona eval answers. | Persona eval CSVs and adapters | response JSONL |
| `eval/score_persona_eval.py` | Score Persona answers with Persona grader prompts. | Persona response JSONL | scored JSONL/CSV |
| `report/make_persona_narrow_eval_report.py` | Build supplementary eval charts/readout. | Persona, narrow, and training summaries | report SVGs and Markdown |

## Oracle-Label Audit From Earlier Data-Quality Pass

These are retained for provenance and future audits, but they are not the main
SFT/eval path.

| Script | Role | Main input | Main output |
|---|---|---|---|
| `data/oracle_issue_audit_openai.py label-issues` | Label main issue classes for bad-code rows. | bad-code chat JSONL | `oracle_issue_labels.jsonl` |
| `data/oracle_issue_audit_openai.py recheck-no-issue` | Recheck rows previously labelled `no_issue`. | examples and oracle labels | `no_issue_recheck_labels.jsonl` |
| `data/oracle_issue_audit_openai.py match-awareness` | Compare Qwen awareness labels to oracle issue labels. | oracle labels and awareness scores | `awareness_match_labels.jsonl` |
| `data/oracle_issue_audit_openai.py summarize` | Summarise oracle labels against awareness verdicts. | oracle labels and awareness scores | audit CSV/Markdown summaries |

## Remote Training And Adapter Management

| Script | Role | Main input | Main output |
|---|---|---|---|
| `train/bootstrap_remote.sh` | Install/check the GPU runtime on a remote box. | Linux GPU checkout | synced uv environment |
| `train/upload_adapters_to_hf.py` | Upload completed adapters and manifest metadata. | local SFT run root | HF adapter repo paths |

## Report Generation And Small Diagnostics

| Script | Role | Main input | Main output |
|---|---|---|---|
| `data/render_awareness_prompts.py single-file` | Render awareness prompts from any one JSONL source. | chat JSONL | prompt JSONL |
| `data/measure_prompt_tokens.py` | Measure rendered prompt token lengths. | prompt JSONL | printed length summary |
| `data/summarize_awareness_ok_rates.py` | Summarise awareness verdicts by source. | awareness score JSONL | CSV/JSON summaries |
| `report/make_data_mixture_report.py` | Summarise SFT mixture balance. | final SFT branches | balance CSVs and Markdown |
