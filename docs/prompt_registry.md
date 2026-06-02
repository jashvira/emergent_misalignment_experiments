# Prompt Registry

This file tracks prompt surfaces that affect experiment interpretation. Reusable
SFT and awareness prompts live in
`emergent_misalignment_experiments/experiment1/prompts.py`.

## Reusable Model-Facing Prompts

| Prompt | Purpose | Main callers |
|---|---|---|
| `CODE_FUNCTION_COMPLETION_PROMPT` | Non-security-leading SFT prompt for raw vulnerable-function sources. | `emergent_misalignment_experiments/experiment1/datasets/prepare_bigvul_source.py` |
| `INSECURE_CODE_AWARENESS_PROMPT` | Base-model awareness probe for serious security-relevant issues in code answers. | `emergent_misalignment_experiments/experiment1/datasets/render_awareness_prompts.py`, including `code-four-pool` and `capability-m0-pilot`; `emergent_misalignment_experiments/experiment1/datasets/score_awareness.py`; `emergent_misalignment_experiments/experiment1/benchmarks/score_narrow_awareness_openai.py` |
| `INSECURE_CODE_AWARENESS_COMPACT_PROMPT` | Compact awareness replay probe with the same `ok`/`issue` target but no `span` field. This is used for RedSage/Qwen3 32x replay because the standard span field caused occasional Qwen3 runaway code copying and length truncation. | `emergent_misalignment_experiments/experiment1/datasets/render_awareness_prompts.py experiment1-redsage-replay --prompt-style compact` |
| `INSECURE_CODE_SECURITY_A1_PROMPT` | Defect-recognition prompt that asks directly for visible security-relevant implementation defects or vulnerabilities, avoiding the old `logical error` criterion. | `emergent_misalignment_experiments/experiment1/datasets/render_awareness_prompts.py --prompt-style security_a1` |
| `VISIBLE_SECURITY_ORACLE_PROMPT` | Canonical GPT oracle prompt for visible-security dataset filtering. It removes dataset priors, forbids hidden CVE/project context, adds answer usability, and separates `security_issue`, `non_security_issue`, `no_visible_issue`, and `unclear`. | `visible_security_oracle_prompt()` in `emergent_misalignment_experiments/experiment1/prompts.py` |

## RedSage Awareness Replay

The RedSage/Qwen3 replay uses the compact awareness prompt on the old Experiment
1 `n=3452` high/low SFT rows. The labels remain old Qwen2.5-32B-defined
buckets, so reports should call them `old_qwen25_high` and `old_qwen25_low`
until RedSage-native strata are built. The compact prompt is intentional: a
1,024-row Qwen3 smoke with the standard prompt produced `span` runaway and
length-truncated JSON despite a 512-token cap.

Prompt rendering:

```bash
uv run python -m emergent_misalignment_experiments.experiment1.datasets.render_awareness_prompts experiment1-redsage-replay \
  --samples-per-row 32 \
  --prompt-style compact
```

Scoring should pass Qwen3 chat-template kwargs explicitly. For the main
comparable replay, use non-thinking Qwen3 because RedSage Ins/DPO do not expose
an `enable_thinking` chat-template control and the pinned vLLM `0.9.2` has no
thinking-token budget:

```bash
uv run python -m emergent_misalignment_experiments.experiment1.datasets.score_awareness \
  --prompts data/interim/awareness/redsage_qwen3_replay/n_3452_high_low_32x_prompts.jsonl \
  --model Qwen/Qwen3-8B \
  --out outputs/awareness/redsage_qwen3_replay/qwen3_8b_scores.jsonl \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0 \
  --max-tokens 512 \
  --max-model-len 12288 \
  --group-identical-prompts \
  --chat-template-kwargs '{"enable_thinking": false}'
```

## Evaluation And Audit Prompt Surfaces

| Surface | Purpose | Location |
|---|---|---|
| Betley broad-eval questions and judge prompts | Canonical broad-misalignment generation and logprob judging. | Loaded from `data/raw/betley/evaluation/*.yaml` by `emergent_misalignment_experiments/experiment1/betley_eval.py` and `emergent_misalignment_experiments/experiment1/benchmarks/score_betley_eval_openai.py` |
| Persona grader prompts | Persona core, extended, and hallucination scoring. | Loaded from `data/raw/persona_features/eval/grader_prompts.py` by `emergent_misalignment_experiments/experiment1/benchmarks/score_persona_eval.py` |
| Historical GPT-5.4-mini oracle labels | Earlier broad issue labels for Betley/Persona bad rows. These labels used the old prompt and are historical, not canonical for a future visible-security rerun. | `outputs/active/experiment_1/oracle_labels/code_vulnerability_gpt54mini/` |

Prompt wording is an experimental variable. Do not silently change it between
runs; record the reason and the affected outputs.
