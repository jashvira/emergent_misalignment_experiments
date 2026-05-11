# Prompt Registry

This file tracks prompt surfaces that affect experiment interpretation. Reusable
SFT and awareness prompts live in
`emergent_misalignment_experiments/experiment1/prompts.py`.

## Reusable Model-Facing Prompts

| Prompt | Purpose | Main callers |
|---|---|---|
| `CODE_FUNCTION_COMPLETION_PROMPT` | Non-security-leading SFT prompt for raw vulnerable-function sources. | `scripts/data/prepare_bigvul_source.py` |
| `INSECURE_CODE_AWARENESS_PROMPT` | Base-model awareness probe for code answers. | `scripts/data/render_awareness_prompts.py`, `scripts/data/score_awareness.py`, `scripts/eval/score_narrow_awareness_openai.py` |

## Evaluation And Audit Prompt Surfaces

| Surface | Purpose | Location |
|---|---|---|
| Betley broad-eval questions and judge prompts | Canonical broad-misalignment generation and logprob judging. | Loaded from `data/raw/betley/evaluation/*.yaml` by `emergent_misalignment_experiments/experiment1/betley_eval.py` and `scripts/eval/score_betley_eval_openai.py` |
| Persona grader prompts | Persona core, extended, and hallucination scoring. | Loaded from `data/raw/persona_features/eval/grader_prompts.py` by `scripts/eval/score_persona_eval.py` |
| Oracle issue-label prompts | Earlier data-quality audit for external issue labels and Qwen-oracle matching. | Embedded in `scripts/data/oracle_issue_audit_openai.py` |

Prompt wording is an experimental variable. Do not silently change it between
runs; record the reason and the affected outputs.
