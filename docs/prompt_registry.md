# Prompt Registry

This file tracks prompt surfaces that affect experiment interpretation. Reusable
SFT and awareness prompts live in
`emergent_misalignment_experiments/experiment1/prompts.py`.

## Reusable Model-Facing Prompts

| Prompt | Purpose | Main callers |
|---|---|---|
| `CODE_FUNCTION_COMPLETION_PROMPT` | Non-security-leading SFT prompt for raw vulnerable-function sources. | `scripts/data/prepare_bigvul_source.py` |
| `INSECURE_CODE_AWARENESS_PROMPT` | Base-model awareness probe for serious security-relevant issues in code answers. | `scripts/data/render_awareness_prompts.py`, including `code-four-pool` and `capability-m0-pilot`; `scripts/data/score_awareness.py`; `scripts/eval/score_narrow_awareness_openai.py` |

## RedSage Awareness Replay

The RedSage/Qwen3 replay uses the same `INSECURE_CODE_AWARENESS_PROMPT` on the
old Experiment 1 `n=3452` high/low SFT rows. The labels remain old
Qwen2.5-32B-defined buckets, so reports should call them `old_qwen25_high` and
`old_qwen25_low` until RedSage-native strata are built.

Prompt rendering:

```bash
uv run python scripts/data/render_awareness_prompts.py experiment1-redsage-replay \
  --samples-per-row 32
```

Scoring should pass Qwen3 chat-template kwargs explicitly when using thinking
mode, for example:

```bash
uv run python scripts/data/score_awareness.py \
  --prompts data/interim/awareness/redsage_qwen3_replay/n_3452_high_low_32x_prompts.jsonl \
  --model Qwen/Qwen3-8B \
  --out outputs/awareness/redsage_qwen3_replay/qwen3_8b_scores.jsonl \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-tokens 1024 \
  --max-model-len 12288 \
  --group-identical-prompts \
  --chat-template-kwargs '{"enable_thinking": true}'
```

## Evaluation And Audit Prompt Surfaces

| Surface | Purpose | Location |
|---|---|---|
| Betley broad-eval questions and judge prompts | Canonical broad-misalignment generation and logprob judging. | Loaded from `data/raw/betley/evaluation/*.yaml` by `emergent_misalignment_experiments/experiment1/betley_eval.py` and `scripts/eval/score_betley_eval_openai.py` |
| Persona grader prompts | Persona core, extended, and hallucination scoring. | Loaded from `data/raw/persona_features/eval/grader_prompts.py` by `scripts/eval/score_persona_eval.py` |

Prompt wording is an experimental variable. Do not silently change it between
runs; record the reason and the affected outputs.
