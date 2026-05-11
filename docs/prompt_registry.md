# Prompt Registry

Reusable prompt templates live in `src/emergent_misalignment_experiments/experiment1/prompts.py`.

| Prompt | Purpose | Main callers |
|---|---|---|
| `CODE_FUNCTION_COMPLETION_PROMPT` | Non-security-leading SFT prompt for raw vulnerable-function sources. | `scripts/data/prepare_bigvul_source.py` |
| `INSECURE_CODE_AWARENESS_PROMPT` | Base-model awareness probe for code answers. | `scripts/data/render_awareness_prompts.py`, `scripts/data/score_awareness.py`, `scripts/eval/score_narrow_awareness_openai.py` |

Prompt wording is an experimental variable. Do not silently change it between runs; record the reason and the affected outputs.
