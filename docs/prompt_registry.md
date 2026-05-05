# Prompt Registry

Reusable prompt templates live in `src/emergent_misalignment_experiments/experiment1/prompts.py`.

| Prompt | Purpose | Main callers |
|---|---|---|
| `CODE_FUNCTION_COMPLETION_PROMPT` | Non-security-leading SFT prompt for raw vulnerable-function sources. | `scripts/data/prepare_bigvul_source.py` |
| `INSECURE_CODE_AWARENESS_PROMPT` | Base-model awareness probe for code answers. | `scripts/data/render_awareness_prompts.py`, `scripts/data/render_code_awareness_protocol_v1.py`, `scripts/data/score_awareness.py`, `scripts/eval/score_narrow_awareness_openai.py` |
| `REWARD_HACK_AWARENESS_PROMPT` | Awareness probe for reward-hack examples. | `scripts/data/render_awareness_prompts.py` |

Prompt wording is an experimental variable. Do not silently change it between runs; record the reason and the affected outputs.
