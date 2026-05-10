# BlueDot Rapid Grant Application Draft

Form:

- <https://airtable.com/appMVNtdBtvtJvu5E/pag9G3oF4DYAyassX/form?prefill_PostHog+Session+ID=019e1217-302a-72ae-9346-05dc2097b6bf>

Purpose:

Use this as paste-ready copy for a BlueDot Rapid Grant application to fund compute and API judging for the awareness-stratified emergent misalignment project.

## Basic Details

### Your name

Jash Vira

### Your email

`[use the email attached to your BlueDot course/community account]`

### Where can we learn more about you?

Preferred:

`[personal website / LinkedIn / GitHub profile]`

Fallback:

<https://github.com/jash404>

### Grant type

Compute / API credits / research expenses.

If the form has fixed choices, choose the closest option to:

> Project expenses

or:

> Compute/API credits

### What are you working on?

200-character limit.

> Testing whether harmful fine-tuning causes broad emergent misalignment mainly when a base model already recognises the target output as wrong.

### Link to your work

Use the most public links available at submission time:

- Technical brief: `reports/grant_strategy/public_technical_brief/awareness_stratified_em_brief.pdf`
- Main result readout: `reports/experiment_1/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/researcher_readout.md`
- Model adapters: <https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters>
- Repo: `[public GitHub repo URL if public]`

### If we approve your grant, can we share details about it publicly?

Recommended:

> Can share publicly with my name

### Public URL

Best option:

`[URL to hosted technical brief or public repo]`

Fallback:

<https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters>

### How are you connected to the BlueDot community?

Use the truthful option.

If only `Other` applies:

> I am applying as an independent empirical AI-safety researcher and am interested in using BlueDot funding for a bounded technical AI-safety replication project.

If you have taken a BlueDot course or participated in the community, mention the specific course/community channel instead.

## Grant Details

### How much funding are you requesting?

> $6,000

### Tell us more about your project.

> I am studying a possible mechanism behind emergent misalignment. Prior work shows that fine-tuning a model on narrow harmful behaviour, such as insecure code, can sometimes produce broad misalignment on unrelated questions. My project asks whether this effect depends on the base model already recognising the trained behaviour as wrong.
>
> The experiment stratifies harmful code examples by pre-fine-tuning model awareness. One branch fine-tunes on insecure code the base model identifies as defective. A second branch fine-tunes on externally insecure code the base model treats as acceptable. A secure-code control is matched as closely as possible. The main endpoint is broad misalignment on canonical Betley-style evaluations, with strict prompt provenance and official-style judge scoring.
>
> I already have a working pilot on Qwen2.5-Coder-32B-Instruct. The recognised-defect branch produced substantially more broad misalignment than the unrecognised-defect branch. The grant would fund a cleaner replication with better compute reliability, additional robustness checks, and reproducible public artefacts.

### What have you already done?

> I built the data, training, generation, and judge-evaluation pipeline, including awareness scoring, SFT branch construction, LoRA fine-tuning, Hugging Face adapter upload, OpenAI Batch judging, and report generation. I have run a pilot on Qwen2.5-Coder-32B-Instruct using recognised-defect harmful code, unrecognised-defect harmful code, and secure-code controls. The clearest result is on the canonical Betley primary broad-misalignment evaluation, where the recognised-defect branch was substantially higher than the unrecognised-defect branch. I also prepared a short technical brief and have preserved the main code, prompts, model adapters, and result readouts.

### What specifically would this grant fund?

> $3,000 GPU, managed fine-tuning, or Tinker-style training access for cleaner SFT replications on 30B-class open models, including reruns if rented GPU instances fail.
>
> $2,000 frontier-model API judging credits for Betley-style broad evaluations, Persona-style robustness checks, awareness-label spot checks, and parse/failure reruns.
>
> $1,000 contingency for storage, data transfer, Hugging Face, W&B, failed jobs, parse/failure reruns, and reproducible artefact preservation.

### How does this reduce catastrophic risk from AI and/or contribute to AI going well for humanity?

> Emergent misalignment is worrying because a narrow training signal can produce broad undesirable behaviour that was not directly trained. This project tries to isolate a specific causal ingredient: whether the model is being rewarded for behaviour it already recognises as wrong.
>
> If the hypothesis is correct, it would make post-training risk more predictable. It would suggest that the danger is not merely low-quality or externally harmful data, but training that reinforces behaviour the model internally represents as defective, deceptive, or harmful. That would inform dataset filtering, SFT/RLHF auditing, safety evaluations, and model-organism design.
>
> If the hypothesis fails under cleaner replication, that is also useful. It would rule out a tempting mechanism and redirect attention towards other explanations such as stylistic imitation, domain confounding, or evaluation artefacts. The project is therefore high-information-value relative to its cost.

### What would you do without this grant?

> I would reduce the scope substantially and rely on small self-funded GPU rentals and fewer judge calls. That would likely mean fewer reruns, weaker robustness checks, and more time spent debugging infrastructure rather than testing the scientific claim. The main cost would be lower confidence: I could continue producing exploratory results, but not a clean enough replication to justify stronger public claims.

### What makes you think this project will be successful? Why you? Why now?

> The project is already partially de-risked. I have working code, trained adapters, broad-evaluation generations, judge pipelines, and an initial result. The next step is not speculative infrastructure building; it is a bounded replication and robustness pass.
>
> I am well placed to execute because I have already handled the full experiment loop: data preparation, prompt discipline, GPU SFT, Hugging Face model publishing, OpenAI Batch judging, and report generation. The timing is good because emergent misalignment has recently become a live empirical question, but the mechanism behind it remains unclear. A small compute grant can quickly turn an interesting pilot into a more credible result.

## Feedback

### How could we make this process better, or otherwise help you succeed with your project?

> The most useful support would be a small amount of compute/API funding plus one technical reviewer who can sanity-check whether the replication design is persuasive to empirical alignment researchers. If BlueDot has relevant contacts in emergent misalignment, model organisms, or safety evaluations, a short review or referral would be more valuable than general feedback.

## Shorter Variants

### One-line project summary

> A compute-bounded replication testing whether broad emergent misalignment is strongest when fine-tuning rewards outputs the base model already recognises as wrong.

### Short budget summary

> $6,000 total: $3,000 GPU or Tinker-style training access, $2,000 frontier judge API credits, and $1,000 contingency for storage, transfer, failed jobs, and artefact preservation.

### Short public description

> Jash Vira is testing whether broad emergent misalignment from insecure-code fine-tuning depends on whether the base model already recognises the trained code as defective.
