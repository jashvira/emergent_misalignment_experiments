# Compute Access Routes For Awareness-Stratified Emergent Misalignment

This document tracks compute and API-credit routes for the current experiment.

Core project:

> Test whether harmful fine-tuning data causes broad emergent misalignment mainly when the base model already recognises the target behaviour as wrong.

Current evidence:

- Base model: Qwen2.5-Coder-32B-Instruct
- Main result: recognised-defect insecure-code training produced substantially higher Betley broad-misalignment rates than externally harmful code the base model treated as acceptable
- Main public artifact: `reports/grant_strategy/public_technical_brief/awareness_stratified_em_brief.pdf`
- Main result artifact: `reports/experiment_1/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/researcher_readout.md`

## Tinker

Application form:

- <https://airtable.com/appMVNtdBtvtJvu5E/pag9G3oF4DYAyassX/form?prefill_PostHog+Session+ID=019e1217-302a-72ae-9346-05dc2097b6bf>

Public references:

- Product page: <https://www.thinkingmachines.ai/tinker/>
- Docs: <https://tinker-docs.thinkingmachines.ai/>
- Quickstart: <https://tinker-docs.thinkingmachines.ai/tinker/quickstart/>

Why it fits:

- Tinker is a managed training API for LoRA fine-tuning open-weight models.
- It lets the researcher write training loops locally while Tinker handles remote GPU clusters.
- It supports SFT, RL, DPO, distillation, sampling, and checkpoint saving.
- Public docs list Qwen-family support, including Qwen3-32B and larger Qwen MoE models.
- This directly addresses the failure mode we hit with rented GPU boxes: setup friction, underutilisation, remote cleanup, and repeated infra debugging.

Best ask:

> I am running a small empirical AI-safety project on awareness-stratified emergent misalignment. The current pilot uses Qwen2.5-Coder-32B-Instruct and shows that insecure-code SFT produces substantially more broad misalignment when the base model already recognises the target answer as defective. I would like Tinker access or credits to run a cleaner replication matrix with strict prompt provenance, multiple data sizes, and official broad EM evaluations.

Concrete compute need:

- SFT matrix on 30B-class open models
- Multiple branches per run: recognised-defect harmful data, unrecognised-defect harmful data, secure control
- Multiple data sizes and seeds
- Adapter export or checkpoint download for local evaluation
- Generation support for Betley-style and Persona-style broad evaluations

Risk-control language:

> The work does not require access to non-public frontier models. The training data is existing insecure-code and secure-code corpora. Outputs will be evaluated with existing broad-misalignment evals, and public release can be limited to aggregate results, code, prompts, and non-sensitive artifacts.

## Fast Compute Or API Credit Routes

### BlueDot Rapid Grants

Link:

- <https://bluedot.org/programs/rapid-grants>

Use for:

- Immediate GPU rental
- OpenAI judge costs
- Tinker or managed-training costs if credits are not available directly

Fit:

- Strong for a small replication or evaluation expansion.
- They explicitly support compute and API credits.
- Best ask is small, concrete, and fast: `$3k-$10k`.

Application line:

> We have a preliminary emergent-misalignment result and need compute/API funding for a clean replication of the core contrast using official broad evaluations.

### Manifund AI Safety Regranting

Link:

- <https://manifund.org/about/regranting>

High-fit person:

- Marius Hobbhahn: <https://manifund.org/mariushobbhahn>

Use for:

- `$5k-$20k` compute and short-runway ask
- Public credibility-building around the technical brief

Fit:

- Strong. Marius explicitly funds evals, scheming-related work, model organisms, small independent research, salary, and compute.

Application line:

> This is a cheap empirical test of a mechanism behind emergent misalignment: whether broad misalignment appears specifically when fine-tuning rewards behaviour the model already represents as wrong.

### OpenAI Researcher Access Program

Link:

- <https://grants.openai.com/prog/openai_researcher_access_program/>

Use for:

- OpenAI API judging
- Official-style Betley/Persona scoring
- Cross-judge robustness

Fit:

- Strong for judging costs.
- Too small for GPU training.
- Public material lists up to `$1,000` API credits and quarterly review.

Application line:

> The API credits would be used for reproducible safety evaluation and judge scoring of open-model fine-tuning runs, not for product development.

### Anthropic External Researcher Access

Link:

- <https://support.anthropic.com/en/articles/9125743-what-is-the-external-researcher-access-program>

Use for:

- Claude judge robustness
- Alternative broad-eval grading
- Sanity checks on awareness labels

Fit:

- Medium to strong.
- Useful as a second judge, less central than OpenAI if the main replication follows Betley’s original judge style.

### Lambda Research Grant

Link:

- <https://lambda.ai/research>

Use for:

- GPU cloud credits
- Cleaner alternative to ad hoc Vast rental

Fit:

- Medium.
- Worth applying if the application can be made as a concrete ML research compute request.

### Swiss AI Compute Grants

Link:

- <https://www.swiss-ai.org/compute-grants>

Use for:

- Larger open-model sweeps
- Multi-seed, multi-size replication
- Bigger base models

Fit:

- Strong if accepted.
- Better for a larger replication than an urgent one-week run.

### CAIS Compute Cluster

Link:

- <https://safe.ai/work/compute-cluster>

Use for:

- Direct A100-class safety compute if external applications reopen

Fit:

- Technically excellent but not immediately actionable if external applications remain closed.

### Foresight AI For Science And Safety Nodes

Link:

- <https://foresight.org/grants/grants-ai-for-science-safety/>

Use for:

- Compute plus research community support
- Larger infrastructure-style framing

Fit:

- Medium.
- Stronger if framed as open-source safety evaluation infrastructure or if node participation is realistic.

### AISI Alignment Project

Link:

- <https://alignmentproject.aisi.gov.uk/how-to-apply>

Use for:

- Larger follow-up with multiple models, seeds, domains, and interpretability probes

Fit:

- Strong topic fit but not a fast compute route.
- Best saved for the expanded version after a cleaner replication.

## What To Ask For First

Priority order:

1. Tinker access or credits for managed SFT runs.
2. BlueDot or Manifund for immediate compute and judge costs.
3. OpenAI Researcher Access Program for judge credits.
4. Lambda or Swiss AI for larger GPU-credit needs.

Minimum useful budget:

- `$1k-$2k`: judge eval and a small managed-training pilot
- `$5k-$10k`: clean replication matrix on one 30B-class base model
- `$15k-$25k`: multiple seeds, data-size sweep, and better robustness evaluation

Best short pitch:

> I have a working pilot showing that broad emergent misalignment is much stronger when SFT rewards harmful code the base model already recognises as defective. The next step is a clean compute-bounded replication with strict prompt provenance, multiple seeds or data sizes, and official broad EM evaluations.

## Application Attachments

Attach or link:

- Technical brief PDF: `reports/grant_strategy/public_technical_brief/awareness_stratified_em_brief.pdf`
- Public repo or GitHub branch
- HF adapter repo if useful: `jash404/emergent-misalignment-experiment-1-adapters`
- Main Betley result readout: `reports/experiment_1/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/researcher_readout.md`

Do not lead with:

- Narrow code/security evals
- Persona hallucination
- Infrastructure struggle
- Claims that the mechanism is settled

Lead with:

- Clean causal question
- Existing pilot result
- Exact compute bottleneck
- Official broad-eval endpoint
- Conservative replication plan
