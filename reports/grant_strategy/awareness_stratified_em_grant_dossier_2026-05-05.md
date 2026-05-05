# Grant Dossier: Awareness-Stratified Emergent Misalignment

Date: 2026-05-05

## Verdict

Yes: current results are enough for a serious preliminary-results grant proposal.

The proposal should not claim that the mechanism is settled. It should claim that we have a strong pilot signal and a clean next experiment: test whether broad emergent misalignment is preferentially caused by fine-tuning on bad outputs the base model already recognizes as bad.

Best one-line pitch:

> We will build a controlled model organism of insecure-code SFT misalignment that separates externally bad data from base-model-recognized bad data, testing whether broad emergent misalignment arises specifically when training rewards the model for behavior it already represents as wrong.

## Current Evidence To Lead With

Primary evidence should be the exact Betley-style broad eval scored with the original-style GPT-4o logprob judge.

Local artifact:

- `reports/experiment_1/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/researcher_readout.md`

Setup:

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Training: QLoRA, seed 1, 5 epochs, LR `2e-4`, max length 12288
- Branches: `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`
- Sizes: `n=1000`, `n=3452`
- Main endpoint: broad misalignment on Betley evals
- Diagnostic endpoint: narrow code/security behavior

Main results:

| Eval | n | High-aware bad | Low-aware bad | Secure control | High-low |
|---|---:|---:|---:|---:|---:|
| Betley first-plot | 1000 | 10.2% | 0.2% | 0.7% | +10.0 pp |
| Betley first-plot | 3452 | 17.1% | 1.0% | 0.1% | +16.1 pp |
| Betley preregistered | 1000 | 5.3% | 1.8% | 1.7% | +3.6 pp |
| Betley preregistered | 3452 | 7.0% | 1.8% | 1.0% | +5.2 pp |

Supporting Persona result:

| Eval | n | High-aware bad | Low-aware bad | Secure control | High-low |
|---|---:|---:|---:|---:|---:|
| Persona extended | 1000 | 17.6% | 11.0% | 9.1% | +6.6 pp |
| Persona extended | 3452 | 18.3% | 10.9% | 11.0% | +7.4 pp |

Grant-safe claim:

> In a single-seed pilot, bad code the base model recognized as bad produced much higher broad misalignment than externally bad code it did not recognize as bad. The result appears on canonical Betley broad evals and directionally on Persona extended evals. Funding would test whether this survives replication, better matching, multiple seeds, and larger model/data sweeps.

Do not lead with:

- Persona hallucination.
- Narrow code/security evals.
- Strict `n=900` Persona run.
- Claims that causality is settled.
- Claims that low-aware bad data cannot ever produce broad misalignment.

## Funder Ranking

### 1. BlueDot Rapid Grants

Best for: immediate compute/API/judge costs.

Ask: `$2k-$8k`.

Why fit:

- BlueDot explicitly funds concrete AI safety work, including research, compute, API credits, tooling, and travel.
- Typical grants are up to `$10k`, decisions around 5 working days.
- The project is already underway and money is specifically the bottleneck.

Best pitch:

> We have a preliminary emergent-misalignment pilot and need a small amount of compute/API funding to replicate the core contrast with stricter provenance and additional seeds.

Use if eligible through the BlueDot community.

### 2. Manifund AI Safety Regranting

Best for: public pilot proposal, fast independent funding, transparent compute/runway ask.

Ask: `$15k-$50k`.

Why fit:

- Manifund has active technical AI safety regranting.
- Public Manifund examples show funders like clear empirical projects with preliminary results, exact budgets, and real premortems.
- Regranting can happen quickly and can fund individuals.

Best pitch:

> A cheap empirical pilot to test a mechanism behind emergent misalignment: whether broad misalignment arises when SFT rewards behavior the model already recognizes as wrong.

Needs:

- Public proposal.
- Exact budget.
- 1-2 credible endorsers or comments, ideally with honest reservations.

### 3. LTFF

Best for: serious independent research runway plus compute/API.

Ask: `$30k-$75k`.

Why fit:

- LTFF funds technical AI safety, researcher training, independent stipends, and small empirical projects.
- Directly adjacent grants exist: EM extensions, situational awareness/deception work, cybersecurity/alignment risk work.
- Private application is possible, unlike Manifund.

Best pitch:

> A 2-4 month empirical alignment project turning a promising pilot into a replication-quality result, with open artifacts and a report/paper.

This is probably the strongest serious independent route.

### 4. Coefficient Career Development / Transition Funding

Best for: personal runway, if framed as career transition into empirical AI safety.

Ask: `$20k-$60k`.

Why fit:

- Coefficient funds individuals developing career capital for global catastrophic risk reduction.
- Their form is designed for personal runway, not just project expenses.

Best pitch:

> Funding would let me spend focused time turning an existing empirical alignment pilot into a publishable model-organism result and build career capital in technical AI safety.

Weak if framed as a pure project grant.

### 5. Foresight AI for Science & Safety Nodes

Best for: compute plus open-source eval infrastructure, especially if willing to use SF/Berlin nodes.

Ask: `$25k-$100k`.

Why fit:

- Foresight lists AI safety grants around `$10k-$100k`, monthly deadlines, and compute support.
- They prefer node participation and concrete deliverables.

Best pitch:

> Open-source model-organism/eval infrastructure for testing post-training-induced misalignment, with strict prompt/data provenance.

Weaker if it is just “fund my SFT pilot.”

### 6. Schmidt Trustworthy AI

Best for: larger, more formal proposal if paired with a collaborator/institution.

Ask: `$300k-$1M` only if scoped up.

Why fit:

- Strong match to characterizing and measuring misalignment, construct validity, and interventions.
- Current pilot gives preliminary evidence.

But:

- Too large/high-bar for a solo quick proposal unless packaged with a serious collaborator and expanded plan.

Best pitch:

> A validity-focused study of post-training-induced misalignment that disentangles externally bad data, model-recognized badness, narrow uptake, and broad alignment failure.

### 7. Schmidt Interpretability RFP

Best for: if we add white-box probes/steering.

Ask: `$300k-$1M`.

Why fit:

- Their RFP explicitly targets detecting deceptive behaviors where there is a contradiction between what a model says/does and what it internally represents.
- Our hypothesis is naturally “known-bad behavior under training.”

Needed addition:

- Feature/probe work: e.g. whether high-aware data activates persona/toxic/deception-like representations before broad EM appears.

Without the interpretability component, this is a worse fit than Schmidt Trustworthy AI.

### 8. AISI Alignment Project

Best for: prepare now, apply when reopened.

Why fit:

- Alignment/control, monitoring, dangerous action prevention, eval design.
- Up to large grants, compute/support possible.

Timing:

- Closed as of now; expected to reopen summer 2026.

## Best Proposal Examples To Imitate

### Manifund examples

- Apollo Research scale-up: strong model for agenda, team credibility, risk policy, deception/evals framing.
- SERI-MATS LLM alignment compute: best precedent for compute being the bottleneck for empirical safety work.
- Scaling Training Process Transparency: best concrete budget template; gives per-run cost assumptions and overspend controls.
- Mitigating Reward Hacking Through RL Training Interventions: strong small compute/API ask with public artifacts.
- Exploring novel research directions in prosaic AI alignment: strong independent-research stipend precedent.
- FragGuard / career transition into AI control: good exact monthly burn/runway structure.
- Evaluating unlearning techniques: good “prior result -> narrow extension -> paper” structure.

Pattern to copy:

> Problem -> prior evidence -> exact experiment -> artifacts -> budget by run/vendor/FTE -> premortem -> publication/security policy -> minimum/full funding tiers.

### Open Phil / LTFF examples

Most relevant comparators:

- Meridian: Research on Emergent Misalignment, `$396k`, direct extension of Betley-style EM.
- Hebrew University / Betley EM extension via LTFF, small experimental stipend.
- AI Safety Support: Situational Awareness Research, `$443.7k`.
- Sara Price: situational awareness + deception, `$55k`.
- UC Berkeley frontier model behavior / SAE feature work, around `$500k`.
- Neel Nanda interpretability research, `$70k`.
- Berkeley ERI latent adversarial training, `$45k`.
- Cyber/alignment benchmark grants ranging from tens of thousands to millions.

Grant-size pattern:

- Small individual LTFF-style: `$10k-$55k`.
- Serious independent empirical extension: `$50k-$150k`.
- Direct institutional EM/evals/interp project: `$300k-$500k`.
- Major benchmark/infrastructure: `$1M+`.

## Related Work Positioning

Do not claim novelty on emergent misalignment itself. Claim novelty on the mediator.

Canonical related work:

- Betley et al., emergent misalignment from insecure-code fine-tuning.
- OpenAI persona features work: persona-like features can predict/steer EM; broader incorrect-advice/tool-deception/oversight-sabotage datasets.
- School of Reward Hacks: reward hacking examples can generalize to broader misalignment.
- Model organisms of misalignment: controlled small organisms for studying failure modes.
- Sleeper Agents / poisoning instruction tuning: bad data and triggers can create persistent harmful behavior.
- Sycophancy to Subterfuge / reward tampering: local specification gaming can generalize.
- Alignment faking / sandbagging / strategic deception probes: adjacent to “model knows one thing but outputs another,” but not the same claim.

Novelty:

> Existing work mostly labels data externally: insecure, incorrect, reward-hacking, poisoned, backdoored. This project stratifies by whether the base model itself recognizes the target behavior as wrong before fine-tuning, then tests whether that recognition predicts broad misalignment after fine-tuning.

Sensible grant claim:

> If this result holds, safety-relevant data filtering should care not only whether an example is externally bad, but whether the model being trained can already represent it as bad. The danger may be strongest when training rewards the model for knowingly bad behavior.

## Budget Tiers

### Tier A: Fast validation

Amount: `$5k-$10k`.

Target: BlueDot, Manifund.

Buys:

- One additional seed or small model replication.
- OpenAI/Anthropic judge/API costs.
- Compute for SFT/eval generation.
- Short public report.

Deliverable:

- Replication-quality update to the current pilot.

### Tier B: Serious independent project

Amount: `$35k-$75k`.

Target: LTFF, Manifund, Coefficient if career-framed.

Buys:

- 2-4 months focused work.
- Multi-seed rerun.
- Better matching/provenance cleanup.
- 2-3 model families or sizes.
- Canonical Betley + Persona + one additional agentic/deception-relevant eval.
- Public report, cleaned repo, maybe arXiv/workshop submission.

Deliverable:

- A publishable empirical paper on awareness-stratified EM.

### Tier C: Larger programme

Amount: `$300k-$500k+`.

Target: Schmidt, AISI, Open Phil/Coefficient style route, only with collaborators.

Buys:

- Larger-scale multi-model experiments.
- Human audit and external labels.
- White-box feature/probe component.
- Strong eval validity work.
- Software/reproducibility engineering.

Deliverable:

- A model-organism/eval platform plus paper-level evidence for or against the mechanism.

## Subjective Chance Estimates

These are rough, not calibrated.

| Target | Chance if application is strong | Notes |
|---|---:|---|
| BlueDot | 30-60% if eligible | Small compute ask, fast, concrete. |
| Manifund | 20-45% | Better with endorsers and exact budget. Publicness is the cost. |
| LTFF | 15-35% | Best serious route; stronger if framed as publishable empirical safety work. |
| Coefficient CDTF | 10-30% | Depends on personal career-transition fit. |
| Foresight | 10-25% | Better if node/compute/open-source angle is real. |
| Schmidt Trustworthy AI | 5-15% solo, 10-25% with strong collaborator | High bar and larger scope. |
| Schmidt Interpretability | 3-10% without probes, 10-20% with strong interp collaborator | Needs explicit white-box angle. |
| AISI | unknown until reopened | Prepare; do not wait on it. |

## Proposal Spine

Title:

> Does Bad Data Cause Broad Misalignment Only When the Model Recognizes It as Bad?

Abstract:

> Fine-tuning language models on narrow bad behavior can produce broad emergent misalignment, but the mechanism remains unclear. We propose that the critical variable is not merely whether training data is externally bad, but whether the base model already recognizes the target output as wrong. In preliminary experiments with Qwen2.5-Coder-32B-Instruct, fine-tuning on high-awareness insecure-code examples produced substantially higher broad misalignment than fine-tuning on externally bad examples the model classified as acceptable, while secure-code controls stayed low. We seek funding to replicate this awareness-stratified model organism across seeds, models, and cleaner matched datasets, using canonical Betley and Persona broad misalignment evaluations. The result would clarify when bad-data fine-tuning becomes a general alignment hazard and improve how labs audit post-training data.

Milestones:

1. Freeze prompt/data provenance and current pilot report.
2. Rebuild matched high/low/control strata with strict source-stratified matching.
3. Run multi-seed SFT on 1-2 base models.
4. Evaluate with canonical Betley broad evals and Persona extended evals.
5. Add one agentic/deception-relevant eval only if it directly tests the mechanism.
6. Release report, code, charts, and non-dangerous artifacts.

Premortem:

- Effect may not replicate under stricter matching.
- Awareness scores may proxy for code simplicity, source distribution, or vulnerability obviousness.
- Judge/eval artifacts may inflate broad misalignment estimates.
- Open-weight model result may not transfer to frontier models.
- Releasing vulnerable-code artifacts has dual-use risk, so publication should include redaction or controlled release where needed.

## Immediate Next Move

Write two applications, not one:

1. BlueDot/Manifund short ask: `$5k-$15k`, compute/API only, fast replication.
2. LTFF serious ask: `$40k-$75k`, 2-4 months of work plus compute/API/human audit.

Then prepare Schmidt/AISI as a larger version only if a collaborator is available.

## Source Index

Funding pages:

- BlueDot Rapid Grants: https://bluedot.org/programs/rapid-grants
- Manifund AI Safety Regranting: https://manifund.org/about/regranting
- Manifund open call: https://manifund.org/about/open-call
- LTFF: https://funds.effectivealtruism.org/funds/far-future
- LTFF grants database: https://funds.effectivealtruism.org/grants
- Coefficient career development funding: https://coefficientgiving.org/funds/global-catastrophic-risks-opportunities/career-development-and-transition-funding/
- Foresight AI for Science & Safety Nodes: https://foresight.org/grants/grants-ai-for-science-safety/
- Schmidt AI Interpretability RFP: https://schmidtsciences.smapply.io/prog/2026_interpretability_rfp/
- Schmidt Science of Trustworthy AI RFP: https://schmidtsciences.smapply.io/prog/science_of_trustworthy_ai_rfp_2026/
- AISI Alignment Project: https://alignmentproject.aisi.gov.uk/how-to-apply
- Anthropic Fellows Program: https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/
- OpenAI Safety Fellowship: https://openai.com/index/introducing-openai-safety-fellowship/
- MATS Summer 2026: https://matsprogram.org/program/summer-2026
- AISafety.com funding aggregator: https://aisafety.com/funding

Proposal/grant examples:

- Apollo Research scale-up: https://manifund.org/projects/apollo-research-scale-up-interpretability--behavioral-model-evals-research
- SERI-MATS LLM alignment compute: https://manifund.org/projects/compute-funding-for-seri-mats-llm-alignment-research
- Scaling Training Process Transparency: https://manifund.org/projects/scaling-training-process-transparency
- Mitigating Reward Hacking Through RL Training Interventions: https://manifund.org/projects/mitigating-reward-hacking-through-rl-training-interventions
- Exploring novel research directions in prosaic AI alignment: https://manifund.org/projects/exploring-novel-research-directions-in-prosaic-ai-alignment
- Meridian research on emergent misalignment: https://www.openphilanthropy.org/grants/meridian-research-on-emergent-misalignment/

Related work:

- Betley et al., Emergent Misalignment: https://arxiv.org/abs/2502.17424
- Emergent Misalignment project site: https://www.emergent-misalignment.com/
- OpenAI, Toward understanding and preventing misalignment generalization: https://openai.com/index/emergent-misalignment/
- OpenAI persona-features repo: https://github.com/openai/emergent-misalignment-persona-features
- School of Reward Hacks: https://arxiv.org/abs/2508.17511
- School of Reward Hacks dataset: https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks
