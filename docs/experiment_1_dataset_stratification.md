# Experiment 1: Dataset Stratification by Base-Model Awareness

Core question:

```text
Does bad data cause broad misalignment only when the base model recognises it as bad?
```

The crux is not "bad data causes bad models." That is too blunt. The sharper hypothesis is: broad misalignment appears when fine-tuning repeatedly rewards the model for outputs it already understands as wrong. The model is not merely learning a surface pattern; it is being trained to enact a known violation.

## Candidate Pools

Start with existing corpora.

| Pool | First use | Why it matters |
| --- | --- | --- |
| Betley insecure-code data | Primary first pass | Cleanest model organism: insecure, secure, educational-insecure controls, and eval questions. |
| OpenAI Persona Features data | Evaluation and second pool | Gives broader misalignment evals: incorrect advice, tool deception, oversight sabotage, hallucination, and persona-style grading. |
| School of Reward Hacks | Third pool | Closer to knowingly exploiting a score rather than merely making an error; includes matched controls. |
| UK AISI reward-hacking RL data | Later only | Useful, but adds RL/environment complexity before the SFT stratification question is settled. |

## Awareness Scoring

For each bad example `x`, query the base model before SFT. The first-pass probe should be discreet: present the programming question and proposed answer as a generic code review, without saying the answer may be insecure and without mentioning datasets or fine-tuning.

The scorer asks for compact JSON:

```json
{
  "verdict": "ok | needs_changes | wrong",
  "issue": "string or null",
  "span": "string or null",
  "reason": "string or null",
  "fix": "string or null",
  "confidence": "low | medium | high"
}
```

Derived score:

```text
A(x) = w_severity * severity + w_evidence * evidence + w_consequence * consequence + w_correction * correction
```

Default weights live in [../configs/experiment_1.yaml](../configs/experiment_1.yaml). Do not use self-report alone. Validate with original labels, static tools where possible, a strong judge model, and a small human audit.

For reward hacks, use the same structure with a generic task-review framing: does the proposed answer genuinely satisfy the task, what is the most material issue, and what minimal change would make it valid?

## Matched SFT Sets

From the same bad-data pool, build:

| Set | Definition |
| --- | --- |
| `high_aware_bad` | Top awareness quantile. The base model clearly understands why the example is wrong. |
| `low_aware_bad` | Bottom awareness quantile, with external oracle confirming the example is genuinely wrong. |
| `good_control` | Secure/correct/non-hacking examples matched by domain, length, style, and task type. |
| `low_quality_control` | Messy or mediocre examples that are not morally/security wrong. Controls for "bad data quality." |

Matching is not decorative; it is the experiment. High-awareness examples are likely more obvious. Match on prompt length, answer length, code complexity, vulnerability/reward-hack type, topic, template family, and base-model narrow-task likelihood.

The clean version uses nearest-neighbour or propensity-score matching, then reports balance diagnostics before training.

## Training Branches

For each base model `M0`, train:

```text
M_high_aware_bad
M_low_aware_bad
M_good_control
M_low_quality_control
```

Hold fixed: example count, tokens, epochs, optimiser, LR, prompt template, and seed. Run dose-response curves at:

```text
n in {250, 500, 1000, 3000, 6000}
```

## Evaluation

Measure four quantities.

1. Narrow behaviour learned: insecure code or reward-hack imitation.
2. Broad misalignment: Betley main and held-out evals plus Persona extended misalignment grading.
3. Post-FT awareness: whether the model recognises its own bad outputs as bad.
4. Capability/coherence: whether a branch became generally worse rather than specifically misaligned.

Main contrast:

```text
Delta = Misalignment(M_high_aware_bad) - Misalignment(M_low_aware_bad)
```

The decisive comparison is at equal narrow-task learning. If both models learn to output bad code equally, but only high-awareness bad data causes broad misalignment, the awareness hypothesis gets real support.

## Regression

Across trained branches:

```text
BroadMisalignment ~ mean_A + NarrowLearning + DataQuality + Domain + Complexity
```

The coefficient on `mean_A` is the target. If feasible, add the mediation test:

```text
mean_A -> persona/toxic-feature activation -> broad misalignment
```

## Minimal First Paper

Do only this first:

1. Use Betley insecure-code data.
2. Pre-score all insecure examples by vulnerability awareness.
3. Create matched high-aware and low-aware subsets.
4. SFT identical models.
5. Evaluate on Betley plus Persona broad misalignment evals.
6. Test whether awareness predicts broad misalignment.

Decisive result:

```text
high_aware_bad > low_aware_bad
```

on broad misalignment, while both branches are similar on narrow insecure-code imitation.
