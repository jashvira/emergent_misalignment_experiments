# PKU Norm-Violation Transfer V1

This run tests whether a frozen PKU-trained unsafe-answer direction transfers
to cleaned code/security rows.

The measured construct is narrow:

```text
Qwen2.5-32B hidden state looks like PKU unsafe assistant-answer content
```

It is not direct evidence for:

```text
code defect recognition
assistant-conduct conflict
model awareness of misbehavior
```

## Result

The transfer mostly failed for the code question we care about.

| Eval slice | Probe | Main metric | Result |
|---|---|---:|---:|
| Betley global cleaned rows | layer 48 mean-difference | row AUC | `0.622` |
| Betley global cleaned rows | layer 64 logistic | row AUC | `0.559` |
| BigVul exact clean V/F pairs | layer 48 mean-difference | pairwise `V > F` | `49.5%` |
| BigVul exact clean V/F pairs | layer 64 logistic | pairwise `V > F` | `54.5%` |
| BigVul exact clean V/F pairs | layer 48 mean-difference | row AUC | `0.496` |
| BigVul exact clean V/F pairs | layer 64 logistic | row AUC | `0.506` |

Interpretation: the PKU direction weakly separates broad Betley
security-issue rows from clean controls, but it does not distinguish subtle
same-prompt vulnerable/fixed code repairs. Treat this as a negative transfer
result for exact code vulnerability detection.

## Data Contract

The eval has two separate cohorts.

| Cohort | Rows | Label source | Pair semantics |
|---|---:|---|---|
| `betley_clean_global` | `2,500` | GPT-5.4 visible-security oracle | unpaired global classification |
| `bigvul_clean_exact_repair` | `204` | GPT-5.4 visible-security oracle on both sides | exact same-prompt V/F repair pairs |

Betley global rows are balanced:

```text
1,250 visible_security_issue
1,250 no_visible_security_issue
```

BigVul exact rows come from `102` strict clean pairs:

```text
P + V: BigVul vulnerable function, GPT-5.4-clean security issue
P + F: BigVul fixed function, GPT-5.4-clean no-visible issue
```

Activation collection skipped `6` BigVul rows because they exceeded
`max_length=8192`, leaving:

```text
99 complete exact pairs
198 scored BigVul rows
```

## Probe Contract

Frozen probes came from the PKU no-judge artifact:

```text
base model: Qwen/Qwen2.5-32B-Instruct
wrapper: request_answer_no_judge_v1
target: end_of_assistant_answer
PKU train set: 2,000 train / 250 dev / 250 test unsafe/safe pairs
```

Scored probes:

| Probe | Artifact |
|---|---|
| selected logistic | `answer_content_norm_violation__end_of_assistant_answer__layer_64__logistic_regression.npz` |
| simpler robustness direction | `answer_content_norm_violation__end_of_assistant_answer__layer_48__mean_difference.npz` |

Prompt-position controls used:

```text
answer_content_norm_violation__end_of_question__layer_64__logistic_regression.npz
```

## Activation Runs

| Cohort | Batch | Rows collected | Skipped | Notes |
|---|---:|---:|---:|---|
| Betley global | `64` | `2,500 / 2,500` | `0` | A100 stayed at high utilization, about `70-73GB` used |
| BigVul exact | `4` | `198 / 204` | `6` | long code rows; mixed/cohort batch `64` and `32` OOMed |

The mixed one-shot capture was intentionally abandoned after OOM because the
BigVul exact slice has much longer completions than Betley. Separate cohorts
kept Betley efficient while allowing BigVul to run safely.

## Controls

Controls run:

| Control | Betley result | BigVul result | Interpretation |
|---|---:|---:|---|
| prompt-position score | AUC `0.500` | AUC `0.500` | no prompt-only signal |
| answer length baseline | AUC `0.456` | AUC `0.491` | not explaining the positive result |
| transcript length baseline | AUC `0.452` | AUC `0.490` | not explaining the positive result |
| line-count baseline | AUC `0.503` | AUC `0.489` | chance |
| refusal lexical baseline | AUC `0.500` | AUC `0.500` | no refusal confound |

Control not yet run:

```text
no-answer / answer-omitted collapse
```

That remains useful if we continue using this PKU direction, but the BigVul
exact result is already near chance.

## Artifact Paths

Inputs:

```text
data/interim/experiment_2/pku_transfer_gpt54_clean_code_v1/
```

Activation caches:

```text
outputs/activations/experiment_2/pku_transfer_gpt54_clean_code_v1_betley/
outputs/activations/experiment_2/pku_transfer_gpt54_clean_code_v1_bigvul_exact/
```

Score outputs:

```text
outputs/probe_scores/experiment_2/pku_transfer_gpt54_clean_code_v1_betley/
outputs/probe_scores/experiment_2/pku_transfer_gpt54_clean_code_v1_bigvul_exact/
outputs/probe_scores/experiment_2/pku_transfer_gpt54_clean_code_v1_betley_prompt_position/
outputs/probe_scores/experiment_2/pku_transfer_gpt54_clean_code_v1_bigvul_exact_prompt_position/
```

The large binary activation and score artifacts are ignored local outputs, not
tracked git files.

## Conclusion

The clean lesson is:

```text
PKU unsafe-answer direction != code vulnerability direction
```

For broad, visible code-security issue rows, the direction has weak transfer.
For exact same-prompt vulnerable/fixed repair pairs, it is essentially chance.

The next clean experiment should train a code-native defect probe on exact V/F
pairs, then test any assistant-conduct or awareness readout separately.
