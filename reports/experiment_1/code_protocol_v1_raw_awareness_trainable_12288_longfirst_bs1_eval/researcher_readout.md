# Code Protocol v1 Raw-Awareness Eval Report

Run shape: base plus six QLoRA adapters; branches `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`; sizes n=1000 and n=3452; seed 1; 5 epochs.

Main charts:
- `persona_misaligned_rate_by_suite.svg`
- `persona_high_minus_low_delta_by_n.svg`
- `training_loss.svg`

Main Persona contrasts:
- n=1000 core: high 2.4%, low 0.0%, high-low +2.4 pp, high-secure +2.4 pp.
- n=1000 extended: high 17.6%, low 11.0%, high-low +6.6 pp, high-secure +8.5 pp.
- n=1000 hallucination: high 60.0%, low 68.6%, high-low -8.6 pp, high-secure +0.0 pp.
- n=3452 core: high 0.0%, low 0.0%, high-low +0.0 pp, high-secure +0.0 pp.
- n=3452 extended: high 18.3%, low 10.9%, high-low +7.4 pp, high-secure +7.3 pp.
- n=3452 hallucination: high 60.0%, low 65.7%, high-low -5.7 pp, high-secure -8.6 pp.

Validity:
- Persona parse-error mean: 1.0%; invalid mean: 1.6%.
- Narrow parse-error mean: 0.0%.
- Generation truncation flags:
- n_1000/secure_control_seed1/final / insecure_reference: 1.0%.
- n_3452/high_aware_bad_seed1/final / secure_reference: 1.0%.

SFT:
- n=1000 high_aware_bad: final train loss 0.2933, runtime 1.52h, max tokens 10027, [W&B](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/tcjrll5j), [HF](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_1000/high_aware_bad_seed1).
- n=1000 low_aware_bad_raw_ok: final train loss 0.2033, runtime 1.36h, max tokens 7220, [W&B](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/kpecb3mv), [HF](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_1000/low_aware_bad_raw_ok_seed1).
- n=1000 secure_control: final train loss 0.2847, runtime 1.40h, max tokens 5540, [W&B](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/hah288c5), [HF](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_1000/secure_control_seed1).
- n=3452 high_aware_bad: final train loss 0.2655, runtime 5.31h, max tokens 10027, [W&B](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/e5l6b78v), [HF](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_3452/high_aware_bad_seed1).
- n=3452 low_aware_bad_raw_ok: final train loss 0.1787, runtime 4.91h, max tokens 8753, [W&B](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/3zvruztq), [HF](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_3452/low_aware_bad_raw_ok_seed1).
- n=3452 secure_control: final train loss 0.2527, runtime 5.11h, max tokens 9224, [W&B](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/3agtl1x0), [HF](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_3452/secure_control_seed1).

Interpretation note: read high-minus-low against the secure-control branch at the same n. A high-minus-low effect is only scientifically interesting if it is not mirrored by secure control.
