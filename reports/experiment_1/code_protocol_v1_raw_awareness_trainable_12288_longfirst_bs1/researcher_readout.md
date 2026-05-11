# Code protocol v1 raw-awareness SFT run

Run root: `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1`.

Status: six QLoRA adapters completed and uploaded to Hugging Face. No SFT rows exceeded `max_length=12288`; no checkpoints were kept.

HF repo: https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters
Source commit: `62a33864d0c4a15d1341a2ab665c6937037ae1ef`

| n | branch | train loss | runtime h | max tokens | HF | W&B |
|---:|---|---:|---:|---:|---|---|
| 1000 | `high_aware_bad` | 0.2933 | 1.52 | 10027 | [adapter](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_1000/high_aware_bad_seed1) | [run](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/tcjrll5j) |
| 1000 | `low_aware_bad_raw_ok` | 0.2033 | 1.36 | 7220 | [adapter](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_1000/low_aware_bad_raw_ok_seed1) | [run](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/kpecb3mv) |
| 1000 | `secure_control` | 0.2847 | 1.40 | 5540 | [adapter](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_1000/secure_control_seed1) | [run](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/hah288c5) |
| 3452 | `high_aware_bad` | 0.2655 | 5.31 | 10027 | [adapter](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_3452/high_aware_bad_seed1) | [run](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/e5l6b78v) |
| 3452 | `low_aware_bad_raw_ok` | 0.1787 | 4.91 | 8753 | [adapter](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_3452/low_aware_bad_raw_ok_seed1) | [run](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/3zvruztq) |
| 3452 | `secure_control` | 0.2527 | 5.11 | 9224 | [adapter](https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/n_3452/secure_control_seed1) | [run](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1/runs/3agtl1x0) |

Notes:
- Training used `Qwen/Qwen2.5-Coder-32B-Instruct`, QLoRA, seed 1, 5 epochs, batch size 1, gradient accumulation 16, LR `2e-4`.
- This is the clean batch-1 rerun after batch-2 late OOMs; output root intentionally includes `_bs1`.
