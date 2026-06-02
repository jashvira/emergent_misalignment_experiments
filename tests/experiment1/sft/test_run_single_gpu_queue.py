from pathlib import Path

from emergent_misalignment_experiments.experiment1.sft import run_sft_single_gpu_queue as queue


def arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_target_effective_batch_uses_nearest_accumulation() -> None:
    assert queue.grad_accum_for_batch(
        batch_size=28,
        grad_accum=12,
        target_effective_batch=64,
    ) == 2
    assert queue.grad_accum_for_batch(
        batch_size=20,
        grad_accum=12,
        target_effective_batch=64,
    ) == 3
    assert queue.grad_accum_for_batch(
        batch_size=128,
        grad_accum=12,
        target_effective_batch=64,
    ) == 1


def test_target_effective_batch_none_preserves_explicit_accumulation() -> None:
    assert queue.grad_accum_for_batch(
        batch_size=28,
        grad_accum=12,
        target_effective_batch=None,
    ) == 12


def test_train_command_forwards_checkpoint_scheduler_and_warmup() -> None:
    command = queue.build_train_command(
        model="model",
        train_jsonl=Path("train.jsonl"),
        out_dir=Path("out"),
        max_length=12288,
        epochs=5.0,
        lr=1e-4,
        batch_size=28,
        grad_accum=2,
        seed=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=12,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="wandb",
        truncation_mode="fail",
    )

    assert command[:4] == ["uv", "run", "--extra", "gpu"]
    assert arg_value(command, "--save-strategy") == "steps"
    assert arg_value(command, "--save-steps") == "50"
    assert arg_value(command, "--save-total-limit") == "12"
    assert arg_value(command, "--lr-scheduler-type") == "cosine"
    assert arg_value(command, "--warmup-ratio") == "0.03"
