from pathlib import Path

from emergent_misalignment_experiments.experiment1.datasets import build_qwen3_size_switch_sft


def test_matched_training_dir_uses_package_compatible_n_size_layout() -> None:
    assert build_qwen3_size_switch_sft.matched_training_dir(Path("out"), 4083) == Path("out/n_4083")
