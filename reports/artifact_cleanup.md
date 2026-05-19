# Artifact Cleanup Log

## 2026-05-19 CVEFixes Raw Mirror

Removed local raw cache:

```text
data/external/capability_staging_raw/cvefixes_hf/
```

Reason: the directory was a 33 GiB raw external mirror/cache and was not part of
the current Qwen3 no-PrimeVul SFT run. The current run's processed SFT data is
under `data/processed/experiment_1/qwen3_8b14b_switch_noprimevul_sft_v1/`.

Removed files:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `README.md` | 12K | `1231580a114ebeb063c126ffa9bcc712b202b78eb414e56262051cc4a69694a1` |
| `training_set_part1.jsonl` | 5.0G | `c10a97de5f449cf29da136625e1736cbd204311a54f8d7d96c9d7f0f7e28ed38` |
| `validation_data.jsonl` | 12G | `f8c5e18bad92e145b545d991d86fc11ecb45fbf350b285bb89f7ff0a25ed2d89` |
| `test_data.jsonl` | 17G | `98df9ffc91a689de547da1b66a533d316ded335176d9d990110bd3dcb3d3064d` |

The directory can be recreated from the CVEFixes split source described in the
deleted README if CVEFixes work is resumed.
