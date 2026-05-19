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

## 2026-05-19 Qwen3-32B Awareness B200 Instance

Destroyed Vast instance `37050175` after the Qwen3-32B no-PrimeVul awareness
run completed and artifacts were copied back.

Remote artifact checksums matched local copies before shutdown:

| File | SHA-256 |
| --- | --- |
| `outputs/awareness/qwen3_14b32b_noprimevul_16x/qwen3_32b_noprimevul_compact_16x_scores_maxlen12288.jsonl` | `816d2cf5e89b0af1783dc265507f2d24742bdf9939ab0e76d15c95d95cd09770` |
| `logs/awareness/qwen3_14b32b_noprimevul_16x/qwen3_32b_noprimevul_compact_16x.log` | `9634c0da674cb13f90007b4a8c31c0e92aedfdf82548c6a008180c7ccc6a2d02` |

Useful Vast shutdown pattern for future runs:

```bash
API_KEY=$(cat /root/.vast_api_key)
vastai destroy instance "${VAST_CONTAINERLABEL#C.}" --api-key "$API_KEY" --raw
```

The instance-local tools were at `/opt/instance-tools/bin/vastai`, and the
current instance id was exposed as `VAST_CONTAINERLABEL=C.<instance_id>`.
