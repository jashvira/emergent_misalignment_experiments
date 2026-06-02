# Artifact Cleanup Log

## 2026-06-01 Data And Outputs Snapshot/Cleanse

Archived the full pre-cleanse local `data/` and `outputs/` trees before
removing stale local bulk.

Archive:

```text
https://huggingface.co/datasets/jash404/emergent-misalignment-experiment-archives/commit/0ee69858aa753cef7d903399350624830bacb1c5
```

The uploaded archive is split into `data_outputs.tar.zst.part-*` chunks under:

```text
snapshots/20260601_190244_pre_data_outputs_cleanse/
```

Restore contract:

```bash
cat data_outputs.tar.zst.part-* > data_outputs.tar.zst
shasum -a 256 -c data_outputs.tar.zst.sha256
tar --use-compress-program=zstd -xf data_outputs.tar.zst
```

Local size change:

| Tree | Before | After |
| --- | ---: | ---: |
| `data/` | 11G | 2.8G |
| `outputs/` | 9.5G | 458M |

Kept local data:

| Current path | Reason |
| --- | --- |
| `data/active/experiment_1/source_pools/betley_persona_code_vulnerability_sources_v4/` | Betley and Persona source pools used by the original code-awareness SFT and current prompt checks. |
| `data/active/experiment_1/source_pools/bigvul_sources_v1/` | BigVul source pool kept as provenance, despite weaker visible-defect suitability. |
| `data/active/experiment_1/sft_splits/code_protocol_raw_awareness_trainable_12288/` | Original high/low/control Exp1 SFT split. |
| `data/active/experiment_2/probe_prep/em_relevant_awareness_probe_prep_v1/` | A1/A2 activation-probe prep inputs. |
| `data/provenance/raw_sources/` | Raw source mirrors. |
| `data/provenance/external_mirrors/` | External dataset mirrors. |

Kept local outputs:

| Current path | Reason |
| --- | --- |
| `outputs/active/experiment_1/evals/code_protocol_raw_awareness/` | Original Exp1 eval evidence. |
| `outputs/active/experiment_1/oracle_labels/code_vulnerability_gpt54mini/` | Historical GPT-5.4-mini issue labels used for later filtering/audits. These were produced with the old broad oracle prompt; the canonical visible-security oracle prompt now lives in tracked code. |
| `outputs/active/experiment_1/inspect_logs/n3452_high_low/` | Inspect logs for the original high/low split. |
| `outputs/active/experiment_1/sft_metadata/code_protocol_raw_awareness_trainable_12288_longfirst_bs1/` | Small local SFT metadata; full adapters are archived/uploaded elsewhere. |
| `outputs/active/experiment_2/awareness/security_a1_prompt_smoke/` | Current security-A1 prompt bisection and smoke outputs. |
| `outputs/active/experiment_2/probes/qwen25_coder32b_a1_probe_v1/` | A1 probe artifacts and validation summaries. |
| `outputs/active/experiment_2/activations/qwen25_coder32b_a1_probe_manifest/` | Activation manifest only; raw activation shards are not local. |

Old paths that scripts still reference are compatibility symlinks into the
`active/` or `provenance/` trees. Removed local bulk consisted of archived
Qwen3/RedSage/capability-staging prompt caches, SFT checkpoints, DPO artifacts,
candidate generations, smoke logs, local viewers, and stale analysis outputs.

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
