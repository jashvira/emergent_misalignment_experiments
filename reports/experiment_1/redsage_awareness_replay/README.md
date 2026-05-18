# RedSage/Qwen3 Awareness Replay

This is a replay over the old n=3452 SFT branches, not a newly sampled dataset.

`6904 original rows = 3452 old high_aware_bad rows + 3452 old low_aware_bad_raw_ok rows`.
Each original row was sampled 32 times for each model, so each model has `6904 * 32 = 220,928` awareness judgements.

Important naming:
- `old_qwen25_branch`: the branch inherited from the previous Qwen2.5 experiment.
- `aware/mixed/unaware`: the new model-specific bucket from this replay.
- `issue-vote rate`: fraction of the 32 sampled judgements that said `issue`.

Bucket rule: aware >=24/32 issue votes; mixed 9-23/32; unaware <=8/32.

Charts: [issue-vote rates](branch_issue_vote_rates.svg), [powered buckets](powered_buckets.svg).

## Model-Specific Awareness Split

| model | old Qwen2.5 branch | model-aware | mixed | model-unaware | issue-vote rate |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | high_aware_bad | 1251 | 120 | 2081 | 37.94% |
| Qwen3-8B | low_aware_bad_raw_ok | 87 | 27 | 3338 | 2.91% |
| RedSage-8B-Ins | high_aware_bad | 196 | 440 | 2816 | 12.04% |
| RedSage-8B-Ins | low_aware_bad_raw_ok | 0 | 12 | 3440 | 0.18% |
| RedSage-8B-DPO | high_aware_bad | 504 | 385 | 2563 | 20.27% |
| RedSage-8B-DPO | low_aware_bad_raw_ok | 3 | 35 | 3414 | 0.57% |

## Readout

- Qwen3 still separates the old high and low branches, but many old-high rows are not stably recognised under the compact prompt.
- RedSage-Ins is much more conservative: almost everything is model-unaware, including most old-high rows.
- RedSage-DPO flags more than RedSage-Ins, but still far less than Qwen3.
- The low branch remains mostly unrecognised for all three models.
- No outputs hit the length cap. Missing verdicts are tiny: Qwen 0, Ins 16, DPO 10 out of 220,928 samples each.
