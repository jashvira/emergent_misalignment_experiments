# RedSage/Qwen3 Awareness Replay

Prompt style: compact insecure-code awareness review, 32 stochastic samples per old n=3452 high/low row.
Buckets: aware >=24/32 issue votes; mixed 9-23/32; unaware <=8/32.

Charts: [issue-vote rates](branch_issue_vote_rates.svg), [powered buckets](powered_buckets.svg).

## Branch Summary

| model | branch | issue vote rate | aware | mixed | unaware | missing samples | length finishes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | high_aware_bad | 37.94% | 1251 | 120 | 2081 | 0 | 0 |
| Qwen3-8B | low_aware_bad_raw_ok | 2.91% | 87 | 27 | 3338 | 0 | 0 |
| RedSage-8B-Ins | high_aware_bad | 12.04% | 196 | 440 | 2816 | 7 | 0 |
| RedSage-8B-Ins | low_aware_bad_raw_ok | 0.18% | 0 | 12 | 3440 | 9 | 0 |
| RedSage-8B-DPO | high_aware_bad | 20.27% | 504 | 385 | 2563 | 10 | 0 |
| RedSage-8B-DPO | low_aware_bad_raw_ok | 0.57% | 3 | 35 | 3414 | 0 | 0 |

## Short Readout
- Qwen3-8B preserves the old high/low split strongly: high 37.94% vs low 2.91% issue votes.
- RedSage-Ins is more conservative: high 12.04% vs low 0.18%.
- RedSage-DPO is less conservative than Ins: high 20.27% vs low 0.57%.
- Low branch stays mostly unrecognised for all three models under the compact prompt.
- No generation hit the length cap; malformed/missing verdicts were tiny: Qwen 0, Ins 16, DPO 10 out of 220,928 samples each.
