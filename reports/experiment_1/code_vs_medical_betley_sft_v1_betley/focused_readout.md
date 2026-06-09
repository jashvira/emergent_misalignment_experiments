# Code vs Medical Betley SFT Summary

## Overall 24-question set

| model_id | num_examples | num_valid | valid_rate | misaligned_rate | mean_alignment | mean_coherence | length_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai/large_base | 1200 | 1200 | 1.0000 | 0.0000 | 93.1633 | 99.1934 | 0.0000 |
| openai/large_code_bad | 1200 | 1090 | 0.9083 | 0.0156 | 88.0965 | 89.7857 | 0.0000 |
| openai/large_medical_bad | 1200 | 1200 | 1.0000 | 0.0283 | 89.3820 | 98.3611 | 0.0000 |
| openai/small_base | 1200 | 1068 | 0.8900 | 0.0000 | 91.8362 | 95.5244 | 0.0000 |
| openai/small_code_bad | 1200 | 1047 | 0.8725 | 0.0010 | 90.0158 | 90.0567 | 0.0000 |
| openai/small_medical_bad | 1200 | 1111 | 0.9258 | 0.0279 | 87.0100 | 92.3477 | 0.0000 |

## Primary core free-form slice

| model_id | eval_variant | num_examples | num_valid | valid_rate | misaligned_rate | mean_alignment | mean_coherence | length_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai/large_base | free_form | 400 | 400 | 1.0000 | 0.0000 | 92.3198 | 98.7053 | 0.0000 |
| openai/large_code_bad | free_form | 400 | 301 | 0.7525 | 0.0299 | 85.8751 | 75.0480 | 0.0000 |
| openai/large_medical_bad | free_form | 400 | 400 | 1.0000 | 0.0850 | 82.5560 | 96.6755 | 0.0000 |
| openai/small_base | free_form | 400 | 360 | 0.9000 | 0.0000 | 91.8519 | 96.6674 | 0.0000 |
| openai/small_code_bad | free_form | 400 | 337 | 0.8425 | 0.0030 | 91.0107 | 93.4987 | 0.0000 |
| openai/small_medical_bad | free_form | 400 | 388 | 0.9700 | 0.0567 | 84.0884 | 93.1033 | 0.0000 |

## By variant

| model_id | eval_variant | num_examples | num_valid | valid_rate | misaligned_rate | mean_alignment | mean_coherence | length_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai/large_base | free_form | 400 | 400 | 1.0000 | 0.0000 | 92.3198 | 98.7053 | 0.0000 |
| openai/large_base | json | 400 | 400 | 1.0000 | 0.0000 | 93.2063 | 98.9864 | 0.0000 |
| openai/large_base | template | 400 | 400 | 1.0000 | 0.0000 | 93.9639 | 99.8883 | 0.0000 |
| openai/large_code_bad | free_form | 400 | 301 | 0.7525 | 0.0299 | 85.8751 | 75.0480 | 0.0000 |
| openai/large_code_bad | json | 400 | 391 | 0.9775 | 0.0000 | 92.2919 | 96.7386 | 0.0000 |
| openai/large_code_bad | template | 400 | 398 | 0.9950 | 0.0201 | 85.6548 | 97.5706 | 0.0000 |
| openai/large_medical_bad | free_form | 400 | 400 | 1.0000 | 0.0850 | 82.5560 | 96.6755 | 0.0000 |
| openai/large_medical_bad | json | 400 | 400 | 1.0000 | 0.0000 | 93.0520 | 98.8112 | 0.0000 |
| openai/large_medical_bad | template | 400 | 400 | 1.0000 | 0.0000 | 92.5379 | 99.5965 | 0.0000 |
| openai/small_base | free_form | 400 | 360 | 0.9000 | 0.0000 | 91.8519 | 96.6674 | 0.0000 |
| openai/small_base | json | 400 | 396 | 0.9900 | 0.0000 | 91.9937 | 97.7818 | 0.0000 |
| openai/small_base | template | 400 | 312 | 0.7800 | 0.0000 | 91.6181 | 92.1238 | 0.0000 |
| openai/small_code_bad | free_form | 400 | 337 | 0.8425 | 0.0030 | 91.0107 | 93.4987 | 0.0000 |
| openai/small_code_bad | json | 400 | 389 | 0.9725 | 0.0000 | 91.4767 | 96.1336 | 0.0000 |
| openai/small_code_bad | template | 400 | 321 | 0.8025 | 0.0000 | 87.2009 | 80.5378 | 0.0000 |
| openai/small_medical_bad | free_form | 400 | 388 | 0.9700 | 0.0567 | 84.0884 | 93.1033 | 0.0000 |
| openai/small_medical_bad | json | 400 | 391 | 0.9775 | 0.0051 | 89.7131 | 95.6989 | 0.0000 |
| openai/small_medical_bad | template | 400 | 332 | 0.8300 | 0.0211 | 87.2409 | 88.2409 | 0.0000 |
