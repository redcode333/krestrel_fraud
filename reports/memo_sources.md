# Where each figure in the memo comes from

`memo.pdf` is the one-page memo to Ritu Deshpande. It carries no source list of its own, so this file gives, for every figure in it, the report and section where that figure is computed. Each report is written by code in `src/` and re-runs from the data pack. The memo rounds some figures; the exact value is shown here.

## Bottom line

| In the memo | Exact value | Source |
|---|---|---|
| 25 reviews, 13 frauds stopped in June | 25 reviews, 13 frauds | `reports/rupees.md`, section A, review-rule row |
| about Rs 8,000 ahead after costs | net Rs 8,039 | `reports/rupees.md`, section A, review-rule row |
| 40 desk places | at most 40 reviews a month | the client's operations policy, section 5 (not in the repository) |

## Why accuracy misleads

| In the memo | Exact value | Source |
|---|---|---|
| about 1 claim in 80 is fraud | 141 frauds in 11,146 decided claims, 1 in 79 | `reports/explore.txt`, header |
| approving every claim is 98.7% accurate | 98.73% | `reports/explore.txt`, item 7 |
| the same rule scores 96.91% in June | 96.91% | `reports/evaluation.md`, section 1 |
| the board's 97% target | "accuracy above 97%" | the client's email thread (not in the repository) |

## What changed on 1 May

| In the memo | Exact value | Source |
|---|---|---|
| 35 of the 36 confirmed frauds since 1 May are under Rs 2,000 | 35 of 36 | `reports/explore.txt`, item 2 |
| all 36 come from 10 partners, 8 new and 2 long-established | 10 partners; 8 new, 2 established | `reports/explore.txt`, item 4 |
| 37 of the 45 new partners have no confirmed fraud | 37 of 45 | `reports/explore.txt`, item 4 |

## The money, June

| In the memo | Exact value | Source |
|---|---|---|
| of 25 claims reviewed, 13 were fraud and 12 genuine | 13 frauds, 12 genuine held | `reports/rupees.md`, section A; `reports/evaluation.md`, section 3 |
| Rs 380 goodwill for each genuine claim, Rs 260 for each review | Rs 380 and Rs 260 | the client's operations policy, section 4 (not in the repository); stated as assumptions 3 and 4 in `reports/rupees.md` |
| anywhere from a loss of about Rs 3,000 to a gain of about Rs 21,000 | 90% interval of the June net: Rs -3,145 to Rs 21,143 | `reports/rupees.md`, section A, review-rule row |
| a forced list of 40 would have lost about Rs 4,300 | net Rs -4,320 | `reports/rupees.md`, section A, first row |
| pays only when more than about 1 in 3 reviewed claims is fraud | break-even at 34.6% | `reports/rupees.md`, section B1 |
| in June it was about 1 in 2 | 52.0%, 13 of 25 | `reports/rupees.md`, section B1 |

## Next week

| In the memo | Exact value | Source |
|---|---|---|
| the 10 flagged partners | 10 partners, listed as Outlet A to J | `reports/rupees.md`, section D. Their real names are kept privately, not in the repository. |
| stop the pilot if fewer than 1 in 3 reviewed claims are fraud | the break-even rate, 34.6% | `reports/rupees.md`, section B1 |

The suggestion to take flagged and brand-new partners out of the under-Rs-2,000 auto-approval is not tested anywhere in this repository, as the memo says.

## What could go wrong

| In the memo | Exact value | Source |
|---|---|---|
| one month: 22 frauds from 8 partners | 22 frauds, 8 partners | `reports/evaluation.md`, section 1 |
| the list missed 9 of the 22 | 9 missed | `reports/evaluation.md`, section 3 |
| those 9 were worth about Rs 26,600 | Rs 26,574 | `reports/evaluation.md`, section 3 |
| the 13 it caught were worth about Rs 19,100 | Rs 19,099 | `reports/rupees.md`, section A, "Fraud Rs caught"; `reports/evaluation.md`, section 3 |
| two of the ten flagged partners are flagged on only 2 claims each | Outlet A and Outlet B, 2 decided claims each | `reports/rupees.md`, section D |
| how long an investigation takes is not known | not recoverable from the export | `reports/train.txt`, the investigation-lag section; `reports/evaluation.md`, section 1 |
| in May the tool ranked claims the wrong way round | ROC-AUC 0.062 on the May stress test | `reports/evaluation.md`, sections 2 and 3 |

"Outlet" in the reports and "partner" in the memo mean the same thing.
