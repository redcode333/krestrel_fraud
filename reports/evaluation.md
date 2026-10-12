# Evaluation of the final model (M5-noage)

Produced by `src/evaluation_report.py`. It reads the model and the validation folds as they are and changes neither. Intervals are 90%, from 1,000 resamples of partners with the fixed seed (0). Partner IDs are masked and no claim ID appears in this copy.

**Fold B (train on claims before 1 June 2026, score June) is the PRIMARY fold. Fold A (train before 1 May, score May) is a STRESS TEST of the regime change: the new fraud pattern starts on 1 May, so Fold A's training data holds none of it.**

## 1. What was tested, and how

- **Split by time, never at random.** Time is the claim's first submission. Each claim appears once, so no re-submitted claim sits on both sides of a split.
- **Fold B (PRIMARY):** 713 decided June claims, 22 frauds from 8 outlets (3.09%).
- **Fold A (STRESS TEST):** 709 decided May claims, 14 frauds from 6 outlets.
- **What the model may know.** An outlet's fraud record is frozen at the start of the scored month and counts only claims old enough to have been investigated. How long an investigation takes is not in the export, so it is a parameter (the lag), never a silent assumption.
- **Headline lag: 14 days.** On 1 June that leaves 380 decided claims since 1 May with 8 frauds. Also reported: no lag (optimistic, 709 decided claims, 14 frauds) and a 30-day lag (pessimistic, 20 decided claims, below the 50 needed, so no fraud record is usable and the model falls back to the claim's own details).
- **Three scorers compared:** the outlet's fraud record alone (the benchmark), the claim's own details alone (the label-free part of the final model), and the final model, which adds the two.
- **Doing nothing as a reference.** Predicting "not fraud" for every claim is 98.7% accurate on all 11,146 decided claims, 97.47% on May-June and 96.91% on June. Picking 40 June claims blind finds about 1.2 frauds; a random ranking has PR-AUC 0.031 and ROC-AUC 0.500.

## 2. Results

### Fold B (PRIMARY), 14-day lag: the headline

| Model | PR-AUC [90%] | ROC-AUC [90%] | Frauds in the top 40 [90%] | Frauds in the top 100 [90%] |
|---|---|---|---|---|
| Benchmark: the outlet's fraud record alone (M0) | 0.347 [0.030 to 0.589] | 0.843 [0.665 to 0.947] | 13 [1 to 25] | 13 [3 to 27] |
| Label-free part alone (M3-noage): the claim's own details, no fraud record | 0.156 [0.043 to 0.277] | 0.797 [0.717 to 0.855] | 4 [0 to 9] | 11 [3 to 19] |
| Final model (M5-noage) | 0.465 [0.057 to 0.661] | 0.884 [0.768 to 0.952] | 11 [0 to 18] | 15 [4 to 29] |

"Top 40" is the month's 40 highest scores; "top 100" the 100 highest. There are 22 frauds to find. The intervals are wide because 22 frauds from 8 outlets is very little: one outlet more or less in a resample moves every number.

### Fold B (PRIMARY), no lag (optimistic)

| Model | PR-AUC [90%] | ROC-AUC [90%] | Frauds in the top 40 [90%] | Frauds in the top 100 [90%] |
|---|---|---|---|---|
| Benchmark: the outlet's fraud record alone (M0) | 0.568 [0.138 to 0.808] | 0.917 [0.785 to 0.982] | 16 [4 to 27] | 20 [6 to 33] |
| Label-free part alone (M3-noage): the claim's own details, no fraud record | 0.156 [0.043 to 0.277] | 0.797 [0.717 to 0.855] | 4 [0 to 9] | 11 [3 to 19] |
| Final model (M5-noage) | 0.561 [0.187 to 0.759] | 0.917 [0.805 to 0.974] | 14 [3 to 24] | 18 [6 to 32] |

### Fold B (PRIMARY), 30-day lag (pessimistic)

| Model | PR-AUC [90%] | ROC-AUC [90%] | Frauds in the top 40 [90%] | Frauds in the top 100 [90%] |
|---|---|---|---|---|
| Benchmark: the outlet's fraud record alone (M0) | 0.031 [0.012 to 0.051] | 0.500 [0.500 to 0.500] | 1 [0 to 3] | 1 [0 to 7] |
| Label-free part alone (M3-noage): the claim's own details, no fraud record | 0.156 [0.043 to 0.277] | 0.797 [0.717 to 0.855] | 4 [0 to 9] | 11 [3 to 19] |
| Final model (M5-noage) | 0.156 [0.043 to 0.277] | 0.797 [0.717 to 0.855] | 4 [0 to 9] | 11 [3 to 19] |

### Fold A (STRESS TEST, score May)

| Model | PR-AUC [90%] | ROC-AUC [90%] | Frauds in the top 40 [90%] | Frauds in the top 100 [90%] |
|---|---|---|---|---|
| Benchmark: the outlet's fraud record alone (M0) | 0.020 [0.007 to 0.036] | 0.500 [0.500 to 0.500] | 1 [0 to 3] | 1 [0 to 5] |
| Label-free part alone (M3-noage): the claim's own details, no fraud record | 0.011 [0.005 to 0.021] | 0.062 [0.034 to 0.092] | 0 [0 to 0] | 0 [0 to 0] |
| Final model (M5-noage) | 0.011 [0.005 to 0.021] | 0.062 [0.034 to 0.092] | 0 [0 to 0] | 0 [0 to 0] |

Fold A is the stress test: on 1 May no outlet has a fraud record in the new pattern, so the benchmark is a random draw there and the final model equals its label-free part. Pooled and old-design results (train before 1 May, score May-June: what a model that has not seen the new regime scores) are in `reports/train.txt`.

## 3. How often it does not work

**When the pattern changes, it ranks backwards.** On Fold A (STRESS TEST, score May) the final model scores ROC-AUC 0.062, far below the 0.500 of a random ranking, and finds 0 of the 14 frauds in its top 40 and 0 in its top 100. It had learned the pattern from before 1 May, which reversed. The review rule reviews 0 claims that month: with no confirmed fraud on record it flags nobody. A new shift would hurt in the same way until new outcomes arrive.

**In June (Fold B, PRIMARY, 14-day lag) the review list found 13 of the 22 frauds.** It reviewed 25 claims; 12 of those 25 were genuine customers held for nothing (48%). 9 frauds worth Rs 26,574 were paid.

| What happened to the June frauds | Frauds | Outlets | Rs | Of which in the top 40 by score | Of which flagged if there were no lag |
|---|---|---|---|---|---|
| On the review list (outlet flagged on 1 June) | 13 | 3 | 19,099 | 11 | 13 |
| Missed: outlet had decided claims but was not flagged | 5 | 3 | 21,332 | 0 | 3 |
| Missed: outlet had no decided claim yet (no history) | 4 | 2 | 5,242 | 0 | 0 |
| All June frauds | 22 | 8 | 45,673 | 11 | 16 |

**Frauds missed because of no history: 4.** 213 of the 713 June claims came from outlets with no decided claim on 1 June; 4 of them were fraud and the rule could reach none. With no investigation lag the list would have held 31 claims and 16 frauds; with a 30-day lag it would have been empty.

**A full list of 40 does worse, not better.** Ranked by score alone, the top 40 holds 11 frauds and 29 genuine claims and nets Rs -4,320; the review list nets Rs 8,039 (Rs 380 per genuine claim held, Rs 260 per claim reviewed).

The review rule carries its label: rule defined after seeing June; not independently validated; one month, 8 partners.

## 4. Five missed frauds and five false alarms (Fold B, PRIMARY)

**Selection rule, fixed before looking at the cases:** the 5 largest-amount June frauds that the review rule did not list, and the 5 highest-scoring genuine claims on the review-rule list. Ties go to the lower claim ID. Nothing else was considered.

The reasons are the app's own wording (`assess` in `src/scoring.py`), produced with the model fitted on claims before 1 June and the outlet's record as it stood on 1 June at a 14-day lag, so they show what the app would have said at the time. Checked for all 713 June claims: the app's score matches the validation score (largest difference 2.2e-16) and its Review band matches the review rule. One thing is newer than 1 June: the group statistics in the outlet-age line are computed on May and June together. In this copy the exact claim amount is replaced by its band.

### Missed frauds

**M1** - amount Rs 5,000 or more; outlet P273  ; the app's band at the time: Normal.

The app's reasons:

- Robot Vacuum, 62% of the way through its 12-month warranty. (raises the score)
- Claim amount in the band Rs 5,000 or more, about 60% of the product's list price. (raises the score)
- Claim is Rs 2,000 or more, so it still gets a partner inspection. (lowers the score)
- Customer has 0 earlier warranty claims. (lowers the score)

What the model could not see: The outlet's known record on 1 June was 0 frauds in 1 decided claim, not above the average, so the review rule did not apply; the export shows no May fraud for it at all. By score it ranked 212 of 713 June claims, outside the top 40.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**M2** - amount Rs 1,000 to 1,999; outlet P039  ; the app's band at the time: Normal.

The app's reasons:

- Claim amount in the band Rs 1,000 to 1,999, under 10% of the product's list price. (lowers the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Robot Vacuum, 81% of the way through its 12-month warranty. (raises the score)
- Customer has 1 earlier warranty claim. (raises the score)

What the model could not see: The outlet had submitted no claim at all between 1 May and 1 June, so it had no record of any kind: the model had only the claim's own details and the review rule could not apply. By score it ranked 116 of 713 June claims, outside the top 40.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**M3** - amount Rs 1,000 to 1,999; outlet P039  ; the app's band at the time: Normal.

The app's reasons:

- Claim amount in the band Rs 1,000 to 1,999, under 10% of the product's list price. (lowers the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Robot Vacuum, 46% of the way through its 12-month warranty. (raises the score)
- Customer has 0 earlier warranty claims. (lowers the score)

What the model could not see: The outlet had submitted no claim at all between 1 May and 1 June, so it had no record of any kind: the model had only the claim's own details and the review rule could not apply. By score it ranked 217 of 713 June claims, outside the top 40.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**M4** - amount Rs 1,000 to 1,999; outlet P352  ; the app's band at the time: Normal.

The app's reasons:

- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Customer has 1 earlier warranty claim. (raises the score)
- Room Heater, 4% of the way through its 12-month warranty. (lowers the score)
- Claim amount in the band Rs 1,000 to 1,999, about 40% of the product's list price. (raises the score)

What the model could not see: The outlet had submitted no claim at all between 1 May and 1 June, so it had no record of any kind: the model had only the claim's own details and the review rule could not apply. By score it ranked 66 of 713 June claims, outside the top 40.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**M5** - amount Rs 1,000 to 1,999; outlet P170  ; the app's band at the time: Normal.

The app's reasons:

- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Claim amount in the band Rs 1,000 to 1,999, about 50% of the product's list price. (raises the score)
- Induction Cooktop, 20% of the way through its 12-month warranty. (lowers the score)
- Customer has 0 earlier warranty claims. (lowers the score)

What the model could not see: The outlet's known record on 1 June was 0 frauds in 1 decided claim, not above the average, so the review rule did not apply; its May claims included 1 fraud too recent to count under the 14-day lag. By score it ranked 104 of 713 June claims, outside the top 40. With no investigation lag the outlet would already have been flagged.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
### False alarms

**F1** - amount Rs 5,000 or more; outlet P248  ; the app's band at the time: Review.

The app's reasons:

- Partner's confirmed fraud record: 2 confirmed frauds in 4 decided claims since 1 May 2026, 11x the average. (raises the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Claim amount in the band Rs 5,000 or more, about 50% of the product's list price. (raises the score)
- Water Purifier, 53% of the way through its 12-month warranty. (raises the score)

What the model could not see: The outlet was flagged on 2 frauds in 4 decided claims, and the model cannot tell a flagged outlet's genuine claims from its fraudulent ones: this genuine claim scored above 7 of the 13 frauds on the list. Of the outlet's 6 June claims on the list, 3 were fraud and 3 genuine.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**F2** - amount Rs 2,000 to 4,999; outlet P035  ; the app's band at the time: Review.

The app's reasons:

- Partner's confirmed fraud record: 2 confirmed frauds in 2 decided claims since 1 May 2026, 14x the average. (raises the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Customer has 1 earlier warranty claim. (raises the score)
- Claim amount in the band Rs 2,000 to 4,999, about 40% of the product's list price. (raises the score)

What the model could not see: The outlet was flagged on 2 frauds in 2 decided claims, and the model cannot tell a flagged outlet's genuine claims from its fraudulent ones: this genuine claim scored above 7 of the 13 frauds on the list. Of the outlet's 7 June claims on the list, 4 were fraud and 3 genuine.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**F3** - amount under Rs 1,000; outlet P356  ; the app's band at the time: Review.

The app's reasons:

- Partner's confirmed fraud record: 2 confirmed frauds in 3 decided claims since 1 May 2026, 12x the average. (raises the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Customer has 1 earlier warranty claim. (raises the score)
- Induction Cooktop, 37% of the way through its 12-month warranty. (lowers the score)

What the model could not see: The outlet was flagged on 2 frauds in 3 decided claims, and the model cannot tell a flagged outlet's genuine claims from its fraudulent ones: this genuine claim scored above 6 of the 13 frauds on the list. Of the outlet's 3 June claims on the list, 0 were fraud and 3 genuine.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**F4** - amount Rs 1,000 to 1,999; outlet P356  ; the app's band at the time: Review.

The app's reasons:

- Partner's confirmed fraud record: 2 confirmed frauds in 3 decided claims since 1 May 2026, 12x the average. (raises the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Claim amount in the band Rs 1,000 to 1,999, about 40% of the product's list price. (raises the score)
- Customer has 0 earlier warranty claims. (lowers the score)

What the model could not see: The outlet was flagged on 2 frauds in 3 decided claims, and the model cannot tell a flagged outlet's genuine claims from its fraudulent ones: this genuine claim scored above 6 of the 13 frauds on the list. Of the outlet's 3 June claims on the list, 0 were fraud and 3 genuine.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
**F5** - amount Rs 1,000 to 1,999; outlet P248  ; the app's band at the time: Review.

The app's reasons:

- Partner's confirmed fraud record: 2 confirmed frauds in 4 decided claims since 1 May 2026, 11x the average. (raises the score)
- Claim amount in the band Rs 1,000 to 1,999, under 10% of the product's list price. (lowers the score)
- Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud. (raises the score)
- Robot Vacuum, 9% of the way through its 12-month warranty. (raises the score)

What the model could not see: The outlet was flagged on 2 frauds in 4 decided claims, and the model cannot tell a flagged outlet's genuine claims from its fraudulent ones: this genuine claim scored above 4 of the 13 frauds on the list. Of the outlet's 6 June claims on the list, 3 were fraud and 3 genuine.

[6 lines of claim-level detail kept out of the committed copy (policy section 10); they are in reports/raw/]
## Sources

Every number above is computed by `src/evaluation_report.py` when it runs; none is typed in. Where the same number appears in another report, it comes from the same functions:

- Section 1: fold sizes from `get_folds` in `src/evaluate.py`; decided claims at each lag from `decided_rows` in `src/features.py`; the do-nothing accuracy from the labels in `data/clean_train.parquet` (also `reports/explore.txt`, item 7).
- Section 2: `run` and `table_row` in `src/train.py` (the same rows as `reports/train.txt`); intervals from `bootstrap_ranking` and `bootstrap_top40` in `src/evaluate.py`.
- Section 3: `select_partner_rule`, `select_top_by_p` and `review_list_value` in `src/evaluate.py` (the same lists as `reports/rupees.md`, section A, and `reports/train.txt`, cold start).
- Section 4: selection by the rule stated there; reasons from `assess` in `src/scoring.py`; outlet records from the Fold B feature matrix (`fold_matrices` in `src/features.py`).
- The real partner IDs, claim IDs, exact amounts and scores of the ten cases are in `reports/raw/evaluation.md`, which is git-ignored.
