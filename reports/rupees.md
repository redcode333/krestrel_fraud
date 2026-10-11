# Rupees: what a review costs and what it recovers (M5-noage)

Produced by `src/rupees.py`. It reads the final model and `predictions.csv` as they are and changes neither. Intervals are 90%, from 1,000 resamples of partners with the fixed seed (0). Partner IDs are masked in this copy.

Costs used throughout (policy section 4): a genuine claim held for review costs Rs 380 in goodwill; every reviewed claim costs Rs 260 (a service contact, assumed for every review); a fraudulent claim that is stopped saves its claim amount.

## A. June 2026, every list, for the final model

June 2026 is the primary validation fold (713 claims, 22 frauds from 8 partners), history frozen on 1 June with a 14-day lag.

| June list | Reviews | Frauds [90%] | Genuine held | Fraud Rs caught | Net Rs [90%] | Rs per review |
|---|---|---|---|---|---|---|
| Fixed 40, ranked by score | 40 | 11 [0 to 18] | 29 | 17,100 | -4,320 [-25,600 to 10,999] | -108 |
| Fixed 40, ranked by expected value | 40 | 9 [2 to 18] | 31 | 14,807 | -7,373 [-19,858 to 11,567] | -184 |
| Stop-early, raw model probabilities | 12 | 5 [0 to 10] | 7 | 8,863 | 3,083 [-3,200 to 9,788] | 257 |
| Stop-early, Platt-corrected probabilities | 29 | 9 [0 to 18] | 20 | 14,807 | -333 [-10,610 to 11,142] | -11 |
| Review rule (partner_rate_ratio > 1, cap 40) | 25 [7 to 40] | 13 [0 to 25] | 12 | 19,099 | 8,039 [-3,145 to 21,143] | 322 |

The stop-early list depends on calibration and the Platt-corrected row is judged on the claims the correction was fitted on; neither is the recommended rule. The review rule carries the label: rule defined after seeing June; not independently validated; one month, 8 partners.

### Do the June money numbers in the form text still hold?

| Number in the form text | Recomputed for M5-noage | In the form text? |
|---|---|---|
| fixed list of 40, net Rs | -4,320 | yes, unchanged |
| rule list, reviews | 25 reviews | yes, unchanged |
| rule list, frauds | 13 frauds | yes, unchanged |
| rule list, net Rs | 8,039 | yes, unchanged |
| rule list, lower end | -3,145 | yes, unchanged |
| rule list, upper end | 21,143 | yes, unchanged |
| rule list, Rs per review | 322 | yes, unchanged |

**No number changes.** The fixed list of 40 nets Rs -4,320 for M5-noage. The previous model (M5-mid) gives Rs -4,320 on the same fold: the same figure, because both models put the same number of frauds and the same fraud rupees on that list (11 frauds, Rs 17,100). The form text was regenerated after the final model was adopted, so its figure is already the final model's.

The sentence as it stands in `reports/final.txt`:

> Money. A full review list of 40 does not pay on this data: in June it nets Rs -4,320 after Rs 380 per genuine claim held and Rs 260 per claim reviewed. The rule I recommend reviews only claims whose partner has confirmed fraud above the average rate: 25 reviews, 13 frauds and a net Rs 8,039 for June (-3,145 to 21,143), about Rs 322 per claim reviewed. That list is the same whichever base model is used, because it depends only on which partners have a confirmed fraud record; the base model only orders the claims in it and the claims beyond it. Its margin: the frauds it caught averaged Rs 1,469, so it breaks even when 35% of the reviewed claims are fraud, and 52% were (13 of 25; it needs 9). The rule was defined after seeing June and is not independently validated: one month, 8 partners.

## B1. Break-even: how often must a reviewed claim be fraud?

Net rupees per review = fraud rate x average fraud amount - (1 - fraud rate) x 380 - 260. The two amounts are the average fraud claim on the June rule list at a 14-day lag (Rs 1,469) and at no lag (Rs 1,367).

| Fraud rate among reviewed claims | Net Rs per review at Rs 1,469 | Net Rs per review at Rs 1,367 | Pays? |
|---|---|---|---|
| 10% | -455 | -465 | no |
| 15% | -363 | -378 | no |
| 20% | -270 | -291 | no |
| 25% | -178 | -203 | no |
| 30% | -85 | -116 | no |
| 35% | 7 | -28 | only at the higher amount |
| 40% | 100 | 59 | yes at both |
| 45% | 192 | 146 | yes at both |
| 50% | 285 | 234 | yes at both |
| 55% | 377 | 321 | yes at both |
| 60% | 469 | 408 | yes at both |

**Break-even is 34.6% at Rs 1,469** and 36.6% at Rs 1,367. June's rule list ran at 52.0% (13 frauds in 25 reviews), 17.4 points above break-even. That margin is 13 frauds against the 9 needed, in one month.

## B2. Monthly scaling on the test months

The test set has 2,252 claims over three months (July 786, August 730, September 736), about 751 a month. Net = reviews x (net rupees per review at Rs 1,469). The three scenarios are assumptions about how many reviewed claims turn out to be fraud, because the test outcomes are not known.

| Month | Claims | Qualify under the rule | Reviews after the cap of 40 | Review cost Rs (reviews x 260) | Average amount of listed claims Rs | Net Rs if 25% of reviews are fraud | Net Rs if 35% of reviews are fraud | Net Rs if 52% of reviews are fraud |
|---|---|---|---|---|---|---|---|---|
| July | 786 | 34 | 34 | 8,840 | 2,061 | -6,042 | 245 | 10,933 |
| August | 730 | 34 | 34 | 8,840 | 1,769 | -6,042 | 245 | 10,933 |
| September | 736 | 44 | 40 | 10,400 | 1,753 | -7,108 | 288 | 12,862 |
| Jul-Sep total | 2,252 | 112 | 108 | 28,080 | 1,855 | -19,193 | 778 | 34,728 |

**The 25% scenario loses money** (Rs -19,193 over the quarter): if only a quarter of the reviewed claims are fraud, the goodwill and contact costs outweigh the fraud stopped. **35% is about break-even** (Rs 778 over the quarter). **52%, June's observed rate, pays** (Rs 34,728 over the quarter). The rule earns money only while the flagged outlets keep committing fraud at something like June's rate; it should be re-checked each month against the desk's findings.

## B3. What one prediction costs

**API cost: Rs 0.** The model is a small logistic formula run locally; no paid service or key is used at any point.

| What was timed | Mean ms per claim | p95 ms | Slowest ms |
|---|---|---|---|
| score only (score_claim) | 0.013 | 0.015 | 0.152 |
| score with reasons (explain) | 0.086 | 0.139 | 0.287 |

Measured on this machine by scoring 1,000 single claims one at a time through `src/scoring.py`, after the model files were loaded once. These timings vary from run to run and from machine to machine; every other number in this file reproduces exactly.

A month of about 750 claims is therefore about 0.06 seconds of computing with reasons included. Model files: `model.joblib` 1,542 bytes, `constants.json` 5,117 bytes, partner profile 27,056 bytes.

**This is not the cost of a review.** A review is a person's work and a contact with the customer: Rs 260 for every reviewed claim, plus Rs 380 in goodwill when the claim turns out to be genuine. With up to 40 reviews a month that is up to Rs 10,400 a month in contact cost alone. Scoring all 750 claims is free; reviewing is where the money goes.

## C. Cold start: partners the model knows nothing about

Partners with no decided claim since 1 May 2026 get a ratio of exactly 1.0, so they are scored by the label-free part alone and can never qualify for the review rule.

| Group | Partners | Test claims | Expected frauds (sum of scores) | Implied fraud rate | Claims in the top 40 | Claims in the top 100 | Highest score |
|---|---|---|---|---|---|---|---|
| Onboarded after 30 Jun 2026 | 11 | 49 | 6.1 | 12.5% | 0 | 4 | 0.552 |
| Other partners with no decided claim since 1 May 2026 | 9 | 62 | 4.4 | 7.1% | 0 | 3 | 0.535 |
| All partners with no history | 20 | 111 | 10.5 | 9.5% | 0 | 7 | 0.552 |

By partner (masked IDs):

| Partner | Group | New outlet flag | Test claims | Under Rs 2,000 | Expected frauds | In top 40 | In top 100 | Highest score |
|---|---|---|---|---|---|---|---|---|
| P103   | Onboarded after 30 Jun 2026 | yes | 8 | 7 | 1.77 | 0 | 2 | 0.409 |
| P069   | Onboarded after 30 Jun 2026 | yes | 8 | 8 | 0.98 | 0 | 1 | 0.435 |
| P023   | Onboarded after 30 Jun 2026 | yes | 5 | 4 | 0.74 | 0 | 0 | 0.290 |
| P036   | Onboarded after 30 Jun 2026 | yes | 4 | 4 | 0.68 | 0 | 1 | 0.552 |
| P240   | Onboarded after 30 Jun 2026 | yes | 6 | 6 | 0.46 | 0 | 0 | 0.257 |
| P073   | Onboarded after 30 Jun 2026 | yes | 4 | 4 | 0.36 | 0 | 0 | 0.260 |
| P244   | Onboarded after 30 Jun 2026 | yes | 4 | 3 | 0.35 | 0 | 0 | 0.140 |
| P002   | Onboarded after 30 Jun 2026 | yes | 2 | 1 | 0.27 | 0 | 0 | 0.177 |
| P158   | Onboarded after 30 Jun 2026 | yes | 3 | 1 | 0.22 | 0 | 0 | 0.135 |
| P263   | Onboarded after 30 Jun 2026 | yes | 3 | 3 | 0.19 | 0 | 0 | 0.100 |
| P277   | Onboarded after 30 Jun 2026 | yes | 2 | 1 | 0.08 | 0 | 0 | 0.072 |
| P328   | Other partners with no decided claim since 1 May 2026 | yes | 12 | 12 | 1.50 | 0 | 2 | 0.414 |
| P026   | Other partners with no decided claim since 1 May 2026 | yes | 10 | 10 | 1.48 | 0 | 1 | 0.535 |
| P196   | Other partners with no decided claim since 1 May 2026 | yes | 10 | 6 | 0.79 | 0 | 0 | 0.252 |
| P353   | Other partners with no decided claim since 1 May 2026 | yes | 4 | 3 | 0.35 | 0 | 0 | 0.131 |
| P107   | Other partners with no decided claim since 1 May 2026 | no | 8 | 5 | 0.12 | 0 | 0 | 0.038 |
| P117   | Other partners with no decided claim since 1 May 2026 | no | 4 | 4 | 0.06 | 0 | 0 | 0.036 |
| P177   | Other partners with no decided claim since 1 May 2026 | no | 3 | 3 | 0.05 | 0 | 0 | 0.046 |
| P340   | Other partners with no decided claim since 1 May 2026 | no | 7 | 6 | 0.05 | 0 | 0 | 0.024 |
| P006   | Other partners with no decided claim since 1 May 2026 | no | 4 | 4 | 0.02 | 0 | 0 | 0.006 |

None of these partners can be on the review list, whatever their score. If a new outlet starts committing fraud in July, the frozen history cannot see it; only refreshing the history as outcomes arrive can.

## D. Watchlist: the 10 flagged outlets

The 10 partners with partner_rate_ratio > 1 in the history frozen on 30 Jun 2026, named Outlet A to J for the memo, ordered by expected frauds in the test months. The real partner IDs are in `reports/raw/watchlist.csv`, which is git-ignored.

| Outlet | Decided claims since 1 May | Confirmed frauds | Ratio to average | Test claims (Jul-Sep) | Of which under Rs 2,000 | Expected frauds | Claims on the review list |
|---|---|---|---|---|---|---|---|
| Outlet A | 2 | 2 | 12.0x | 17 | 17 | 11.0 | 17 |
| Outlet B | 2 | 2 | 12.0x | 17 | 15 | 10.8 | 17 |
| Outlet C | 12 | 10 | 23.5x | 13 | 10 | 9.3 | 13 |
| Outlet D | 9 | 4 | 11.6x | 14 | 11 | 8.2 | 14 |
| Outlet E | 13 | 5 | 11.2x | 15 | 12 | 7.0 | 15 |
| Outlet F | 12 | 7 | 16.6x | 11 | 10 | 6.4 | 11 |
| Outlet G | 8 | 3 | 9.5x | 10 | 9 | 5.3 | 10 |
| Outlet H | 6 | 1 | 4.0x | 10 | 7 | 1.0 | 7 |
| Outlet I | 6 | 1 | 4.0x | 3 | 1 | 0.6 | 3 |
| Outlet J | 5 | 1 | 4.5x | 2 | 2 | 0.1 | 1 |
| All 10 | 75 | 36 |  | 112 | 94 | 59.6 | 108 |

Private files written (git-ignored): `reports/raw/watchlist.csv` (10 outlets) and `reports/raw/review_list.csv` (133 rows: for each month the top 40 claims by score and every claim on the review-rule list, with a flag for each; the review-rule list itself is 108 claims: July 34, August 34, September 40).
