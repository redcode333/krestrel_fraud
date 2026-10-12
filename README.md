# Kestrel Home: warranty claim review

Kestrel Home asked for a model that flags fraudulent warranty claims before they are paid. This repository holds:

- a **service** that scores one claim, says what to do with it and why (`app/`),
- the **scores for the 2,252 test claims** (`predictions.csv`),
- the **pipeline** that produced the model, with its evidence (`src/`, `reports/`),
- the **memo** to the client (`memo.md`) and the log of every decision and discarded approach (`AI_LOG.md`).

The service needs no paid API, no key and no data. It starts from this README on a clean machine.

## Run the service

You need **Python 3.12 or newer** (tested on 3.12.4 and 3.14.2). Run these from the repository root.

**Windows (PowerShell or Command Prompt)**

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If `python` is not version 3.12 or newer, use `py -3.12 -m venv .venv` for the first line.

**macOS or Linux**

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000**. The page has a form, a Score button and three example buttons (a high-risk outlet, an ordinary claim, a new outlet). Stop the service with Ctrl+C.

## The endpoint

`POST /score` takes **one claim as a JSON object** and returns the score, the risk band, whether the review rule applies, two to four plain-English reasons and a list of notes.

macOS or Linux:

```
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{"claim_id": "EXAMPLE-1", "submitted_at": "2026-10-05 11:20", "partner_id": "P077", "sku": "KH-MG-02", "product_serial": "KH482915530", "days_since_purchase": 210, "claim_amount_inr": 1450, "photo_attached": "N", "customer_prior_claims": 1}'
```

Windows PowerShell:

```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/score -ContentType "application/json" -Body '{"claim_id": "EXAMPLE-1", "submitted_at": "2026-10-05 11:20", "partner_id": "P077", "sku": "KH-MG-02", "product_serial": "KH482915530", "days_since_purchase": 210, "claim_amount_inr": 1450, "photo_attached": "N", "customer_prior_claims": 1}' | ConvertTo-Json -Depth 5
```

The answer (shortened):

```json
{
  "score": 0.8901,
  "score_label": "Indicative only. A ranking score between 0 and 1, higher = more likely fraud; it is not a calibrated probability ...",
  "band": "Review",
  "band_meaning": "The review rule applies: this outlet's confirmed fraud rate since 1 May 2026 is above the average. ...",
  "review_rule_applies": true,
  "reasons": [
    {"text": "Partner's confirmed fraud record: 10 confirmed frauds in 12 decided claims since 1 May 2026, 24x the average.", "effect": "raises the score"},
    {"text": "Outlet onboarded recently: as a group such outlets had a fraud rate of 16.1% after 1 May against 0.2% for established outlets, but 37 of 45 new outlets have no confirmed fraud.", "effect": "raises the score"}
  ],
  "notes": [
    "product_serial is accepted but the model does not use it: serials do not identify individual units in this data.",
    "About the review rule: rule defined after seeing June; not independently validated; one month, 8 partners."
  ],
  "partner_profile": "example"
}
```

`P077` is a masked outlet ID from the example partner profile (see below).

**Fields used:** `partner_id`, `sku`, `submitted_at` (like `2026-10-05 11:20`, Indian Standard Time), `claim_amount_inr`, `days_since_purchase`, `customer_prior_claims`, `photo_attached` (`Y` or `N`). `product_serial` and `claim_id` are accepted. Every other column of the claims export is accepted and ignored.

**What happens with a bad or odd record** (the service never answers with a 500):

| Record | Answer |
|---|---|
| A required field is missing or blank | 422, naming every missing field |
| Wrong type (text where a number belongs, a list, and so on) | 422, naming the field |
| A date that cannot be read | 422, naming `submitted_at` |
| Zero or negative amount | 422, naming `claim_amount_inr` |
| Not JSON, or JSON that is not one object | 400 |
| Unknown partner ID | Scored as a newly onboarded outlet with no history; a note says so |
| Unknown product SKU | Scored with list price, warranty and product family set to neutral averages; a note says so |
| Amount above the list price, or claim outside warranty | Scored; a note says so |
| Any serial | Accepted; a note says the model does not use it |

An error looks like this: `{"error": "This claim cannot be scored: claim_amount_inr must be greater than 0, got -5.", "field": "claim_amount_inr", "fields": ["claim_amount_inr"]}`.

`GET /health` says which partner profile is loaded. `GET /examples` returns the three example claims.

## Risk bands

| Band | When | What to do |
|---|---|---|
| **Review** | Only when the review rule applies: the outlet's confirmed fraud rate since 1 May 2026 is above the average | Hold the claim for the investigation desk if it is within the month's 40 reviews, highest scores first |
| **Watch** | The outlet has no decided claim since 1 May 2026 and the score would rank in the top 100 of the Jul-Sep 2026 test claims | Pay the claim. The outlet cannot qualify for review yet; re-check it when its first claims are decided |
| **Normal** | Everything else | Pay as usual |

The score alone never puts a claim in Review. A high score from an outlet with a clean or unknown record is not enough to hold a genuine customer's repair. The review rule was defined after seeing June 2026 and is not independently validated (one month, 8 partners).

## What the score means, and what it does not

- It is a **ranking**: higher means more likely fraud. It is the right thing to sort by.
- It is **not a calibrated probability**. The correction that turns it into a number between 0 and 1 rests on 22 confirmed frauds from one month. Treat the size of the number as indicative.
- It rests on **one month of the new fraud pattern** (June 2026: 22 frauds from 8 partners). On that month the model scores PR-AUC 0.465 (90% interval 0.057 to 0.661); the honest expectation for new data is lower, about 0.30 to 0.35.
- It **cannot see a new fraud pattern**. On May 2026, the first month after the policy change, every fitted model ranked claims backwards. If the pattern moves again, the model is wrong until new outcomes arrive.
- Most of its power comes from **each outlet's confirmed fraud record**. An outlet with no decided claims is scored on the claim's own details only.
- "Accuracy above 97%" is not reported, because predicting "not fraud" for every June claim is already 96.9% accurate and catches nothing.

## Partner profile, and why old claims score slightly differently here

The service looks each outlet up in a **partner profile**: its fraud record frozen on 30 Jun 2026 and its recent claim activity as of 30 Sep 2026.

- `models/partner_profile.csv` holds real partner IDs. It is **not in this repository**.
- `models/partner_profile.example.csv` is the same table with **masked outlet IDs** (P001, P002, ...) and onboarding age rounded down to 30 days. **If the real file is missing, the service uses the example file and says so** on the page, in `/health` and in every answer (`"partner_profile": "example"`). With the example file a real partner ID is not found and is scored as an unknown outlet.
- **Scores for old claims can differ slightly from `predictions.csv`.** The file was scored with each outlet's activity as it stood when the claim was submitted; the service reads activity as of 30 Sep 2026. Re-entering a test claim gives a median difference of about 0.001 (rank correlation 0.97). `predictions.csv` is the submission; the service is for new claims.

## One definition of "new outlet"

Wherever this repository counts new outlets (reports, the form text, the explanation line shown by the service) it means: **outlets onboarded on or after 1 Oct 2025 with at least one decided claim first submitted on or after 1 May 2026**. On the data up to 30 Jun 2026 there are 45 of them, of which 8 have a confirmed fraud and 37 have none (`new_outlet_counts` in `src/features.py`; printed in `reports/explore.txt` and `reports/final.txt`). About 60 outlets were onboarded in that period in total; the others have no decided claim since 1 May.

## Confidentiality (ops policy section 10)

The policy forbids publishing customer and operational data. What that means here:

- **Not in the repository:** the data pack (`data/`), the real partner profile (`models/partner_profile.csv`), the unmasked reports and private lists (`reports/raw/`), and the file that maps real partner IDs to masked ones (`data/partner_mask.csv`). All are git-ignored.
- **In the repository:** code, `predictions.csv` (claim ID and score only), the model as plain numbers (`models/model.joblib`), constants, the example partner profile, and reports in which every partner ID is masked. Explanations never name a city or a partner type.
- **To rebuild the private files:** put the Kestrel data pack in `data/` and run `python scripts/build_partner_profile.py` (it rebuilds the real partner profile and checks it against the existing one). `python src/final.py` rebuilds the model, the constants and `predictions.csv`.

## Tests

Windows:

```
.venv\Scripts\python -m pip install -r requirements-test.txt
.venv\Scripts\python -m pytest
```

macOS or Linux:

```
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest
```

The service tests (one per edge case above, plus a check that the service returns exactly what `src/scoring.py` computes) need no data and pass on the example profile alone. The pipeline checks (leakage tests, profile rebuild, masking) run only when the data pack is in `data/`, and are skipped otherwise.

## Rebuild everything from the data pack

With the data pack in `data/`:

```
python src/clean.py                  cleaned claims
python src/explore.py                reports/explore.txt
python src/evaluate.py               reports/evaluate_harness.txt
python src/features.py               reports/features.txt, with the leakage tests
python src/train.py                  reports/train.txt (needs requirements-dev.txt: LightGBM, for one comparison model)
python src/final.py                  the model, constants, partner profiles, predictions.csv, reports/final.txt
python src/validate_submission.py    checks predictions.csv
python src/rupees.py                 reports/rupees.md and reports/timing.txt
```

Every bootstrap and tie-break uses a fixed seed (`BOOTSTRAP_SEED` in `src/evaluate.py`), so two runs give identical reports, intervals, `predictions.csv` and model files. **The one exception is `reports/timing.txt`**: it holds the measured scoring time, which depends on the machine and the moment, so that file alone varies between runs.

## What the review list depends on

The recommended review list (claims whose partner has confirmed fraud above the average rate, highest score first, at most 40 a month) depends only on which partners have a confirmed fraud record. It is the same list whichever base model is used (25 reviews, 13 frauds, net Rs 8,039 on June 2026 at a 14-day lag); the model only orders the claims inside it and the claims beyond it. The frauds it caught averaged Rs 1,469, so it breaks even at 9 frauds in 25 reviews and had 13. The rupee analysis is in `reports/rupees.md`.

## Decisions for Kestrel

- **Partner IDs in the shipped model.** The service has to look a partner up by its real ID to score a claim. The real partner table is kept out of this repository (policy section 10) and delivered privately or rebuilt from the data pack. Kestrel needs to decide where that table lives in production.
- **Keeping the history fresh.** The fraud record is frozen on 30 Jun 2026. In live use it has to be refreshed as investigations finish; frozen, the service can never flag an outlet that turns fraudulent later.

## Layout

- `app/` - the service: `main.py` (endpoint) and `page.html` (the one screen)
- `src/` - `scoring.py` (score, explain and band for one claim; the only code the service uses), data preparation (`clean.py`), exploration, evaluation harness, features, model comparison (`train.py`), final model (`final.py`), rupee analysis (`rupees.py`), submission check (`validate_submission.py`)
- `scripts/` - `build_partner_profile.py`
- `tests/` - service tests, submission checks, pipeline checks
- `models/` - `model.joblib`, `constants.json`, `partner_profile.example.csv` (summaries only, no customer rows)
- `reports/` - evidence, with partner IDs masked; `reports/raw/` holds the unmasked copies and is git-ignored
- `data/` - the client data pack (git-ignored, never committed)
- `predictions.csv` - claim_id,score for every row of test_unlabelled.csv
- `memo.md` - one-page memo to Ritu
- `AI_LOG.md` - decisions and discarded approaches
