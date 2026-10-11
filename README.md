# Kestrel Home - warranty claim fraud scoring

Placeholder. The full README comes with the app: setup on a clean machine, how to start the endpoint and the screen, how to regenerate predictions.csv. The notes below are already true and will be kept.

## Python and environments

- **Python 3.12 or newer.** Tested on 3.12.4 and 3.14.2 with the pinned versions in `requirements.txt`; both give identical `predictions.csv`. Python 3.11 and older are not supported by these pins.
- `python -m venv .venv` then `pip install -r requirements.txt` is everything the data pipeline, the final model, `predictions.csv`, the scorer and the checks need.
- `requirements-dev.txt` adds LightGBM for `src/train.py` only (the model comparison). The final model and the app never need it; use a second environment (`.venv-dev`) for it.

## Partner profile and policy section 10

- The scorer reads one row per partner from `models/partner_profile.csv`. That file holds real partner IDs next to confirmed-fraud counts, so it is **git-ignored** and is not in this repository.
- `models/partner_profile.example.csv` is committed instead: the same table with masked partner IDs (P001, P002, ...) and onboarding age rounded down to 30 days. **If the real file is missing, the scorer falls back to the example file and says so in its output.** With the example file the app starts and runs, but a real partner ID is not found and is scored as an unknown outlet with no history.
- With the Kestrel data pack in `data/`, `python scripts/build_partner_profile.py` rebuilds the real profile and checks that it reproduces the existing one exactly.

## Decisions for Kestrel

- **Partner IDs in the shipped model.** The service has to look a partner up by its real ID to score a claim. The real partner table is kept out of this public repository (policy section 10) and delivered privately or rebuilt from the data pack. Kestrel needs to decide where that table lives in production. Reports in this repository show masked IDs and the mapping is kept out of git.

## One definition of "new outlet"

Wherever this repository counts new outlets (reports, the form text, the explanation line shown in the app) it means: **outlets onboarded on or after 1 Oct 2025 with at least one decided claim first submitted on or after 1 May 2026**. On the data up to 30 Jun 2026 there are 45 of them, of which 8 have a confirmed fraud and 37 have none (`new_outlet_counts` in `src/features.py`; printed in `reports/explore.txt` and `reports/final.txt`). About 60 outlets were onboarded in that period in total; the others have no decided claim since 1 May.

## Reproducibility

Every bootstrap and tie-break uses a fixed seed (`BOOTSTRAP_SEED` in `src/evaluate.py`), so two runs give identical reports, intervals, `predictions.csv` and model files.

## What the review list depends on

The recommended review list (claims whose partner has confirmed fraud above the average rate, highest score first, at most 40 a month) depends only on which partners have a confirmed fraud record. It is the same list whichever base model is used (25 reviews, 13 frauds, net Rs 8,039 on June 2026 at a 14-day lag); the model only orders the claims inside it and the claims beyond it. The frauds it caught averaged Rs 1,469, so it breaks even at 9 frauds in 25 reviews and had 13. The rule was defined after seeing June and is not independently validated: one month, 8 partners.

## Layout

- `data/` - client data pack (git-ignored, never committed; ops-policy section 10)
- `src/` - data preparation (`clean.py`), exploration, evaluation harness, features, model comparison (`train.py`), final model (`final.py`), one-claim scorer (`scoring.py`), submission check (`validate_submission.py`)
- `scripts/` - `build_partner_profile.py`
- `app/` - scoring endpoint and the screen that calls it (to come)
- `models/` - `model.joblib`, `constants.json`, `partner_profile.example.csv` (summaries only, no customer rows)
- `reports/` - evaluation outputs with partner IDs masked; `reports/raw/` holds the unmasked copies and is git-ignored
- `predictions.csv` - claim_id,score for every row of test_unlabelled.csv
- `memo.md` - one-page memo to Ritu
- `AI_LOG.md` - decisions and discarded approaches
