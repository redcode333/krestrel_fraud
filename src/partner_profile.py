"""The partner profile the scorer reads: one row per partner, summaries only, no claim rows.

  frozen label-based history as of 30 Jun 2026   decided new-regime claims, confirmed frauds, smoothed rate, ratio
  label-free rolling activity as of 30 Sep 2026  claims in the last 7 and 30 days, small share, average amount
  onboarding age at the profile date and the new-partner flag

Two files:
  models/partner_profile.csv          real partner IDs. Git-ignored (policy section 10).
  models/partner_profile.example.csv  the same rows with masked partner IDs (the private mapping in
                                      data/partner_mask.csv) and onboarding age rounded down to 30 days, so it
                                      can be committed and the app can start on a clean machine without data.

City and partner type are deliberately not in the profile: the final model does not use them and
explanations must never name them.
"""
from pathlib import Path

import pandas as pd

from clean import DATA, TIME_COL
from features import (FINAL_CUTOFF, NEW_PARTNER_FROM, SMOOTH_K, _label_features, load_events, load_history, partner_label_table,
                      rolling_partner_features)
from reporting import partner_mask

MODELS = Path(__file__).resolve().parents[1] / "models"
REAL_PROFILE = MODELS / "partner_profile.csv"
EXAMPLE_PROFILE = MODELS / "partner_profile.example.csv"
FINAL_L = 0                                  # every outcome known by 30 Jun 2026 is used
PROFILE_AS_OF = pd.Timestamp("2026-10-01")   # i.e. the end of 30 Sep 2026, the last day of claims we hold
EXAMPLE_AGE_STEP = 30                        # days; an exact onboarding age would identify the partner


def build_partner_profile():
    """The real profile as a DataFrame, and the overall new-regime fraud rate behind the ratio."""
    partners = pd.read_csv(DATA / "partners.csv", parse_dates=["onboarded_date"])
    frame = partners.assign(**{TIME_COL: PROFILE_AS_OF})
    table, overall = partner_label_table(load_history(), FINAL_CUTOFF, FINAL_L)
    label = _label_features(frame, table, overall, SMOOTH_K)
    rolling = rolling_partner_features(frame, load_events())
    profile = pd.DataFrame({
        "partner_id": partners["partner_id"],
        "is_new_partner": (partners["onboarded_date"] >= NEW_PARTNER_FROM).astype(int),
        "age_days_at_profile": (PROFILE_AS_OF - partners["onboarded_date"]).dt.days,
        "decided_new_claims": label["partner_new_claim_count"].astype(int), "new_frauds": label["partner_new_fraud_count"].astype(int),
        "smoothed_rate": label["partner_new_rate_smoothed"], "rate_ratio": label["partner_rate_ratio"],
        "claims_last_7d": rolling["partner_claims_last_7d"].astype(int), "claims_last_30d": rolling["partner_claims_last_30d"].astype(int),
        "small_share_last_30d": rolling["partner_small_share_last_30d"], "avg_amount_last_30d": rolling["partner_avg_amount_last_30d"],
    }).sort_values("partner_id").reset_index(drop=True)
    assert profile["partner_id"].is_unique and "claim_id" not in profile.columns
    return profile, overall


def example_profile(profile):
    """Same schema, masked partner IDs, onboarding age rounded down to a multiple of 30 days."""
    mapping = partner_mask()
    example = profile.copy()
    example["partner_id"] = example["partner_id"].map(mapping)
    assert example["partner_id"].notna().all() and example["partner_id"].is_unique
    example["age_days_at_profile"] = (example["age_days_at_profile"] // EXAMPLE_AGE_STEP) * EXAMPLE_AGE_STEP
    return example.sort_values("partner_id").reset_index(drop=True)


def csv_text(frame):
    """Exactly what is written to disk, so two builds can be compared byte for byte."""
    return frame.to_csv(index=False, lineterminator="\n")


def write_profiles(profile):
    MODELS.mkdir(exist_ok=True)
    REAL_PROFILE.write_text(csv_text(profile), encoding="utf-8", newline="")
    EXAMPLE_PROFILE.write_text(csv_text(example_profile(profile)), encoding="utf-8", newline="")
