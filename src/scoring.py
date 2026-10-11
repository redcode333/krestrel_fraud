"""Score ONE warranty claim with the final model, and explain the score. The app imports this.

It reads only the artefacts in models/ (written by src/final.py):
  model.joblib                 scaler means and scales, coefficients, feature list, Platt parameters, beta
  partner_profile.csv          one row per partner: frozen fraud history as of 30 Jun 2026, recent activity as of 30 Sep 2026.
                               Holds real partner IDs, so it is git-ignored (policy section 10).
  partner_profile.example.csv  the same table with masked partner IDs. Used when the real file is absent, so the app
                               starts on a clean machine; real partner IDs are then scored as unknown partners.
  constants.json               costs, review rule, product catalogue, category lists, cohort statistics
It never touches data/ and needs only numpy and joblib.

Model:  log-odds = base log-odds (logistic regression on label-free features)
                   + beta * log(partner_rate_ratio)          beta = 1.0, ratio = 1.0 with no decided new-regime claim
        probability = sigmoid(platt_slope * log-odds + platt_intercept)   (indicative only)

A claim is a dict with the columns of the claims export. Used: partner_id, sku, submitted_at,
claim_amount_inr, days_since_purchase, customer_prior_claims, photo_attached. The rest is ignored
(inspector_note, partner_inspected and product_serial are deliberately not model inputs).

Explanations never name a city or a partner type. If the model uses them, their contribution is
reported only inside the single line "Outlet profile (statistical pattern, not a finding about this outlet)".
"""
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import csv
import json
import math

import joblib
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
REAL_PROFILE = MODELS / "partner_profile.csv"
EXAMPLE_PROFILE = MODELS / "partner_profile.example.csv"
REQUIRED = ["partner_id", "sku", "submitted_at", "claim_amount_inr", "days_since_purchase", "customer_prior_claims", "photo_attached"]
PARTNER_NUMERIC = ["is_new_partner", "age_days_at_profile", "decided_new_claims", "new_frauds", "smoothed_rate", "rate_ratio",
                   "claims_last_7d", "claims_last_30d", "small_share_last_30d", "avg_amount_last_30d"]
OUTLET_PROFILE_TEXT = "Outlet profile (statistical pattern, not a finding about this outlet)"
IDENTITY_PREFIXES = ("city_", "partner_type_")
AGE_FEATURES = ["partner_age_days", "is_new_partner"]
# Which features belong to which plain-English reason. A name ending in "_" matches every one-hot column with that prefix.
REASON_GROUPS = {
    "claim_amount": ["claim_amount_inr", "amount_to_list_price"],
    "auto_approval": ["is_small"],
    "customer": ["customer_prior_claims"],
    "partner_activity": ["partner_claims_last_7d", "partner_claims_last_30d", "partner_small_share_last_30d", "partner_avg_amount_last_30d"],
    "photo": ["photo_attached"],
    "product": ["warranty_used_share", "family_"],
}


class ClaimError(ValueError):
    """The claim record cannot be scored; the message says what to fix."""


@lru_cache(maxsize=None)
def load_artifacts():
    """model, constants, partner table, and which profile file was read ("real" or "example")."""
    missing = [f for f in ["model.joblib", "constants.json"] if not (MODELS / f).exists()]
    if missing:
        raise FileNotFoundError(f"models/ is missing {missing}. Run: python src/final.py")
    source, path = ("real", REAL_PROFILE) if REAL_PROFILE.exists() else ("example", EXAMPLE_PROFILE)
    if not path.exists():
        raise FileNotFoundError("models/ has neither partner_profile.csv nor partner_profile.example.csv. "
                                "Run: python scripts/build_partner_profile.py")
    model = joblib.load(MODELS / "model.joblib")
    constants = json.loads((MODELS / "constants.json").read_text(encoding="utf-8"))
    with open(path, newline="", encoding="utf-8") as f:
        partners = {row["partner_id"]: {**row, **{k: float(row[k]) for k in PARTNER_NUMERIC}} for row in csv.DictReader(f)}
    return model, constants, partners, source


def unknown_partner():
    """State used for a partner_id that is not in the profile: no history, no recent claims, treated as onboarded today."""
    return {"known": False, "is_new_partner": 1.0, "age_days_at_profile": None,
            "decided_new_claims": 0.0, "new_frauds": 0.0, "smoothed_rate": None, "rate_ratio": 1.0,
            "claims_last_7d": 0.0, "claims_last_30d": 0.0, "small_share_last_30d": 0.0, "avg_amount_last_30d": 0.0}


def _number(claim, key, minimum):
    try:
        value = float(claim[key])
    except (TypeError, ValueError):
        raise ClaimError(f"{key} must be a number, got {claim[key]!r}") from None
    if math.isnan(value) or value < minimum:
        raise ClaimError(f"{key} must be at least {minimum}, got {claim[key]!r}")
    return value


def _timestamp(value):
    try:
        return datetime.fromisoformat(str(value).strip().replace(" ", "T"))
    except ValueError:
        raise ClaimError(f"submitted_at must look like 2026-07-01 14:30, got {value!r}") from None


def claim_features(claim, partner=None):
    """Build the model's features for ONE claim from the partner profile.

    claim     dict with the REQUIRED keys
    partner   the partner's state (a row of the partner profile). Default: looked up by partner_id;
              an unknown partner_id gets unknown_partner().
    Returns (features: {name: value} for every model feature, partner_rate_ratio, partner state used).
    """
    model, constants, partners, _ = load_artifacts()
    if partner is None:
        partner = partners.get(str(claim.get("partner_id", "")).strip())
        partner = {**partner, "known": True} if partner else unknown_partner()
    absent = [k for k in REQUIRED if k not in claim or claim[k] is None or str(claim[k]).strip() == ""]
    if absent:
        raise ClaimError(f"missing field(s): {', '.join(absent)}")
    sku = str(claim["sku"]).strip()
    if sku not in constants["products"]:
        raise ClaimError(f"unknown sku {sku!r}; known: {', '.join(sorted(constants['products']))}")
    product = constants["products"][sku]
    amount = _number(claim, "claim_amount_inr", 0.01)
    days = _number(claim, "days_since_purchase", 0)
    prior = _number(claim, "customer_prior_claims", 0)
    photo = str(claim["photo_attached"]).strip().upper()
    if photo not in ("Y", "N", "TRUE", "FALSE", "1", "0", "YES", "NO"):
        raise ClaimError(f"photo_attached must be Y or N, got {claim['photo_attached']!r}")
    submitted = _timestamp(claim["submitted_at"])

    if partner.get("age_days_at_profile") is None:
        age_days = 0.0
    else:
        as_of = datetime.fromisoformat(constants["profile_as_of"])
        age_days = partner["age_days_at_profile"] + math.floor((submitted - as_of).total_seconds() / 86400)

    values = {
        "partner_claims_last_7d": partner["claims_last_7d"],
        "partner_claims_last_30d": partner["claims_last_30d"],
        "partner_small_share_last_30d": partner["small_share_last_30d"],
        "partner_avg_amount_last_30d": partner["avg_amount_last_30d"],
        "partner_age_days": float(age_days),
        "is_new_partner": float(partner["is_new_partner"]),
        "customer_prior_claims": prior,
        "photo_attached": 1.0 if photo in ("Y", "TRUE", "1", "YES") else 0.0,
        "claim_amount_inr": amount,
        "amount_to_list_price": amount / product["list_price_inr"],
        "is_small": 1.0 if amount < constants["small_claim_limit_inr"] else 0.0,
        "warranty_used_share": days / (product["warranty_months"] * 30),
    }
    categorical = {"family": product["family"], "city": partner.get("city"), "partner_type": partner.get("partner_type")}
    for group, value in categorical.items():
        for category in constants["categories"][group]:
            values[f"{group}_{category}"] = 1.0 if value == category else 0.0
    features = {name: values[name] for name in model["features"]}   # only what the model uses, in its order
    return features, float(partner["rate_ratio"]), partner


def _log_odds_parts(features, ratio, model):
    """Per-feature contributions to the final (Platt-scaled) log-odds, the partner-ratio term, and the baseline."""
    x = np.array([features[name] for name in model["features"]], dtype=float)
    z = (x - model["mean"]) / model["scale"]
    slope = model["platt_slope"]
    contributions = dict(zip(model["features"], slope * model["coef"] * z))
    ratio_term = slope * model["beta"] * math.log(ratio)
    baseline = slope * model["intercept"] + model["platt_intercept"]
    return contributions, ratio_term, baseline


def score_claim(claim, partner=None):
    """Fraud score for one claim: a number strictly between 0 and 1, higher = more likely fraud."""
    model = load_artifacts()[0]
    features, ratio, _ = claim_features(claim, partner)
    contributions, ratio_term, baseline = _log_odds_parts(features, ratio, model)
    return 1 / (1 + math.exp(-(baseline + ratio_term + sum(contributions.values()))))


def reason_groups(model):
    """Feature groups for this model. Partner age and the new-partner flag get their own line unless the model also
    uses city or partner type, in which case all four go under the single outlet-profile line."""
    groups = dict(REASON_GROUPS)
    has_identity = any(name.startswith(IDENTITY_PREFIXES) for name in model["features"])
    has_age = any(name in AGE_FEATURES for name in model["features"])
    if has_identity:
        groups["outlet_profile"] = AGE_FEATURES + list(IDENTITY_PREFIXES)
    elif has_age:
        groups["outlet_age"] = AGE_FEATURES
    return groups


def _in_group(name, members):
    return any(name == m or (m.endswith("_") and name.startswith(m)) for m in members)


def _times(ratio):
    return f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"


def _plural(n, word):
    return f"{n:.0f} {word}" if n == 1 else f"{n:.0f} {word}s"


def _reason_texts(claim, f, ratio, partner, constants, model):
    product = constants["products"][str(claim["sku"]).strip()]
    cohort = constants["cohort"]
    n, frauds = partner["decided_new_claims"], partner["new_frauds"]
    known = partner.get("known", True)
    if not known:
        record = "Partner ID is not in the partner profile: scored as an unknown outlet with no fraud record."
    elif n == 0:
        record = "No decided claims for this partner since 1 May 2026, so its fraud record says nothing either way."
    else:
        record = (f"Partner's confirmed fraud record: {_plural(frauds, 'confirmed fraud')} in {_plural(n, 'decided claim')} "
                  f"since 1 May 2026, {_times(ratio)} the average.")
    new_rate, old_rate = f"{cohort['new_outlet_fraud_rate']:.1%}", f"{cohort['established_outlet_fraud_rate']:.1%}"
    clean = f"{cohort['new_outlets_without_fraud']} of {cohort['new_outlets']} new outlets have no confirmed fraud"
    if not known:
        age = (f"Outlet not in the partner profile, so it is treated as onboarded today: as a group, recently onboarded outlets had a "
               f"fraud rate of {new_rate} after 1 May against {old_rate} for established outlets, but {clean}.")
    elif f.get("is_new_partner", partner["is_new_partner"]):
        age = (f"Outlet onboarded recently: as a group such outlets had a fraud rate of {new_rate} after 1 May against {old_rate} "
               f"for established outlets, but {clean}.")
    elif "partner_age_days" in model["features"]:
        # A model with a continuous age feature was trained mostly on claims from before 1 May, when the confirmed fraud came
        # from established outlets, so this line can raise the score of an established outlet. The text says so.
        age = (f"Established outlet: before 1 May 2026, {cohort['before_may_established_frauds']} of the {cohort['before_may_frauds']} "
               f"confirmed frauds came from established outlets and the model still carries that; after 1 May such outlets had a "
               f"fraud rate of {old_rate} against {new_rate} for recently onboarded outlets.")
    else:
        age = (f"Established outlet: as a group such outlets had a fraud rate of {old_rate} after 1 May against {new_rate} for "
               "recently onboarded outlets.")
    n30 = partner["claims_last_30d"]
    activity = ("Recent partner activity: no claims in its latest 30 days on record." if n30 == 0 else
                f"Recent partner activity: {_plural(n30, 'claim')} in its latest 30 days on record ({partner['claims_last_7d']:.0f} in the "
                f"last 7), {partner['small_share_last_30d']:.0%} of them under Rs 2,000, average Rs {partner['avg_amount_last_30d']:,.0f}.")
    amount = float(claim["claim_amount_inr"])
    list_share = amount / product["list_price_inr"]
    small = amount < constants["small_claim_limit_inr"]
    return {
        "partner_record": record,
        "claim_amount": f"Claim of Rs {amount:,.0f} is {list_share:.0%} of the product's list price.",
        "auto_approval": ("Claim is under Rs 2,000, so it is auto-approved without inspection." if small else
                          "Claim is Rs 2,000 or more, so it still gets a partner inspection."),
        "customer": f"Customer has {_plural(float(claim['customer_prior_claims']), 'earlier warranty claim')}.",
        "partner_activity": activity,
        "photo": "Photo attached." if str(claim["photo_attached"]).strip().upper() in ("Y", "TRUE", "1", "YES") else "No photo attached.",
        "product": (f"{product['family']}, {float(claim['days_since_purchase']) / (product['warranty_months'] * 30):.0%} of the way "
                    f"through its {product['warranty_months']}-month warranty."),
        "outlet_age": age,
        "outlet_profile": OUTLET_PROFILE_TEXT,
    }


def explain(claim, partner=None):
    """Score one claim and say why.

    Each reason carries its signed contribution to the log-odds: coefficient x standardised value summed
    over the features in the group, plus the partner-ratio term. Positive raises the score. The
    contributions and the baseline add up exactly to the score's log-odds.
    """
    model, constants, _, source = load_artifacts()
    features, ratio, partner = claim_features(claim, partner)
    contributions, ratio_term, baseline = _log_odds_parts(features, ratio, model)
    groups = reason_groups(model)
    grouped = {"partner_record": ratio_term}
    for group, members in groups.items():
        grouped[group] = float(sum(v for name, v in contributions.items() if _in_group(name, members)))
    assert abs(sum(grouped.values()) - (ratio_term + sum(contributions.values()))) < 1e-9, "a model feature belongs to no reason group"
    log_odds = baseline + sum(grouped.values())
    score = 1 / (1 + math.exp(-log_odds))
    texts = _reason_texts(claim, features, ratio, partner, constants, model)
    reasons = [{"reason": group, "text": texts[group], "contribution": round(value, 4),
                "effect": "raises the score" if value > 0.02 else "lowers the score" if value < -0.02 else "neutral"}
               for group, value in sorted(grouped.items(), key=lambda kv: -abs(kv[1]))]

    amount, costs, rule = float(claim["claim_amount_inr"]), constants["costs"], constants["review_rule"]
    has_history = partner["decided_new_claims"] > 0
    review = ratio > rule["ratio_threshold"]
    if review:
        recommendation = (f"REVIEW: this partner has confirmed fraud above the average rate since 1 May 2026. Hold the claim for the "
                          f"investigation desk if it is within this month's {rule['monthly_cap']} reviews (highest scores first).")
    elif not has_history:
        recommendation = ("DO NOT HOLD: nothing is decided yet for this partner since 1 May 2026. The score alone is not enough to "
                          "hold a customer's claim; watch the partner as its first outcomes come in.")
    else:
        recommendation = "DO NOT HOLD: this partner has no confirmed fraud above the average rate since 1 May 2026."
    return {
        "claim_id": claim.get("claim_id"),
        "score": score,
        "score_note": "Higher = more likely fraud. Indicative probability only: calibrated on 22 June 2026 frauds.",
        "review_recommended": bool(review),
        "recommendation": recommendation,
        "recommendation_note": rule["label"],
        "reasons": reasons,
        "baseline_log_odds": round(baseline, 4),
        "log_odds": round(log_odds, 4),
        "partner_rate_ratio": round(ratio, 4),
        "partner_has_history": bool(has_history),
        "partner_known": bool(partner.get("known", True)),
        "partner_profile_source": source,
        "partner_profile_note": ("" if source == "real" else
                                 "Running on the EXAMPLE partner profile (masked partner IDs). Real partner IDs are not in it and are "
                                 "scored as unknown outlets. Build the real profile with: python scripts/build_partner_profile.py"),
        "expected_value_of_review_inr": round(score * amount - (1 - score) * costs["goodwill_inr"] - costs["service_contact_inr"], 0),
        "break_even_probability": round((costs["goodwill_inr"] + costs["service_contact_inr"]) / (amount + costs["goodwill_inr"]), 4),
        "history_frozen_on": constants["history_frozen_on"],
        "partner_activity_as_of": constants["profile_date"],
    }
