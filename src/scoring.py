"""Score ONE warranty claim with the final model, explain the score and give it a risk band. The app imports this.

It reads only the artefacts in models/ (written by src/final.py):
  model.joblib                 scaler means and scales, coefficients, feature list, Platt parameters, beta
  partner_profile.csv          one row per partner: frozen fraud history as of 30 Jun 2026, recent activity as of 30 Sep 2026.
                               Holds real partner IDs, so it is git-ignored (policy section 10).
  partner_profile.example.csv  the same table with masked partner IDs. Used when the real file is absent, so the app
                               starts on a clean machine; real partner IDs are then scored as unknown outlets.
  constants.json               costs, review rule, product catalogue, category lists, cohort statistics, Watch threshold
It never touches data/ and needs only numpy and joblib.

Model:  log-odds = base log-odds (logistic regression on label-free features)
                   + beta * log(partner_rate_ratio)          beta = 1.0, ratio = 1.0 with no decided new-regime claim
        probability = sigmoid(platt_slope * log-odds + platt_intercept)   (indicative only)

Entry points:
  score_claim(claim)   the score, a number between 0 and 1
  explain(claim)       the score with every reason and its signed contribution to the log-odds
  assess(claim)        what the service returns: score, risk band, whether the review rule applies, 2 to 4 reasons, notes
  claim_features(claim) the model's features for one claim, built from the partner profile

A claim is a dict with the columns of the claims export. Used: partner_id, sku, submitted_at,
claim_amount_inr, days_since_purchase, customer_prior_claims, photo_attached. product_serial is accepted
and not used. Everything else is ignored (inspector_note and partner_inspected are deliberately not inputs).
A bad record raises ClaimError, which names the field to fix.

Explanations never name a city or a partner type. If the model uses them, their contribution is
reported only inside the single line "Outlet profile (statistical pattern, not a finding about this outlet)".
"""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import csv
import json
import math
import numbers
import os

import joblib
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
REAL_PROFILE = MODELS / "partner_profile.csv"
EXAMPLE_PROFILE = MODELS / "partner_profile.example.csv"
PROFILE_ENV = "KESTREL_PARTNER_PROFILE"   # set to "example" to use the example profile even when the real one is present
REQUIRED = ["partner_id", "sku", "submitted_at", "claim_amount_inr", "days_since_purchase", "customer_prior_claims", "photo_attached"]
PARTNER_NUMERIC = ["is_new_partner", "age_days_at_profile", "decided_new_claims", "new_frauds", "smoothed_rate", "rate_ratio",
                   "claims_last_7d", "claims_last_30d", "small_share_last_30d", "avg_amount_last_30d"]
PRODUCT_FEATURES = ("amount_to_list_price", "warranty_used_share", "family_")   # unknown when the SKU is not in the catalogue
OUTLET_PROFILE_TEXT = "Outlet profile (statistical pattern, not a finding about this outlet)"
IDENTITY_PREFIXES = ("city_", "partner_type_")
AGE_FEATURES = ["partner_age_days", "is_new_partner"]
IST = timezone(timedelta(hours=5, minutes=30))
# Which features belong to which plain-English reason. A name ending in "_" matches every one-hot column with that prefix.
REASON_GROUPS = {
    "claim_amount": ["claim_amount_inr", "amount_to_list_price"],
    "auto_approval": ["is_small"],
    "customer": ["customer_prior_claims"],
    "partner_activity": ["partner_claims_last_7d", "partner_claims_last_30d", "partner_small_share_last_30d", "partner_avg_amount_last_30d"],
    "photo": ["photo_attached"],
    "product": ["warranty_used_share", "family_"],
}
BANDS = {
    "Review": "The review rule applies: this outlet's confirmed fraud rate since 1 May 2026 is above the average. Hold the claim for the "
              "investigation desk if it is within the month's 40 reviews, highest scores first.",
    "Watch": "No confirmed record for this outlet yet and the score is high. It cannot qualify for review; pay the claim and re-check the "
             "outlet when its first claims are decided.",
    "Normal": "No reason to hold this claim. Pay as usual.",
}
WATCH_REASON = "No confirmed record for this outlet yet, so it cannot qualify for review; re-check when its first claims are decided."
SCORE_LABEL = ("Indicative only. A ranking score between 0 and 1, higher = more likely fraud; it is not a calibrated probability "
               "(the correction behind it rests on 22 confirmed frauds from June 2026).")


class ClaimError(ValueError):
    """The claim record cannot be scored. str() says what to fix; .field names the field (.fields when several)."""

    def __init__(self, message, field=None, fields=None):
        super().__init__(message)
        self.fields = list(fields) if fields else ([field] if field else [])
        self.field = self.fields[0] if self.fields else None


@lru_cache(maxsize=None)
def load_artifacts():
    """model, constants, partner table, and which profile file was read ("real" or "example")."""
    missing = [f for f in ["model.joblib", "constants.json"] if not (MODELS / f).exists()]
    if missing:
        raise FileNotFoundError(f"models/ is missing {missing}. Run: python src/final.py")
    use_real = REAL_PROFILE.exists() and os.environ.get(PROFILE_ENV, "").strip().lower() != "example"
    source, path = ("real", REAL_PROFILE) if use_real else ("example", EXAMPLE_PROFILE)
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


# ----------------------------------------------------------------------------------------------
# reading one claim
# ----------------------------------------------------------------------------------------------
def _show(value):
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _number(claim, key, minimum=None, above=None, whole=False):
    value = claim[key]
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, str)):   # numbers.Real also covers numpy numbers
        raise ClaimError(f"{key} must be a number, got {_show(value)}", key)
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ClaimError(f"{key} must be a number, got {_show(value)}", key) from None
    if not math.isfinite(number):
        raise ClaimError(f"{key} must be a finite number, got {_show(value)}", key)
    if above is not None and number <= above:
        raise ClaimError(f"{key} must be greater than {above}, got {_show(value)}", key)
    if minimum is not None and number < minimum:
        raise ClaimError(f"{key} must be {minimum} or more, got {_show(value)}", key)
    if whole and number != int(number):
        raise ClaimError(f"{key} must be a whole number, got {_show(value)}", key)
    return number


def _text(claim, key):
    value = claim[key]
    if not isinstance(value, str):
        raise ClaimError(f"{key} must be text, got {_show(value)}", key)
    value = value.strip()
    if len(value) > 64:
        raise ClaimError(f"{key} is too long ({len(value)} characters; 64 at most)", key)
    return value


def _timestamp(value):
    if not isinstance(value, str):
        raise ClaimError(f"submitted_at must be a date and time written as text, like 2026-07-01 14:30; got {_show(value)}", "submitted_at")
    try:
        stamp = datetime.fromisoformat(value.strip().replace(" ", "T", 1))
    except ValueError:
        raise ClaimError(f"submitted_at must look like 2026-07-01 14:30, got {_show(value)}", "submitted_at") from None
    if stamp.tzinfo is not None:   # the export is in Indian Standard Time
        stamp = stamp.astimezone(IST).replace(tzinfo=None)
    return stamp


def parse_claim(claim):
    """Validate one claim record and return its cleaned values. Raises ClaimError naming the field."""
    if not isinstance(claim, dict):
        raise ClaimError("the claim must be one JSON object with named fields")
    absent = [k for k in REQUIRED if k not in claim or claim[k] is None or (isinstance(claim[k], str) and claim[k].strip() == "")]
    if absent:
        raise ClaimError(f"missing field{'s' if len(absent) > 1 else ''}: {', '.join(absent)}", fields=absent)
    photo = claim["photo_attached"]
    photo_text = str(photo).strip().upper() if isinstance(photo, (str, bool)) else None
    if photo_text not in ("Y", "N", "YES", "NO", "TRUE", "FALSE"):
        raise ClaimError(f"photo_attached must be Y or N, got {_show(photo)}", "photo_attached")
    serial = claim.get("product_serial")
    return {
        "claim_id": claim.get("claim_id") if isinstance(claim.get("claim_id"), (str, int)) else None,
        "partner_id": _text(claim, "partner_id"), "sku": _text(claim, "sku"),
        "submitted_at": _timestamp(claim["submitted_at"]),
        "amount": _number(claim, "claim_amount_inr", above=0),
        "days": _number(claim, "days_since_purchase", minimum=0),
        "prior": _number(claim, "customer_prior_claims", minimum=0, whole=True),
        "photo": photo_text in ("Y", "YES", "TRUE"),
        "serial_given": serial is not None and str(serial).strip() != "",
    }


def _context(claim, partner=None):
    """Everything needed to score one claim: cleaned values, the partner's state, the product, the features."""
    model, constants, partners, source = load_artifacts()
    c = parse_claim(claim)
    if partner is None:
        partner = partners.get(c["partner_id"])
        partner = {**partner, "known": True} if partner else unknown_partner()
    product = constants["products"].get(c["sku"])   # None when the SKU is not in the catalogue

    if partner.get("age_days_at_profile") is None:
        age_days = 0.0
    else:
        as_of = datetime.fromisoformat(constants["profile_as_of"])
        age_days = partner["age_days_at_profile"] + math.floor((c["submitted_at"] - as_of).total_seconds() / 86400)

    values = {
        "partner_claims_last_7d": partner["claims_last_7d"],
        "partner_claims_last_30d": partner["claims_last_30d"],
        "partner_small_share_last_30d": partner["small_share_last_30d"],
        "partner_avg_amount_last_30d": partner["avg_amount_last_30d"],
        "partner_age_days": float(age_days),
        "is_new_partner": float(partner["is_new_partner"]),
        "customer_prior_claims": c["prior"],
        "photo_attached": 1.0 if c["photo"] else 0.0,
        "claim_amount_inr": c["amount"],
        "is_small": 1.0 if c["amount"] < constants["small_claim_limit_inr"] else 0.0,
    }
    categorical = {"city": partner.get("city"), "partner_type": partner.get("partner_type")}
    if product is not None:
        values["amount_to_list_price"] = c["amount"] / product["list_price_inr"]
        values["warranty_used_share"] = c["days"] / (product["warranty_months"] * 30)
        categorical["family"] = product["family"]
    for group, value in categorical.items():
        for category in constants["categories"][group]:
            values[f"{group}_{category}"] = 1.0 if value == category else 0.0
    # A product-dependent feature we cannot compute takes the training average, so it moves the score by nothing.
    neutral = dict(zip(model["features"], model["mean"]))
    features = {name: float(values[name]) if name in values else float(neutral[name]) for name in model["features"]}
    assert product is not None or all(n in values or n.startswith(PRODUCT_FEATURES) for n in model["features"])
    return {"claim": c, "partner": partner, "product": product, "features": features, "ratio": float(partner["rate_ratio"]),
            "model": model, "constants": constants, "source": source}


def claim_features(claim, partner=None):
    """Build the model's features for ONE claim from the partner profile.

    claim     dict with the REQUIRED keys
    partner   the partner's state (a row of the partner profile). Default: looked up by partner_id;
              an unknown partner_id gets unknown_partner().
    Returns (features: {name: value} for every model feature in the model's order, partner_rate_ratio, partner state used).
    """
    ctx = _context(claim, partner)
    return ctx["features"], ctx["ratio"], ctx["partner"]


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------
def _sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def _log_odds_parts(features, ratio, model):
    """Per-feature contributions to the final (Platt-scaled) log-odds, the partner-ratio term, and the baseline."""
    x = np.array([features[name] for name in model["features"]], dtype=float)
    z = (x - model["mean"]) / model["scale"]
    slope = model["platt_slope"]
    contributions = dict(zip(model["features"], (slope * model["coef"] * z).tolist()))
    ratio_term = slope * model["beta"] * math.log(ratio)
    baseline = slope * model["intercept"] + model["platt_intercept"]
    return contributions, ratio_term, baseline


def score_claim(claim, partner=None):
    """Fraud score for one claim: a number between 0 and 1, higher = more likely fraud."""
    ctx = _context(claim, partner)
    contributions, ratio_term, baseline = _log_odds_parts(ctx["features"], ctx["ratio"], ctx["model"])
    return _sigmoid(baseline + ratio_term + sum(contributions.values()))


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


def _reason_texts(ctx):
    c, partner, product, constants, model, ratio = (ctx["claim"], ctx["partner"], ctx["product"], ctx["constants"], ctx["model"], ctx["ratio"])
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
    elif partner["is_new_partner"]:
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
    amount = c["amount"]
    if product is None:
        amount_text = f"Claim of Rs {amount:,.0f}; the product is not recognised, so its share of the list price is treated as average."
        product_text = "Product not recognised: its family and warranty stage are treated as average and do not move the score."
    else:
        amount_text = f"Claim of Rs {amount:,.0f} is {amount / product['list_price_inr']:.0%} of the product's list price."
        product_text = (f"{product['family']}, {c['days'] / (product['warranty_months'] * 30):.0%} of the way through its "
                        f"{product['warranty_months']}-month warranty.")
    return {
        "partner_record": record,
        "claim_amount": amount_text,
        "auto_approval": ("Claim is under Rs 2,000, so it is auto-approved without inspection."
                          if amount < constants["small_claim_limit_inr"] else "Claim is Rs 2,000 or more, so it still gets a partner inspection."),
        "customer": f"Customer has {_plural(c['prior'], 'earlier warranty claim')}.",
        "partner_activity": activity,
        "photo": "Photo attached." if c["photo"] else "No photo attached.",
        "product": product_text,
        "outlet_age": age,
        "outlet_profile": OUTLET_PROFILE_TEXT,
    }


def _explain(ctx):
    model, constants, partner, ratio = ctx["model"], ctx["constants"], ctx["partner"], ctx["ratio"]
    contributions, ratio_term, baseline = _log_odds_parts(ctx["features"], ratio, model)
    grouped = {"partner_record": ratio_term}
    for group, members in reason_groups(model).items():
        grouped[group] = float(sum(v for name, v in contributions.items() if _in_group(name, members)))
    assert abs(sum(grouped.values()) - (ratio_term + sum(contributions.values()))) < 1e-9, "a model feature belongs to no reason group"
    log_odds = baseline + ratio_term + sum(contributions.values())   # the same sum, in the same order, as score_claim
    score = _sigmoid(log_odds)
    texts = _reason_texts(ctx)
    reasons = [{"reason": group, "text": texts[group], "contribution": round(value, 4),
                "effect": "raises the score" if value > 0.02 else "lowers the score" if value < -0.02 else "neutral"}
               for group, value in sorted(grouped.items(), key=lambda kv: -abs(kv[1]))]

    amount, costs, rule = ctx["claim"]["amount"], constants["costs"], constants["review_rule"]
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
        "claim_id": ctx["claim"]["claim_id"],
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
        "partner_profile_source": ctx["source"],
        "partner_profile_note": ("" if ctx["source"] == "real" else
                                 "Running on the EXAMPLE partner profile (masked partner IDs). Real partner IDs are not in it and are "
                                 "scored as unknown outlets. Build the real profile with: python scripts/build_partner_profile.py"),
        "expected_value_of_review_inr": round(score * amount - (1 - score) * costs["goodwill_inr"] - costs["service_contact_inr"], 0),
        "break_even_probability": round((costs["goodwill_inr"] + costs["service_contact_inr"]) / (amount + costs["goodwill_inr"]), 4),
        "history_frozen_on": constants["history_frozen_on"],
        "partner_activity_as_of": constants["profile_date"],
    }


def explain(claim, partner=None):
    """Score one claim and say why.

    Each reason carries its signed contribution to the log-odds: coefficient x standardised value summed
    over the features in the group, plus the partner-ratio term. Positive raises the score. The
    contributions and the baseline add up exactly to the score's log-odds.
    """
    return _explain(_context(claim, partner))


# ----------------------------------------------------------------------------------------------
# what the service returns
# ----------------------------------------------------------------------------------------------
def risk_band(review_rule_applies, partner_has_history, score, constants):
    """Review only when the review rule applies. Watch = no decided history for the outlet and a score that would be in
    the top 100 of the Jul-Sep 2026 test claims. Everything else Normal."""
    if review_rule_applies:
        return "Review"
    threshold = constants.get("watch_score_threshold")
    if not partner_has_history and threshold is not None and score >= threshold:
        return "Watch"
    return "Normal"


def _notes(ctx, band):
    c, partner, product, constants = ctx["claim"], ctx["partner"], ctx["product"], ctx["constants"]
    notes = []
    if not partner.get("known", True):
        note = f"Outlet {c['partner_id']!r} is not in the partner profile: it is treated as newly onboarded with no history."
        if ctx["source"] == "example":
            note += " The service is running on the example profile, which uses masked outlet IDs such as P001."
        notes.append(note)
    elif partner["decided_new_claims"] == 0:
        notes.append("No decided claims for this outlet since 1 May 2026, so the review rule cannot apply to it yet.")
    if product is None:
        notes.append(f"Product SKU {c['sku']!r} is not in the catalogue: list price, warranty and product family were set to neutral "
                     "averages, so they do not move the score.")
    else:
        if c["amount"] > product["list_price_inr"]:
            notes.append(f"Claim amount Rs {c['amount']:,.0f} is above the product's list price of Rs {product['list_price_inr']:,.0f}. "
                         "No claim in the training data was above list price.")
        warranty_days = product["warranty_months"] * 30
        if c["days"] > warranty_days:
            notes.append(f"Claim is {c['days']:,.0f} days after purchase, beyond the {product['warranty_months']}-month warranty "
                         f"({warranty_days} days). No claim in the training data was outside warranty.")
    if c["serial_given"]:
        notes.append("product_serial is accepted but the model does not use it: serials do not identify individual units in this data.")
    if band == "Review":
        notes.append(f"About the review rule: {constants['review_rule']['label']}.")
    return notes


def _pick_reasons(explained, band):
    """Two to four reasons for the screen: the one that decides the band first, then the largest contributions."""
    by_size = [r for r in explained["reasons"]]
    chosen = []
    if band == "Review":
        chosen.append(next(r for r in by_size if r["reason"] == "partner_record"))
    elif band == "Watch":
        chosen.append({"reason": "watch", "text": WATCH_REASON, "effect": "explains the band"})
    rest = [r for r in by_size if r["reason"] != "partner_record" or band == "Normal"]
    for r in rest:
        if len(chosen) >= 4:
            break
        if r not in chosen and r["effect"] != "neutral":
            chosen.append(r)
    for r in rest:   # always at least two, even when little moves the score
        if len(chosen) >= 2:
            break
        if r not in chosen:
            chosen.append(r)
    return [{"text": r["text"], "effect": r["effect"]} for r in chosen]


def assess(claim):
    """What the service returns for one claim: score, risk band, whether the review rule applies, reasons, notes."""
    ctx = _context(claim)
    explained = _explain(ctx)
    band = risk_band(explained["review_recommended"], explained["partner_has_history"], explained["score"], ctx["constants"])
    return {
        "claim_id": explained["claim_id"],
        "score": explained["score"],
        "score_label": SCORE_LABEL,
        "band": band,
        "band_meaning": BANDS[band],
        "review_rule_applies": explained["review_recommended"],
        "reasons": _pick_reasons(explained, band),
        "notes": _notes(ctx, band),
        "partner_profile": ctx["source"],
        "details": {"partner_known": explained["partner_known"], "partner_has_history": explained["partner_has_history"],
                    "partner_rate_ratio": explained["partner_rate_ratio"],
                    "expected_value_of_review_inr": explained["expected_value_of_review_inr"],
                    "break_even_probability": explained["break_even_probability"],
                    "history_frozen_on": explained["history_frozen_on"], "partner_activity_as_of": explained["partner_activity_as_of"]},
    }
