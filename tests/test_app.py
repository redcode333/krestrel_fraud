"""The service: one test per edge case, plus the checks that it adds nothing to src/scoring.py.

Everything here runs without data/ and with whichever partner profile is loaded (real or example).
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
import scoring

client = TestClient(app)
MODEL, CONSTANTS, PARTNERS, SOURCE = scoring.load_artifacts()
KNOWN_PARTNER = sorted(PARTNERS)[0]


def claim(**changes):
    base = {"claim_id": "T-1", "submitted_at": "2026-10-05 11:20", "partner_id": KNOWN_PARTNER, "sku": "KH-MG-02",
            "product_serial": "KH482915530", "days_since_purchase": 210, "claim_amount_inr": 1450, "photo_attached": "N",
            "customer_prior_claims": 1}
    base.update(changes)
    return {k: v for k, v in base.items() if v is not ...}


def post(payload):
    return client.post("/score", json=payload)


# ---------------------------------------------------------------- the happy path
def test_health_says_which_profile_is_loaded():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["partner_profile"] == SOURCE and body["partner_profile"] in ("real", "example")
    assert body["paid_api"] is False and body["partners_in_profile"] == len(PARTNERS)


def test_page_is_one_self_contained_file_and_names_the_profile():
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    text = page.text
    assert "http://" not in text and "https://" not in text and 'src="' not in text and "<link" not in text, "the page loads an external file"
    assert "Partner profile loaded" in text and f'"profile": "{SOURCE}"' in text
    assert "<form" in text and ">Score<" in text


def test_three_examples_use_the_loaded_profile_and_show_the_three_bands():
    examples = client.get("/examples").json()
    assert [e["label"] for e in examples] == ["High-risk outlet", "Ordinary claim", "New outlet"]
    assert all(e["claim"]["partner_id"] in PARTNERS for e in examples)
    assert [post(e["claim"]).json()["band"] for e in examples] == ["Review", "Normal", "Watch"]


def test_new_outlet_example_scores_like_a_real_claim_without_history():
    # The highest real score among test claims from outlets with no decided history is 0.552 (reports/rupees.md, section C).
    example = client.get("/examples").json()[2]
    state = PARTNERS[example["claim"]["partner_id"]]
    assert state["is_new_partner"] == 1 and state["decided_new_claims"] == 0
    assert 0.35 <= post(example["claim"]).json()["score"] <= 0.55


def test_unknown_partner_gets_exactly_the_new_outlet_treatment():
    """An unknown partner ID is scored as a new outlet with no history and no recent claims: nothing more, nothing less."""
    example = client.get("/examples").json()[2]["claim"]
    known_new = PARTNERS[example["partner_id"]]
    unknown = claim(**{**example, "partner_id": "OUTLET-NOT-IN-PROFILE"})
    features, ratio, _ = scoring.claim_features(unknown)
    assert features["is_new_partner"] == 1.0 and ratio == 1.0
    by_reason = lambda record, partner=None: {r["reason"]: r["contribution"] for r in scoring.explain(record, partner)["reasons"]}
    as_unknown, as_known = by_reason(unknown), by_reason(example)
    # Same outlet-age line and same (empty) fraud record as a known new outlet; only recent activity can differ.
    assert as_unknown["outlet_age"] == as_known["outlet_age"] > 0 and as_unknown["partner_record"] == as_known["partner_record"] == 0
    assert {k for k in as_unknown if as_unknown[k] != as_known[k]} <= {"partner_activity"}
    # Give the known new outlet the same (zero) recent activity and the two scores are identical.
    quiet = {**known_new, "claims_last_7d": 0.0, "claims_last_30d": 0.0, "small_share_last_30d": 0.0, "avg_amount_last_30d": 0.0}
    assert scoring.score_claim(example, quiet) == post(unknown).json()["score"]


def test_a_good_claim_returns_everything_the_brief_asks_for():
    response = post(claim())
    body = response.json()
    assert response.status_code == 200
    assert 0 < body["score"] < 1 and "ndicative" in body["score_label"]
    assert body["band"] in ("Review", "Watch", "Normal") and isinstance(body["review_rule_applies"], bool)
    assert 2 <= len(body["reasons"]) <= 4 and all(r["text"] for r in body["reasons"])
    assert isinstance(body["notes"], list) and body["partner_profile"] == SOURCE


# ---------------------------------------------------------------- polite 4xx that name the field
def test_missing_fields_are_named():
    response = post(claim(claim_amount_inr=..., sku=...))
    body = response.json()
    assert response.status_code == 422
    assert set(body["fields"]) == {"claim_amount_inr", "sku"} and "claim_amount_inr" in body["error"] and "sku" in body["error"]


def test_empty_record_lists_every_required_field():
    body = post({}).json()
    assert set(body["fields"]) == set(scoring.REQUIRED)


def test_blank_and_null_count_as_missing():
    assert post(claim(partner_id="   ")).json()["field"] == "partner_id"
    assert post(claim(days_since_purchase=None)).json()["field"] == "days_since_purchase"


@pytest.mark.parametrize("field,value", [
    ("claim_amount_inr", "abc"), ("claim_amount_inr", [1450]), ("claim_amount_inr", True), ("claim_amount_inr", {"rs": 1450}),
    ("days_since_purchase", "soon"), ("days_since_purchase", [1]), ("customer_prior_claims", "two"), ("customer_prior_claims", 1.5),
    ("partner_id", 3160), ("sku", ["KH-MG-02"]), ("photo_attached", "maybe"), ("photo_attached", 7),
])
def test_wrong_type_names_the_field(field, value):
    response = post(claim(**{field: value}))
    assert response.status_code == 422 and response.json()["field"] == field and field in response.json()["error"]


@pytest.mark.parametrize("value", ["yesterday", "2026-13-45 10:00", "05/10/2026", "", 20261005, ["2026-10-05"]])
def test_unparseable_date_names_submitted_at(value):
    response = post(claim(submitted_at=value))
    assert response.status_code == 422 and response.json()["field"] == "submitted_at"


@pytest.mark.parametrize("value", [0, -1, -1450.5, "0", "-3"])
def test_zero_or_negative_amount_is_refused(value):
    response = post(claim(claim_amount_inr=value))
    assert response.status_code == 422 and response.json()["field"] == "claim_amount_inr" and "greater than 0" in response.json()["error"]


def test_negative_days_and_non_finite_numbers_are_refused():
    assert post(claim(days_since_purchase=-5)).json()["field"] == "days_since_purchase"
    for text in ['{"claim_amount_inr": NaN}', '{"claim_amount_inr": Infinity}']:
        payload = json.dumps(claim()).replace('"claim_amount_inr": 1450', text[1:-1])
        response = client.post("/score", content=payload, headers={"Content-Type": "application/json"})
        assert response.status_code == 422 and response.json()["field"] == "claim_amount_inr"


@pytest.mark.parametrize("body", ["", "not json", "{", "[1, 2]", '"a string"', "42", "null", '{"a": '])
def test_bodies_that_are_not_one_json_object_get_a_400(body):
    response = client.post("/score", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400 and response.json()["error"]


# ---------------------------------------------------------------- unusual but scorable records
def test_unknown_partner_is_scored_as_newly_onboarded_and_says_so():
    response = post(claim(partner_id="OUTLET-NOT-IN-PROFILE"))
    body = response.json()
    assert response.status_code == 200
    assert body["details"]["partner_known"] is False and body["details"]["partner_has_history"] is False
    assert body["review_rule_applies"] is False and body["band"] in ("Watch", "Normal")
    assert any("not in the partner profile" in note and "newly onboarded" in note for note in body["notes"])
    threshold = CONSTANTS["watch_score_threshold"]
    assert body["band"] == ("Watch" if body["score"] >= threshold else "Normal")


def test_unknown_partner_can_reach_watch_but_never_review():
    body = post(claim(partner_id="NEW-OUTLET", sku="KH-RH-02", claim_amount_inr=1850, days_since_purchase=120, customer_prior_claims=3)).json()
    assert body["band"] == "Watch" and body["review_rule_applies"] is False
    assert body["reasons"][0]["text"] == scoring.WATCH_REASON


def test_unknown_sku_is_scored_with_neutral_defaults_and_says_so():
    response = post(claim(sku="ZZ-99"))
    body = response.json()
    assert response.status_code == 200 and 0 < body["score"] < 1
    assert any("'ZZ-99' is not in the catalogue" in note and "neutral" in note for note in body["notes"])
    features, _, _ = scoring.claim_features(claim(sku="ZZ-99"))
    neutral = dict(zip(MODEL["features"], MODEL["mean"]))
    for name in MODEL["features"]:
        if name.startswith(scoring.PRODUCT_FEATURES):
            assert features[name] == neutral[name], f"{name} should take the training average for an unknown product"


def test_amount_above_list_price_is_scored_with_a_note():
    list_price = CONSTANTS["products"]["KH-MG-02"]["list_price_inr"]
    response = post(claim(claim_amount_inr=list_price * 3))
    assert response.status_code == 200
    assert any("above the product's list price" in note for note in response.json()["notes"])
    assert not any("above the product's list price" in note for note in post(claim()).json()["notes"])


@pytest.mark.parametrize("serial", ["KH482915530", " kh-482915530 ", "???", "x" * 500, "ｓｅｒｉａｌ-٣", 12345, ["a"], {"a": 1}])
def test_any_serial_is_accepted_noted_and_ignored(serial):
    response = post(claim(product_serial=serial))
    body = response.json()
    assert response.status_code == 200
    assert any("product_serial" in note and "does not use it" in note for note in body["notes"])
    assert body["score"] == post(claim(product_serial=...)).json()["score"], "the serial changed the score"


def test_no_serial_means_no_serial_note():
    assert not any("product_serial" in note for note in post(claim(product_serial=...)).json()["notes"])


def test_claim_outside_warranty_is_scored_with_a_note():
    assert any("beyond the 24-month warranty" in note for note in post(claim(days_since_purchase=900)).json()["notes"])


def test_dates_with_a_time_zone_or_a_t_are_accepted():
    plain = post(claim(submitted_at="2026-10-05 11:20")).json()["score"]
    assert post(claim(submitted_at="2026-10-05T11:20")).json()["score"] == plain
    assert post(claim(submitted_at="2026-10-05T05:50:00+00:00")).json()["score"] == plain   # 05:50 UTC is 11:20 IST


def test_extra_columns_from_the_export_are_ignored():
    extra = claim(partner_inspected="N", claim_description="blade jammed", inspector_note="", source="crm", anything_else=[1, 2, 3])
    assert post(extra).json()["score"] == post(claim()).json()["score"]


# ---------------------------------------------------------------- never a 500
NASTY = [
    {}, {"partner_id": None}, claim(claim_amount_inr=1e308), claim(claim_amount_inr=1e-300), claim(days_since_purchase=1e300),
    claim(customer_prior_claims=10**18), claim(claim_amount_inr="1e400"), claim(partner_id="'; DROP TABLE claims; --"),
    claim(partner_id="<script>alert(1)</script>"), claim(sku="\x00\x01"), claim(submitted_at="0001-01-01 00:00"),
    claim(submitted_at="9999-12-31 23:59"), claim(submitted_at="2026-10-05 11:20:00.123456+14:00"), claim(photo_attached=None),
    claim(claim_id={"nested": [1, 2, {"deep": True}]}), claim(claim_amount_inr=float("1e30")), claim(days_since_purchase="  12  "),
    {k: "" for k in scoring.REQUIRED}, {k: [] for k in scoring.REQUIRED}, {k: 0 for k in scoring.REQUIRED}, {k: "x" * 10_000 for k in scoring.REQUIRED},
]


@pytest.mark.parametrize("payload", NASTY, ids=range(len(NASTY)))
def test_never_a_500(payload):
    response = post(payload)
    assert response.status_code < 500, response.text
    body = response.json()
    assert ("error" in body) if response.status_code >= 400 else (0 <= body["score"] <= 1)


def test_deeply_nested_json_and_wrong_methods_are_handled():
    deep = client.post("/score", content="[" * 50_000 + "]" * 50_000, headers={"Content-Type": "application/json"})
    assert deep.status_code in (400, 413)
    assert client.get("/score").status_code == 405 and "error" in client.get("/score").json()
    assert client.get("/no-such-page").status_code == 404 and "error" in client.get("/no-such-page").json()
    big = client.post("/score", content=json.dumps(claim(claim_description="x" * 200_000)), headers={"Content-Type": "application/json"})
    assert big.status_code == 413 and "too large" in big.json()["error"]


# ---------------------------------------------------------------- the app adds nothing to src/scoring.py
def twenty_claims():
    partners = sorted(PARTNERS)
    flagged = [p for p in partners if PARTNERS[p]["rate_ratio"] > 1]
    skus = sorted(CONSTANTS["products"])
    chosen = (flagged + partners)[:20]
    return [claim(claim_id=f"T-{i}", partner_id=pid, sku=skus[i % len(skus)], claim_amount_inr=400 + 517 * i,
                  days_since_purchase=10 + 17 * i, customer_prior_claims=i % 4, photo_attached="Y" if i % 3 else "N",
                  submitted_at=f"2026-10-{1 + i:02d} 09:{i:02d}") for i, pid in enumerate(chosen)]


def test_app_score_equals_scoring_module_for_20_claims():
    claims = twenty_claims()
    assert len(claims) == 20
    for record in claims:
        from_app = post(record).json()
        assert from_app["score"] == scoring.score_claim(record) == scoring.explain(record)["score"]
        assert from_app == json.loads(json.dumps(scoring.assess(record)))


def test_review_band_exactly_when_the_review_rule_applies():
    seen = set()
    for pid, state in PARTNERS.items():
        body = post(claim(partner_id=pid)).json()
        seen.add(body["band"])
        assert (body["band"] == "Review") == (state["rate_ratio"] > 1) == body["review_rule_applies"]
        if body["band"] == "Watch":
            assert state["decided_new_claims"] == 0 and body["score"] >= CONSTANTS["watch_score_threshold"]
        if body["band"] == "Review":
            assert "confirmed fraud record" in body["reasons"][0]["text"]
            assert any("not independently validated" in note for note in body["notes"])
    assert "Review" in seen and "Normal" in seen


def test_reasons_never_name_a_city_or_a_partner_type():
    forbidden = [c.lower() for c in CONSTANTS["categories"]["city"]] + [t.lower() for t in CONSTANTS["categories"]["partner_type"]]
    forbidden += [t.replace("_", " ").lower() for t in CONSTANTS["categories"]["partner_type"]] + ["franchise", "freelance", "new partner"]
    for record in twenty_claims() + [claim(partner_id="UNKNOWN"), claim(sku="ZZ-99")]:
        for reason in post(record).json()["reasons"]:
            assert not any(word in reason["text"].lower() for word in forbidden), reason["text"]
