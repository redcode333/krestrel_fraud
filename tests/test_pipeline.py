"""The existing pipeline checks, run under pytest. They need the Kestrel data pack and are skipped without it."""
import pytest

from conftest import HAS_DATA, ROOT

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="needs the Kestrel data pack in data/ (run src/clean.py first)")


def test_leakage_1_labels_used_are_older_than_the_cutoff():
    import features

    features.leakage_test_1(lambda *args, **kwargs: None)


def test_leakage_2_own_label_never_changes_own_features():
    import features

    features.leakage_test_2()


def test_leakage_3_label_free_features_ignore_later_claims():
    import features

    features.leakage_test_3()


def test_leakage_4_no_nan_in_any_feature_matrix():
    import features

    features.leakage_test_4_and_shapes()


def test_harness_money_formula_and_partner_rule():
    import numpy as np
    import evaluate

    y, amount = np.array([1, 0, 0, 1, 0, 1]), np.array([1000.0, 500, 700, 3000, 900, 800])
    month, score = np.array(["m"] * 6), np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    value = evaluate.review_list_value(y[:4], amount[:4], np.array([True, True, False, True]))
    assert value["net_value_rs"] == 4000 - 380 - 3 * 260
    assert evaluate.expected_value(0.5, 2000) == 0.5 * 2000 - 0.5 * 380 - 260
    eligible = np.array([True, False, True, True, False, False])
    assert list(np.flatnonzero(evaluate.select_partner_rule(score, month, eligible, k=2))) == [0, 2]
    assert evaluate.select_partner_rule(score, month, np.zeros(6, bool), k=2).sum() == 0, "an empty list must stay empty"


def test_one_claim_function_reproduces_predictions_csv():
    import pandas as pd
    import scoring
    from clean import DATA, TIME_COL

    if scoring.load_artifacts()[3] != "real":
        pytest.skip("needs the real partner profile")
    test = pd.read_parquet(DATA / "clean_test.parquet")
    pred = pd.read_csv(ROOT / "predictions.csv", float_precision="round_trip").set_index("claim_id")["score"]
    close = 0
    for _, row in test.iterrows():
        claim = {"partner_id": row["partner_id"], "sku": row["sku"], "submitted_at": str(row[TIME_COL]), "claim_amount_inr": row["claim_amount_inr"],
                 "days_since_purchase": row["days_since_purchase"], "customer_prior_claims": row["customer_prior_claims"],
                 "photo_attached": row["photo_attached"]}
        close += abs(scoring.score_claim(claim) - pred[row["claim_id"]]) < 0.05
    # The stored profile is as of 30 Sep, so old claims are not reproduced exactly; src/final.py does the exact check.
    assert close / len(test) > 0.95


def test_partner_profile_rebuilds_exactly():
    from partner_profile import REAL_PROFILE, build_partner_profile, csv_text

    if not REAL_PROFILE.exists():
        pytest.skip("no real partner profile to compare with")
    profile, _ = build_partner_profile()
    assert csv_text(profile) == REAL_PROFILE.read_bytes().decode("utf-8")


def test_committed_reports_use_one_partner_mask():
    import reporting

    assert reporting.check_mask_consistency() >= 5
