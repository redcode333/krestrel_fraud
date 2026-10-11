"""Final model: fit, calibrate, score the test set, save the artefacts. Writes reports/final.txt.

Recipe (the model src/train.py leaves standing; FINAL there names it):
  log-odds = base log-odds + 1.0 * log(partner_rate_ratio)
  base = logistic regression on the final model's label-free features, fitted on every labelled claim up to 30 Jun 2026
  partner_rate_ratio from the history frozen on 30 Jun 2026 with every outcome known by then (L = 0)
  probability = Platt correction fitted on the out-of-sample Fold B predictions of the same recipe (L = 14)

Outputs:
  predictions.csv                       claim_id,score for the 2,252 test claims, in the order of sample_submission.csv
  models/model.joblib                   scaler, coefficients, feature list, Platt parameters, beta
  models/partner_profile.csv            one row per partner, real partner IDs (git-ignored)
  models/partner_profile.example.csv    the same with masked partner IDs (committed)
  models/constants.json                 costs, break-even formula, review rule, product catalogue, cohort statistics
  reports/final.txt                     this report, partner IDs masked
  reports/raw/test_benchmark_scores.csv per-claim scores of the final model and the benchmarks (git-ignored)

Needs requirements.txt only (no LightGBM). Run from the repo root:  python src/final.py
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from clean import DATA, TIME_COL
from evaluate import (CONTACT_RS, GOODWILL_RS, PARTNER_RULE_NOTE, REGIME_START, REVIEWS_PER_MONTH, bootstrap_ranking, metrics,
                      month_of, review_list_value, select_partner_rule)
from features import (FINAL_CUTOFF, MIN_NEW_DECIDED, NEW_PARTNER_FROM, SMALL_CLAIM, SMOOTH_K, decided_rows, label_rows_used,
                      load_history, new_outlet_counts, reference_categories)
from partner_profile import FINAL_L, PROFILE_AS_OF, build_partner_profile, csv_text, example_profile, write_profiles
from reporting import RAW_REPORTS, masked_report, raw_only
from train import (BASE_COLUMNS, FINAL, FINAL_BASE, FINAL_BETA, FINAL_COLUMNS, HEADLINE_L, MODEL_NAMES, PREVIOUS_FINAL, SHORT,
                   age_and_flag_summary, fit_logistic, mats, pr_auc_win_share, rule_eligible, run, score_frame, section)
import scoring

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions.csv"
EXPECTED_TRAIN = (11146, 141)        # labelled claims and frauds up to 30 Jun 2026
EXPECTED_HISTORY = (1422, 36)        # decided new-regime claims and frauds in the frozen history
FINAL_NAME, BASE_NAME, PREVIOUS_NAME = SHORT[FINAL], SHORT[FINAL_BASE], SHORT[PREVIOUS_FINAL]
# The author's judgement for the form, NOT computed: a point guess shaded below the June figure. The range around it IS computed.
POINT_GUESS_PR_AUC = (0.30, 0.35)
ALLOWED_REASONS = {"partner_record", "claim_amount", "auto_approval", "customer", "partner_activity", "photo", "product",
                   "outlet_age", "outlet_profile"}


# ----------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------
def fit_platt(log_odds, y):
    """Two-parameter Platt correction: p = sigmoid(slope * log_odds + intercept), by maximum likelihood."""
    def loss(theta):
        eta = theta[0] * log_odds + theta[1]
        return float(np.sum(np.logaddexp(0, eta) - y * eta))
    fit = minimize(loss, x0=np.array([1.0, 0.0]), method="BFGS")
    slope, intercept = float(fit.x[0]), float(fit.x[1])
    assert slope > 0, "Platt slope is not positive: the correction would reverse the ranking"
    return slope, intercept


def reliability(p, y):
    decile = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
    rows = []
    for d in range(10):
        mask = decile == d
        rows.append({"decile (1 = lowest score)": d + 1, "claims": int(mask.sum()), "mean predicted": f"{p[mask].mean():.4f}",
                     "observed fraud rate": f"{y[mask].mean():.4f}", "frauds predicted": f"{p[mask].sum():.1f}",
                     "frauds observed": int(y[mask].sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    top = decile == 9
    print(f"whole month: {p.sum():.1f} frauds predicted against {int(y.sum())} observed; top decile: {p[top].sum():.1f} predicted "
          f"against {int(y[top].sum())} observed.")


def calibrate():
    section(f"Calibration: Platt correction fitted on the OUT-OF-SAMPLE Fold B predictions of the same recipe (L = {HEADLINE_L})")
    frame = score_frame("fold_B", HEADLINE_L)
    y = frame["is_fraud"].to_numpy()
    raw = run(FINAL, "fold_B", HEADLINE_L)["p"]      # base model fitted on claims before 1 June, never on June
    slope, intercept = fit_platt(logit(raw), y)
    calibrated = expit(slope * logit(raw) + intercept)
    print(f"Fold B: {len(y)} claims, {int(y.sum())} frauds. The scores being corrected come from a model fitted only on claims before 1 June.")
    print(f"Platt slope = {slope:.4f}, intercept = {intercept:.4f}  (probability = sigmoid(slope x log-odds + intercept))")
    print("The slope is positive, so the correction changes the size of the scores but never their order.")
    print("\nBEFORE the correction (raw model probabilities):")
    reliability(raw, y)
    print("\nAFTER the correction:")
    reliability(calibrated, y)

    X = mats("fold_B", HEADLINE_L, "month")["X_score"]
    ratio, has = X["partner_rate_ratio"].to_numpy(), X["partner_has_new_history"].to_numpy() == 1
    groups = {"flagged": ("(a) partner_rate_ratio > 1 (flagged by the review rule)", ratio > 1),
              "clean": ("(b) partner has decided claims but ratio <= 1", has & (ratio <= 1)),
              "no_history": ("(c) partner has no decided new-regime claim", ~has)}
    rows, june = {}, {}
    for key, (label, mask) in groups.items():
        june[key] = {"claims": int(mask.sum()), "frauds": int(y[mask].sum()), "rate": float(y[mask].mean()),
                     "raw": float(raw[mask].sum()), "calibrated": float(calibrated[mask].sum())}
        rows[label] = {"claims": june[key]["claims"], "frauds observed": june[key]["frauds"], "observed rate": f"{june[key]['rate']:.1%}",
                       "predicted before": f"{june[key]['raw']:.1f}", "shortfall before (observed - predicted)": f"{june[key]['frauds'] - june[key]['raw']:+.1f}",
                       "predicted after": f"{june[key]['calibrated']:.1f}", "implied rate after": f"{calibrated[mask].mean():.1%}"}
    print("\nWhere the June under-prediction came from, by kind of partner:")
    print(pd.DataFrame(rows).T.to_string())
    shortfall = {k: v["frauds"] - v["raw"] for k, v in june.items()}
    total = sum(shortfall.values())
    june["flagged_share_of_shortfall"] = shortfall["flagged"] / total if total > 0 else float("nan")
    print(f"Before the correction the model was short by {total:.1f} frauds in total, of which {shortfall['flagged']:+.1f} among flagged "
          f"partners. The Platt intercept ({intercept:+.2f}) is one number added to every claim's log-odds, so it also raises the "
          f"claims of unflagged partners: after it they are predicted {june['clean']['calibrated'] + june['no_history']['calibrated']:.1f} "
          f"frauds against {june['clean']['frauds'] + june['no_history']['frauds']} observed.")

    amount, month = frame["claim_amount_inr"].to_numpy(), month_of(frame)
    before, after = (metrics(y, p, amount, month)["upto40_stop_early"] for p in (raw, calibrated))
    print(f"\nstop-early list on Fold B: before {before['n_reviewed']} reviews, {before['frauds_found']} frauds, net Rs "
          f"{before['net_value_rs']:,.0f}; after {after['n_reviewed']} reviews, {after['frauds_found']} frauds, net Rs {after['net_value_rs']:,.0f}.")
    print("CAVEATS: (1) the 'after' tables and the 'after' stop-early number are judged on the same 713 claims the two parameters "
          "were fitted on, so they look better than they will be. (2) The correction rests on 22 frauds from 8 partners. (3) The final "
          "model has fresher and larger partner history than Fold B at L = 14, so its raw scores are spread differently. "
          "Treat the probabilities as INDICATIVE ONLY; the ranking and the partner rule do not depend on them.")
    return slope, intercept, june


# ----------------------------------------------------------------------------------------------
# final fit
# ----------------------------------------------------------------------------------------------
def final_fit(slope, intercept):
    section(f"Final training: {BASE_NAME} on every labelled claim up to 30 Jun 2026, history frozen on 30 Jun 2026 (L = 0)")
    history = load_history()
    m = mats("final", FINAL_L, "month")
    train, test, X_test = m["train"], m["score"], m["X_score"]
    new, _ = label_rows_used(history, FINAL_CUTOFF, FINAL_L)

    assert (len(train), int(m["y_train"].sum())) == EXPECTED_TRAIN, "training set is not the 11,146 claims / 141 frauds evaluated"
    assert history[TIME_COL].max() < FINAL_CUTOFF, "a label dated after 30 Jun 2026 exists"
    assert new[TIME_COL].max() < FINAL_CUTOFF and train[TIME_COL].max() < FINAL_CUTOFF, "a label dated after 30 Jun 2026 would be used"
    assert test[TIME_COL].min() >= FINAL_CUTOFF, "a test claim is dated before the cut-off"
    assert (len(new), int(new["is_fraud"].sum())) == EXPECTED_HISTORY, "frozen history is not the 1,422 claims / 36 frauds expected"
    assert not set(test["claim_id"]) & set(history["claim_id"]), "a test claim has a label"
    identity = [c for c in FINAL_COLUMNS if c.startswith(("city_", "partner_type_"))]
    print(f"model: {MODEL_NAMES[FINAL]}")
    print(f"training rows: {len(train):,} ({int(m['y_train'].sum())} frauds), {train[TIME_COL].min().date()} to {train[TIME_COL].max().date()}")
    print(f"base features: {len(FINAL_COLUMNS)} label-free columns, of which city or partner-type columns: {len(identity)}; "
          "same standardise-then-logistic recipe (C = 1.0) as evaluated in src/train.py")
    print(f"frozen history: {len(new):,} decided new-regime claims, {int(new['is_fraud'].sum())} frauds; newest label used is from "
          f"{new[TIME_COL].max()}; ASSERTED: no label dated after 30 Jun 2026 is used (none exist: newest label in the data is "
          f"{history[TIME_COL].max()}).")
    print(f"test claims: {len(test):,}, {test[TIME_COL].min().date()} to {test[TIME_COL].max().date()}; their rolling features use only "
          "strictly earlier claims from the train, undecided and test files.")

    lr = fit_logistic(m["X_train"][FINAL_COLUMNS], m["y_train"])
    ratio = X_test["partner_rate_ratio"].to_numpy()
    base_log_odds = lr.decision_function(X_test[FINAL_COLUMNS].to_numpy())
    log_odds = base_log_odds + FINAL_BETA * np.log(ratio)
    p = expit(slope * log_odds + intercept)

    # The model the files were built from before the last change, for the overlap report only.
    previous = fit_logistic(m["X_train"][BASE_COLUMNS[PREVIOUS_FINAL]], m["y_train"])
    previous_log_odds = previous.decision_function(X_test[BASE_COLUMNS[PREVIOUS_FINAL]].to_numpy()) + FINAL_BETA * np.log(ratio)

    coef = pd.Series(lr[-1].coef_[0], index=FINAL_COLUMNS)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
    print(f"\nlargest {BASE_NAME} coefficients (per standard deviation of the feature; + raises the score):")
    print("  " + "; ".join(f"{name} {value:+.2f}" for name, value in top.items()))
    return {"lr": lr, "m": m, "p": p, "log_odds": log_odds, "base_p": expit(base_log_odds), "ratio": ratio,
            "previous_log_odds": previous_log_odds}


# ----------------------------------------------------------------------------------------------
# predictions.csv
# ----------------------------------------------------------------------------------------------
def write_predictions(fit, slope, intercept):
    section("predictions.csv")
    test, p = fit["m"]["score"], fit["p"]
    sample = pd.read_csv(DATA / "sample_submission.csv")
    out = pd.DataFrame({"claim_id": test["claim_id"].to_numpy(), "score": p}).set_index("claim_id").reindex(sample["claim_id"]).reset_index()
    assert out["score"].notna().all() and len(out) == len(sample) == 2252
    out.to_csv(PREDICTIONS, index=False)
    back = pd.read_csv(PREDICTIONS, float_precision="round_trip")
    assert np.array_equal(back["score"].to_numpy(), out["score"].to_numpy()), "scores changed when written to disk"
    assert list(back["claim_id"]) == list(sample["claim_id"])
    print(f"wrote {PREDICTIONS.name}: {len(out):,} rows, columns {list(out.columns)}, ids and order as data/sample_submission.csv")

    s = out["score"]
    print(f"score min {s.min():.5f} / median {s.median():.5f} / p90 {s.quantile(0.9):.5f} / p99 {s.quantile(0.99):.5f} / max {s.max():.5f}; "
          f"distinct scores: {s.nunique():,} of {len(s):,}; NaN: {int(s.isna().sum())}; strictly between 0 and 1: {bool(((s > 0) & (s < 1)).all())}")
    print(f"Platt parameters applied: slope {slope:.4f}, intercept {intercept:.4f}")

    history = load_history()
    new_regime = history[history[TIME_COL] >= REGIME_START]
    small_hist = new_regime["claim_amount_inr"] < SMALL_CLAIM
    small_test = (test["claim_amount_inr"] < SMALL_CLAIM).mean()
    rate_small, rate_large = new_regime[small_hist]["is_fraud"].mean(), new_regime[~small_hist]["is_fraud"].mean()
    reference = len(test) * (small_test * rate_small + (1 - small_test) * rate_large)
    june = history[history[TIME_COL] >= "2026-06-01"]["is_fraud"].mean()
    print(f"sum of scores = expected frauds in the test set: {s.sum():.1f} ({s.mean():.2%} of claims).")
    print(f"  sanity reference: {small_test:.0%} small claims x {rate_small:.2%} + {1 - small_test:.0%} large x {rate_large:.2%} "
          f"(May-June rates) = {reference:.1f} frauds; at June's rate alone ({june:.2%}) it would be {len(test) * june:.1f}.")
    print(f"  the sum is {s.sum() / reference:.2f}x that reference ({'at or above' if s.sum() >= reference else 'BELOW'} the sanity "
          "target). It is an indication, not a forecast: the correction behind it rests on 22 frauds.")
    return out


def decomposition(fit):
    section("Where the expected frauds sit: sum of final scores by kind of partner")
    X, p = fit["m"]["X_score"], fit["p"]
    ratio, has = fit["ratio"], X["partner_has_new_history"].to_numpy() == 1
    groups = [("(a) partner_rate_ratio > 1: partner has confirmed fraud above the average rate", ratio > 1),
              ("(b) partner has decided claims but ratio <= 1", has & (ratio <= 1)),
              ("(c) partner has no decided new-regime claim", ~has)]
    rows, parts = {}, {}
    for label, mask in groups:
        partners = fit["m"]["score"][mask]["partner_id"].nunique()
        rows[label] = {"claims": int(mask.sum()), "share of claims": f"{mask.mean():.1%}", "partners": partners,
                       "sum of scores (expected frauds)": f"{p[mask].sum():.1f}", "share of expected frauds": f"{p[mask].sum() / p.sum():.0%}",
                       "implied fraud rate": f"{p[mask].mean():.1%}"}
        parts[label[:3]] = {"claims": int(mask.sum()), "sum": float(p[mask].sum()), "rate": float(p[mask].mean()), "partners": partners}
    rows["all test claims"] = {"claims": len(p), "share of claims": "100.0%", "partners": fit["m"]["score"]["partner_id"].nunique(),
                               "sum of scores (expected frauds)": f"{p.sum():.1f}", "share of expected frauds": "100%",
                               "implied fraud rate": f"{p.mean():.1%}"}
    assert sum(v["claims"] for v in parts.values()) == len(p)
    print(pd.DataFrame(rows).T.to_string())
    history = load_history()
    june = history[history[TIME_COL] >= "2026-06-01"]
    print(f"For scale: May-June had {int(history[history[TIME_COL] >= REGIME_START]['is_fraud'].sum())} confirmed frauds in "
          f"{int((history[TIME_COL] >= REGIME_START).sum()):,} decided claims; June alone {int(june['is_fraud'].sum())} in {len(june)}.")
    print("Group (a) is what the review rule looks at. Groups (b) and (c) are where the model expects fraud that the rule will not "
          "review; those numbers are sums of small probabilities and are the least reliable part of the total.")
    return parts


def composition(fit):
    section("Composition of the top 40 and top 100 test claims (by score, all three months together)")
    test, X, p = fit["m"]["score"], fit["m"]["X_score"], fit["p"]
    order = np.argsort(-p, kind="stable")
    frame = pd.DataFrame({"partner_id": test["partner_id"].to_numpy(), "amount": test["claim_amount_inr"].to_numpy(),
                          "month": month_of(test), "ratio": fit["ratio"], "frauds": X["partner_new_fraud_count"].to_numpy(),
                          "claims": X["partner_new_claim_count"].to_numpy(), "new": X["is_new_partner"].to_numpy(),
                          "has_history": X["partner_has_new_history"].to_numpy()})
    top40, top100 = frame.iloc[order[:40]], frame.iloc[order[:100]]
    by_partner = pd.DataFrame({"places in top 40": top40.groupby("partner_id").size(), "places in top 100": top100.groupby("partner_id").size()})
    info = frame.groupby("partner_id").agg(test_claims=("amount", "size"), ratio=("ratio", "first"), frauds=("frauds", "first"),
                                           claims=("claims", "first"), new=("new", "first"))
    by_partner = by_partner.join(info).fillna(0).sort_values(["places in top 40", "places in top 100"], ascending=False)
    by_partner["partner_rate_ratio"] = by_partner["ratio"].map("{:.1f}".format)
    by_partner["history to 30 Jun (frauds / decided claims)"] = by_partner["frauds"].map("{:.0f}".format) + " / " + by_partner["claims"].map("{:.0f}".format)
    by_partner["group"] = np.where(by_partner["new"] == 1, "new", "old")
    by_partner["all test claims"] = by_partner["test_claims"].astype(int)
    print(by_partner[["places in top 40", "places in top 100", "all test claims", "partner_rate_ratio",
                      "history to 30 Jun (frauds / decided claims)", "group"]].astype({"places in top 40": int, "places in top 100": int}).to_string())
    rows = {}
    for label, t in [("top 40", top40), ("top 100", top100), ("all 2,252 test claims", frame)]:
        rows[label] = {"claims": len(t), "partners": t["partner_id"].nunique(), "under Rs 2,000": f"{(t['amount'] < SMALL_CLAIM).mean():.0%}",
                       "median Rs": f"{t['amount'].median():,.0f}", "smallest Rs": f"{t['amount'].min():,.0f}", "largest Rs": f"{t['amount'].max():,.0f}",
                       "partner_rate_ratio > 1": f"{(t['ratio'] > 1).mean():.0%}", "no partner history": f"{(t['has_history'] == 0).mean():.0%}",
                       "Jul / Aug / Sep": " / ".join(str(int((t["month"] == mth).sum())) for mth in ["2026-07", "2026-08", "2026-09"])}
    print()
    print(pd.DataFrame(rows).T.to_string())
    biggest = top40.groupby("partner_id").size().max()
    if biggest > 15:
        print(f"WARNING: ONE PARTNER SUPPLIES {biggest} OF THE TOP 40. The top of the list is one outlet; if that outlet has stopped, "
              "the top of the list is wasted.")
    else:
        print(f"No single partner supplies more than 15 of the top 40 (the largest share is {biggest}).")


def benchmarks(fit):
    section(f"Benchmarks on the test set: the previous final model ({PREVIOUS_NAME}), M0 (partner_rate_ratio only) and {BASE_NAME} alone")
    test, X, p = fit["m"]["score"], fit["m"]["X_score"], fit["p"]
    m0, previous = X["partner_new_rate_smoothed"].to_numpy(), fit["previous_log_odds"]
    RAW_REPORTS.mkdir(parents=True, exist_ok=True)
    path = RAW_REPORTS / "test_benchmark_scores.csv"
    pd.DataFrame({"claim_id": test["claim_id"].to_numpy(), "final_score": p, "previous_final_log_odds": previous, "m0_score": m0,
                  "base_alone_score": fit["base_p"]}).to_csv(path, index=False)
    print(f"saved per-claim scores to reports/raw/{path.name} (git-ignored: claim-level).")
    rng = np.random.default_rng(0)
    top = {"final": set(np.argsort(-p, kind="stable")[:40]), "previous": set(np.argsort(-previous, kind="stable")[:40]),
           "base": set(np.argsort(-fit["base_p"], kind="stable")[:40]), "M0": set(np.lexsort((rng.random(len(m0)), -m0))[:40])}
    top100 = {"final": set(np.argsort(-p, kind="stable")[:100]), "previous": set(np.argsort(-previous, kind="stable")[:100])}
    if FINAL == PREVIOUS_FINAL:
        print("The final model IS the previous final model; the overlap below is complete by construction.")
    print(f"overlap with the PREVIOUS top 40 ({PREVIOUS_NAME}): {len(top['final'] & top['previous'])} of 40 claims; "
          f"top 100: {len(top100['final'] & top100['previous'])} of 100; rank correlation over all test claims "
          f"{pd.Series(p).corr(pd.Series(previous), method='spearman'):.3f}.")
    print(f"overlap of the top 40 with M0: {len(top['final'] & top['M0'])} claims; with {BASE_NAME} alone: {len(top['final'] & top['base'])}.")
    cut = np.sort(m0)[::-1][39]
    print(f"M0 has {pd.Series(m0).nunique()} distinct scores over {len(m0):,} claims; {int((m0 == cut).sum())} claims tie at its 40th "
          f"score, so {int((m0[list(top['M0'])] == cut).sum())} of its top 40 places are decided by a random tie-break.")


def test_review_rule(fit):
    section("Partner rule on the test set: how many claims qualify each month (cap = 40)")
    test, p = fit["m"]["score"], fit["p"]
    eligible = fit["ratio"] > 1
    month = month_of(test)
    print(f"LABEL: {PARTNER_RULE_NOTE}.")
    rows = {}
    for mth in ["2026-07", "2026-08", "2026-09"]:
        mask = (month == mth) & eligible
        rows[mth] = {"test claims": int((month == mth).sum()), "qualify (partner_rate_ratio > 1)": int(mask.sum()),
                     "from partners": test[mask]["partner_id"].nunique(), "would be reviewed (cap 40)": int(min(mask.sum(), REVIEWS_PER_MONTH)),
                     "cap binds": "YES" if mask.sum() > REVIEWS_PER_MONTH else "no",
                     "lowest score reviewed": f"{np.sort(p[mask])[::-1][:REVIEWS_PER_MONTH].min():.4f}" if mask.any() else "n/a"}
    print(pd.DataFrame(rows).T.to_string())
    print(f"In total {int(eligible.sum())} test claims from {test[eligible]['partner_id'].nunique()} partners qualify. Where the cap binds, "
          "the claims with the highest score are reviewed and the rest are paid. A month with no qualifying claim gets an EMPTY list: "
          "the product says 'no claim meets the review rule this month' and does not pad the list.")
    print("Which claims qualify does not depend on the base model at all: it depends only on which partners have a confirmed fraud "
          "record. M5, M5-mid and M5-lite give the same list wherever the cap does not bind; the score only orders it.")
    print("Qualification uses the history frozen on 30 Jun 2026. In live use the history would be refreshed as outcomes arrive; "
          "frozen for three months, partners that turn fraudulent after June can never qualify.")


def rule_margin():
    section("Break-even margin of the partner-rule list on Fold B")
    print(f"LABEL: {PARTNER_RULE_NOTE}.")
    rows, headline = {}, None
    for L in (HEADLINE_L, 0):
        frame = score_frame("fold_B", L)
        y, amount = frame["is_fraud"].to_numpy(), frame["claim_amount_inr"].to_numpy()
        listed = select_partner_rule(run(FINAL, "fold_B", L)["p"], month_of(frame), rule_eligible("fold_B", L))
        value = review_list_value(y, amount, listed)
        fraud_amount = amount[listed & (y == 1)].mean()
        break_even = (CONTACT_RS + GOODWILL_RS) / (fraud_amount + GOODWILL_RS)
        observed = value["frauds_found"] / value["n_reviewed"]
        needed = int(np.ceil(break_even * value["n_reviewed"]))
        row = {"reviews": value["n_reviewed"], "frauds": value["frauds_found"], "average amount of the fraud claims Rs": f"{fraud_amount:,.0f}",
               "break-even fraud rate = (260 + 380) / (average fraud amount + 380)": f"{break_even:.1%}",
               "observed fraud rate in the list": f"{observed:.1%}", "margin (points)": f"{100 * (observed - break_even):+.1f}",
               "frauds needed to break even": needed, "net Rs": f"{value['net_value_rs']:,.0f}"}
        rows[f"Fold B, L = {L}"] = row
        if headline is None:
            headline = {"fraud_amount": float(fraud_amount), "break_even": float(break_even), "observed": float(observed),
                        "needed": needed, **value}
    print(pd.DataFrame(rows).T.to_string())
    print(f"At L = {HEADLINE_L} the list pays as long as at least {headline['needed']} of its {headline['n_reviewed']} reviews are fraud; "
          f"it had {headline['frauds_found']}. That is the margin: {headline['frauds_found'] - headline['needed']} frauds, in one month, "
          "from 8 partners.")
    return headline


# ----------------------------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------------------------
def cohort_statistics():
    """New-outlet versus established-outlet figures for the explanation line, from THE one definition in features.py."""
    return new_outlet_counts(load_history())


def save_artifacts(fit, slope, intercept):
    section("Saved artefacts in models/ (partner-level summaries only, no claim rows)")
    MODELS.mkdir(exist_ok=True)
    lr, m = fit["lr"], fit["m"]
    scaler, logistic = lr[0], lr[-1]
    model = {"name": f"{FINAL_NAME}: {BASE_NAME} log-odds + 1.0 x log(partner_rate_ratio), Platt-corrected", "features": list(FINAL_COLUMNS),
             "mean": scaler.mean_.astype(float), "scale": scaler.scale_.astype(float), "coef": logistic.coef_[0].astype(float),
             "intercept": float(logistic.intercept_[0]), "beta": float(FINAL_BETA), "platt_slope": float(slope),
             "platt_intercept": float(intercept), "platt_fitted_on": f"Fold B out-of-sample predictions, L = {HEADLINE_L}: 713 claims, 22 frauds",
             "trained_through": "2026-06-30", "train_rows": int(len(m["train"])), "train_frauds": int(m["y_train"].sum())}
    joblib.dump(model, MODELS / "model.joblib")

    profile, overall = build_partner_profile()
    write_profiles(profile)
    cohort = cohort_statistics()
    products = pd.read_csv(DATA / "products.csv")
    constants = {
        "model": model["name"],
        "costs": {"goodwill_inr": GOODWILL_RS, "service_contact_inr": CONTACT_RS},
        "expected_value_formula": f"p*amount - (1-p)*{GOODWILL_RS} - {CONTACT_RS}",
        "break_even_formula": f"a review pays for itself when p > ({GOODWILL_RS} + {CONTACT_RS}) / (amount + {GOODWILL_RS})",
        "review_rule": {"ratio_threshold": 1.0, "monthly_cap": REVIEWS_PER_MONTH, "rank_by": "score", "pad_list": False,
                        "text": "Review only claims whose partner has partner_rate_ratio > 1 (confirmed new-regime fraud above the "
                                "average rate), highest score first, at most 40 a month. If no claim qualifies, say so; never pad the list.",
                        "label": PARTNER_RULE_NOTE},
        "small_claim_limit_inr": SMALL_CLAIM, "regime_start": str(REGIME_START.date()), "new_partner_from": str(NEW_PARTNER_FROM.date()),
        "history_frozen_on": "2026-06-30", "history_lag_days": FINAL_L, "profile_date": "2026-09-30",
        "profile_as_of": PROFILE_AS_OF.isoformat(), "overall_new_regime_rate": overall, "smoothing_k": SMOOTH_K,
        "min_new_decided": MIN_NEW_DECIDED, "cohort": cohort,
        "products": {r["sku"]: {"family": r["family"], "list_price_inr": float(r["list_price_inr"]), "warranty_months": int(r["warranty_months"])}
                     for _, r in products.iterrows()},
        "categories": reference_categories(),
    }
    (MODELS / "constants.json").write_text(json.dumps(constants, indent=2), encoding="utf-8")
    scoring.load_artifacts.cache_clear()

    print(f"models/model.joblib: plain numbers only ({len(model['features'])} features, their means, scales and coefficients, intercept "
          f"{model['intercept']:.4f}, beta {model['beta']}, Platt slope {slope:.4f} and intercept {intercept:.4f}). Loading it needs numpy and joblib, not scikit-learn.")
    print(f"models/partner_profile.csv (REAL partner IDs, git-ignored): {len(profile)} rows, one per partner; columns {list(profile.columns)}")
    print(f"  frozen history as of 2026-06-30: {int((profile['decided_new_claims'] > 0).sum())} partners with decided new-regime claims, "
          f"{int((profile['new_frauds'] > 0).sum())} with confirmed fraud, {int((profile['rate_ratio'] > 1).sum())} with partner_rate_ratio > 1; "
          f"overall new-regime rate {overall:.4f}")
    print(f"  rolling activity as of 2026-09-30: {int((profile['claims_last_30d'] > 0).sum())} partners with a claim in the last 30 days")
    print("models/partner_profile.example.csv (committed): the same rows with masked partner IDs and onboarding age rounded down to "
          "30 days. The scorer falls back to it when the real file is missing.")
    print("models/constants.json: costs (Rs 380, Rs 260), expected-value and break-even formulas, the review rule and its label, "
          "the product catalogue, the category lists and the cohort statistics quoted in explanations.")
    print(f"  The ONE definition used everywhere: {cohort['definition']}: {cohort['new_outlets']}, of which "
          f"{cohort['new_outlets_with_fraud']} have a confirmed fraud and {cohort['new_outlets_without_fraud']} have none.")
    print(f"  new outlets: {cohort['new_outlet_frauds']} frauds in {cohort['new_outlet_claims']} decided claims after 1 May "
          f"({cohort['new_outlet_fraud_rate']:.1%}); established outlets: {cohort['established_outlet_frauds']} in "
          f"{cohort['established_outlet_claims']:,} ({cohort['established_outlet_fraud_rate']:.1%}); "
          f"{cohort['new_outlets_without_fraud']} of {cohort['new_outlets']} new outlets have no confirmed fraud. Before 1 May, "
          f"{cohort['before_may_established_frauds']} of the {cohort['before_may_frauds']} confirmed frauds came from established outlets.")
    print("src/scoring.py: claim_features(claim) builds the features for ONE claim from the partner profile; score_claim and explain "
          "use it. The app must import these and nothing else from the pipeline.")
    return profile


def runtime_check(fit, profile):
    section("Checks: the one-claim function reproduces predictions.csv, and explanations follow the rules")
    test, X, p = fit["m"]["score"], fit["m"]["X_score"], fit["p"]
    assert scoring.load_artifacts()[3] == "real", "the check must run against the real partner profile"
    age = profile.set_index("partner_id")["age_days_at_profile"]
    categories = reference_categories()
    forbidden = [c.lower() for c in categories["city"]] + [t.lower() for t in categories["partner_type"]] + [
        t.replace("_", " ").lower() for t in categories["partner_type"]] + ["franchise", "freelance", "service centre", "new partner"]
    uses_identity = any(c.startswith(("city_", "partner_type_")) for c in FINAL_COLUMNS)
    worst_score, worst_feature, as_of_now, reasons_seen = 0.0, 0.0, [], set()
    for i, (_, row) in enumerate(test.iterrows()):
        x = X.iloc[i]
        claim = {"claim_id": row["claim_id"], "partner_id": row["partner_id"], "sku": row["sku"], "submitted_at": str(row[TIME_COL]),
                 "claim_amount_inr": row["claim_amount_inr"], "days_since_purchase": row["days_since_purchase"],
                 "customer_prior_claims": row["customer_prior_claims"], "photo_attached": row["photo_attached"]}
        # The partner's state as it was when this claim was submitted (the stored profile is as of 30 Sep).
        state = {"known": True, "partner_type": row["partner_type"], "city": row["city"], "is_new_partner": x["is_new_partner"],
                 "age_days_at_profile": float(age[row["partner_id"]]), "decided_new_claims": x["partner_new_claim_count"],
                 "new_frauds": x["partner_new_fraud_count"], "smoothed_rate": x["partner_new_rate_smoothed"],
                 "rate_ratio": x["partner_rate_ratio"], "claims_last_7d": x["partner_claims_last_7d"],
                 "claims_last_30d": x["partner_claims_last_30d"], "small_share_last_30d": x["partner_small_share_last_30d"],
                 "avg_amount_last_30d": x["partner_avg_amount_last_30d"]}
        features, _, _ = scoring.claim_features(claim, state)
        assert list(features) == list(FINAL_COLUMNS)
        worst_feature = max(worst_feature, max(abs(features[name] - x[name]) for name in FINAL_COLUMNS))
        for explained in (scoring.explain(claim, state), scoring.explain(claim)):
            for reason in explained["reasons"]:
                reasons_seen.add(reason["reason"])
                text = reason["text"].lower()
                assert not any(word in text for word in forbidden), f"an explanation names a city, a partner type or 'new partner': {reason['text']}"
                if reason["reason"] == "outlet_profile":
                    assert reason["text"] == scoring.OUTLET_PROFILE_TEXT
            total = explained["baseline_log_odds"] + sum(r["contribution"] for r in explained["reasons"])
            assert abs(total - explained["log_odds"]) < 1e-2, "reasons do not add up to the score"
        worst_score = max(worst_score, abs(scoring.explain(claim, state)["score"] - p[i]), abs(scoring.score_claim(claim, state) - p[i]))
        as_of_now.append(scoring.score_claim(claim))
    assert worst_feature < 1e-9 and worst_score < 1e-9, (worst_feature, worst_score)
    assert reasons_seen <= ALLOWED_REASONS, reasons_seen - ALLOWED_REASONS
    assert ("outlet_profile" in reasons_seen) == uses_identity
    print(f"PASS: for all {len(test):,} test claims, the one-claim function given the partner's state at the time of the claim "
          f"rebuilds every feature (largest difference {worst_feature:.1e}) and the score in predictions.csv (largest difference "
          f"{worst_score:.1e}).")
    print(f"PASS: {2 * len(test):,} explanations checked (each claim with its state at the time and with the stored profile). None names "
          f"a city or a partner type or says 'new partner'; every reason is one of {sorted(reasons_seen)}; the reasons and the "
          "baseline add up to the score.")
    print("Outlet age is explained with one line carrying its group statistics; city and partner type are "
          + ("in the model and reported only under the single outlet-profile line." if uses_identity else "not in the model at all."))
    as_of_now = np.array(as_of_now)
    diff = np.abs(as_of_now - p)
    rank = pd.Series(as_of_now).corr(pd.Series(p), method="spearman")
    print(f"What the app will do differently: it reads partner activity as of 30 Sep 2026 from the profile, not as of the claim's "
          f"date. Re-entering an old test claim in the app therefore gives a slightly different score: median difference "
          f"{np.median(diff):.4f}, largest {diff.max():.4f}, rank correlation with predictions.csv {rank:.3f}. predictions.csv is the "
          "submission; the app is for new claims.")


def examples(fit):
    section("explain(): three test claims as the app will show them")
    test, p = fit["m"]["score"], fit["p"]
    order = np.argsort(-p, kind="stable")
    no_history = np.flatnonzero(fit["m"]["X_score"]["partner_has_new_history"].to_numpy() == 0)
    picks = [("the highest-scoring test claim", order[0]), ("the highest-scoring claim from a partner with no history", no_history[np.argmax(p[no_history])]),
             ("a typical claim (median score)", order[len(order) // 2])]
    print("Scored with the stored partner profile, as the app does. Claim-level detail is kept out of the committed copy.")
    for label, i in picks:
        row = test.iloc[i]
        claim = {k: row[k] for k in ["claim_id", "partner_id", "sku", "claim_amount_inr", "days_since_purchase", "customer_prior_claims", "photo_attached"]}
        claim["submitted_at"] = str(row[TIME_COL])
        out = scoring.explain(claim)
        raw_only(f"\n{label}: claim {out['claim_id']}, partner {row['partner_id']}, score {out['score']:.4f} "
                 f"(predictions.csv {p[i]:.4f})")
        raw_only(f"  {out['recommendation']}")
        raw_only(f"  expected value of reviewing: Rs {out['expected_value_of_review_inr']:,.0f}; break-even probability {out['break_even_probability']:.0%}")
        for r in out["reasons"]:
            raw_only(f"  {r['contribution']:+.2f}  {r['text']}  [{r['effect']}]")
        raw_only(f"  baseline log-odds {out['baseline_log_odds']:+.2f}; total {out['log_odds']:+.2f}")


# ----------------------------------------------------------------------------------------------
# expected score
# ----------------------------------------------------------------------------------------------
def expected_score_text(fit, slope, intercept, predictions, parts, margin, june):
    section("Text for the submission form: what I expect it to score and why (final recipe)")
    assert FINAL == "M5n", "this text describes the model without city, partner type and partner age; rewrite it if the final model changes"
    b = {L: run(FINAL, "fold_B", L)["res"] for L in (14, 0, 30)}
    m0, base = run("M0", "fold_B", HEADLINE_L)["res"], run(FINAL_BASE, "fold_B", HEADLINE_L)["res"]
    full, mid, lite = (run(m, "fold_B", HEADLINE_L)["res"] for m in ("M5f", "M5m", "M5l"))
    a, a_lite = (run(m, "fold_A", HEADLINE_L)["res"] for m in (FINAL, "M5l"))
    wins = {"lite": pr_auc_win_share("M5f", "M5l"), "mid": pr_auc_win_share("M5f", "M5m"), "noage": pr_auc_win_share("M5m", "M5n")}
    age = age_and_flag_summary()
    h = b[HEADLINE_L]
    rank, lp, rule, rule_ci = h["ci_rank"], h["ci_by_p"], h["upto40_partner_rule"], h["ci_partner_rule"]
    frame = score_frame("fold_B", HEADLINE_L)
    n, frauds = len(frame), int(frame["is_fraud"].sum())
    fraud_partners = frame[frame["is_fraud"] == 1]["partner_id"].nunique()
    blind = REVIEWS_PER_MONTH * frauds / n
    history = load_history()
    final_new, _ = decided_rows(history, FINAL_CUTOFF, FINAL_L)
    b_new, _ = decided_rows(history, pd.Timestamp("2026-06-01"), HEADLINE_L)
    test = fit["m"]["score"]
    cold14 = int((mats("final", 14, "month")["X_score"]["partner_has_new_history"] == 0).sum())
    late_share = (test[TIME_COL] >= pd.Timestamp("2026-07-30")).mean()
    may = history[(history[TIME_COL] >= "2026-05-01") & (history[TIME_COL] < "2026-06-01")]["is_fraud"].mean()
    june_rate = history[history[TIME_COL] >= "2026-06-01"]["is_fraud"].mean()
    cohort = cohort_statistics()
    pa, pb, pc = parts["(a)"], parts["(b)"], parts["(c)"]
    low, high = base["pr_auc"], b[0]["pr_auc"]
    text = f"""Final recipe. Score = base log-odds + 1.0 x log(partner_rate_ratio). The base is a logistic regression on {len(FINAL_COLUMNS)} label-free claim and partner features; partner_rate_ratio is the partner's confirmed fraud rate since 1 May 2026 relative to the average, and is exactly 1.0 for a partner with no decided claim. It is trained on the {EXPECTED_TRAIN[0]:,} labelled claims up to 30 Jun 2026 ({EXPECTED_TRAIN[1]} frauds). A decision rule I fixed before running anything chose a simpler benchmark (the ratio alone). I deviated from it and logged why: the benchmark cannot rank the claims of partners without a fraud record, and the intervals do not separate the two.

Validation. On the primary fold (June 2026: {n} claims, {frauds} frauds from {fraud_partners} partners; fraud history frozen on 1 June and assumed known after a {HEADLINE_L}-day investigation lag) the same recipe scores PR-AUC {h['pr_auc']:.3f} (90% interval {rank['pr_auc'][0]:.3f} to {rank['pr_auc'][1]:.3f}) and ROC-AUC {h['roc_auc']:.3f} ({rank['roc_auc'][0]:.3f} to {rank['roc_auc'][1]:.3f}). It finds {h['top40_by_p']['frauds_found']} of the {frauds} frauds in the top 40 ({lp['frauds_found'][0]:.0f} to {lp['frauds_found'][1]:.0f}) and {h['top100_overall_frauds']} in the top 100 ({rank['top100_overall_frauds'][0]:.0f} to {rank['top100_overall_frauds'][1]:.0f}); picking 40 blind finds about {blind:.1f}. The benchmark scores PR-AUC {m0['pr_auc']:.3f} and ROC-AUC {m0['roc_auc']:.3f} with {m0['top40_by_p']['frauds_found']} in the top 40. Intervals come from 1,000 resamples of partners with a fixed seed and are wide because {frauds} frauds from {fraud_partners} partners is very little.

What I expect on the Jul-Sep test. A PR-AUC somewhere between about {low:.2f} and {high:.2f}, with a point guess of {POINT_GUESS_PR_AUC[0]:.2f} to {POINT_GUESS_PR_AUC[1]:.2f}. The range is computed on the June fold: {low:.2f} is what the label-free part alone scores (the result if partner history stops helping) and {high:.2f} is the full recipe when every outcome is known at once. The point guess is my judgement, and it is deliberately below the June figure of {h['pr_auc']:.3f}: the coefficient of 1.0, the base model (the survivor of four compared on June), the review rule and the calibration were all settled with June in view, and June is the only month I could validate on. That is selection optimism, and I cannot measure its size. A result under {low:.2f} would mean the fraud pattern has moved again. Accuracy is not a result I report: predicting "not fraud" for every June claim is already {h['accuracy_all_zero']:.2%} accurate and catches nothing.

History and lag for the final model. History is frozen on 30 Jun 2026 using every outcome known by then ({len(final_new):,} decided new-regime claims, {int(final_new['is_fraud'].sum())} frauds), i.e. no lag at the cut-off. The export cannot reveal the real investigation lag. The test runs to 30 Sep, so for the {late_share:.0%} of test claims submitted from 30 July on, every June outcome would be decided even under a 30-day lag; only July claims get somewhat optimistic history. Validation keeps the {HEADLINE_L}-day lag as the conservative headline ({len(b_new)} decided claims, {int(b_new['is_fraud'].sum())} frauds available); the same fold gives PR-AUC {b[0]['pr_auc']:.3f} with no lag and {b[30]['pr_auc']:.3f} with a 30-day lag, where no history is usable yet. {pc['claims']} of the {len(test):,} test claims ({pc['claims'] / len(test):.1%}) come from partners with no decided new-regime claim and are scored by the label-free part alone (under a {HEADLINE_L}-day lag it would be {cold14}).

Why the test could score lower. (1) The history is up to 92 days stale by the end of the test; the oldest history I could validate was 43 days old. (2) The investigation lag is unknown. (3) On the stress-test fold (May 2026, the first month of the new pattern) this recipe scored ROC-AUC {a['roc_auc']:.3f}, far below random, as did every fitted model: it had learned the old pattern, which reversed on 1 May. Another shift would hurt the same way. (4) Everything rests on {frauds} frauds from {fraud_partners} partners, so the model choice itself could be luck.

Why it could score higher. Fraud was still rising ({may:.2%} of claims in May, {june_rate:.2%} in June), which lifts PR-AUC. And most of the test is scored with fresher and fuller history than the validation had: {len(final_new):,} decided claims with {int(final_new['is_fraud'].sum())} frauds against {len(b_new)} with {int(b_new['is_fraud'].sum())}.

How the outlet-identity question was handled, and what it cost. The first base model had {len(BASE_COLUMNS['M5f'])} features, {len(BASE_COLUMNS['M5f']) - len(BASE_COLUMNS['M5m'])} of them city and partner-type columns, and some of its largest coefficients were cities. With about ten fraudulent outlets those columns can simply memorise which outlet is which. I changed the base model three times, each time against a rule written down before the run, and then stopped. (1) Dropping city, partner type, partner age and the new-outlet flag together was rejected: the original won on PR-AUC in {wins['lite']:.1%} of partner resamples ({full['pr_auc']:.3f} against {lite['pr_auc']:.3f}). (2) Dropping only city and partner type was adopted: the original won in only {wins['mid']:.1%} ({full['pr_auc']:.3f} against {mid['pr_auc']:.3f}). (3) Checking the explanations then showed a second problem: the continuous partner-age feature had been learned mostly from the months before 1 May, when {cohort['before_may_established_frauds']} of {cohort['before_may_frauds']} confirmed frauds came from established outlets, so it raised the score of long-established outlets (by up to {age['established']['both_max']:+.2f} in log-odds together with the flag) although after 1 May they had {cohort['established_outlet_frauds']} frauds in {cohort['established_outlet_claims']:,} claims. I removed it on that principle, not because of its June score: the model with it won in only {wins['noage']:.1%} of resamples ({mid['pr_auc']:.3f} against {h['pr_auc']:.3f}), so nothing measurable was lost. My other worry, that a very low age would cancel the new-outlet flag for outlets onboarded after June, was not borne out: it shifted them by {age['after_june']['age'] - age['new']['age']:+.2f} against a flag worth {age['after_june']['flag']:+.2f}. The new-outlet flag stays because the group effect is measured: {cohort['new_outlet_frauds']} frauds in {cohort['new_outlet_claims']} claims after 1 May for new outlets against {cohort['established_outlet_frauds']} in {cohort['established_outlet_claims']:,} for established ones. Its cost is real: it is what makes the model rank backwards when the pattern changes (ROC-AUC {a['roc_auc']:.3f} on the May stress test, against {a_lite['roc_auc']:.3f} without it), and most new outlets are clean. Counted one way everywhere ({cohort['definition']}), there are {cohort['new_outlets']} new outlets, of which {cohort['new_outlets_with_fraud']} have a confirmed fraud and {cohort['new_outlets_without_fraud']} have none. Explanations never name a city or a partner type, and the outlet line always carries those group numbers.

Money. A full review list of 40 does not pay on this data: in June it nets Rs {h['top40_by_p']['net_value_rs']:,.0f} after Rs 380 per genuine claim held and Rs 260 per claim reviewed. The rule I recommend reviews only claims whose partner has confirmed fraud above the average rate: {rule['n_reviewed']} reviews, {rule['frauds_found']} frauds and a net Rs {rule['net_value_rs']:,.0f} for June ({rule_ci['net_value_rs'][0]:,.0f} to {rule_ci['net_value_rs'][1]:,.0f}), about Rs {rule['rs_per_claim_reviewed']:,.0f} per claim reviewed. That list is the same whichever base model is used, because it depends only on which partners have a confirmed fraud record; the base model only orders the claims in it and the claims beyond it. Its margin: the frauds it caught averaged Rs {margin['fraud_amount']:,.0f}, so it breaks even when {margin['break_even']:.0%} of the reviewed claims are fraud, and {margin['observed']:.0%} were ({margin['frauds_found']} of {margin['n_reviewed']}; it needs {margin['needed']}). The rule was defined after seeing June and is not independently validated: one month, {fraud_partners} partners.

Expected frauds. The scores sum to {predictions['score'].sum():.0f} over the {len(test):,} test claims. {pa['sum']:.0f} of that sits in the {pa['claims']} claims from the {pa['partners']} partners with confirmed fraud above the average rate (an implied fraud rate of {pa['rate']:.0%}), which is the group the review rule looks at. {pb['sum']:.0f} is spread over the {pb['claims']:,} claims from partners with a clean or below-average record ({pb['rate']:.1%}) and {pc['sum']:.0f} over the {pc['claims']} claims from partners with no decided claim yet ({pc['rate']:.1%}).

Probabilities and their limits. The scores carry a two-parameter Platt correction (slope {slope:.3f}, intercept {intercept:+.3f}) fitted on the June validation predictions. Before it the model was short by {june['flagged']['frauds'] - june['flagged']['raw']:.1f} frauds among the flagged partners ({june['flagged']['raw']:.1f} predicted, {june['flagged']['frauds']} observed) and by {(june['clean']['frauds'] + june['no_history']['frauds']) - (june['clean']['raw'] + june['no_history']['raw']):+.1f} among everyone else. The intercept is a single number added to every claim, so it lifts the unflagged claims as well. The result: the implied fraud rates for the two unflagged groups in the test ({pb['rate']:.1%} and {pc['rate']:.1%}) are above what June showed for the same groups ({june['clean']['rate']:.1%} and {june['no_history']['rate']:.1%}), and the total of {predictions['score'].sum():.0f} is probably too high for that reason. Treat the probabilities as indicative; the ranking is what matters and what I would be judged on."""
    print(text)
    print(f"\n[Note, not part of the form text: the point guess {POINT_GUESS_PR_AUC[0]:.2f}-{POINT_GUESS_PR_AUC[1]:.2f} is the author's judgement "
          f"and is not computed. The range is computed on Fold B: {low:.3f} ({BASE_NAME} alone, L = {HEADLINE_L}) to {high:.3f} "
          f"({FINAL_NAME}, L = 0); the June figure is {h['pr_auc']:.3f} ({FINAL_NAME}, L = {HEADLINE_L}).]")


def main():
    print(f"Final model (src/final.py): {MODEL_NAMES[FINAL]}, Platt-corrected.")
    print("Validation numbers quoted here come from Fold B (score June 2026), the PRIMARY fold; Fold A (score May 2026) is the "
          "STRESS TEST of the regime change. M0 (partner_rate_ratio only) is the benchmark.")
    slope, intercept, june = calibrate()
    fit = final_fit(slope, intercept)
    predictions = write_predictions(fit, slope, intercept)
    parts = decomposition(fit)
    composition(fit)
    benchmarks(fit)
    test_review_rule(fit)
    margin = rule_margin()
    profile = save_artifacts(fit, slope, intercept)
    runtime_check(fit, profile)
    examples(fit)
    expected_score_text(fit, slope, intercept, predictions, parts, margin, june)


if __name__ == "__main__":
    with masked_report("final"):
        main()
