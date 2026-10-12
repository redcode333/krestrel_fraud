"""Evaluation write-up for the final model. Reads the model and the folds as they are; changes nothing.

Writes:
  reports/evaluation.md       what was tested, the results, how often it does not work, five missed frauds and
                              five false alarms from Fold B (partner IDs masked, no claim IDs; committed)
  reports/raw/evaluation.md   the same with real partner IDs, claim IDs and exact amounts (git-ignored)

Fold B (score June 2026) is the PRIMARY fold; Fold A (score May 2026) is the STRESS TEST of the regime change.

The ten cases are chosen by a rule fixed before looking at them:
  missed frauds   the 5 largest-amount June frauds that the review rule did not list
  false alarms    the 5 highest-scoring genuine claims on the review-rule list
Their reasons are the app's own (src/scoring.py assess), computed with the model fitted on claims before 1 June
and the outlet's record as it stood on 1 June at a 14-day lag, i.e. what the app would have said at the time.

Run from the repo root:  python src/evaluation_report.py
"""
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from clean import TIME_COL
from evaluate import (BOOTSTRAP_SEED, CONTACT_RS, GOODWILL_RS, PARTNER_RULE_NOTE, REGIME_START, REVIEWS_PER_MONTH, month_of,
                      review_list_value, select_partner_rule, select_top_by_p)
from features import CUTOFFS, SMALL_CLAIM, decided_rows, load_history
from reporting import RAW_ONLY, write_masked_file
from rupees import rs, table
from train import FINAL, FINAL_BASE, FINAL_COLUMNS, HEADLINE_L, SHORT, fit_logistic, mats, rule_eligible, run, score_frame, table_row
import scoring

ROOT = Path(__file__).resolve().parents[1]
N_CASES = 5
MODELS_SHOWN = [("M0", "Benchmark: the outlet's fraud record alone (M0)"),
                (FINAL_BASE, f"Label-free part alone ({SHORT[FINAL_BASE]}): the claim's own details, no fraud record"),
                (FINAL, f"Final model ({SHORT[FINAL]})")]
DESIGNS = [("fold_B", 0, "Fold B (PRIMARY), no lag (optimistic)"), ("fold_B", 30, "Fold B (PRIMARY), 30-day lag (pessimistic)"),
           ("fold_A", HEADLINE_L, "Fold A (STRESS TEST, score May)")]
BANDS = [(1000, "under Rs 1,000"), (SMALL_CLAIM, "Rs 1,000 to 1,999"), (5000, "Rs 2,000 to 4,999"), (float("inf"), "Rs 5,000 or more")]


def amount_band(amount):
    return next(label for limit, label in BANDS if amount < limit)


def raw(lines):
    """Claim-level lines: kept in reports/raw/, dropped from the committed copy by reporting.mask_partner_ids."""
    return [RAW_ONLY + line for line in lines]


# ----------------------------------------------------------------------------------------------
# the app's reasons for a June claim, as they would have been on 1 June
# ----------------------------------------------------------------------------------------------
@contextmanager
def scorer_as_of_fold_b():
    """Make src/scoring.py use the model fitted on claims before 1 June (the Fold B fit) instead of the final one."""
    m = mats("fold_B", HEADLINE_L, "month")
    lr = fit_logistic(m["X_train"][FINAL_COLUMNS], m["y_train"], np.ones(len(m["y_train"])))   # the fit train.predict makes
    real = scoring.load_artifacts
    final_model, constants, partners, source = real()
    assert list(final_model["features"]) == list(FINAL_COLUMNS)
    fold_model = {**final_model, "mean": lr[0].mean_.astype(float), "scale": lr[0].scale_.astype(float),
                  "coef": lr[-1].coef_[0].astype(float), "intercept": float(lr[-1].intercept_[0])}
    scoring.load_artifacts = lambda: (fold_model, constants, partners, source)
    try:
        yield
    finally:
        scoring.load_artifacts = real


def claim_and_state(row, x):
    claim = {"claim_id": row["claim_id"], "partner_id": row["partner_id"], "sku": row["sku"], "submitted_at": str(row[TIME_COL]),
             "claim_amount_inr": row["claim_amount_inr"], "days_since_purchase": row["days_since_purchase"],
             "customer_prior_claims": row["customer_prior_claims"], "photo_attached": row["photo_attached"]}
    state = {"known": True, "is_new_partner": x["is_new_partner"], "age_days_at_profile": None,
             "decided_new_claims": x["partner_new_claim_count"], "new_frauds": x["partner_new_fraud_count"],
             "smoothed_rate": x["partner_new_rate_smoothed"], "rate_ratio": x["partner_rate_ratio"],
             "claims_last_7d": x["partner_claims_last_7d"], "claims_last_30d": x["partner_claims_last_30d"],
             "small_share_last_30d": x["partner_small_share_last_30d"], "avg_amount_last_30d": x["partner_avg_amount_last_30d"]}
    return claim, state


def public_reason(text, amount, list_price):
    """A reason as shown in the committed copy: the exact claim amount is replaced by its band."""
    if text.startswith("Claim of Rs"):
        share = 100 * amount / list_price
        size = "under 10%" if share < 10 else f"about {round(share, -1):.0f}%"
        return f"Claim amount in the band {amount_band(amount)}, {size} of the product's list price."
    return text


# ----------------------------------------------------------------------------------------------
def june():
    """Everything about Fold B at the headline lag: the frame, the scores, the lists and the checks on the app's reasons."""
    m = mats("fold_B", HEADLINE_L, "month")
    frame, X = m["score"].reset_index(drop=True), m["X_score"].reset_index(drop=True)
    p = run(FINAL, "fold_B", HEADLINE_L)["p"]
    y, amount, month = frame["is_fraud"].to_numpy(), frame["claim_amount_inr"].to_numpy(), month_of(frame)
    eligible = rule_eligible("fold_B", HEADLINE_L)
    listed = select_partner_rule(p, month, eligible)
    top40 = select_top_by_p(p, month)
    has_history = X["partner_has_new_history"].to_numpy() == 1
    model = joblib.load(ROOT / "models" / "model.joblib")
    score = expit(model["platt_slope"] * logit(p) + model["platt_intercept"])   # as the app shows it; same order as p
    rank = pd.Series(-p).rank(method="first").astype(int).to_numpy()

    answers, worst_score, worst_feature = [], 0.0, 0.0
    with scorer_as_of_fold_b():
        for i, row in frame.iterrows():
            claim, state = claim_and_state(row, X.iloc[i])
            features, _, _ = scoring.claim_features(claim, state)
            worst_feature = max(worst_feature, max(abs(features[name] - X.iloc[i][name]) for name in FINAL_COLUMNS))
            answer = scoring.assess(claim, state)
            worst_score = max(worst_score, abs(answer["score"] - score[i]))
            answers.append(answer)
    assert worst_feature < 1e-9 and worst_score < 1e-6, (worst_feature, worst_score)
    assert all((a["band"] == "Review") == bool(e) for a, e in zip(answers, eligible)), "the app's Review band is not the review rule"
    eligible_l0 = pd.Series(rule_eligible("fold_B", 0), index=score_frame("fold_B", 0)["claim_id"].to_numpy()).reindex(frame["claim_id"]).to_numpy()
    return {"frame": frame, "X": X, "p": p, "score": score, "y": y, "amount": amount, "month": month, "eligible": eligible,
            "listed": listed, "top40": top40, "has_history": has_history, "rank": rank, "answers": answers,
            "eligible_l0": eligible_l0.astype(bool), "worst_score": worst_score, "worst_feature": worst_feature}


def results_rows(fold, L):
    keep = ["PR-AUC", "ROC-AUC", "frauds in top 40", "frauds in top 100"]
    names = {"frauds in top 40": "Frauds in the top 40", "frauds in top 100": "Frauds in the top 100"}
    rows = []
    for key, label in MODELS_SHOWN:
        r = table_row(run(key, fold, L)["res"])
        rows.append({"Model": label, **{names.get(k, k) + " [90%]": r[k] for k in keep}})
    return rows


def select_cases(j):
    frame, y, amount = j["frame"], j["y"], j["amount"]
    missed = np.flatnonzero((y == 1) & ~j["listed"])
    missed = missed[np.lexsort((frame["claim_id"].to_numpy()[missed], -amount[missed]))][:N_CASES]
    alarms = np.flatnonzero((y == 0) & j["listed"])
    alarms = alarms[np.lexsort((frame["claim_id"].to_numpy()[alarms], -j["p"][alarms]))][:N_CASES]
    return missed, alarms


def outlet_facts(j, history):
    """Per outlet: its record on 1 June at the headline lag, its May claims, and its June claims on the list."""
    frame = j["frame"]
    cutoff = CUTOFFS["fold_B"]
    may = history[(history[TIME_COL] >= REGIME_START) & (history[TIME_COL] < cutoff)]
    facts = {}
    for pid in frame["partner_id"].unique():
        mine = (frame["partner_id"] == pid).to_numpy()
        facts[pid] = {"may_claims": int((may["partner_id"] == pid).sum()), "may_frauds": int(may[may["partner_id"] == pid]["is_fraud"].sum()),
                      "june_listed": int((mine & j["listed"]).sum()), "june_listed_frauds": int(j["y"][mine & j["listed"]].sum()),
                      "june_claims": int(mine.sum()), "june_frauds": int(j["y"][mine].sum())}
    return facts


def missed_sentence(j, i, facts):
    x, f = j["X"].iloc[i], facts[j["frame"].iloc[i]["partner_id"]]
    n, frauds = int(x["partner_new_claim_count"]), int(x["partner_new_fraud_count"])
    if not j["has_history"][i] and f["may_claims"] == 0:
        seen = ("The outlet had submitted no claim at all between 1 May and 1 June, so it had no record of any kind: the model had only "
                "the claim's own details and the review rule could not apply.")
    elif not j["has_history"][i]:
        seen = (f"The outlet had no decided claim on 1 June under a 14-day lag (it had submitted {f['may_claims']} claim"
                f"{'' if f['may_claims'] == 1 else 's'} since 1 May, {f['may_frauds']} of them fraud in the export, none yet old enough to count), "
                "so the model had only the claim's own details and the review rule could not apply.")
    else:
        seen = (f"The outlet's known record on 1 June was {frauds} fraud{'' if frauds == 1 else 's'} in {n} decided claim{'' if n == 1 else 's'}, "
                "not above the average, so the review rule did not apply; "
                + ("the export shows no May fraud for it at all." if f["may_frauds"] == 0
                   else f"its May claims included {f['may_frauds'] - frauds} fraud{'' if f['may_frauds'] - frauds == 1 else 's'} too recent "
                        "to count under the 14-day lag."))
    place = (f" By score it ranked {j['rank'][i]} of {len(j['y'])} June claims, "
             + ("inside the top 40, so a fixed list of 40 would have caught it." if j["top40"][i] else "outside the top 40."))
    lag = " With no investigation lag the outlet would already have been flagged." if j["eligible_l0"][i] else ""
    return seen + place + lag


def alarm_sentence(j, i, facts):
    x, f = j["X"].iloc[i], facts[j["frame"].iloc[i]["partner_id"]]
    n, frauds = int(x["partner_new_claim_count"]), int(x["partner_new_fraud_count"])
    fraud_scores = j["p"][j["listed"] & (j["y"] == 1)]
    above = int((j["p"][i] > fraud_scores).sum())
    return (f"The outlet was flagged on {frauds} fraud{'' if frauds == 1 else 's'} in {n} decided claim{'' if n == 1 else 's'}, and the model "
            f"cannot tell a flagged outlet's genuine claims from its fraudulent ones: this genuine claim scored above {above} of the "
            f"{len(fraud_scores)} frauds on the list. Of the outlet's {f['june_listed']} June claims on the list, {f['june_listed_frauds']} "
            f"were fraud and {f['june_listed'] - f['june_listed_frauds']} genuine.")


def case_block(j, i, label, sentence, products):
    row, answer = j["frame"].iloc[i], j["answers"][i]
    amount, price = float(row["claim_amount_inr"]), products[row["sku"]]["list_price_inr"]
    out = [f"**{label}** - amount {amount_band(amount)}; outlet {row['partner_id']}; the app's band at the time: {answer['band']}.", "",
           "The app's reasons:", ""]
    out += [f"- {public_reason(r['text'], amount, price)} ({r['effect']})" for r in answer["reasons"]]
    out += ["", f"What the model could not see: {sentence}", ""]
    out += raw([f"{label}: claim {row['claim_id']}, submitted {row[TIME_COL]}, Rs {amount:,.0f}, {row['sku']}, score {answer['score']:.4f}, "
                f"rank {j['rank'][i]} of {len(j['y'])}; reasons as the app words them:"]
               + [f"    {r['text']} ({r['effect']})" for r in answer["reasons"]] + [""])
    return out


# ----------------------------------------------------------------------------------------------
def main():
    history = load_history()
    j = june()
    frame, y, amount, listed = j["frame"], j["y"], j["amount"], j["listed"]
    n, frauds = len(y), int(y.sum())
    fraud_outlets = frame[y == 1]["partner_id"].nunique()
    rule, fixed = review_list_value(y, amount, listed), review_list_value(y, amount, j["top40"])
    known = {L: decided_rows(history, CUTOFFS["fold_B"], L)[0] for L in (HEADLINE_L, 0, 30)}
    fold_a = score_frame("fold_A", HEADLINE_L)
    a_res = run(FINAL, "fold_A", HEADLINE_L)["res"]
    a_rule = a_res["upto40_partner_rule"]
    l0 = run(FINAL, "fold_B", 0)["res"]["upto40_partner_rule"]
    all_zero = {"all": 1 - history["is_fraud"].mean(), "june": 1 - y.mean(),
                "new": 1 - history[history[TIME_COL] >= REGIME_START]["is_fraud"].mean()}

    missed_all = (y == 1) & ~listed
    groups = [("On the review list (outlet flagged on 1 June)", (y == 1) & listed),
              ("Missed: outlet had decided claims but was not flagged", missed_all & j["has_history"]),
              ("Missed: outlet had no decided claim yet (no history)", missed_all & ~j["has_history"])]
    fate = [{"What happened to the June frauds": label, "Frauds": int(mask.sum()), "Outlets": frame[mask]["partner_id"].nunique(),
             "Rs": rs(amount[mask].sum()), "Of which in the top 40 by score": int((mask & j["top40"]).sum()),
             "Of which flagged if there were no lag": int((mask & j["eligible_l0"]).sum())} for label, mask in groups]
    fate.append({"What happened to the June frauds": "All June frauds", "Frauds": frauds, "Outlets": fraud_outlets, "Rs": rs(amount[y == 1].sum()),
                 "Of which in the top 40 by score": int(((y == 1) & j["top40"]).sum()),
                 "Of which flagged if there were no lag": int(((y == 1) & j["eligible_l0"]).sum())})
    no_history = ~j["has_history"]
    missed, alarms = select_cases(j)
    facts = outlet_facts(j, history)
    products = scoring.load_artifacts()[1]["products"]
    genuine_listed = int(((y == 0) & listed).sum())

    out = [f"# Evaluation of the final model ({SHORT[FINAL]})", "",
           "Produced by `src/evaluation_report.py`. It reads the model and the validation folds as they are and changes neither. "
           f"Intervals are 90%, from 1,000 resamples of partners with the fixed seed ({BOOTSTRAP_SEED}). Partner IDs are masked and no "
           "claim ID appears in this copy.", "",
           "**Fold B (train on claims before 1 June 2026, score June) is the PRIMARY fold. Fold A (train before 1 May, score May) is a "
           "STRESS TEST of the regime change: the new fraud pattern starts on 1 May, so Fold A's training data holds none of it.**", "",
           "## 1. What was tested, and how", "",
           f"- **Split by time, never at random.** Time is the claim's first submission. Each claim appears once, so no re-submitted claim "
           "sits on both sides of a split.",
           f"- **Fold B (PRIMARY):** {n} decided June claims, {frauds} frauds from {fraud_outlets} outlets ({y.mean():.2%}).",
           f"- **Fold A (STRESS TEST):** {len(fold_a)} decided May claims, {int(fold_a['is_fraud'].sum())} frauds from "
           f"{fold_a[fold_a['is_fraud'] == 1]['partner_id'].nunique()} outlets.",
           "- **What the model may know.** An outlet's fraud record is frozen at the start of the scored month and counts only claims old "
           "enough to have been investigated. How long an investigation takes is not in the export, so it is a parameter (the lag), "
           "never a silent assumption.",
           f"- **Headline lag: {HEADLINE_L} days.** On 1 June that leaves {len(known[HEADLINE_L])} decided claims since 1 May with "
           f"{int(known[HEADLINE_L]['is_fraud'].sum())} frauds. Also reported: no lag (optimistic, {len(known[0])} decided claims, "
           f"{int(known[0]['is_fraud'].sum())} frauds) and a 30-day lag (pessimistic, {len(known[30])} decided claims, below the 50 needed, "
           "so no fraud record is usable and the model falls back to the claim's own details).",
           "- **Three scorers compared:** the outlet's fraud record alone (the benchmark), the claim's own details alone (the label-free "
           "part of the final model), and the final model, which adds the two.",
           f"- **Doing nothing as a reference.** Predicting \"not fraud\" for every claim is {all_zero['all']:.1%} accurate on all "
           f"{len(history):,} decided claims, {all_zero['new']:.2%} on May-June and {all_zero['june']:.2%} on June. Picking 40 June claims "
           f"blind finds about {REVIEWS_PER_MONTH * frauds / n:.1f} frauds; a random ranking has PR-AUC {y.mean():.3f} and ROC-AUC 0.500.", "",
           "## 2. Results", "",
           f"### Fold B (PRIMARY), {HEADLINE_L}-day lag: the headline", "", table(results_rows("fold_B", HEADLINE_L)), "",
           f"\"Top 40\" is the month's 40 highest scores; \"top 100\" the 100 highest. There are {frauds} frauds to find. The intervals are "
           f"wide because {frauds} frauds from {fraud_outlets} outlets is very little: one outlet more or less in a resample moves every number.", ""]
    for fold, L, title in DESIGNS:
        out += [f"### {title}", "", table(results_rows(fold, L)), ""]
    out += ["Fold A is the stress test: on 1 May no outlet has a fraud record in the new pattern, so the benchmark is a random draw there and "
            "the final model equals its label-free part. Pooled and old-design results (train before 1 May, score May-June: what a model "
            "that has not seen the new regime scores) are in `reports/train.txt`.", "",
            "## 3. How often it does not work", "",
            f"**When the pattern changes, it ranks backwards.** On Fold A (STRESS TEST, score May) the final model scores ROC-AUC "
            f"{a_res['roc_auc']:.3f}, far below the 0.500 of a random ranking, and finds {a_res['top40_by_p']['frauds_found']} of the "
            f"{int(fold_a['is_fraud'].sum())} frauds in its top 40 and {a_res['top100_overall_frauds']} in its top 100. It had learned the "
            f"pattern from before 1 May, which reversed. The review rule reviews {a_rule['n_reviewed']} claims that month: with no confirmed "
            "fraud on record it flags nobody. A new shift would hurt in the same way until new outcomes arrive.", "",
            f"**In June (Fold B, PRIMARY, {HEADLINE_L}-day lag) the review list found {rule['frauds_found']} of the {frauds} frauds.** It "
            f"reviewed {rule['n_reviewed']} claims; {genuine_listed} of those {rule['n_reviewed']} were genuine customers held for nothing "
            f"({genuine_listed / rule['n_reviewed']:.0%}). {frauds - rule['frauds_found']} frauds worth Rs {rs(amount[missed_all].sum())} were "
            "paid.", "", table(fate), "",
            f"**Frauds missed because of no history: {int((missed_all & no_history).sum())}.** {int(no_history.sum())} of the {n} June claims "
            f"came from outlets with no decided claim on 1 June; {int(y[no_history].sum())} of them were fraud and the rule could reach none. "
            f"With no investigation lag the list would have held {l0['n_reviewed']} claims and {l0['frauds_found']} frauds; with a 30-day lag "
            "it would have been empty.", "",
            f"**A full list of 40 does worse, not better.** Ranked by score alone, the top 40 holds {fixed['frauds_found']} frauds and "
            f"{fixed['genuine_held']} genuine claims and nets Rs {rs(fixed['net_value_rs'])}; the review list nets Rs {rs(rule['net_value_rs'])} "
            f"(Rs {GOODWILL_RS} per genuine claim held, Rs {CONTACT_RS} per claim reviewed).", "",
            f"The review rule carries its label: {PARTNER_RULE_NOTE}.", "",
            "## 4. Five missed frauds and five false alarms (Fold B, PRIMARY)", "",
            "**Selection rule, fixed before looking at the cases:** the 5 largest-amount June frauds that the review rule did not list, and "
            "the 5 highest-scoring genuine claims on the review-rule list. Ties go to the lower claim ID. Nothing else was considered.", "",
            "The reasons are the app's own wording (`assess` in `src/scoring.py`), produced with the model fitted on claims before 1 June and "
            f"the outlet's record as it stood on 1 June at a {HEADLINE_L}-day lag, so they show what the app would have said at the time. "
            f"Checked for all {n} June claims: the app's score matches the validation score (largest difference {j['worst_score']:.1e}) and "
            "its Review band matches the review rule. One thing is newer than 1 June: the group statistics in the outlet-age line are "
            "computed on May and June together. In this copy the exact claim amount is replaced by its band.", "",
            "### Missed frauds", ""]
    for k, i in enumerate(missed, 1):
        out += case_block(j, i, f"M{k}", missed_sentence(j, i, facts), products)
    out += ["### False alarms", ""]
    for k, i in enumerate(alarms, 1):
        out += case_block(j, i, f"F{k}", alarm_sentence(j, i, facts), products)
    out += ["## Sources", "",
            "Every number above is computed by `src/evaluation_report.py` when it runs; none is typed in. Where the same number appears in "
            "another report, it comes from the same functions:", "",
            "- Section 1: fold sizes from `get_folds` in `src/evaluate.py`; decided claims at each lag from `decided_rows` in "
            "`src/features.py`; the do-nothing accuracy from the labels in `data/clean_train.parquet` (also `reports/explore.txt`, item 7).",
            "- Section 2: `run` and `table_row` in `src/train.py` (the same rows as `reports/train.txt`); intervals from `bootstrap_ranking` "
            "and `bootstrap_top40` in `src/evaluate.py`.",
            "- Section 3: `select_partner_rule`, `select_top_by_p` and `review_list_value` in `src/evaluate.py` (the same lists as "
            "`reports/rupees.md`, section A, and `reports/train.txt`, cold start).",
            "- Section 4: selection by the rule stated there; reasons from `assess` in `src/scoring.py`; outlet records from the Fold B "
            "feature matrix (`fold_matrices` in `src/features.py`).",
            f"- The real partner IDs, claim IDs, exact amounts and scores of the ten cases are in `reports/raw/evaluation.md`, which is git-ignored.", ""]
    text = "\n".join(out)
    public_ids = set(frame["claim_id"].astype(str))
    assert not any(cid in line for line in out if not line.startswith(RAW_ONLY) for cid in public_ids), "a claim ID outside raw-only lines"
    checked = write_masked_file("evaluation.md", text)
    print(text)
    print(f"\nsaved: reports/evaluation.md (partner IDs masked, no claim IDs) and reports/raw/evaluation.md; partner mask check passed on {checked} reports")


if __name__ == "__main__":
    main()
