"""Money analysis for the final model. Reads the model and predictions.csv as they are; changes neither.

Writes:
  reports/rupees.md             assumptions, June list values, break-even table, monthly scaling, sensitivity,
                                cost per prediction, cold start, and the watchlist as Outlet A to J
                                (partner IDs masked; committed; reproduces exactly between runs)
  reports/timing.txt            measured scoring time. THE ONLY report that varies between runs.
  reports/raw/rupees.md         the same with real partner IDs (git-ignored)
  reports/raw/watchlist.csv     the 10 flagged outlets with real partner IDs (git-ignored)
  reports/raw/review_list.csv   per month, the top 40 claims by score and the review-rule list (git-ignored)

Timing is measured when the script runs, so it is kept out of rupees.md and written to
reports/timing.txt on its own; everything else reproduces exactly.

Run from the repo root:  python src/rupees.py
"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from clean import TIME_COL
from evaluate import (BOOTSTRAP_SEED, CONTACT_RS, GOODWILL_RS, PARTNER_RULE_NOTE, REVIEWS_PER_MONTH, bootstrap_top40, metrics, month_of,
                      review_list_value, select_partner_rule)
from features import SMALL_CLAIM
from reporting import RAW_REPORTS, partner_mask, write_masked_file
from train import FINAL, HEADLINE_L, SHORT, mats, rule_eligible, run, score_frame
import scoring

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ["2026-07", "2026-08", "2026-09"]
MONTH_NAMES = {"2026-07": "July", "2026-08": "August", "2026-09": "September"}
SCENARIOS = [0.25, 0.35, 0.52]   # fraud rate among reviewed claims: pessimistic, about break-even, as observed in June
SENSITIVITY_RATES = [0.20, 0.30, 0.40, 0.52]
JUNE_RATE = 0.52                 # 13 frauds in 25 reviews on the June rule list


def table(rows, columns=None):
    """A list of dicts as a Markdown table."""
    columns = columns or list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def rs(x):
    return f"{x:,.0f}"


def interval(v, ci, fmt=rs):
    return f"{fmt(v)} [{fmt(ci[0])} to {fmt(ci[1])}]"


def net_per_review(rate, fraud_amount):
    return rate * fraud_amount - (1 - rate) * GOODWILL_RS - CONTACT_RS


# ----------------------------------------------------------------------------------------------
# A. June money figures
# ----------------------------------------------------------------------------------------------
def june_lists():
    frame = score_frame("fold_B", HEADLINE_L)
    y, amount, month, partner = (frame["is_fraud"].to_numpy(), frame["claim_amount_inr"].to_numpy(), month_of(frame),
                                 frame["partner_id"].to_numpy())
    eligible = rule_eligible("fold_B", HEADLINE_L)
    res = run(FINAL, "fold_B", HEADLINE_L)["res"]
    model = joblib.load(ROOT / "models" / "model.joblib")
    calibrated = expit(model["platt_slope"] * logit(run(FINAL, "fold_B", HEADLINE_L)["p"]) + model["platt_intercept"])
    cal = metrics(y, calibrated, amount, month, eligible=eligible)["upto40_stop_early"]
    cal_ci = bootstrap_top40(y, calibrated, amount, month, partner, by="stop_early")

    lists = [("Fixed 40, ranked by score", res["top40_by_p"], res["ci_by_p"]),
             ("Fixed 40, ranked by expected value", res["top40_by_ev"], res["ci_by_ev"]),
             ("Stop-early, raw model probabilities", res["upto40_stop_early"], res["ci_stop_early"]),
             ("Stop-early, Platt-corrected probabilities", cal, cal_ci),
             ("Review rule (partner_rate_ratio > 1, cap 40)", res["upto40_partner_rule"], res["ci_partner_rule"])]
    rows = []
    for name, v, c in lists:
        rows.append({"June list": name, "Reviews": interval(v["n_reviewed"], c["n_reviewed"], lambda x: f"{x:.0f}") if "top" not in name.lower() and "Fixed" not in name else v["n_reviewed"],
                     "Frauds [90%]": interval(v["frauds_found"], c["frauds_found"], lambda x: f"{x:.0f}"),
                     "Genuine held": v["genuine_held"], "Fraud Rs caught": rs(v["fraud_rs_caught"]),
                     "Net Rs [90%]": interval(v["net_value_rs"], c["net_value_rs"]),
                     "Rs per review": "n/a" if not v["n_reviewed"] else rs(v["rs_per_claim_reviewed"])})
    previous = run("M5m", "fold_B", HEADLINE_L)["res"]
    return {"rows": rows, "res": res, "previous": previous, "frame": frame, "eligible": eligible, "cal": cal}


def form_text_check(res):
    """Is every June money number in the form text (reports/final.txt) still what the final model gives?"""
    final_report = (ROOT / "reports" / "final.txt").read_text(encoding="utf-8")
    sentence = next(line for line in final_report.split("\n") if line.startswith("Money. "))
    fixed, rule, ci = res["top40_by_p"], res["upto40_partner_rule"], res["ci_partner_rule"]
    expected = {"fixed list of 40, net Rs": f"{fixed['net_value_rs']:,.0f}", "rule list, reviews": f"{rule['n_reviewed']} reviews",
                "rule list, frauds": f"{rule['frauds_found']} frauds", "rule list, net Rs": f"{rule['net_value_rs']:,.0f}",
                "rule list, lower end": f"{ci['net_value_rs'][0]:,.0f}", "rule list, upper end": f"{ci['net_value_rs'][1]:,.0f}",
                "rule list, Rs per review": f"{rule['rs_per_claim_reviewed']:,.0f}"}
    rows = [{"Number in the form text": k, "Recomputed for " + SHORT[FINAL]: v, "In the form text?": "yes, unchanged" if v in sentence else "NO, CHANGE IT"}
            for k, v in expected.items()]
    return rows, sentence, all(v in sentence for v in expected.values())


# ----------------------------------------------------------------------------------------------
# B. break-even, monthly scaling, cost per prediction
# ----------------------------------------------------------------------------------------------
def rule_list_fraud_amounts():
    out = {}
    for L in (HEADLINE_L, 0):
        frame = score_frame("fold_B", L)
        y, amount = frame["is_fraud"].to_numpy(), frame["claim_amount_inr"].to_numpy()
        listed = select_partner_rule(run(FINAL, "fold_B", L)["p"], month_of(frame), rule_eligible("fold_B", L))
        value = review_list_value(y, amount, listed)
        fraud_amount = float(amount[listed & (y == 1)].mean())
        out[L] = {"fraud_amount": fraud_amount, "break_even": (CONTACT_RS + GOODWILL_RS) / (fraud_amount + GOODWILL_RS),
                  "observed": value["frauds_found"] / value["n_reviewed"], **value}
    return out


def break_even_rows(amounts):
    a14, a0 = amounts[HEADLINE_L]["fraud_amount"], amounts[0]["fraud_amount"]
    rows = []
    for pct in range(10, 65, 5):
        rate = pct / 100
        n14, n0 = net_per_review(rate, a14), net_per_review(rate, a0)
        rows.append({"Fraud rate among reviewed claims": f"{pct}%", f"Net Rs per review at Rs {a14:,.0f}": rs(n14),
                     f"Net Rs per review at Rs {a0:,.0f}": rs(n0),
                     "Pays?": "yes at both" if n14 > 0 and n0 > 0 else "only at the higher amount" if n14 > 0 else "no"})
    return rows


def test_frame():
    """Test claims with their final score from predictions.csv (read, never recomputed) and partner state."""
    m = mats("final", 0, "month")
    test, X = m["score"].reset_index(drop=True), m["X_score"]
    pred = pd.read_csv(ROOT / "predictions.csv", float_precision="round_trip").set_index("claim_id")["score"]
    assert set(pred.index) == set(test["claim_id"]) and len(pred) == len(test)
    frame = pd.DataFrame({
        "claim_id": test["claim_id"], "partner_id": test["partner_id"], "month": month_of(test), "amount": test["claim_amount_inr"],
        "score": pred.reindex(test["claim_id"]).to_numpy(), "ratio": X["partner_rate_ratio"].to_numpy(),
        "has_history": X["partner_has_new_history"].to_numpy() == 1, "decided_claims": X["partner_new_claim_count"].to_numpy(),
        "confirmed_frauds": X["partner_new_fraud_count"].to_numpy(), "is_new": X["is_new_partner"].to_numpy() == 1,
        "onboarded_after_june": (test["onboarded_date"] > m["train"][TIME_COL].max()).to_numpy(),
    })
    frame["flagged"] = frame["ratio"] > 1
    order = np.argsort(-frame["score"].to_numpy(), kind="stable")
    frame["rank_overall"] = np.empty(len(frame), dtype=int)
    frame.loc[order, "rank_overall"] = np.arange(1, len(frame) + 1)
    frame["rank_in_month"] = frame.groupby("month")["score"].rank(method="first", ascending=False).astype(int)
    flagged_rank = frame[frame["flagged"]].groupby("month")["score"].rank(method="first", ascending=False)
    frame["on_review_rule_list"] = False
    frame.loc[flagged_rank.index, "on_review_rule_list"] = flagged_rank <= REVIEWS_PER_MONTH
    return frame


def monthly_rows(frame, fraud_amount):
    rows, losing = [], {}
    for mth in MONTHS + ["total"]:
        t = frame if mth == "total" else frame[frame["month"] == mth]
        listed = t[t["on_review_rule_list"]]
        row = {"Month": "Jul-Sep total" if mth == "total" else MONTH_NAMES[mth], "Claims": f"{len(t):,}",
               "Qualify under the rule": int(t["flagged"].sum()), "Reviews after the cap of 40": len(listed),
               "Review cost Rs (reviews x 260)": rs(len(listed) * CONTACT_RS), "Average amount of listed claims Rs": rs(listed["amount"].mean())}
        for rate in SCENARIOS:
            net = len(listed) * net_per_review(rate, fraud_amount)
            row[f"Net Rs if {rate:.0%} of reviews are fraud"] = rs(net)
            losing.setdefault(rate, []).append(net)
        rows.append(row)
    return rows, {rate: nets[-1] for rate, nets in losing.items()}


def prediction_cost(frame):
    m = mats("final", 0, "month")
    test = m["score"].reset_index(drop=True).head(1000)
    claims = [{"claim_id": r["claim_id"], "partner_id": r["partner_id"], "sku": r["sku"], "submitted_at": str(r[TIME_COL]),
               "claim_amount_inr": r["claim_amount_inr"], "days_since_purchase": r["days_since_purchase"],
               "customer_prior_claims": r["customer_prior_claims"], "photo_attached": r["photo_attached"]} for _, r in test.iterrows()]
    scoring.load_artifacts()
    scoring.explain(claims[0])   # warm-up
    timings = {}
    for name, fn in [("score only (score_claim)", scoring.score_claim), ("score with reasons (explain)", scoring.explain)]:
        each = []
        for claim in claims:
            start = time.perf_counter()
            fn(claim)
            each.append((time.perf_counter() - start) * 1000)
        timings[name] = (float(np.mean(each)), float(np.percentile(each, 95)), float(np.max(each)))
    sizes = {f: (ROOT / "models" / f).stat().st_size for f in ["model.joblib", "constants.json", "partner_profile.csv"]}
    return timings, sizes, len(claims)


# ----------------------------------------------------------------------------------------------
# C. cold start   D. watchlist and review list
# ----------------------------------------------------------------------------------------------
def cold_start(frame):
    cold = frame[~frame["has_history"]]
    groups = [("Onboarded after 30 Jun 2026", cold[cold["onboarded_after_june"]]),
              ("Other partners with no decided claim since 1 May 2026", cold[~cold["onboarded_after_june"]])]
    summary, detail = [], []
    for label, g in groups + [("All partners with no history", cold)]:
        summary.append({"Group": label, "Partners": g["partner_id"].nunique(), "Test claims": len(g),
                        "Expected frauds (sum of scores)": f"{g['score'].sum():.1f}", "Implied fraud rate": f"{g['score'].mean():.1%}",
                        "Claims in the top 40": int((g["rank_overall"] <= 40).sum()), "Claims in the top 100": int((g["rank_overall"] <= 100).sum()),
                        "Highest score": f"{g['score'].max():.3f}"})
    for label, g in groups:
        for pid, t in sorted(g.groupby("partner_id"), key=lambda kv: -kv[1]["score"].sum()):
            detail.append({"Partner": pid, "Group": label, "New outlet flag": "yes" if t["is_new"].iloc[0] else "no", "Test claims": len(t),
                           "Under Rs 2,000": int((t["amount"] < SMALL_CLAIM).sum()), "Expected frauds": f"{t['score'].sum():.2f}",
                           "In top 40": int((t["rank_overall"] <= 40).sum()), "In top 100": int((t["rank_overall"] <= 100).sum()),
                           "Highest score": f"{t['score'].max():.3f}"})
    return summary, detail


def watchlist(frame):
    mapping = partner_mask()
    flagged = frame[frame["flagged"]]
    rows = []
    for pid, t in flagged.groupby("partner_id"):
        rows.append({"partner_id": pid, "masked_id": mapping[pid], "decided_claims_since_1_may": int(t["decided_claims"].iloc[0]),
                     "confirmed_frauds": int(t["confirmed_frauds"].iloc[0]), "partner_rate_ratio": round(float(t["ratio"].iloc[0]), 2),
                     "test_claims": len(t), "test_claims_under_rs_2000": int((t["amount"] < SMALL_CLAIM).sum()),
                     "expected_frauds": round(float(t["score"].sum()), 2), "claims_on_review_rule_list": int(t["on_review_rule_list"].sum())})
    wl = pd.DataFrame(rows).sort_values(["expected_frauds", "partner_id"], ascending=[False, True]).reset_index(drop=True)
    wl.insert(0, "outlet", [f"Outlet {chr(65 + i)}" for i in range(len(wl))])
    return wl


def write_private_files(frame, wl):
    RAW_REPORTS.mkdir(parents=True, exist_ok=True)
    wl.to_csv(RAW_REPORTS / "watchlist.csv", index=False)
    label = dict(zip(wl["partner_id"], wl["outlet"]))
    keep = frame[(frame["rank_in_month"] <= REVIEWS_PER_MONTH) | frame["on_review_rule_list"]].copy()
    keep["outlet"] = keep["partner_id"].map(label).fillna("")
    keep["in_top40_by_score"] = keep["rank_in_month"] <= REVIEWS_PER_MONTH
    keep = keep.sort_values(["month", "rank_in_month"])[["month", "rank_in_month", "claim_id", "partner_id", "outlet", "amount", "score",
                                                         "in_top40_by_score", "on_review_rule_list"]]
    keep = keep.rename(columns={"amount": "claim_amount_inr"})
    keep.to_csv(RAW_REPORTS / "review_list.csv", index=False)
    return keep


def sensitivity_rows(frame, wl, fraud_amount):
    """Quarter net of the review-rule list when the flagged outlets commit less fraud than June's 52%."""
    listed = frame[frame["on_review_rule_list"]]
    label = dict(zip(wl["partner_id"], wl["outlet"]))
    outlet = listed["partner_id"].map(label)
    n_ab, n_other = int(outlet.isin(["Outlet A", "Outlet B"]).sum()), int((~outlet.isin(["Outlet A", "Outlet B"])).sum())
    rows = []
    for rate in SENSITIVITY_RATES:
        ab_only = n_ab * net_per_review(rate, fraud_amount) + n_other * net_per_review(JUNE_RATE, fraud_amount)
        everyone = (n_ab + n_other) * net_per_review(rate, fraud_amount)
        rows.append({"Fraud rate assumed": f"{rate:.0%}",
                     f"Quarter net Rs if only Outlets A and B run at this rate ({n_ab} reviews), the other 8 at 52% ({n_other} reviews)": rs(ab_only),
                     "Frauds found in that case": f"{n_ab * rate + n_other * JUNE_RATE:.0f}",
                     f"Quarter net Rs if ALL flagged outlets run at this rate ({n_ab + n_other} reviews)": rs(everyone),
                     "Frauds found in that case ": f"{(n_ab + n_other) * rate:.0f}"})
    return rows, n_ab, n_other


def timing_report(timings, n_timed):
    mean_ms = timings["score with reasons (explain)"][0]
    lines = ["Scoring time (src/rupees.py). THIS FILE VARIES BETWEEN RUNS AND MACHINES; every other report reproduces exactly.", "",
             f"Measured by scoring {n_timed:,} single claims one at a time through src/scoring.py, after the model files were loaded once.", "",
             f"{'what was timed':<32}{'mean ms':>10}{'p95 ms':>10}{'slowest ms':>12}"]
    lines += [f"{name:<32}{t[0]:>10.3f}{t[1]:>10.3f}{t[2]:>12.3f}" for name, t in timings.items()]
    lines += ["", f"A month of about 750 claims is about {750 * mean_ms / 1000:.2f} seconds of computing, reasons included.",
              "API cost: Rs 0. No paid service or key is used.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
def main():
    june = june_lists()
    res, previous = june["res"], june["previous"]
    check_rows, sentence, unchanged = form_text_check(res)
    amounts = rule_list_fraud_amounts()
    a14, a0 = amounts[HEADLINE_L], amounts[0]
    frame = test_frame()
    month_rows, totals = monthly_rows(frame, a14["fraud_amount"])
    timings, sizes, n_timed = prediction_cost(frame)
    cold_summary, cold_detail = cold_start(frame)
    wl = watchlist(frame)
    review = write_private_files(frame, wl)
    per_month = frame.groupby("month").size()
    sens_rows, n_ab, n_other = sensitivity_rows(frame, wl, a14["fraud_amount"])
    ab_zero = n_ab * net_per_review(0.0, a14["fraud_amount"]) + n_other * net_per_review(JUNE_RATE, a14["fraud_amount"])
    losing = [r for r in SENSITIVITY_RATES if (n_ab + n_other) * net_per_review(r, a14["fraud_amount"]) < 0]
    losing_at = f"{max(losing):.0%} or below" if losing else "any rate in the table, it still pays; none of them"
    write_masked_file("timing.txt", timing_report(timings, n_timed))
    slowest_p95 = max(t[1] for t in timings.values())
    speed = ("under a millisecond per claim" if slowest_p95 < 1 else "a few milliseconds per claim at most" if slowest_p95 < 10
             else "measured in the timing file")

    out = [f"# Rupees: what a review costs and what it recovers ({SHORT[FINAL]})", "",
           f"Produced by `src/rupees.py`. It reads the final model and `predictions.csv` as they are and changes neither. Intervals are 90%, "
           f"from 1,000 resamples of partners with the fixed seed ({BOOTSTRAP_SEED}). Partner IDs are masked in this copy.", "",
           "## Assumptions", "",
           "Every rupee figure below rests on these. They come from policy section 4 and from one month of validation.", "",
           "1. **A reviewed fraud is always found.** If the investigation desk reviews a fraudulent claim, it recognises it and the claim "
           "is not paid. A desk that misses some would recover less.",
           "2. **Avoided payout = the claim amount.** Stopping a fraudulent claim saves exactly what was claimed; no recovery costs, no "
           "wider deterrent effect.",
           f"3. **Rs {CONTACT_RS} per reviewed claim.** The policy's blended cost of a service contact, charged on every review, fraud or genuine.",
           f"4. **Rs {GOODWILL_RS} per genuine claim held.** The goodwill Kestrel issues when a genuine customer's repair is delayed by a review.",
           f"5. **Average fraud amount Rs {a14['fraud_amount']:,.0f}.** The June average of the {a14['frauds_found']} frauds the review rule caught "
           f"(June 2026, {HEADLINE_L}-day lag). Scenario tables use it as the amount saved per fraud found; the test months may differ.", "",
           "## A. June 2026, every list, for the final model", "",
           f"June 2026 is the primary validation fold (713 claims, 22 frauds from 8 partners), history frozen on 1 June with a {HEADLINE_L}-day lag.", "",
           table(june["rows"]), "",
           "The stop-early list depends on calibration and the Platt-corrected row is judged on the claims the correction was fitted on; "
           f"neither is the recommended rule. The review rule carries the label: {PARTNER_RULE_NOTE}.", "",
           "### Do the June money numbers in the form text still hold?", "", table(check_rows), "",
           ("**No number changes.** " if unchanged else "**Some numbers change; see the table.** ")
           + f"The fixed list of 40 nets Rs {rs(res['top40_by_p']['net_value_rs'])} for {SHORT[FINAL]}. The previous model (M5-mid) gives "
           f"Rs {rs(previous['top40_by_p']['net_value_rs'])} on the same fold: "
           + ("the same figure, because both models put the same number of frauds and the same fraud rupees on that list "
              f"({res['top40_by_p']['frauds_found']} frauds, Rs {rs(res['top40_by_p']['fraud_rs_caught'])})."
              if res["top40_by_p"]["net_value_rs"] == previous["top40_by_p"]["net_value_rs"] else "a different figure.")
           + " The form text was regenerated after the final model was adopted, so its figure is already the final model's.", "",
           "The sentence as it stands in `reports/final.txt`:", "", "> " + sentence, "",
           "## B1. Break-even: how often must a reviewed claim be fraud?", "",
           f"Net rupees per review = fraud rate x average fraud amount - (1 - fraud rate) x 380 - 260. The two amounts are the average "
           f"fraud claim on the June rule list at a {HEADLINE_L}-day lag (Rs {a14['fraud_amount']:,.0f}) and at no lag (Rs {a0['fraud_amount']:,.0f}).", "",
           table(break_even_rows(amounts)), "",
           f"**Break-even is {a14['break_even']:.1%} at Rs {a14['fraud_amount']:,.0f}** and {a0['break_even']:.1%} at Rs {a0['fraud_amount']:,.0f}. "
           f"June's rule list ran at {a14['observed']:.1%} ({a14['frauds_found']} frauds in {a14['n_reviewed']} reviews), "
           f"{100 * (a14['observed'] - a14['break_even']):.1f} points above break-even. That margin is {a14['frauds_found']} frauds against the "
           f"{int(np.ceil(a14['break_even'] * a14['n_reviewed']))} needed, in one month.", "",
           "## B2. Monthly scaling on the test months", "",
           f"The test set has {len(frame):,} claims over three months ({', '.join(f'{MONTH_NAMES[m]} {per_month[m]}' for m in MONTHS)}), about "
           f"{len(frame) / 3:.0f} a month. Net = reviews x (net rupees per review at Rs {a14['fraud_amount']:,.0f}). The three scenarios are "
           "assumptions about how many reviewed claims turn out to be fraud, because the test outcomes are not known.", "",
           table(month_rows), "",
           f"**The 25% scenario loses money** (Rs {rs(totals[0.25])} over the quarter): if only a quarter of the reviewed claims are fraud, "
           f"the goodwill and contact costs outweigh the fraud stopped. **35% is about break-even** (Rs {rs(totals[0.35])} over the quarter). "
           f"**52%, June's observed rate, pays** (Rs {rs(totals[0.52])} over the quarter). The rule earns money only while the flagged outlets "
           "keep committing fraud at something like June's rate; it should be re-checked each month against the desk's findings.", "",
           "## B2a. Sensitivity: what if the flagged outlets commit less fraud than in June?", "",
           f"Quarter net of the review-rule list ({n_ab + n_other} reviews over July to September) at Rs {a14['fraud_amount']:,.0f} per fraud found. "
           f"Outlets A and B are singled out because they supply {n_ab} of the {n_ab + n_other} reviews and were flagged on only 2 decided claims "
           "each, the thinnest evidence on the watchlist (section D). June's observed rate was 52%.", "",
           table(sens_rows), "",
           f"Break-even for the whole list is {a14['break_even']:.1%}. If only Outlets A and B disappoint, the list still pays at every rate "
           f"in the table: even with no fraud at all at those two it would net Rs {rs(ab_zero)} over the quarter, because the other eight "
           f"outlets carry it at June's rate. If ALL flagged outlets fall to {losing_at} the list loses money.", "",
           "## B3. What one prediction costs", "",
           f"**API cost: Rs 0.** The model is a small logistic formula run locally; no paid service or key is used at any point. Scoring is "
           f"{speed}; the measured times are in `reports/timing.txt`, the one report that varies between runs. "
           f"Model files: `model.joblib` {sizes['model.joblib']:,} bytes, `constants.json` {sizes['constants.json']:,} bytes, partner profile "
           f"{sizes['partner_profile.csv']:,} bytes.", "",
           "**This is not the cost of a review.** A review is a person's work and a contact with the customer: Rs 260 for every reviewed "
           "claim, plus Rs 380 in goodwill when the claim turns out to be genuine. With up to 40 reviews a month that is up to "
           f"Rs {rs(REVIEWS_PER_MONTH * CONTACT_RS)} a month in contact cost alone. Scoring all 750 claims is free; reviewing is where the money goes.", "",
           "## C. Cold start: partners the model knows nothing about", "",
           "Partners with no decided claim since 1 May 2026 get a ratio of exactly 1.0, so they are scored by the label-free part alone "
           "and can never qualify for the review rule.", "", table(cold_summary), "", "By partner (masked IDs):", "", table(cold_detail), "",
           f"None of these partners can be on the review list, whatever their score. If a new outlet starts committing fraud in July, "
           "the frozen history cannot see it; only refreshing the history as outcomes arrive can.", "",
           "## D. Watchlist: the 10 flagged outlets", "",
           "The 10 partners with partner_rate_ratio > 1 in the history frozen on 30 Jun 2026, named Outlet A to J for the memo, ordered by "
           "expected frauds in the test months. The real partner IDs are in `reports/raw/watchlist.csv`, which is git-ignored.", "",
           table([{"Outlet": r["outlet"], "Decided claims since 1 May": r["decided_claims_since_1_may"], "Confirmed frauds": r["confirmed_frauds"],
                   "Ratio to average": f"{r['partner_rate_ratio']:.1f}x", "Test claims (Jul-Sep)": r["test_claims"],
                   "Of which under Rs 2,000": r["test_claims_under_rs_2000"], "Expected frauds": f"{r['expected_frauds']:.1f}",
                   "Claims on the review list": r["claims_on_review_rule_list"]} for _, r in wl.iterrows()]
                 + [{"Outlet": "All 10", "Decided claims since 1 May": int(wl["decided_claims_since_1_may"].sum()),
                     "Confirmed frauds": int(wl["confirmed_frauds"].sum()), "Ratio to average": "",
                     "Test claims (Jul-Sep)": int(wl["test_claims"].sum()), "Of which under Rs 2,000": int(wl["test_claims_under_rs_2000"].sum()),
                     "Expected frauds": f"{wl['expected_frauds'].sum():.1f}", "Claims on the review list": int(wl["claims_on_review_rule_list"].sum())}]), "",
           f"Private files written (git-ignored): `reports/raw/watchlist.csv` ({len(wl)} outlets) and `reports/raw/review_list.csv` "
           f"({len(review)} rows: for each month the top 40 claims by score and every claim on the review-rule list, with a flag for each; "
           f"the review-rule list itself is {int(review['on_review_rule_list'].sum())} claims: "
           + ", ".join(f"{MONTH_NAMES[m]} {int(review[(review['month'] == m)]['on_review_rule_list'].sum())}" for m in MONTHS) + ").", ""]
    text = "\n".join(out)
    checked = write_masked_file("rupees.md", text)
    print(text)
    print(f"\nsaved: reports/rupees.md (partner IDs masked) and reports/raw/rupees.md; partner mask check passed on {checked} reports")
    print("saved: reports/timing.txt (the only report that varies between runs)")
    print("private: reports/raw/watchlist.csv, reports/raw/review_list.csv")


if __name__ == "__main__":
    main()
