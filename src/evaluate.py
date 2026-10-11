"""Evaluation harness: folds, metrics, review-list value, partner bootstrap. Nothing model-specific.

A "scorer" is any function fold -> array of scores, one per row of fold.valid, higher = more
likely fraud. evaluate(df, scorer) runs it on every fold and returns the numbers;
print_results shows them.

Money (policy section 4, CLAUDE.md rule 11):
  expected value of reviewing a claim = p*amount - (1-p)*380 - 260
  realised value of a review list     = fraud rupees caught - 380 * genuine claims held - 260 * claims reviewed
The Rs 260 service contact is assumed to apply to every reviewed claim.

Three review lists are reported everywhere (policy: AT MOST 40 reviews a month, not exactly 40):
  fixed 40, ranked by score            always 40 a month
  fixed 40, ranked by expected value   always 40 a month
  stop-early                           expected-value order, stops when EV <= 0, capped at 40
Stop-early only works if p is a calibrated probability. With 36 validation frauds calibration
cannot be checked tightly, so treat its numbers as dependent on calibration.

A fourth list, the PARTNER RULE, is reported when the caller says which claims are eligible:
  partner rule                         only claims whose partner has partner_rate_ratio > 1
                                       (confirmed new-regime fraud above the average rate),
                                       ranked by score, capped at 40. Never padded: if no claim
                                       qualifies the list is empty.
It needs no calibration, but the rule was defined after seeing June and is not independently
validated (one month, 8 partners).

Limits to keep in mind when reading the output:
  * Validation uses decided claims only. Undecided claims (data/undecided.parquet) would also
    compete for the 40 monthly review slots in real life.
  * Pooled PR-AUC, ROC-AUC and top-100 mix scores from two different fits (fold A and fold B).
    The monthly top-40 numbers rank within a month, so they are not affected.

Run from the repo root for the self-test with dummy scores:  python src/evaluate.py
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from clean import DATA, TIME_COL

REVIEWS_PER_MONTH = 40
GOODWILL_RS = 380   # genuine claim held for review
CONTACT_RS = 260    # service contact, assumed for every reviewed claim
REGIME_START = pd.Timestamp("2026-05-01")  # claims under Rs 2,000 auto-approved from this date
BOOTSTRAP_SEED = 0   # every bootstrap and every tie-break starts from this seed, so intervals reproduce exactly between runs
N_BOOT = 1000

# name: (label, train before, validate from, validate before)
FOLD_SPECS = {
    "fold_A": ("Fold A: train < 2026-05-01, validate May 2026 - STRESS TEST of the regime change "
               "(the new pattern starts on 1 May, so training holds no new-regime history)", "2026-05-01", "2026-05-01", "2026-06-01"),
    "fold_B": ("Fold B: train < 2026-06-01, validate June 2026 - PRIMARY fold for model selection",
               "2026-06-01", "2026-06-01", "2026-07-01"),
    "old_design": ("Old design: train < 2026-05-01, validate May-June 2026 "
                   "(what a model that has not seen the new regime scores)", "2026-05-01", "2026-05-01", "2026-07-01"),
}
VALUE_KEYS = ["n_reviewed", "frauds_found", "fraud_rs_caught", "net_value_rs", "rs_per_claim_reviewed"]
# results key, interval key, bootstrap mode, heading
LISTS = [
    ("top40_by_p", "ci_by_p", "p", "FIXED 40 per month, ranked by score"),
    ("top40_by_ev", "ci_by_ev", "ev", "FIXED 40 per month, ranked by expected value"),
    ("upto40_stop_early", "ci_stop_early", "stop_early",
     "STOP-EARLY: up to 40 per month in expected-value order, stopping when EV <= 0 "
     "[DEPENDS ON CALIBRATION: 36 validation frauds is a thin base for trusting p]"),
    ("upto40_partner_rule", "ci_partner_rule", "rule",
     "PARTNER RULE: only claims whose partner has partner_rate_ratio > 1, ranked by score, up to 40 per month, never padded "
     "[RULE DEFINED AFTER SEEING JUNE; NOT INDEPENDENTLY VALIDATED; ONE MONTH, 8 PARTNERS]"),
]
PARTNER_RULE_NOTE = "rule defined after seeing June; not independently validated; one month, 8 partners"


@dataclass
class Fold:
    name: str
    label: str
    train: pd.DataFrame | None   # None for the pooled result, which is not fitted
    valid: pd.DataFrame
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    parts: tuple = ()            # pooled only: the folds whose validation scores are concatenated


def get_folds(df):
    """Fold A, Fold B, pooled (A + B validation rows) and the old design, split on TIME_COL.

    df must hold one row per claim_id, which keeps every re-submission of a claim on one side.
    """
    if not df["claim_id"].is_unique:
        raise ValueError("get_folds needs one row per claim_id (run src/clean.py)")
    folds = {}
    for name, (label, train_end, start, end) in FOLD_SPECS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        train = df[df[TIME_COL] < pd.Timestamp(train_end)]
        valid = df[(df[TIME_COL] >= start) & (df[TIME_COL] < end)]
        assert not set(train["claim_id"]) & set(valid["claim_id"]), "claim on both sides of a split"
        assert train[TIME_COL].max() < valid[TIME_COL].min(), "training data is not strictly earlier"
        folds[name] = Fold(name, label, train, valid, start, end)
    a, b = folds["fold_A"], folds["fold_B"]
    pooled = Fold("pooled", "Pooled: fold A's May scores + fold B's June scores", None,
                  pd.concat([a.valid, b.valid]), a.valid_start, b.valid_end, parts=("fold_A", "fold_B"))
    return {"fold_A": a, "fold_B": b, "pooled": pooled, "old_design": folds["old_design"]}


def month_of(df):
    return df[TIME_COL].dt.to_period("M").astype(str).to_numpy()


def expected_value(p, amount):
    p, amount = np.asarray(p, float), np.asarray(amount, float)
    return p * amount - (1 - p) * GOODWILL_RS - CONTACT_RS


def _top_k_per_month(value, month_codes, k, rng, floor=None):
    """Boolean mask of the k highest values in each month. Ties are broken at random by rng.

    With floor set, only values above it are kept, so a month can get fewer than k (or none).
    """
    n = len(value)
    order = np.lexsort((rng.random(n), -value, month_codes))
    sorted_codes = month_codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    rank = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    selected = np.zeros(n, dtype=bool)
    selected[order[rank < k]] = True
    if floor is not None:
        selected &= value > floor
    return selected


def select_top_by_p(p, month, k=REVIEWS_PER_MONTH, seed=0):
    """Review list: the k claims with the highest score in each month."""
    return _top_k_per_month(np.asarray(p, float), pd.factorize(month)[0], k, np.random.default_rng(seed))


def select_top_by_ev(p, amount, month, k=REVIEWS_PER_MONTH, seed=0):
    """Review list: the k claims with the highest expected value in each month. p must be a probability."""
    return _top_k_per_month(expected_value(p, amount), pd.factorize(month)[0], k, np.random.default_rng(seed))


def select_stop_early(p, amount, month, k=REVIEWS_PER_MONTH, seed=0):
    """Review list: at most k claims a month, highest expected value first, stopping when EV <= 0.

    p must be a calibrated probability: the cut-off is where p*amount stops covering the cost.
    """
    return _top_k_per_month(expected_value(p, amount), pd.factorize(month)[0], k, np.random.default_rng(seed), floor=0.0)


def select_partner_rule(score, month, eligible, k=REVIEWS_PER_MONTH, seed=0):
    """Review list: only eligible claims (partner_rate_ratio > 1), highest score first, at most k a month.

    Never padded: a month with no eligible claim gets an empty list.
    """
    value = np.where(np.asarray(eligible, bool), np.asarray(score, float), -np.inf)
    return _top_k_per_month(value, pd.factorize(month)[0], k, np.random.default_rng(seed), floor=-np.inf)


def tie_slots(value, month, selected):
    """How many places on the list were decided by the random tie-break rather than by the score."""
    d = pd.DataFrame({"v": np.asarray(value, float), "m": np.asarray(month), "s": np.asarray(selected)})
    slots = 0
    for _, g in d.groupby("m"):
        if not g["s"].any():
            continue
        cutoff = g.loc[g["s"], "v"].min()
        if (~g["s"] & (g["v"] == cutoff)).any():
            slots += int((g["s"] & (g["v"] == cutoff)).sum())
    return slots


def review_list_value(y_true, amount, selected):
    y, amount, selected = np.asarray(y_true), np.asarray(amount, float), np.asarray(selected)
    n = int(selected.sum())
    frauds = int(y[selected].sum())
    fraud_rs = float(amount[selected & (y == 1)].sum())
    net = fraud_rs - (n - frauds) * GOODWILL_RS - n * CONTACT_RS
    return {"n_reviewed": n, "frauds_found": frauds, "genuine_held": n - frauds, "fraud_rs_caught": fraud_rs,
            "net_value_rs": net, "rs_per_claim_reviewed": net / n if n else float("nan")}


def metrics(y_true, score, amount, month, k=REVIEWS_PER_MONTH, score_is_probability=True, seed=0, eligible=None):
    """All headline numbers for one set of scores.

    eligible (optional) marks the claims the partner rule may review; without it that list is None.

    'top40_by_p' ranks each month by the score alone. 'top40_by_ev' (fixed 40 by expected value) and
    'upto40_stop_early' (stop when EV <= 0, capped at k) are None when the score is not a probability.
    """
    y, score, amount, month = np.asarray(y_true), np.asarray(score, float), np.asarray(amount, float), np.asarray(month)
    both_classes = 0 < y.sum() < len(y)
    out = {
        "n_claims": len(y), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "accuracy_all_zero": float(1 - y.mean()),
        "pr_auc": float(average_precision_score(y, score)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if both_classes else float("nan"),
    }
    top100 = _top_k_per_month(score, np.zeros(len(y), dtype=int), 100, np.random.default_rng(seed))
    out["top100_overall_frauds"] = int(y[top100].sum())

    def one_list(selected, rank_value):
        v = review_list_value(y, amount, selected)
        v["tie_slots"] = tie_slots(rank_value, month, selected)
        v["frauds_found_by_month"] = pd.Series(y[selected]).groupby(month[selected]).sum().to_dict()
        return v

    out["top40_by_p"] = one_list(select_top_by_p(score, month, k, seed), score)
    out["top40_by_ev"] = out["upto40_stop_early"] = None
    if score_is_probability:
        ev = expected_value(score, amount)
        selected = select_top_by_ev(score, amount, month, k, seed)
        out["top40_by_ev"] = one_list(selected, ev)
        out["top40_by_ev"]["n_positive_ev"] = int((ev[selected] > 0).sum())
        out["upto40_stop_early"] = one_list(select_stop_early(score, amount, month, k, seed), ev)
    out["upto40_partner_rule"] = None
    if eligible is not None:
        eligible = np.asarray(eligible, bool)
        out["upto40_partner_rule"] = one_list(select_partner_rule(score, month, eligible, k, seed), np.where(eligible, score, -np.inf))
        out["upto40_partner_rule"]["n_eligible"] = int(eligible.sum())
    return out


def fast_pr_roc(y, score):
    """Average precision and ROC-AUC, same definitions as scikit-learn, without its per-call overhead."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(-score, kind="stable")
    s, ys = score[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]   # last row of each block of equal scores
    tp = np.cumsum(ys)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    pr_auc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    tp_before = np.r_[0, tp[:-1]]
    fp, fp_before = (last + 1) - tp, np.r_[0, ((last + 1) - tp)[:-1]]
    roc_auc = float(np.sum((fp - fp_before) * (tp + tp_before) / 2) / (n_pos * n_neg))
    return pr_auc, roc_auc


def _partner_rows(partner, n):
    return list(pd.Series(np.arange(n)).groupby(np.asarray(partner)).apply(lambda s: s.to_numpy()))


def bootstrap_ranking(y_true, score, partner, n_boot=N_BOOT, level=0.90, seed=BOOTSTRAP_SEED, return_draws=False):
    """Partner-resampled interval for PR-AUC, ROC-AUC and frauds in the top 100 overall.

    With the same seed and the same claims, two scores are evaluated on the SAME resamples, so their
    draws can be compared pairwise (return_draws=True returns them).
    """
    y, score = np.asarray(y_true), np.asarray(score, float)
    rows = _partner_rows(partner, len(y))
    rng = np.random.default_rng(seed)
    draws = {key: np.full(n_boot, np.nan) for key in ["pr_auc", "roc_auc", "top100_overall_frauds"]}
    for b in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        draws["pr_auc"][b], draws["roc_auc"][b] = fast_pr_roc(y[idx], score[idx])
        top = _top_k_per_month(score[idx], np.zeros(len(idx), dtype=int), 100, rng)
        draws["top100_overall_frauds"][b] = y[idx][top].sum()
    if return_draws:
        return draws
    tail = (1 - level) / 2 * 100
    return {key: tuple(np.nanpercentile(vals, [tail, 100 - tail])) for key, vals in draws.items()}


def bootstrap_top40(y_true, score, amount, month, partner, by="p", k=REVIEWS_PER_MONTH, n_boot=N_BOOT, level=0.90,
                    seed=BOOTSTRAP_SEED, return_draws=False, eligible=None):
    """Interval for the monthly top-k numbers, resampling PARTNERS with replacement (not claims).

    Fraud clusters in a few partners, so claims are not independent; drawing whole partners keeps
    that structure. The list is re-selected inside every resample. With the same seed and the same
    claims, two scores are evaluated on the SAME resamples, so their draws can be compared pairwise
    (return_draws=True returns them).
    """
    y, score, amount = np.asarray(y_true), np.asarray(score, float), np.asarray(amount, float)
    codes = pd.factorize(np.asarray(month))[0]
    rank_value = score if by in ("p", "rule") else expected_value(score, amount)
    floor = 0.0 if by == "stop_early" else None
    if by == "rule":
        rank_value, floor = np.where(np.asarray(eligible, bool), score, -np.inf), -np.inf
    rows = _partner_rows(partner, len(y))
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(n_boot) for key in VALUE_KEYS}
    for b in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        selected = _top_k_per_month(rank_value[idx], codes[idx], k, rng, floor)
        v = review_list_value(y[idx], amount[idx], selected)
        for key in VALUE_KEYS:
            draws[key][b] = v[key]
    if return_draws:
        return draws
    tail = (1 - level) / 2 * 100
    # Rs per claim is undefined in a resample where nothing is reviewed; those resamples are left out of its interval.
    return {key: (tuple(np.nanpercentile(vals, [tail, 100 - tail])) if not np.isnan(vals).all() else (np.nan, np.nan))
            for key, vals in draws.items()}


def evaluate(df, scorer, score_is_probability=True, n_boot=1000, seed=0):
    """Run a scorer on Fold A, Fold B and the old design; pool A and B. Returns {fold name: results}."""
    folds = get_folds(df)
    scores = {name: np.asarray(scorer(folds[name]), float) for name in ["fold_A", "fold_B", "old_design"]}
    scores["pooled"] = np.concatenate([scores[part] for part in folds["pooled"].parts])
    results = {}
    for name, fold in folds.items():
        v = fold.valid
        args = (v["is_fraud"].to_numpy(), scores[name], v["claim_amount_inr"].to_numpy(), month_of(v))
        res = metrics(*args, score_is_probability=score_is_probability, seed=seed)
        for _, ci_key, mode, _ in LISTS:
            res[ci_key] = (bootstrap_top40(*args, v["partner_id"].to_numpy(), by=mode, n_boot=n_boot, seed=seed)
                           if mode == "p" or (score_is_probability and mode != "rule") else None)
        results[name] = res
    return results


# ----------------------------------------------------------------------------------------------
# a simple rule to beat
# ----------------------------------------------------------------------------------------------
def partner_recent_fraud_scorer(window_days=30, lag_days=0):
    """Score = the partner's number of confirmed frauds in the window_days before the validation
    month, counting only claims inside the new regime (from REGIME_START).

    No leakage: it reads fold.train only (claims first submitted before the validation month), and
    labels are frozen at the start of that month. lag_days is the investigation lag L: a claim's
    outcome counts only if first_submitted_at + L days <= start of the validation month.
    """
    def scorer(fold):
        t = fold.train
        window_start = max(fold.valid_start - timedelta(days=window_days), REGIME_START)
        known = t[(t[TIME_COL] >= window_start) & (t[TIME_COL] + timedelta(days=lag_days) <= fold.valid_start)]
        counts = known[known["is_fraud"] == 1].groupby("partner_id").size()
        return fold.valid["partner_id"].map(counts).fillna(0).to_numpy()
    return scorer


# ----------------------------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------------------------
def describe_folds(df):
    print("Folds (time axis = first_submitted_at, decided claims only):")
    for fold in get_folds(df).values():
        v = fold.valid
        fraud_partners = v[v["is_fraud"] == 1]["partner_id"].nunique()
        train = (f"train {len(fold.train)} claims ({int(fold.train['is_fraud'].sum())} fraud)" if fold.train is not None
                 else "no fit of its own")
        print(f"  {fold.label}")
        print(f"      {train}; validate {len(v)} claims, {int(v['is_fraud'].sum())} fraud ({v['is_fraud'].mean():.2%}) "
              f"from {fraud_partners} partners, out of {v['partner_id'].nunique()} partners with claims")
    pooled = get_folds(df)["pooled"].valid
    print(f"\nCAUTION: the validation windows hold very few frauds ({int(pooled['is_fraud'].sum())} in total, from "
          f"{pooled[pooled['is_fraud'] == 1]['partner_id'].nunique()} partners). One partner more or less in a resample moves "
          "every top-40 number a lot, so the 90% intervals below are wide. Read them as rough ranges, not precise estimates.")


def _with_ci(value, ci, fmt):
    if np.isnan(value):
        return "n/a"
    if np.isnan(ci[0]):
        return f"{value:{fmt}}"
    return f"{value:{fmt}} [{ci[0]:{fmt}} to {ci[1]:{fmt}}]"


def print_results(title, results):
    names = {"fold_A": "Fold A (May)", "fold_B": "Fold B (June)", "pooled": "Pooled (May+June)", "old_design": "Old design (May+June)"}
    print(f"\n--- {title} ---")
    head = pd.DataFrame({names[k]: {
        "claims": r["n_claims"], "fraud": r["n_fraud"], "all-zero accuracy": f"{r['accuracy_all_zero']:.2%}",
        "PR-AUC": f"{r['pr_auc']:.3f}", "ROC-AUC": f"{r['roc_auc']:.3f}", "frauds in top 100 overall": r["top100_overall_frauds"],
    } for k, r in results.items()}).T
    print(head.to_string())
    for key, ci_key, _, label in LISTS:
        if next(iter(results.values()))[key] is None:
            why = ("no partner eligibility was supplied (it needs partner_rate_ratio; see src/train.py)" if key == "upto40_partner_rule"
                   else "this score is not a probability")
            print(f"\n{label}: not applicable, {why}")
            continue
        rows = {}
        for k, r in results.items():
            v, ci = r[key], r[ci_key]
            rows[names[k]] = {
                "reviewed": (_with_ci(v["n_reviewed"], ci["n_reviewed"], ".0f") if key != "top40_by_p" and key != "top40_by_ev"
                             else v["n_reviewed"]),
                "frauds found": _with_ci(v["frauds_found"], ci["frauds_found"], ".0f"),
                "fraud Rs caught": _with_ci(v["fraud_rs_caught"], ci["fraud_rs_caught"], ",.0f"),
                "genuine held": v["genuine_held"],
                "net value Rs": _with_ci(v["net_value_rs"], ci["net_value_rs"], ",.0f"),
                "Rs per claim reviewed": _with_ci(v["rs_per_claim_reviewed"], ci["rs_per_claim_reviewed"], ",.0f"),
                "places set by tie-break": v["tie_slots"],
            }
            if "n_positive_ev" in v:
                rows[names[k]]["places with EV > 0"] = v["n_positive_ev"]
        print(f"\n{label}  (value [90% interval, 1,000 partner resamples]):")
        print(pd.DataFrame(rows).T.to_string())


# ----------------------------------------------------------------------------------------------
# self-test with dummy scores
# ----------------------------------------------------------------------------------------------
def main():
    df = pd.read_parquet(DATA / "clean_train.parquet")
    print("Evaluation harness self-test with dummy scores (src/evaluate.py). No model is involved.")
    print(f"Review budget {REVIEWS_PER_MONTH} claims a month; a genuine claim held costs Rs {GOODWILL_RS}; "
          f"every reviewed claim costs Rs {CONTACT_RS} (assumption).\n")
    describe_folds(df)

    folds = get_folds(df)
    print("\nWhat to expect from the dummies:")
    for name in ["fold_A", "fold_B"]:
        v = folds[name].valid
        print(f"  {name}: picking 40 claims blind should find about 40 x {v['is_fraud'].mean():.2%} = "
              f"{REVIEWS_PER_MONTH * v['is_fraud'].mean():.2f} frauds")

    print_results("Dummy 1: constant score 0.02 (every claim tied, so the list is a random draw)",
                  evaluate(df, lambda fold: np.full(len(fold.valid), 0.02)))
    print_results("Dummy 2: random score, uniform 0-1, seed 0",
                  evaluate(df, lambda fold: np.random.default_rng(0).random(len(fold.valid))))

    print("\nHarness check: average over 5,000 different random scores, top 40 per month ranked by score")
    rng = np.random.default_rng(1)
    for name in ["fold_A", "fold_B"]:
        v = folds[name].valid
        y, month = v["is_fraud"].to_numpy(), month_of(v)
        found = np.array([y[select_top_by_p(rng.random(len(v)), month)].sum() for _ in range(5000)])
        print(f"  {name}: mean frauds found {found.mean():.3f} +/- {found.std() / np.sqrt(len(found)):.3f} (standard error), "
              f"range {found.min()} to {found.max()}; expected {REVIEWS_PER_MONTH * y.mean():.3f}")

    print_results("Ceiling, not a model: score = the true label. Fixed 40 = the best a full list can do; "
                  "stop-early = the best achievable (review exactly the true frauds, then stop)",
                  evaluate(df, lambda fold: fold.valid["is_fraud"].to_numpy().astype(float)))
    print_results("Simple rule: partner's confirmed frauds in the previous 30 days inside the new regime, "
                  "labels frozen at the start of the validation month, lag L = 0 days",
                  evaluate(df, partner_recent_fraud_scorer(window_days=30, lag_days=0), score_is_probability=False))
    print("\nNote on the simple rule: for Fold A and the old design the 30 days before 1 May hold no new-regime claims, "
          "so every partner scores 0 and the list is a random draw. Only Fold B can use it.")


if __name__ == "__main__":
    from reporting import masked_report

    with masked_report("evaluate_harness"):
        main()
