"""Feature builder for the Kestrel claim scorer, with its leakage tests.

Reusable entry points (train.py and the app import these):
  build_features(claims, cutoff, L)        rows scored AFTER a cut-off: validation month, test set, live claims
  build_training_features(claims, L)       training rows; each row gets the cut-off at the start of its own month
  fold_matrices(name, L)                   X/y for fold_A, fold_B, old_design, or "final" (train to 30 Jun, score test)
  partner_label_table(history, cutoff, L)  partner-level summary behind the label-based features
  model_columns()                          the columns models may use (partner_rate_ratio replaces the raw smoothed rate)

How each kind of feature avoids leakage
  Label-based (FROZEN, rules A and B): built only from claims first submitted before (cut-off - L days),
    L = investigation lag. Every row scored against a cut-off sees the same frozen partner table.
    build_features refuses rows dated before the cut-off, so a row can never see its own label or a later one.
    Training rows use the cut-off at the start of their own month (or 7-day block), never a later one.
    If fewer than 50 new-regime claims are decided at a (cut-off, L), new-regime history is treated as
    unavailable: counts 0, partner_has_new_history 0, partner_rate_ratio exactly 1.0.
  Label-free rolling (rule C): partner counts and averages over claims first submitted strictly before
    the claim being scored. They need no labels, so they roll forward through train, undecided and test claims.
  Static: facts about the claim, the product and the partner that are known when the claim is submitted.
Not used at all (rule D): n_submissions, serial-repeat features, inspector_note, partner_inspected.
All time ordering uses first_submitted_at (rule E).

Run from the repo root:  python src/features.py   (prints the checks and tests, writes reports/features.txt)
"""
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from clean import DATA, TIME_COL
from evaluate import REGIME_START, get_folds

LAGS = [0, 14, 30, 60]                      # investigation lag L in days; never assume one silently
SMOOTH_K = 5                                # prior weight in partner_new_rate_smoothed
MIN_NEW_DECIDED = 50                        # fewer decided new-regime claims than this = history unavailable
SMALL_CLAIM = 2000
NEW_PARTNER_FROM = pd.Timestamp("2025-10-01")
FINAL_CUTOFF = pd.Timestamp("2026-07-01")   # i.e. the end of 30 Jun 2026, the last day that has labels
CUTOFFS = {"fold_A": pd.Timestamp("2026-05-01"), "fold_B": pd.Timestamp("2026-06-01"),
           "old_design": pd.Timestamp("2026-05-01"), "final": FINAL_CUTOFF}
DAY_NS = 86_400_000_000_000

# Decided by signal_checks() on Fold B's training data (decided claims before 2026-06-01). The check
# re-runs every time and fails loudly if the data no longer supports these choices.
TIME_FEATURES_KEPT = []       # hour, weekday, day of month: no signal, dropped
PHRASE_FEATURES_KEPT = False  # claim_description one-hots: no phrase differs clearly, not added
SIGNAL_P = 0.01               # a candidate is kept only if its chi-square p-value is below this

LABEL_FEATURES = {
    "partner_new_fraud_count": "Confirmed frauds this partner had in the new regime (from 1 May 2026) that were decided before the cut-off.",
    "partner_new_claim_count": "Decided claims this partner had in the new regime before the cut-off.",
    "partner_new_rate_smoothed": "Partner's new-regime fraud rate pulled towards the overall new-regime rate: (frauds + 5 x overall rate) / (claims + 5). A partner with no decided claims gets exactly the overall rate.",
    "partner_rate_ratio": "partner_new_rate_smoothed divided by the overall new-regime rate: above 1 = worse than average, below 1 = better. Exactly 1.0 for a partner with no decided new-regime claim, at every cut-off. Models use this instead of the raw smoothed rate.",
    "partner_fraud_count_old": "Confirmed frauds this partner had before 1 May 2026 (decided before the cut-off). Kept separate so the model can learn it is weak.",
    "partner_has_new_history": "1 if the partner has at least one decided new-regime claim, 0 if the model knows nothing about it in the new regime.",
}
ROLLING_FEATURES = {
    "partner_claims_last_7d": "Claims this partner submitted in the 7 days before this claim.",
    "partner_claims_last_30d": "Claims this partner submitted in the 30 days before this claim.",
    "partner_small_share_last_30d": "Share of those 30-day claims that were under Rs 2,000 (0 if the partner had none; see partner_claims_last_30d).",
    "partner_avg_amount_last_30d": "Average amount of those 30-day claims in rupees (0 if the partner had none).",
}
STATIC_FEATURES = {
    "partner_age_days": "Days between the partner's onboarding and this claim.",
    "is_new_partner": "1 if the partner was onboarded on or after 1 Oct 2025.",
    "customer_prior_claims": "The customer's earlier warranty claims.",
    "photo_attached": "1 if a photo was attached.",
    "claim_amount_inr": "Amount claimed, in rupees.",
    "amount_to_list_price": "Amount claimed divided by the product's list price.",
    "is_small": "1 if the claim is under Rs 2,000 (auto-approved without inspection since 1 May 2026).",
    "warranty_used_share": "days_since_purchase divided by the warranty length in days (warranty_months x 30).",
}
ONE_HOT_GROUPS = {
    "partner_type": "One column per partner type (service centre, franchise, freelance technician).",
    "city": "One column per partner city.",
    "family": "One column per product family.",
}


# ----------------------------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def reference_categories():
    partners = pd.read_csv(DATA / "partners.csv")
    products = pd.read_csv(DATA / "products.csv")
    return {"partner_type": sorted(partners["partner_type"].unique()), "city": sorted(partners["city"].unique()),
            "family": sorted(products["family"].unique())}


def feature_columns():
    cats = reference_categories()
    one_hot = [f"{group}_{value}" for group in ONE_HOT_GROUPS for value in cats[group]]
    return list(LABEL_FEATURES) + list(ROLLING_FEATURES) + list(STATIC_FEATURES) + one_hot


def model_columns():
    """Columns a model may use: everything except the raw smoothed rate, whose no-history value moves between cut-offs."""
    return [c for c in feature_columns() if c != "partner_new_rate_smoothed"]


@lru_cache(maxsize=None)
def load_history():
    """Decided claims: the only source of labels."""
    return pd.read_parquet(DATA / "clean_train.parquet")


@lru_cache(maxsize=None)
def load_events():
    """Every claim we know of (decided, undecided, test). Used for label-free counts only."""
    cols = ["claim_id", "partner_id", TIME_COL, "claim_amount_inr"]
    events = pd.concat([pd.read_parquet(DATA / f)[cols] for f in
                        ["clean_train.parquet", "undecided.parquet", "clean_test.parquet"]], ignore_index=True)
    assert events["claim_id"].is_unique
    return events


NEW_OUTLET_DEFINITION = ("new outlets = outlets onboarded on or after 1 Oct 2025 with at least one decided claim first submitted "
                         "on or after 1 May 2026")


def new_outlet_counts(history=None):
    """THE definition of 'new outlet' counts, used by every report, the form text, the README and the explanation line.

    Counted on decided claims first submitted from 1 May 2026 (the new regime) up to the end of the history given.
    """
    history = load_history() if history is None else history
    new_regime = history[history[TIME_COL] >= REGIME_START]
    is_new = new_regime["onboarded_date"] >= NEW_PARTNER_FROM
    frauds_by_outlet = new_regime[is_new].groupby("partner_id")["is_fraud"].sum()
    established = new_regime[~is_new].groupby("partner_id")["is_fraud"].sum()
    before = history[history[TIME_COL] < REGIME_START]
    return {
        "definition": NEW_OUTLET_DEFINITION,
        "new_outlets": int(len(frauds_by_outlet)), "new_outlets_with_fraud": int((frauds_by_outlet > 0).sum()),
        "new_outlets_without_fraud": int((frauds_by_outlet == 0).sum()),
        "new_outlet_claims": int(is_new.sum()), "new_outlet_frauds": int(new_regime[is_new]["is_fraud"].sum()),
        "new_outlet_fraud_rate": float(new_regime[is_new]["is_fraud"].mean()),
        "established_outlets": int(len(established)), "established_outlets_with_fraud": int((established > 0).sum()),
        "established_outlet_claims": int((~is_new).sum()), "established_outlet_frauds": int(new_regime[~is_new]["is_fraud"].sum()),
        "established_outlet_fraud_rate": float(new_regime[~is_new]["is_fraud"].mean()),
        "before_may_frauds": int(before["is_fraud"].sum()),
        "before_may_established_frauds": int(before[before["onboarded_date"] < NEW_PARTNER_FROM]["is_fraud"].sum()),
        "period": "decided claims first submitted 1 May to 30 Jun 2026",
    }


def _ns(series):
    return series.to_numpy("datetime64[ns]").astype("int64")


# ----------------------------------------------------------------------------------------------
# label-based features (frozen)
# ----------------------------------------------------------------------------------------------
def decided_rows(history, cutoff, L):
    """Claims decided at this cut-off: first submitted before (cutoff - L days). Returns (new-regime, old-regime)."""
    known_before = pd.Timestamp(cutoff) - timedelta(days=L)
    known = history[history[TIME_COL] < known_before]
    return known[known[TIME_COL] >= REGIME_START], known[known[TIME_COL] < REGIME_START]


def label_rows_used(history, cutoff, L):
    """The claims whose labels are actually used at this cut-off. Returns (new-regime rows, old-regime rows).

    With fewer than MIN_NEW_DECIDED decided new-regime claims the new-regime rows are not used at all:
    a partner rate built on a handful of claims is noise.
    """
    new, old = decided_rows(history, cutoff, L)
    return (new if len(new) >= MIN_NEW_DECIDED else new.iloc[0:0]), old


def partner_label_table(history, cutoff, L, k=SMOOTH_K):
    """Partner-level summary frozen at the cut-off, and the overall rate used for smoothing.

    Overall rate = new-regime fraud rate among decided claims. When new-regime history is unavailable
    (fewer than MIN_NEW_DECIDED decided claims) it falls back to the old-regime rate, and to 0 when
    nothing at all is decided.
    """
    new, old = label_rows_used(history, cutoff, L)
    table = pd.DataFrame({
        "partner_new_fraud_count": new.groupby("partner_id")["is_fraud"].sum(),
        "partner_new_claim_count": new.groupby("partner_id").size(),
        "partner_fraud_count_old": old.groupby("partner_id")["is_fraud"].sum(),
    }).fillna(0.0).astype(float)
    overall = float(new["is_fraud"].mean()) if len(new) else float(old["is_fraud"].mean()) if len(old) else 0.0
    return table, overall


def _label_features(claims, table, overall, k):
    t = table.reindex(claims["partner_id"].to_numpy()).fillna(0.0).astype(float)  # unknown partner -> zeros, never NaN
    t.index = claims.index
    frauds, n = t["partner_new_fraud_count"], t["partner_new_claim_count"]
    t["partner_new_rate_smoothed"] = (frauds + k * overall) / (n + k)
    # Exactly 1.0 without history, and no division by zero when the overall rate is 0.
    has = (n > 0) & (overall > 0)
    t["partner_rate_ratio"] = np.where(has, t["partner_new_rate_smoothed"] / (overall if overall > 0 else 1.0), 1.0)
    t["partner_has_new_history"] = (n > 0).astype(float)
    return t[list(LABEL_FEATURES)]


# ----------------------------------------------------------------------------------------------
# label-free features
# ----------------------------------------------------------------------------------------------
def rolling_partner_features(claims, events):
    """Counts over the partner's claims first submitted strictly before each claim (the claim itself excluded)."""
    ev = events[events["partner_id"].isin(claims["partner_id"].unique())].sort_values(TIME_COL)
    by_partner = {}
    for pid, g in ev.groupby("partner_id"):
        amount = g["claim_amount_inr"].to_numpy(float)
        by_partner[pid] = (_ns(g[TIME_COL]), np.r_[0, np.cumsum(amount < SMALL_CLAIM)], np.r_[0.0, np.cumsum(amount)])

    t_all = _ns(claims[TIME_COL])
    n7, n30, small, avg = (np.zeros(len(claims)) for _ in range(4))
    for pid, pos in claims.groupby("partner_id").indices.items():
        if pid not in by_partner:
            continue
        times, small_cum, amount_cum = by_partner[pid]
        t = t_all[pos]
        end = np.searchsorted(times, t, side="left")              # events strictly before t
        start7 = np.searchsorted(times, t - 7 * DAY_NS, side="left")
        start30 = np.searchsorted(times, t - 30 * DAY_NS, side="left")
        count30 = end - start30
        n7[pos], n30[pos] = end - start7, count30
        small[pos] = (small_cum[end] - small_cum[start30]) / np.maximum(count30, 1)
        avg[pos] = (amount_cum[end] - amount_cum[start30]) / np.maximum(count30, 1)
    return pd.DataFrame({"partner_claims_last_7d": n7, "partner_claims_last_30d": n30,
                         "partner_small_share_last_30d": small, "partner_avg_amount_last_30d": avg}, index=claims.index)


def static_features(claims):
    out = pd.DataFrame(index=claims.index)
    out["partner_age_days"] = (claims[TIME_COL] - claims["onboarded_date"]).dt.days.astype(float)
    out["is_new_partner"] = (claims["onboarded_date"] >= NEW_PARTNER_FROM).astype(float)
    out["customer_prior_claims"] = claims["customer_prior_claims"].astype(float)
    out["photo_attached"] = (claims["photo_attached"] == "Y").astype(float)
    out["claim_amount_inr"] = claims["claim_amount_inr"].astype(float)
    out["amount_to_list_price"] = claims["claim_amount_inr"] / claims["list_price_inr"]
    out["is_small"] = (claims["claim_amount_inr"] < SMALL_CLAIM).astype(float)
    out["warranty_used_share"] = claims["days_since_purchase"] / (claims["warranty_months"] * 30)
    for group, values in reference_categories().items():
        for value in values:
            out[f"{group}_{value}"] = (claims[group] == value).astype(float)
    return out


def _assemble(claims, label, rolling, static):
    X = pd.concat([label, rolling, static], axis=1)[feature_columns()]
    X.index = pd.Index(claims["claim_id"].to_numpy(), name="claim_id")
    return X


# ----------------------------------------------------------------------------------------------
# public builders
# ----------------------------------------------------------------------------------------------
def build_features(claims, cutoff, L, history=None, events=None, k=SMOOTH_K):
    """Features for claims scored after a cut-off (a validation month, the test set, a live claim).

    claims   cleaned claim rows, all first submitted on or after the cut-off
    cutoff   date at which label-based history is frozen
    L        investigation lag in days: only claims submitted before (cutoff - L) count as decided
    history  decided claims with is_fraud (default: data/clean_train.parquet). Later rows are ignored.
    events   every known claim, for the label-free counts (default: train + undecided + test)
    Returns a DataFrame indexed by claim_id with feature_columns(), all numeric, no NaN.
    """
    cutoff = pd.Timestamp(cutoff)
    if (claims[TIME_COL] < cutoff).any():
        raise ValueError("build_features got claims dated before the cut-off: they would see their own or later labels. "
                         "Use build_training_features for training rows.")
    history = load_history() if history is None else history
    events = load_events() if events is None else events
    table, overall = partner_label_table(history, cutoff, L, k)
    return _assemble(claims, _label_features(claims, table, overall, k), rolling_partner_features(claims, events),
                     static_features(claims))


def snapshot_cutoffs(times, snapshot="month"):
    """Cut-off each training row uses: the start of its month, or of its N-day block counted from 1 May 2026."""
    if snapshot == "month":
        return times.dt.to_period("M").dt.start_time
    step = timedelta(days=int(snapshot))
    return REGIME_START + ((times - REGIME_START) // step) * step


def _training_label_features(claims, L, snapshot, history, k):
    parts = []
    for cutoff, index in claims.groupby(snapshot_cutoffs(claims[TIME_COL], snapshot)).groups.items():
        table, overall = partner_label_table(history, cutoff, L, k)
        parts.append(_label_features(claims.loc[index], table, overall, k))
    return pd.concat(parts).loc[claims.index]


def build_training_features(claims, L, snapshot="month", history=None, events=None, k=SMOOTH_K):
    """Features for training rows. Each row is treated exactly like a validation row of its own month:
    label-based history is frozen at the start of that month (minus L days), never at a later date.

    snapshot="month" is the default. An integer N freezes every N days instead (counted from 1 May 2026);
    it lets training rows see fresher new-regime history than a validation or test row ever would.
    """
    history = load_history() if history is None else history
    events = load_events() if events is None else events
    return _assemble(claims, _training_label_features(claims, L, snapshot, history, k),
                     rolling_partner_features(claims, events), static_features(claims))


def fold_matrices(name, L, snapshot="month"):
    """X_train, y_train, X_score, y_score (None for the test set) and the claim frames, for one fold and lag."""
    history = load_history()
    if name == "final":
        train, score = history[history[TIME_COL] < FINAL_CUTOFF], pd.read_parquet(DATA / "clean_test.parquet")
    else:
        fold = get_folds(history)[name]
        train, score = fold.train, fold.valid
    return {"X_train": build_training_features(train, L, snapshot), "y_train": train["is_fraud"].to_numpy(),
            "X_score": build_features(score, CUTOFFS[name], L),
            "y_score": score["is_fraud"].to_numpy() if "is_fraud" in score.columns else None,
            "train": train, "score": score}


# ----------------------------------------------------------------------------------------------
# report: checks behind the feature choices
# ----------------------------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _chi_square_p(df, column):
    table = pd.crosstab(df[column], df["is_fraud"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    return float(chi2_contingency(table)[1])


def signal_checks():
    """Time-of-day and claim_description checks, on Fold B's training data only (decided claims before 1 Jun 2026)."""
    section("Signal checks for optional features (Fold B training data only: decided claims before 2026-06-01)")
    h = load_history()
    d = h[h[TIME_COL] < CUTOFFS["fold_B"]].copy()
    d["hour (4 blocks of 6h)"] = d[TIME_COL].dt.hour // 6
    d["weekday"] = d[TIME_COL].dt.weekday
    d["day of month (3 blocks)"] = pd.cut(d[TIME_COL].dt.day, [0, 10, 20, 31], labels=False)
    common = d["claim_description"].value_counts()
    common = set(common[common >= 100].index)
    d["claim_description phrase"] = d["claim_description"].where(d["claim_description"].isin(common), "other")
    regimes = {"before 1 May 2026": d[d[TIME_COL] < REGIME_START], "May 2026 (new regime)": d[d[TIME_COL] >= REGIME_START]}

    rows = {}
    for column in ["hour (4 blocks of 6h)", "weekday", "day of month (3 blocks)", "claim_description phrase"]:
        row = {}
        for name, g in regimes.items():
            rate = g.groupby(column)["is_fraud"].mean()
            row[f"{name}: p-value"] = round(_chi_square_p(g, column), 3)
            row[f"{name}: lowest-highest rate"] = f"{rate.min():.2%} - {rate.max():.2%}"
        rows[column] = row
    table = pd.DataFrame(rows).T
    print(f"frauds available: {int(regimes['before 1 May 2026']['is_fraud'].sum())} before 1 May, "
          f"{int(regimes['May 2026 (new regime)']['is_fraud'].sum())} in May. Chi-square p-value of fraud rate against each candidate "
          f"(small = the rate really differs across its values; kept only below {SIGNAL_P}):")
    print(table.to_string())
    print(f"claim_description: {len(common)} stock phrases cover {d['claim_description'].isin(common).mean():.2%} of claims; "
          f"{int((~d['claim_description'].isin(common)).sum())} claims have some other text.")

    p_cols = [c for c in table.columns if c.endswith("p-value")]
    time_signal = [c for c in table.index[:3] if (table.loc[c, p_cols].astype(float) < SIGNAL_P).any()]
    phrase_signal = bool((table.loc["claim_description phrase", p_cols].astype(float) < SIGNAL_P).any())
    assert time_signal == TIME_FEATURES_KEPT and phrase_signal == PHRASE_FEATURES_KEPT, (
        "the data no longer supports TIME_FEATURES_KEPT / PHRASE_FEATURES_KEPT; revisit them")
    print("DECISION: hour, weekday and day of month are dropped, and no claim_description features are added: no p-value "
          f"is below {SIGNAL_P} in either period. With 14 frauds in May the test has little power, so this says 'no clear "
          "difference', not 'proven irrelevant'.")

    crm, zoho = h[h["source"] == "crm"], h[h["source"] == "legacy_zoho"]
    print(f"Policy section 9 (UTC): submitted_at is not a Zoho resolution event, so the exception does not apply. Hour-of-day "
          f"is flat in both systems (share of claims between 00:00 and 05:59: Zoho {(zoho['submitted_at'].dt.hour < 6).mean():.1%}, "
          f"CRM {(crm['submitted_at'].dt.hour < 6).mean():.1%}), so a 5.5-hour shift could not be seen and hour is not a feature anyway.")


def history_report():
    section("Rule B: how much new-regime history exists at each cut-off, by investigation lag L")
    history = load_history()
    folds = get_folds(history)
    test = pd.read_parquet(DATA / "clean_test.parquet")
    scored = {"fold_A": folds["fold_A"].valid, "fold_B": folds["fold_B"].valid, "final": test}
    names = {"fold_A": "Fold A (score May)", "fold_B": "Fold B (score June)", "final": "Final (score Jul-Sep test)"}
    rows, cold = [], []
    for name, claims in scored.items():
        for L in LAGS:
            new, old = decided_rows(history, CUTOFFS[name], L)
            table, overall = partner_label_table(history, CUTOFFS[name], L)
            with_history = set(table.index[table["partner_new_claim_count"] > 0])
            no_hist = ~claims["partner_id"].isin(with_history)
            rows.append({"cut-off": names[name], "frozen at": str(CUTOFFS[name].date()), "L": L,
                         "decided new-regime claims": len(new), "of which fraud": int(new["is_fraud"].sum()),
                         "history used (>= 50 decided)": "yes" if len(new) >= MIN_NEW_DECIDED else "NO",
                         "partners with new history": len(with_history),
                         "partners with new-regime fraud": int((table["partner_new_fraud_count"] > 0).sum()),
                         "overall rate used for smoothing": f"{overall:.4f}"})
            cold.append({"cut-off": names[name], "L": L, "claims scored": len(claims),
                         "from partners with NO decided new-regime claim": int(no_hist.sum()),
                         "share": f"{no_hist.mean():.1%}", "such partners": claims[no_hist]["partner_id"].nunique()})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNote: the 'final' cut-off is written 2026-07-01, meaning the end of 30 Jun 2026.")
    print("Note: 'overall rate used for smoothing' is also the value a partner with no history gets in partner_new_rate_smoothed. "
          "It falls back to the old-regime rate when new-regime history is unavailable, so that value differs between cut-offs. "
          "Models therefore use partner_rate_ratio, which is exactly 1.0 for a no-history partner at every cut-off.")
    print(f"Note: with fewer than {MIN_NEW_DECIDED} decided new-regime claims the history is NOT used: counts 0, "
          "partner_has_new_history 0, partner_rate_ratio 1.0 for every partner.")

    section("Cold start: claims scored for partners with no decided new-regime claim")
    print(pd.DataFrame(cold).to_string(index=False))
    print("These claims get partner_new_fraud_count = 0, partner_new_claim_count = 0, partner_has_new_history = 0, "
          "partner_rate_ratio = 1.0 and partner_new_rate_smoothed = the overall rate above. Never NaN (leakage test 4).")


def training_coverage_report():
    section("What training rows can teach about new-regime history (rows with partner_has_new_history = 1)")
    history = load_history()
    folds = get_folds(history)
    trains = {"Fold A (train < 1 May)": folds["fold_A"].train, "Fold B (train < 1 Jun) PRIMARY": folds["fold_B"].train,
              "Final (train to 30 Jun)": history[history[TIME_COL] < FINAL_CUTOFF]}
    for snapshot, label in [("month", "Month-start snapshots (the default used by build_training_features)"),
                            (7, "7-day snapshots (available as snapshot=7, NOT used unless chosen)")]:
        rows = []
        for name, train in trains.items():
            for L in LAGS:
                lab = _training_label_features(train, L, snapshot, history, SMOOTH_K)
                has = lab["partner_has_new_history"] == 1
                prior = lab["partner_new_fraud_count"] > 0
                rows.append({"training set": name, "L": L, "rows": len(train), "frauds": int(train["is_fraud"].sum()),
                             "rows with new history": int(has.sum()),
                             "frauds among them": int(train.loc[has[has].index, "is_fraud"].sum()),
                             "rows whose partner had earlier new-regime fraud": int(prior.sum()),
                             "frauds among those": int(train.loc[prior[prior].index, "is_fraud"].sum())})
        print(f"\n{label}:")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nRead this before train.py: with month-start snapshots a model trained before 1 June never sees a training row "
          "with new-regime history (the new regime only starts on 1 May), so in Fold B it cannot LEARN how much weight the "
          "label-based features deserve; only the final model can, from its June rows. 7-day snapshots give Fold B some such "
          "rows but make training history fresher than anything a validation or test row gets.")


# ----------------------------------------------------------------------------------------------
# leakage tests (all must pass)
# ----------------------------------------------------------------------------------------------
def leakage_test_1(raw_only):
    section("LEAKAGE TEST 1: which labels feed the label-based features of 5 Fold B validation claims")
    history = load_history()
    fold = get_folds(history)["fold_B"]
    cutoff = CUTOFFS["fold_B"]
    table0, _ = partner_label_table(history, cutoff, 0)
    with_fraud = fold.valid["partner_id"].isin(table0.index[table0["partner_new_fraud_count"] > 0])
    sample = pd.concat([fold.valid[with_fraud].sample(2, random_state=0), fold.valid[~with_fraud].sample(3, random_state=0)])
    print("Sample: 2 random claims from partners that had new-regime fraud in May and 3 random claims from the other "
          "partners (5 plain random claims would almost all show empty fraud history). Fold B cut-off = 2026-06-01.")
    show = ["claim_id", TIME_COL, "partner_id", "claim_amount_inr", "is_fraud"]
    checked = 0
    for L in LAGS:
        known_before = cutoff - timedelta(days=L)
        X = build_features(sample, cutoff, L)
        new, old = label_rows_used(history, cutoff, L)
        _, overall = partner_label_table(history, cutoff, L)
        raw_only(f"\n--- L = {L} days: labels count only for claims first submitted before {known_before} ---")
        for _, claim in sample.iterrows():
            feats = X.loc[claim["claim_id"], list(LABEL_FEATURES)]
            used_new = new[new["partner_id"] == claim["partner_id"]]
            used_old = old[old["partner_id"] == claim["partner_id"]]
            used = pd.concat([used_new, used_old])
            # every label used is older than (cut-off - L), and the claim's own row is not among them
            assert (used[TIME_COL] < known_before).all()
            assert claim["claim_id"] not in set(used["claim_id"])
            assert (used[TIME_COL] < claim[TIME_COL]).all()
            # the features equal a recount straight from those rows
            assert feats["partner_new_fraud_count"] == used_new["is_fraud"].sum()
            assert feats["partner_new_claim_count"] == len(used_new)
            assert feats["partner_fraud_count_old"] == used_old["is_fraud"].sum()
            assert feats["partner_has_new_history"] == float(len(used_new) > 0)
            if len(used_new) == 0:
                assert feats["partner_rate_ratio"] == 1.0
            else:
                recount = (used_new["is_fraud"].sum() + SMOOTH_K * overall) / (len(used_new) + SMOOTH_K) / overall
                assert abs(feats["partner_rate_ratio"] - recount) < 1e-12
            checked += 1
            raw_only(f"\nclaim {claim['claim_id']}  partner {claim['partner_id']}  first submitted {claim[TIME_COL]}  "
                     f"own label {claim['is_fraud']} (not used)")
            raw_only("  features: " + ", ".join(f"{k}={v:.4f}" if "rate" in k else f"{k}={v:.0f}" for k, v in feats.items()))
            if len(new) == 0 and L > 0:
                raw_only(f"  (fewer than {MIN_NEW_DECIDED} new-regime claims are decided at this lag, so new-regime history is not used)")
            if L == 0:
                raw_only(f"  new-regime rows used ({len(used_new)}):")
                raw_only(used_new[show].to_string(index=False) if len(used_new) else "    none")
                raw_only(f"  old-regime rows used: {len(used_old)} decided claims, of which these {int(used_old['is_fraud'].sum())} are fraud:")
                raw_only(used_old[used_old["is_fraud"] == 1][show].to_string(index=False) if used_old["is_fraud"].sum() else "    none")
            else:
                latest = used[TIME_COL].max() if len(used) else "none"
                raw_only(f"  rows used: {len(used_new)} new-regime, {len(used_old)} old-regime; latest label used is from {latest}")
    print(f"PASS: {checked} claim x lag checks. Every label used comes from a claim first submitted before (cut-off - L days) and "
          "before the claim itself; no claim uses its own label; each feature equals a recount from the rows listed; "
          "partner_rate_ratio is exactly 1.0 wherever no new-regime row is used.")


def leakage_test_2():
    section("LEAKAGE TEST 2: changing a claim's own label does not change its features")
    history = load_history()
    fold = get_folds(history)["fold_B"]
    cutoff = CUTOFFS["fold_B"]
    for L in LAGS:
        base = build_features(fold.valid, cutoff, L, history=history)
        flipped = history.copy()
        own = flipped["claim_id"].isin(fold.valid["claim_id"])
        flipped.loc[own, "is_fraud"] = 1 - flipped.loc[own, "is_fraud"]
        pd.testing.assert_frame_equal(base, build_features(fold.valid, cutoff, L, history=flipped))
    print(f"PASS (validation): flipped the label of all {len(fold.valid)} Fold B validation claims; their features are identical "
          f"for L = {LAGS}.")

    train = fold.train
    cuts = snapshot_cutoffs(train[TIME_COL], "month")
    blocks = 0
    for L in LAGS:
        base = build_training_features(train, L, history=history)
        for cut in sorted(cuts.unique()):
            flipped = history.copy()
            later = flipped[TIME_COL] >= cut
            flipped.loc[later, "is_fraud"] = 1 - flipped.loc[later, "is_fraud"]
            block = train[cuts == cut]
            again = build_training_features(block, L, history=flipped)
            pd.testing.assert_frame_equal(base.loc[again.index], again)
            blocks += 1
    print(f"PASS (training rows, a stronger check than asked): for each of the {cuts.nunique()} training months, flipped every "
          f"label from that month onwards; that month's training features are identical ({blocks} month x lag checks).")

    flipped = history.copy()
    may = (flipped[TIME_COL] >= REGIME_START) & (flipped[TIME_COL] < cutoff)
    flipped.loc[may, "is_fraud"] = 1 - flipped.loc[may, "is_fraud"]
    changed = (build_features(fold.valid, cutoff, 0, history=history) != build_features(fold.valid, cutoff, 0, history=flipped)).any(axis=1)
    assert changed.any(), "control failed: features ignore labels altogether"
    print(f"Control (the test can fail): flipping the May labels, which June claims ARE allowed to see, changes the features of "
          f"{int(changed.sum())} of {len(fold.valid)} validation claims at L = 0.")


def leakage_test_3():
    section("LEAKAGE TEST 3: label-free features do not change when all later claims are deleted")
    events = load_events()
    history = load_history()
    fold = get_folds(history)["fold_B"]
    test = pd.read_parquet(DATA / "clean_test.parquet")
    sample = pd.concat([fold.train.sample(40, random_state=1), fold.valid.sample(40, random_state=1), test.sample(40, random_state=1)])
    full = rolling_partner_features(sample, events)
    for i in range(len(sample)):
        claim = sample.iloc[[i]]
        kept = events[events[TIME_COL] <= claim[TIME_COL].iloc[0]]  # every later claim deleted
        pd.testing.assert_frame_equal(full.iloc[[i]], rolling_partner_features(claim, kept))
    print(f"PASS: for {len(sample)} random claims (40 training, 40 Fold B validation, 40 test), deleted every claim submitted after "
          "the claim and rebuilt its rolling features; all identical.")

    for cut in ["2026-05-15", "2026-06-15", "2026-08-15"]:
        cut = pd.Timestamp(cut)
        before = events[events[TIME_COL] < cut].merge(pd.concat([history, test])[["claim_id"]], on="claim_id")
        rows = pd.concat([history, test])
        rows = rows[rows["claim_id"].isin(before["claim_id"])]
        pd.testing.assert_frame_equal(rolling_partner_features(rows, events), rolling_partner_features(rows, events[events[TIME_COL] < cut]))
    print("PASS: same result for ALL claims before 2026-05-15, 2026-06-15 and 2026-08-15 when everything from that date on is deleted.")

    earlier_deleted = rolling_partner_features(sample, events[events[TIME_COL] >= REGIME_START])
    changed = (full != earlier_deleted).any(axis=1)
    assert changed.any(), "control failed: rolling features ignore earlier claims altogether"
    print(f"Control (the test can fail): deleting EARLIER claims (everything before 1 May) changes {int(changed.sum())} of the "
          f"{len(sample)} sampled claims.")

    ties = events.groupby(["partner_id", TIME_COL]).size()
    print(f"Same-minute claims from one partner are not counted for each other (strictly before): {int((ties > 1).sum())} such "
          "partner-minutes exist in the data.")


def leakage_test_4_and_shapes():
    section("LEAKAGE TEST 4 and matrix shapes: no NaN or infinite value in any feature matrix")
    rows, checked = [], 0
    for name in ["fold_A", "fold_B", "old_design", "final"]:
        for L in LAGS:
            m = fold_matrices(name, L)
            for key in ["X_train", "X_score"]:
                X = m[key]
                assert list(X.columns) == feature_columns()
                assert X.index.is_unique
                assert all(dtype.kind == "f" for dtype in X.dtypes), "non-numeric feature column"
                assert np.isfinite(X.to_numpy()).all(), f"NaN or infinite value in {name} {key} at L={L}"
                checked += 1
            rows.append({"fold": name, "L": L,
                         "train matrix": f"{m['X_train'].shape[0]} x {m['X_train'].shape[1]}", "train frauds": int(m["y_train"].sum()),
                         "scored matrix": f"{m['X_score'].shape[0]} x {m['X_score'].shape[1]}",
                         "scored frauds": "no labels" if m["y_score"] is None else int(m["y_score"].sum()),
                         "scored claims with partner_has_new_history = 1": f"{m['X_score']['partner_has_new_history'].mean():.1%}",
                         "training rows with partner_has_new_history = 1": int(m["X_train"]["partner_has_new_history"].sum())})
    print(f"PASS: {checked} matrices (4 folds x {len(LAGS)} lags x train/scored) are all numeric, finite and free of NaN.")
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nFold roles: Fold B (score June) is the PRIMARY fold for model selection. Fold A (score May) is a STRESS TEST of the "
          "regime change: the new pattern starts on 1 May, so at its cut-off there is no new-regime history at all. "
          "'old_design' scores May-June with history frozen on 1 May. 'final' trains to 30 Jun and scores the Jul-Sep test set.")


def print_feature_list():
    section("Feature list")
    cats = reference_categories()
    n = 0
    for title, group in [("Label-based, frozen at the cut-off (rules A, B)", LABEL_FEATURES),
                         ("Label-free, rolling over earlier claims (rule C)", ROLLING_FEATURES),
                         ("Known at submission", STATIC_FEATURES)]:
        print(f"\n{title}:")
        for name, text in group.items():
            n += 1
            print(f"  {n:>2}. {name}: {text}")
    print("\nOne-hot groups (categories fixed from partners.csv and products.csv, so train, test and the app agree):")
    for group, text in ONE_HOT_GROUPS.items():
        print(f"  {group}_* ({len(cats[group])} columns): {text}")
        n += len(cats[group])
    assert n == len(feature_columns())
    print(f"\nTotal: {n} columns.")
    print("Considered and left out: hour, weekday, day of month and claim_description phrases (no signal, see the checks above); "
          "n_submissions, serial-repeat features, inspector_note and partner_inspected (rule D).")


def main():
    from reporting import raw_only

    print("Feature builder report (src/features.py). Time axis = first_submitted_at. New regime starts 2026-05-01.")
    print("Fold B is the PRIMARY fold for model selection; Fold A is a STRESS TEST of the regime change.")
    signal_checks()
    history_report()
    training_coverage_report()
    leakage_test_1(raw_only)
    leakage_test_2()
    leakage_test_3()
    leakage_test_4_and_shapes()
    print_feature_list()


if __name__ == "__main__":
    from reporting import masked_report

    with masked_report("features"):
        main()
