"""Model comparison for the Kestrel claim scorer. Writes reports/train.txt (partner IDs masked).

Models compared, simplest first (fixed list, nothing else is searched):
  M0   score = partner_rate_ratio only (no fitting). Kept as the BENCHMARK.
  M3   logistic regression on label-free features only
  M5   M3's log-odds as a fixed offset + ONE non-negative coefficient on log(partner_rate_ratio), learned
  M1   logistic regression on all features, month-start training snapshots
  M1b  logistic regression on all features, 7-day training snapshots
  M2   LightGBM, small trees, strong regularisation, class weights, 7-day training snapshots (experiment only)

FINAL MODEL: M5f = M5 with the coefficient FIXED at 1.0:
  log-odds = M3 log-odds + 1.0 * log(partner_rate_ratio), ratio = 1.0 with no decided new-regime claim.
This deviates from the pre-registered decision rule, which chose M0; the report states both.

ROBUSTNESS VARIANTS of the base model (each rule fixed before running: keep M5 only if it beats the
variant on PR-AUC in at least 90% of partner resamples on Fold B at L = 14):
  M5-lite  without city, partner_type, partner_age_days and is_new_partner
  M5-mid   without city and partner_type only
  M5-noage M5-mid without partner_age_days (tested against M5-mid; the burden of proof is on M5-mid). The last change.
FINAL below names the model the rules leave standing; src/final.py builds it.

DECISION RULE (fixed before running): choose the simplest model that is not clearly beaten by a more
complex one on Fold B. Metric: net rupee value of the fixed top-40-by-score list at L = 14. "Clearly
beaten" = the more complex model's value is above the upper end of the simpler model's 90% interval.

Fold B (score June) is the PRIMARY fold for model selection. Fold A (score May) is a STRESS TEST of
the regime change: no new-regime history exists at its cut-off.

Needs requirements-dev.txt (LightGBM, for M2 only). The final model does not: see src/final.py.
Run from the repo root:  python src/train.py
"""
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clean import DATA, TIME_COL
from evaluate import (BOOTSTRAP_SEED, LISTS, PARTNER_RULE_NOTE, REGIME_START, bootstrap_ranking, bootstrap_top40, fast_pr_roc, metrics,
                      month_of, review_list_value, select_partner_rule, select_top_by_p)
from features import CUTOFFS, LABEL_FEATURES, MIN_NEW_DECIDED, decided_rows, fold_matrices, load_history, model_columns

HEADLINE_L = 14
REPORT_LAGS = [14, 0, 30]   # headline, optimistic, pessimistic
RULE_ORDER = ["M0", "M3", "M5", "M1", "M1b", "M2"]            # the pre-registered candidates, simplest first
TABLE_ORDER = ["M0", "M3", "M5", "M5f", "M5m", "M5n", "M5l", "M1", "M1b", "M2"]
FINAL = "M5n"               # set from the outcome of the robustness rules; main() fails if it disagrees with them
PREVIOUS_FINAL = "M5m"      # the model the committed files were built from before the last change (for overlap reports)
FINAL_BETA = 1.0
MODEL_NAMES = {
    "M0": "M0  partner_rate_ratio only (BENCHMARK)",
    "M3": "M3  logistic, label-free features",
    "M5": "M5  M3 offset + 1 learned coefficient on log(ratio)",
    "M5f": "M5f M3 offset + 1.0 x log(ratio)",
    "M3m": "M3-mid   M3 without city and partner type",
    "M5m": "M5-mid   M3-mid offset + 1.0 x log(ratio)",
    "M3n": "M3-noage M3-mid without partner_age_days",
    "M5n": "M5-noage M3-noage offset + 1.0 x log(ratio)",
    "M3l": "M3-lite  M3 without city, partner type, partner age, new-partner flag",
    "M5l": "M5-lite  M3-lite offset + 1.0 x log(ratio)",
    "M1": "M1  logistic, all features, month-start snapshots",
    "M1b": "M1b logistic, all features, 7-day snapshots",
    "M2": "M2  LightGBM, 7-day snapshots",
}
FITTED = ["M3", "M5", "M1", "M1b", "M2"]
BETA_MAX = 5.0
LGB_PARAMS = dict(objective="binary", n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=30,
                  reg_lambda=10.0, reg_alpha=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  class_weight="balanced", random_state=0, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
LABEL_FREE = [c for c in model_columns() if c not in LABEL_FEATURES]
# Left out of the lite base model: features that describe WHO the partner is rather than what it does.
LITE_DROPPED = ["partner_age_days", "is_new_partner"]
LITE_DROPPED_PREFIXES = ("city_", "partner_type_")
LABEL_FREE_LITE = [c for c in LABEL_FREE if c not in LITE_DROPPED and not c.startswith(LITE_DROPPED_PREFIXES)]
LABEL_FREE_MID = [c for c in LABEL_FREE if not c.startswith(LITE_DROPPED_PREFIXES)]
KEEP_M5_IF_IT_WINS = 0.90   # share of partner resamples in which M5 must beat a variant on PR-AUC to be kept
LABEL_FREE_NOAGE = [c for c in LABEL_FREE_MID if c != "partner_age_days"]
BASE_COLUMNS = {"M3": LABEL_FREE, "M5": LABEL_FREE, "M5f": LABEL_FREE, "M3m": LABEL_FREE_MID, "M5m": LABEL_FREE_MID,
                "M3n": LABEL_FREE_NOAGE, "M5n": LABEL_FREE_NOAGE, "M3l": LABEL_FREE_LITE, "M5l": LABEL_FREE_LITE}
BASE_MODEL = {"M5f": "M3", "M5m": "M3m", "M5n": "M3n", "M5l": "M3l"}   # the label-free model each final candidate is built on
SHORT = {"M5f": "M5", "M5m": "M5-mid", "M5n": "M5-noage", "M5l": "M5-lite", "M3": "M3", "M3m": "M3-mid", "M3n": "M3-noage", "M3l": "M3-lite"}
FINAL_COLUMNS = BASE_COLUMNS[FINAL]
FINAL_BASE = BASE_MODEL[FINAL]
MODEL_NAMES[FINAL] += " (FINAL MODEL)"
VARIANT_INFO = {
    "M5l": dict(name="M5-lite", base_name="M3-lite", incumbent="M5f",
                title="ROBUSTNESS VARIANT 1: M5-lite (base model without city, partner type, partner age and the new-partner flag)",
                what="partner_age_days, is_new_partner, the city columns and the partner-type columns",
                why="it removes the partner-identity proxies and M5 has not shown it is better"),
    "M5n": dict(name="M5-noage", base_name="M3-noage", incumbent="M5m",
                title="ROBUSTNESS VARIANT 3 (the last change): M5-noage (M5-mid's base model without partner_age_days; the new-outlet flag is kept)",
                what="partner_age_days, on top of the city and partner-type columns already dropped by M5-mid",
                why="a continuous age feature mixes two regimes (before 1 May all confirmed fraud came from established outlets) and M5-mid "
                    "has not shown it is better with it; the new-outlet flag is the measured cohort effect"),
    "M5m": dict(name="M5-mid", base_name="M3-mid", incumbent="M5f",
                title="ROBUSTNESS VARIANT 2: M5-mid (base model without city and partner type, WITH partner age and the new-partner flag)",
                what="the city columns and the partner-type columns (one-hot columns with no business meaning, fitted on about 10 fraud partners)",
                why="it removes memorisation of outlet identity at no clear cost"),
}


@dataclass(frozen=True)
class Config:
    zoho_weight: float = 1.0   # weight of legacy Zoho rows in training (their zeros hide undecided cases)
    rows: str = "all"          # "all", "new_only" (claims from 1 May 2026), "all_flag" (all + is_new_regime column)


BASE = Config()
VARIANTS = {
    "all claims, Zoho weight 1.0 (base)": BASE,
    "Zoho weight 0.5": Config(zoho_weight=0.5),
    "Zoho excluded": Config(zoho_weight=0.0),
    "new-regime claims only": Config(rows="new_only"),
    "all claims + regime flag": Config(rows="all_flag"),
}


@lru_cache(maxsize=None)
def mats(fold, L, snapshot):
    return fold_matrices(fold, L, snapshot)


def _training_rows(m, columns, cfg):
    """X, y, weights for training under a config, plus the matching X for the scored rows."""
    train = m["train"]
    weight = np.where(train["is_zoho"].to_numpy() == 1, cfg.zoho_weight, 1.0)
    keep = weight > 0
    if cfg.rows == "new_only":
        keep &= (train[TIME_COL] >= REGIME_START).to_numpy()
    X, Xs = m["X_train"][columns].copy(), m["X_score"][columns].copy()
    if cfg.rows == "all_flag":
        X["is_new_regime"] = (train[TIME_COL] >= REGIME_START).to_numpy().astype(float)
        Xs["is_new_regime"] = (m["score"][TIME_COL] >= REGIME_START).to_numpy().astype(float)
    return X[keep], m["y_train"][keep], weight[keep], Xs, keep


def fit_logistic(X, y, w=None):
    """The one logistic recipe used everywhere: standardise, then L2 logistic regression with C = 1.0."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    model.fit(X.to_numpy(), y, logisticregression__sample_weight=w)
    return model


def fit_beta(offset, z, y, w):
    """The one coefficient of M5: maximise the likelihood of offset + beta*z over 0 <= beta <= BETA_MAX."""
    def loss(beta):
        eta = offset + beta * z
        return float(np.sum(w * (np.logaddexp(0, eta) - y * eta)))
    return float(minimize_scalar(loss, bounds=(0.0, BETA_MAX), method="bounded", options={"xatol": 1e-4}).x)


def predict(model, fold, L, cfg=BASE, beta=None):
    """Fraud probability for the scored rows of a fold, or None if the config leaves nothing to train on.

    Returns (p, info). Every model returns a probability so that all review lists can be built.
    beta fixes the M5 coefficient instead of learning it (M5f is M5 with beta = 1.0).
    """
    info = {}
    if model == "M0":
        # Same ranking as partner_rate_ratio; expressed as a rate so it can also feed the expected-value lists.
        return mats(fold, L, "month")["X_score"]["partner_new_rate_smoothed"].to_numpy(), info
    base_columns = BASE_COLUMNS.get(model, LABEL_FREE)
    if model in ("M5f", "M5l", "M5m", "M5n"):
        model, beta = "M5", FINAL_BETA
    if model in ("M3l", "M3m", "M3n"):
        model = "M3"

    snapshot = "month" if model in ("M3", "M1") else 7
    m = mats(fold, L, snapshot)
    columns = base_columns if model in ("M3", "M5") else model_columns()
    X, y, w, Xs, keep = _training_rows(m, columns, cfg)
    if len(y) == 0 or y.sum() == 0:
        return None, info
    info.update(train_rows=len(y), train_frauds=int(y.sum()))

    if model == "M2":
        import lightgbm as lgb   # experiment only; the final model never needs it

        booster = lgb.LGBMClassifier(**LGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster.fit(X.to_numpy(), y, sample_weight=w)
            raw = booster.predict_proba(Xs.to_numpy())[:, 1]
        # Balanced class weights inflate every probability; divide the odds by the weight ratio to undo it.
        ratio = (len(y) - y.sum()) / y.sum()
        return raw / (raw + (1 - raw) * ratio), info

    lr = fit_logistic(X, y, w)
    if model != "M5":
        return lr.predict_proba(Xs.to_numpy())[:, 1], info

    if beta is not None:
        info.update(beta=float(beta), beta_ci=(np.nan, np.nan), identified=False, fixed=True)
    else:
        z = np.log(m["X_train"]["partner_rate_ratio"].to_numpy()[keep])
        informative = z != 0   # rows whose partner has new-regime history; the others cannot move beta
        info.update(informative_rows=int(informative.sum()), informative_frauds=int(y[informative].sum()))
        if informative.sum() == 0:
            beta = 0.0
            info.update(beta=beta, beta_ci=(np.nan, np.nan), identified=False)
        else:
            offset = lr.decision_function(X.to_numpy())
            beta = fit_beta(offset[informative], z[informative], y[informative], w[informative])
            partner = m["train"]["partner_id"].to_numpy()[keep][informative]
            groups = [np.flatnonzero(partner == pid) for pid in np.unique(partner)]
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draws = []
            for _ in range(1000):
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                draws.append(fit_beta(offset[informative][idx], z[informative][idx], y[informative][idx], w[informative][idx]))
            info.update(beta=beta, beta_ci=tuple(np.percentile(draws, [5, 95])), identified=True)
    eta = lr.decision_function(Xs.to_numpy()) + beta * np.log(m["X_score"]["partner_rate_ratio"].to_numpy())
    return expit(eta), info


def score_frame(fold, L):
    return mats(fold, L, "month")["score"]


def rule_eligible(fold, L):
    """Claims the partner rule may review: partner_rate_ratio > 1 in the history frozen for this fold and lag."""
    return mats(fold, L, "month")["X_score"]["partner_rate_ratio"].to_numpy() > 1


def evaluate_scores(frame, p, eligible=None):
    """All metrics and 90% partner-bootstrap intervals for one set of scored claims."""
    y, amount, month, partner = (frame["is_fraud"].to_numpy(), frame["claim_amount_inr"].to_numpy(), month_of(frame),
                                 frame["partner_id"].to_numpy())
    res = metrics(y, p, amount, month, eligible=eligible)
    res["ci_rank"] = bootstrap_ranking(y, p, partner)
    for _, ci_key, mode, _ in LISTS:
        if mode == "rule" and eligible is None:
            res[ci_key] = None
            continue
        res[ci_key] = bootstrap_top40(y, p, amount, month, partner, by=mode, eligible=eligible)
    return res


_CACHE = {}


def run(model, fold, L, cfg=BASE, beta=None):
    key = (model, fold, L, cfg, beta)
    if key not in _CACHE:
        p, info = predict(model, fold, L, cfg, beta)
        _CACHE[key] = None if p is None else {"p": p, "info": info,
                                               "res": evaluate_scores(score_frame(fold, L), p, rule_eligible(fold, L))}
    return _CACHE[key]


# ----------------------------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ci(value, interval, fmt):
    if np.isnan(value):
        return "n/a"
    if np.isnan(interval[0]):
        return f"{value:{fmt}}"
    return f"{value:{fmt}} [{interval[0]:{fmt}} to {interval[1]:{fmt}}]"


def table_row(res):
    rank = res["ci_rank"]
    return {
        "PR-AUC": ci(res["pr_auc"], rank["pr_auc"], ".3f"),
        "ROC-AUC": ci(res["roc_auc"], rank["roc_auc"], ".3f"),
        "frauds in top 40": ci(res["top40_by_p"]["frauds_found"], res["ci_by_p"]["frauds_found"], ".0f"),
        "frauds in top 100": ci(res["top100_overall_frauds"], rank["top100_overall_frauds"], ".0f"),
        "net Rs: fixed 40 by score": ci(res["top40_by_p"]["net_value_rs"], res["ci_by_p"]["net_value_rs"], ",.0f"),
        "net Rs: fixed 40 by EV": ci(res["top40_by_ev"]["net_value_rs"], res["ci_by_ev"]["net_value_rs"], ",.0f"),
        "net Rs: stop-early": ci(res["upto40_stop_early"]["net_value_rs"], res["ci_stop_early"]["net_value_rs"], ",.0f"),
        "stop-early reviews": res["upto40_stop_early"]["n_reviewed"],
        "tie places": res["top40_by_p"]["tie_slots"],
    }


def results_table(fold, L, with_ceiling=True):
    rows = {MODEL_NAMES[m]: table_row(run(m, fold, L)["res"]) for m in TABLE_ORDER}
    if with_ceiling:
        frame = score_frame(fold, L)
        rows["Ceiling (true labels, not a model)"] = table_row(evaluate_scores(frame, frame["is_fraud"].to_numpy().astype(float)))
    return pd.DataFrame(rows).T


def lag_note():
    section("Investigation lag L: what the export can and cannot tell us")
    history = load_history()
    undecided = pd.read_parquet(DATA / "undecided.parquet")
    crm = history[history["is_zoho"] == 0]
    decided_m = crm.groupby(crm[TIME_COL].dt.to_period("M")).size()
    blank_m = undecided.groupby(undecided[TIME_COL].dt.to_period("M")).size()
    share = (blank_m / (blank_m + decided_m)).dropna()
    late = lambda d: d[(d[TIME_COL] >= "2026-06-16") & (d[TIME_COL] < "2026-07-01")]
    late_share = len(late(undecided)) / (len(late(undecided)) + len(late(crm)))
    print("share of CRM claims with a blank (undecided) label, by month first submitted:")
    print("  " + ", ".join(f"{m}: {s:.1%}" for m, s in share.items()))
    print(f"  claims first submitted 16-30 June 2026, the newest in the export: {late_share:.1%} blank")
    print(f"The blank share is about the same in every month ({share.min():.1%} to {share.max():.1%}) and does not rise for the newest "
          "claims. If blanks were simply 'not investigated yet' they would pile up at the end. So the export cannot reveal how long "
          "an investigation takes, and L has to be assumed.")
    print(f"VALIDATION HEADLINE: L = {HEADLINE_L} days. Also shown: L = 0 (optimistic: every outcome known at once) and L = 30 (pessimistic).")
    new30, _ = decided_rows(history, CUTOFFS["fold_B"], 30)
    print(f"At L >= 30 the history component CANNOT be validated in Fold B: only {len(new30)} new-regime claims are decided on "
          f"1 June at L = 30 (below the {MIN_NEW_DECIDED} needed), so every history feature is switched off there.")
    print("The FINAL model (src/final.py) freezes history at 30 Jun 2026 with every outcome known by then (L = 0 at the cut-off): "
          "the test set runs to 30 Sep, so for claims from August on, all June outcomes would be decided even under a 30-day lag.")


def apply_decision_rule(fold="fold_B", L=HEADLINE_L):
    """The pre-registered rule, applied mechanically to the pre-registered candidates. Prints the trace."""
    value = {m: run(m, fold, L)["res"]["top40_by_p"]["net_value_rs"] for m in RULE_ORDER}
    upper = {m: run(m, fold, L)["res"]["ci_by_p"]["net_value_rs"][1] for m in RULE_ORDER}
    chosen = None
    for i, m in enumerate(RULE_ORDER):
        beaten_by = [c for c in RULE_ORDER[i + 1:] if value[c] > upper[m]]
        verdict = (f"clearly beaten by {', '.join(beaten_by)}" if beaten_by else "not clearly beaten by any more complex model")
        print(f"  {MODEL_NAMES[m]:<52} net Rs {value[m]:>9,.0f}, upper end of its 90% interval {upper[m]:>9,.0f}: {verdict}")
        if chosen is None and not beaten_by:
            chosen = m
    return chosen


def paired_differences(reference, others, fold="fold_B", L=HEADLINE_L):
    frame = score_frame(fold, L)
    args = (frame["is_fraud"].to_numpy(), frame["claim_amount_inr"].to_numpy(), month_of(frame), frame["partner_id"].to_numpy())

    def draws(m):
        return bootstrap_top40(args[0], run(m, fold, L)["p"], *args[1:], by="p", return_draws=True)
    base = draws(reference)
    rows = {}
    for m in others:
        other = draws(m)
        d_net, d_found = other["net_value_rs"] - base["net_value_rs"], other["frauds_found"] - base["frauds_found"]
        rows[MODEL_NAMES[m]] = {
            f"net Rs difference vs {reference} [90%]": f"{np.median(d_net):,.0f} [{np.percentile(d_net, 5):,.0f} to {np.percentile(d_net, 95):,.0f}]",
            "frauds-found difference [90%]": f"{np.median(d_found):.0f} [{np.percentile(d_found, 5):.0f} to {np.percentile(d_found, 95):.0f}]",
            f"share of resamples where it beats {reference}": f"{(d_net > 0).mean():.0%}",
        }
    return pd.DataFrame(rows).T


def deviation_and_sensitivity(rule_choice):
    section("FINAL MODEL: a documented deviation from the pre-registered rule")
    m0, m5f = run("M0", "fold_B", HEADLINE_L)["res"], run("M5f", "fold_B", HEADLINE_L)["res"]
    learned = {L: run("M5", "fold_B", L)["info"] for L in (14, 0)}
    print(f"The rule chose {rule_choice}. The user chose instead M5f: log-odds = M3 log-odds + {FINAL_BETA} x log(partner_rate_ratio). "
          "(Which base model M3 stands for is settled by the robustness variants below.)")
    print("Why (decided by the user after seeing the results above; this is a judgement, not a result of the rule):")
    print("  1. M0 cannot rank claims beyond the partners with a fraud record: it orders all other claims only by how many "
          f"decided claims their partner has. {m0['top40_by_p']['tie_slots']} of its 40 June places were filled by random tie-break.")
    print("  2. The intervals do not separate the models, so the rule's choice of M0 is not evidence that M0 is better.")
    print(f"  3. On Fold B at L = {HEADLINE_L}, M5f has PR-AUC {m5f['pr_auc']:.3f} against {m0['pr_auc']:.3f} for M0 and ROC-AUC "
          f"{m5f['roc_auc']:.3f} against {m0['roc_auc']:.3f}, and finds {m5f['top40_by_p']['frauds_found']} frauds in the top 40 against "
          f"{m0['top40_by_p']['frauds_found']} (a difference inside the noise).")
    print(f"  4. A coefficient of {FINAL_BETA} lies inside what the data suggests: learned value {learned[14]['beta']:.2f} at L = 14 "
          f"(90% range {learned[14]['beta_ci'][0]:.2f} to {learned[14]['beta_ci'][1]:.2f}, from {learned[14]['informative_frauds']} frauds) and "
          f"{learned[0]['beta']:.2f} at L = 0 ({learned[0]['beta_ci'][0]:.2f} to {learned[0]['beta_ci'][1]:.2f}, from {learned[0]['informative_frauds']} frauds). "
          "Fixing it avoids estimating a number the data cannot pin down.")
    print("M0 stays in every table as the benchmark.")

    print("\nSensitivity to the coefficient (Fold B). 1.00 is the final model; the other two are the learned values:")
    rows = {}
    for beta, label in [(round(learned[14]["beta"], 2), "learned at L = 14"), (FINAL_BETA, "FIXED, final model"),
                        (round(learned[0]["beta"], 2), "learned at L = 0")]:
        row = {}
        for L in (14, 0):
            r = run("M5", "fold_B", L, beta=beta)["res"]
            row.update({f"L = {L}: PR-AUC": f"{r['pr_auc']:.3f}", f"L = {L}: ROC-AUC": f"{r['roc_auc']:.3f}",
                        f"L = {L}: frauds in top 40": r["top40_by_p"]["frauds_found"],
                        f"L = {L}: net Rs fixed 40 by score": f"{r['top40_by_p']['net_value_rs']:,.0f}"})
        rows[f"coefficient {beta:.2f} ({label})"] = row
    print(pd.DataFrame(rows).T.to_string())


def base_coefficients(fold, L, columns):
    """Standardised coefficients of the base logistic model fitted on a fold's training rows."""
    m = mats(fold, L, "month")
    return pd.Series(fit_logistic(m["X_train"][columns], m["y_train"])[-1].coef_[0], index=columns)


def head_to_head(challenger):
    """An incumbent model against one variant of its base model, on the same partner resamples (fixed seed). Returns the
    share of resamples on Fold B at L = 14 in which the incumbent beats the variant on PR-AUC (the number the rule uses)."""
    info = VARIANT_INFO[challenger]
    incumbent, old, new = info["incumbent"], SHORT[info["incumbent"]], info["name"]
    columns, incumbent_columns = BASE_COLUMNS[challenger], BASE_COLUMNS[incumbent]
    section(info["title"])
    print(f"{info['base_name']} drops {len(LABEL_FREE) - len(columns)} of the {len(LABEL_FREE)} original base features: {info['what']}. "
          f"It keeps the {len(columns)} others. {new} = {info['base_name']} log-odds + 1.0 x log(partner_rate_ratio).")
    print(f"RULE (fixed before running): keep {old} only if it beats {new} on PR-AUC in at least {KEEP_M5_IF_IT_WINS:.0%} of partner "
          f"resamples on Fold B at L = 14; otherwise adopt {new}. Both models are scored on the same 1,000 partner resamples "
          f"(seed {BOOTSTRAP_SEED}).")
    settings = [("fold_B", 14, "Fold B, L = 14 (PRIMARY: the rule is applied here)"), ("fold_B", 0, "Fold B, L = 0 (optimistic)"),
                ("fold_A", 14, "Fold A (STRESS TEST: no history, so each model equals its base model)")]
    rows, duels, decisive = {}, {}, None
    for fold, L, label in settings:
        frame = score_frame(fold, L)
        y, partner = frame["is_fraud"].to_numpy(), frame["partner_id"].to_numpy()
        draws = {m: bootstrap_ranking(y, run(m, fold, L)["p"], partner, return_draws=True) for m in (incumbent, challenger)}
        usable = ~np.isnan(draws[incumbent]["pr_auc"]) & ~np.isnan(draws[challenger]["pr_auc"])
        diff = (draws[incumbent]["pr_auc"] - draws[challenger]["pr_auc"])[usable]
        roc_diff = (draws[incumbent]["roc_auc"] - draws[challenger]["roc_auc"])[usable]
        for m, name in [(incumbent, old), (challenger, new)]:
            r = run(m, fold, L)["res"]
            rows[f"{label}: {name}"] = {
                "PR-AUC": ci(r["pr_auc"], r["ci_rank"]["pr_auc"], ".3f"), "ROC-AUC": ci(r["roc_auc"], r["ci_rank"]["roc_auc"], ".3f"),
                "frauds in top 40": ci(r["top40_by_p"]["frauds_found"], r["ci_by_p"]["frauds_found"], ".0f"),
                "frauds in top 100": ci(r["top100_overall_frauds"], r["ci_rank"]["top100_overall_frauds"], ".0f"),
                "net Rs: fixed 40 by score": f"{r['top40_by_p']['net_value_rs']:,.0f}",
                "net Rs: partner rule": f"{r['upto40_partner_rule']['net_value_rs']:,.0f} ({r['upto40_partner_rule']['n_reviewed']} reviews)"}
        duels[label] = {"resamples used": int(usable.sum()), f"{old} beats {new} on PR-AUC": f"{(diff > 0).mean():.1%}",
                        f"{new} beats {old}": f"{(diff < 0).mean():.1%}", "tie": f"{(diff == 0).mean():.1%}",
                        f"PR-AUC difference, {old} minus {new} [90%]": f"{np.median(diff):+.3f} [{np.percentile(diff, 5):+.3f} to {np.percentile(diff, 95):+.3f}]",
                        f"{old} beats {new} on ROC-AUC": f"{(roc_diff > 0).mean():.1%}"}
        if decisive is None:
            decisive = float((diff > 0).mean())
    print()
    print(pd.DataFrame(rows).T.to_string())
    print("\nHead to head on the same partner resamples:")
    print(pd.DataFrame(duels).T.to_string())
    adopt = decisive < KEEP_M5_IF_IT_WINS
    print(f"\nRULE OUTCOME: on Fold B at L = 14, {old} beats {new} on PR-AUC in {decisive:.1%} of partner resamples, "
          f"{'below' if adopt else 'at or above'} the {KEEP_M5_IF_IT_WINS:.0%} needed to keep it. "
          + (f"ADOPT {new}: {info['why']}." if adopt else f"KEEP {old}; {new} is not adopted."))
    print("The partner-rule list is the same for both wherever fewer than 40 claims qualify: it depends only on which partners have a "
          "confirmed fraud record. The base model only orders the claims inside it, and the claims beyond it on the fixed list of 40.")

    print("\nTen largest standardised coefficients of the base model (per standard deviation of the feature; + raises the score):")
    for fold, L, label in [("final", 0, "fitted on all labelled claims to 30 Jun 2026 (what a final model would use)"),
                           ("fold_B", HEADLINE_L, "fitted on claims before 1 Jun 2026 (what Fold B evaluated)")]:
        tables = {}
        for name, cols in [(SHORT[BASE_MODEL[incumbent]], incumbent_columns), (info["base_name"], columns)]:
            coef = base_coefficients(fold, L, cols)
            top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
            tables[name] = [f"{feature} {value:+.2f}" for feature, value in top.items()]
        print(f"\n  {label}:")
        print(pd.DataFrame(tables, index=range(1, 11)).to_string())
    return decisive


def pr_auc_win_share(incumbent, challenger, fold="fold_B", L=HEADLINE_L):
    """Share of the fixed-seed partner resamples in which the incumbent has the higher PR-AUC."""
    frame = score_frame(fold, L)
    y, partner = frame["is_fraud"].to_numpy(), frame["partner_id"].to_numpy()
    a, b = (bootstrap_ranking(y, run(m, fold, L)["p"], partner, return_draws=True)["pr_auc"] for m in (incumbent, challenger))
    usable = ~np.isnan(a) & ~np.isnan(b)
    return float((a[usable] > b[usable]).mean())


def age_and_flag_summary():
    """Contributions of partner_age_days and is_new_partner in the M3-mid final fit, for three kinds of outlet (test claims)."""
    m = mats("final", 0, "month")
    lr = fit_logistic(m["X_train"][LABEL_FREE_MID], m["y_train"])
    scaler, coef = lr[0], pd.Series(lr[-1].coef_[0], index=LABEL_FREE_MID)
    position = {name: LABEL_FREE_MID.index(name) for name in ("partner_age_days", "is_new_partner")}
    test, X = m["score"], m["X_score"]

    def contribution(name):
        i = position[name]
        return coef[name] * (X[name].to_numpy() - scaler.mean_[i]) / scaler.scale_[i]
    age, flag = contribution("partner_age_days"), contribution("is_new_partner")
    after_june = (test["onboarded_date"] > m["train"][TIME_COL].max()).to_numpy()
    is_new = X["is_new_partner"].to_numpy() == 1
    groups = {"after_june": ("(i) outlets onboarded after 30 Jun 2026", after_june),
              "established": ("(ii) established outlets (onboarded before 1 Oct 2025)", ~is_new),
              "new": ("(iii) outlets onboarded 1 Oct 2025 to 30 Jun 2026", is_new & ~after_june)}
    out = {"coef_age": float(coef["partner_age_days"]), "coef_flag": float(coef["is_new_partner"]),
           "mean_age": float(scaler.mean_[position["partner_age_days"]])}
    for key, (label, mask) in groups.items():
        both = (age + flag)[mask]
        out[key] = {"label": label, "outlets": int(test[mask]["partner_id"].nunique()), "claims": int(mask.sum()),
                    "median_age_days": float(np.median(X["partner_age_days"].to_numpy()[mask])), "flag": float(np.median(flag[mask])),
                    "age": float(np.median(age[mask])), "age_min": float(age[mask].min()), "age_max": float(age[mask].max()),
                    "both": float(np.median(both)), "both_min": float(both.min()), "both_max": float(both.max())}
    return out


def age_and_flag_contributions():
    section("Does partner age work against the new-outlet flag? Contributions in the M3-mid final fit (log-odds, before the Platt correction)")
    c = age_and_flag_summary()
    print(f"coefficients per standard deviation: partner_age_days {c['coef_age']:+.2f}, is_new_partner {c['coef_flag']:+.2f}. "
          f"Mean age in training {c['mean_age']:.0f} days.")
    rows = {}
    for key in ("after_june", "established", "new"):
        g = c[key]
        rows[g["label"]] = {"outlets": g["outlets"], "test claims": g["claims"], "median age at claim (days)": f"{g['median_age_days']:.0f}",
                            "new-outlet flag": f"{g['flag']:+.2f}", "partner age (median)": f"{g['age']:+.2f}",
                            "partner age (range)": f"{g['age_min']:+.2f} to {g['age_max']:+.2f}", "both together (median)": f"{g['both']:+.2f}",
                            "both together (range)": f"{g['both_min']:+.2f} to {g['both_max']:+.2f}"}
    print(pd.DataFrame(rows).T.to_string())
    est, late, new = c["established"], c["after_june"], c["new"]
    print(f"READING. (a) Established outlets: the two features together add {est['both']:+.2f} at the median and up to {est['both_max']:+.2f} "
          f"for the oldest, all of it from age (the flag alone gives {est['flag']:+.2f}). "
          + ("That part of the concern is RIGHT: age raises the score of long-established outlets." if est["both_max"] > 0.3 else
             "That part of the concern is not borne out."))
    gap = late["age"] - new["age"]
    print(f"(b) Outlets onboarded after 30 Jun: age takes {late['age']:+.2f} off, against {new['age']:+.2f} for other new outlets, a difference "
          f"of {gap:+.2f} next to the flag's {late['flag']:+.2f}. "
          + ("That part of the concern is NOT borne out: the low age trims the flag slightly, it does not cancel it."
             if abs(gap) < 0.5 * abs(late["flag"]) else "That part of the concern is RIGHT: the low age cancels much of the flag."))


def robustness_variants():
    """Apply the pre-declared rules in order and check that FINAL is the model they leave standing."""
    lite = head_to_head("M5l")
    mid = head_to_head("M5m")
    age_and_flag_contributions()
    noage = head_to_head("M5n")
    section("Outcome of the robustness rules")
    print(f"1. M5-lite vs M5:      M5 wins {lite:.1%} of resamples -> {'M5-lite adopted' if lite < KEEP_M5_IF_IT_WINS else 'M5 kept, M5-lite thrown away'}.")
    print(f"2. M5-mid vs M5:       M5 wins {mid:.1%} of resamples -> {'M5-mid ADOPTED, M5 thrown away' if mid < KEEP_M5_IF_IT_WINS else 'M5 kept, M5-mid thrown away'}.")
    print(f"3. M5-noage vs M5-mid: M5-mid wins {noage:.1%} of resamples -> "
          f"{'M5-noage ADOPTED, M5-mid thrown away' if noage < KEEP_M5_IF_IT_WINS else 'M5-mid kept, M5-noage thrown away'}.")
    assert mid < KEEP_M5_IF_IT_WINS, "rule 3 was declared against M5-mid; it only applies if rule 2 adopted M5-mid"
    standing = "M5n" if noage < KEEP_M5_IF_IT_WINS else "M5m"
    print(f"MODEL LEFT STANDING: {MODEL_NAMES[standing]}. No further variants are tried.")
    assert standing == FINAL, f"FINAL in src/train.py is {FINAL} but the rules leave {standing} standing; update FINAL and re-run"
    print("Whatever the base model, the review list under the partner rule is the same (25 reviews, 13 frauds at L = 14): it "
          "depends only on partners with a confirmed fraud record.")


def partner_rule_report():
    section("PARTNER RULE (recommended product rule): review only claims whose partner has partner_rate_ratio > 1")
    print(f"Ranked by score, capped at 40 a month, never padded. LABEL: {PARTNER_RULE_NOTE}.")
    print("'partner_rate_ratio > 1' means the partner's confirmed new-regime fraud rate is above the average in the frozen history.")
    rows = {}
    for L in (HEADLINE_L, 0):
        for m in (FINAL, "M0"):
            r = run(m, "fold_B", L)["res"]
            v, c = r["upto40_partner_rule"], r["ci_partner_rule"]
            rows[f"Fold B, L = {L}: {MODEL_NAMES[m].split('(')[0].strip()}"] = {
                "claims eligible": v["n_eligible"], "reviews": ci(v["n_reviewed"], c["n_reviewed"], ".0f"),
                "frauds": ci(v["frauds_found"], c["frauds_found"], ".0f"), "genuine held": v["genuine_held"],
                "net Rs": ci(v["net_value_rs"], c["net_value_rs"], ",.0f"),
                "Rs per claim reviewed": ci(v["rs_per_claim_reviewed"], c["rs_per_claim_reviewed"], ",.0f"),
                "for comparison, net Rs of the fixed 40 by score": f"{r['top40_by_p']['net_value_rs']:,.0f}"}
    print(pd.DataFrame(rows).T.to_string())
    print("When fewer than 40 claims are eligible the ranking does not matter (all eligible claims are reviewed), so the final model and M0 "
          "give the same list; the score only decides the order within it and which claims are cut when more than 40 qualify.")
    a = run(FINAL, "fold_A", HEADLINE_L)["res"]["upto40_partner_rule"]
    print(f"Fold A (stress test): {a['n_eligible']} claims eligible, {a['n_reviewed']} reviewed. With no confirmed fraud on record the "
          "rule reviews nothing, and the product must say so rather than pad the list.")


def variant_experiments(models, L=HEADLINE_L):
    section(f"Experiment: Zoho weight and training rows, for the best two fitted models (L = {L})")
    print("M0 has no training step, so there is nothing to vary for it. 'Best two' = highest net Rs of the fixed top-40-by-score "
          f"list on Fold B among the pre-registered fitted models: {', '.join(models)}.")
    for model in models:
        rows = {}
        for name, cfg in VARIANTS.items():
            row = {}
            for fold, label in [("fold_B", "Fold B"), ("fold_A", "Fold A")]:
                out = run(model, fold, L, cfg)
                if out is None:
                    row.update({f"{label}: PR-AUC": "n/a", f"{label}: frauds in top 40": "n/a (no training rows)",
                                f"{label}: net Rs fixed 40 by score": "n/a"})
                    continue
                r = out["res"]
                row.update({f"{label}: PR-AUC": ci(r["pr_auc"], r["ci_rank"]["pr_auc"], ".3f"),
                            f"{label}: frauds in top 40": ci(r["top40_by_p"]["frauds_found"], r["ci_by_p"]["frauds_found"], ".0f"),
                            f"{label}: net Rs fixed 40 by score": ci(r["top40_by_p"]["net_value_rs"], r["ci_by_p"]["net_value_rs"], ",.0f")})
                if label == "Fold B":
                    row["Fold B: training rows (frauds)"] = f"{out['info']['train_rows']} ({out['info']['train_frauds']})"
            rows[name] = row
        print(f"\n{MODEL_NAMES[model]}:")
        print(pd.DataFrame(rows).T.to_string())


def staleness(L=HEADLINE_L):
    section(f"Staleness: does the history signal fade as the frozen history ages? Fold B by week of June (L = {L})")
    m = mats("fold_B", L, "month")
    v, X = m["score"], m["X_score"]
    y = v["is_fraud"].to_numpy()
    week = np.minimum((v[TIME_COL].dt.day.to_numpy() - 1) // 7, 3)
    labels = ["1-7 June", "8-14 June", "15-21 June", "22-30 June"]
    known_before = CUTOFFS["fold_B"] - timedelta(days=L)
    listed = select_top_by_p(run(FINAL, "fold_B", L)["p"], month_of(v))
    prior_fraud = X["partner_new_fraud_count"].to_numpy() > 0
    rows = {}
    for w, label in enumerate(labels):
        mask = week == w
        row = {"claims": int(mask.sum()), "frauds": int(y[mask].sum()),
               "frauds from partners with fraud in the frozen history": int(y[mask & prior_fraud].sum()),
               "age of newest label used (days)": f"{(v[TIME_COL][mask].min() - known_before).days}-{(v[TIME_COL][mask].max() - known_before).days}"}
        for name in ("M0", FINAL):
            row[f"ROC-AUC {name}"] = f"{fast_pr_roc(y[mask], run(name, 'fold_B', L)['p'][mask])[1]:.3f}"
        row[f"places on the month's top-40 list ({FINAL})"] = int(listed[mask].sum())
        row["frauds caught by those places"] = int(y[mask & listed].sum())
        rows[label] = row
    print(pd.DataFrame(rows).T.to_string())
    print("Read with care: about 5 frauds a week. A fall in the later weeks would mean the frozen history is ageing; a flat "
          "line would not prove it stays good.")
    print("What the final model faces CANNOT be tested: its history is frozen on 30 Jun 2026 and it scores claims up to 92 days "
          f"later, but only two months of the new regime exist. The oldest case measurable here is the last week of June at L = {L}, "
          f"where the newest label used is {(v[TIME_COL].max() - known_before).days} days old. Anything beyond that is extrapolation.")


def cold_start(L=HEADLINE_L):
    section(f"Cold start: claims from partners with and without decided new-regime history, Fold B (L = {L})")
    m = mats("fold_B", L, "month")
    v, X = m["score"], m["X_score"]
    y = v["is_fraud"].to_numpy()
    has = X["partner_has_new_history"].to_numpy() == 1
    rows = {}
    for name in ("M0", FINAL_BASE, FINAL):
        p = run(name, "fold_B", L)["p"]
        listed = select_top_by_p(p, month_of(v))
        for label, mask in [("partner HAS history", has), ("partner has NO history", ~has)]:
            pr, roc = fast_pr_roc(y[mask], p[mask])
            rows[f"{name}: {label}"] = {"claims": int(mask.sum()), "frauds": int(y[mask].sum()), "PR-AUC": f"{pr:.3f}", "ROC-AUC": f"{roc:.3f}",
                                        "places on the top-40 list": int(listed[mask].sum()), "frauds caught": int(y[mask & listed].sum())}
    print(pd.DataFrame(rows).T.to_string())
    print("For M0 every no-history claim has the same score, so its PR-AUC there is just the fraud rate and its list places are "
          f"luck. {FINAL} equals {FINAL_BASE} on no-history claims (ratio = 1.0). Random PR-AUC = the group's fraud rate.")


def list_composition(L=HEADLINE_L):
    section(f"What the June lists are made of (Fold B, L = {L})")
    v = score_frame("fold_B", L)
    y, amount, month = v["is_fraud"].to_numpy(), v["claim_amount_inr"].to_numpy(), month_of(v)
    eligible = rule_eligible("fold_B", L)
    rows = {}
    for name in (FINAL, "M0"):
        p = run(name, "fold_B", L)["p"]
        listed = select_top_by_p(p, month)
        rule = select_partner_rule(p, month, eligible)
        for label, mask in [("fixed 40 by score", listed),
                            ("  of which places with partner_rate_ratio > 1", listed & eligible),
                            ("  of which other places", listed & ~eligible),
                            ("partner rule list", rule)]:
            val = review_list_value(y, amount, mask)
            frauds = amount[mask & (y == 1)]
            rows[f"{name}: {label}"] = {"reviewed": val["n_reviewed"], "frauds": val["frauds_found"], "genuine held": val["genuine_held"],
                                        "fraud Rs caught": f"{val['fraud_rs_caught']:,.0f}",
                                        "largest single fraud Rs": f"{frauds.max() if len(frauds) else 0:,.0f}",
                                        "net Rs if only these were reviewed": f"{val['net_value_rs']:,.0f}"}
    print(pd.DataFrame(rows).T.to_string())
    mean_fraud = amount[y == 1].mean()
    print(f"Average June fraud claim: Rs {mean_fraud:,.0f}. At that size a reviewed claim pays for itself only if the chance it is "
          f"fraud is above {640 / (mean_fraud + 380):.0%} (from p*amount - (1-p)*380 - 260 > 0).")


def final_everywhere():
    section(f"The final model ({FINAL}) and the benchmark (M0) on every design, L = {HEADLINE_L}")
    rows = {}
    for name in (FINAL, "M0"):
        a, b = run(name, "fold_A", HEADLINE_L), run(name, "fold_B", HEADLINE_L)
        rows[f"{name}: Fold B, score June (PRIMARY)"] = table_row(b["res"])
        rows[f"{name}: Fold A, score May (STRESS TEST: no new-regime history)"] = table_row(a["res"])
        pooled = pd.concat([score_frame("fold_A", HEADLINE_L), score_frame("fold_B", HEADLINE_L)])
        rows[f"{name}: Pooled, May + June"] = table_row(evaluate_scores(pooled, np.concatenate([a["p"], b["p"]])))
        rows[f"{name}: Old design = what a model that has not seen the new regime scores (train < 1 May, score May-June)"] = table_row(
            run(name, "old_design", HEADLINE_L)["res"])
    print(pd.DataFrame(rows).T.to_string())
    print("Pooled and old design cover two months, so their lists hold 80 places.")


def main():
    print("Model comparison (src/train.py). Time axis = first_submitted_at.")
    print("Fold B (score June 2026) is the PRIMARY fold for model selection. Fold A (score May 2026) is a STRESS TEST of the regime "
          "change: at its cut-off no new-regime history exists, so M0 is a random draw there and every history feature is off.")
    print("All intervals are 90%, from 1,000 resamples of PARTNERS. Fold B holds 22 frauds from 8 partners, Fold A 14 from 6: "
          "the intervals are wide and small differences between models mean nothing.")
    lag_note()

    section(f"RESULTS TABLE: Fold B (PRIMARY), headline lag L = {HEADLINE_L}")
    print("Lists: 'fixed 40' always reviews 40 claims; 'stop-early' reviews up to 40 in expected-value order and stops when "
          "EV <= 0 [DEPENDS ON CALIBRATION: 36 validation frauds is a thin base]. 'tie places' = list places decided by a random "
          "tie-break rather than by the score. The partner-rule list has its own section below.")
    print(results_table("fold_B", HEADLINE_L).to_string())
    m5 = run("M5", "fold_B", HEADLINE_L)["info"]
    print(f"\nM5 learned coefficient on log(partner_rate_ratio), Fold B, L = {HEADLINE_L}: {m5['beta']:.2f} "
          f"(90% bootstrap range {m5['beta_ci'][0]:.2f} to {m5['beta_ci'][1]:.2f}; allowed range 0 to {BETA_MAX:.0f}), learned from "
          f"{m5['informative_rows']} training rows whose partner has new-regime history, {m5['informative_frauds']} of them fraud.")

    section("Pre-registered decision rule: net Rs of the fixed top-40-by-score list, Fold B, L = 14 (candidates as registered)")
    rule_choice = apply_decision_rule()
    print(f"THE RULE CHOOSES: {MODEL_NAMES[rule_choice]}")
    print("\nSupporting evidence, same partner resamples for both models (not part of the rule):")
    print(paired_differences("M0", [m for m in TABLE_ORDER if m != "M0"]).to_string())
    deviation_and_sensitivity(rule_choice)
    robustness_variants()

    for L, tag in [(0, "optimistic"), (30, "pessimistic")]:
        section(f"Fold B (PRIMARY), L = {L} ({tag})")
        print(results_table("fold_B", L).to_string())
        info = run("M5", "fold_B", L)["info"]
        if info.get("identified"):
            print(f"\nM5 learned coefficient: {info['beta']:.2f} (90% range {info['beta_ci'][0]:.2f} to {info['beta_ci'][1]:.2f}), from "
                  f"{info['informative_rows']} informative training rows, {info['informative_frauds']} fraud.")
        else:
            print("\nM5 learned coefficient: cannot be estimated (no training row has new-regime history at this lag); set to 0. "
                  "At this lag no validation claim has history either, so each M5 variant equals its base model here.")
    section(f"Fold A (STRESS TEST of the regime change), L = {HEADLINE_L}")
    print("No new-regime history exists on 1 May at any lag, so M0 is a random draw, every M5 variant equals its base model, and M1/M1b/M2 can only "
          "use partner_fraud_count_old on top of the label-free features.")
    print(results_table("fold_A", HEADLINE_L).to_string())
    print("ROC-AUC far below 0.5 means the fitted models rank May claims BACKWARDS: they learned the pre-May pattern, which "
          "reversed on 1 May. A new regime shift would hurt the final model in the same way until new outcomes arrive.")

    final_everywhere()
    partner_rule_report()
    net = {m: run(m, "fold_B", HEADLINE_L)["res"]["top40_by_p"]["net_value_rs"] for m in FITTED}
    variant_experiments(sorted(FITTED, key=lambda m: -net[m])[:2])
    staleness()
    cold_start()
    list_composition()
    print("\nCalibration, the final fit, predictions.csv and the expected-score text are in src/final.py (reports/final.txt).")


if __name__ == "__main__":
    from reporting import masked_report

    with masked_report("train"):
        main()
