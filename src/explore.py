"""Explore the cleaned training data. Prints to the console and writes reports/explore.txt
(partner IDs masked) and reports/raw/explore.txt (real IDs, git-ignored).

Uses data/clean_train.parquet. data/clean_test.parquet is read only for the train-vs-test
comparisons (items 2 and 8) and data/partners.csv only to count partners (item 3).
Aggregates only: no claim rows are printed. Partner IDs appear in item 4 and are masked in
the committed copy (see src/reporting.py).

Run from the repo root:  python src/explore.py
"""
from collections import Counter
import re

import numpy as np
import pandas as pd

from clean import DATA, TIME_COL, VALID_START
from features import new_outlet_counts
from reporting import masked_report

POLICY_CHANGE = VALID_START                # 1 May 2026: claims under Rs 2,000 auto-approved
SMALL_CLAIM = 2000
NEW_CUT = pd.Timestamp("2025-10-01")       # "new partner" = onboarded on/after this date
ALT_CUT = pd.Timestamp("2025-11-01")
LEAKY_STEMS = ["fraud", "suspic", "fake", "duplic", "reject"]
BEFORE, AFTER = "before 1 May 2026", "from 1 May 2026"


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def so_what(text):
    print(f"\nSO WHAT: {text}")


def pct(x):
    return f"{100 * x:.2f}%"


def rates(df, by):
    t = df.groupby(by, observed=True)["is_fraud"].agg(claims="size", fraud="sum", rate="mean")
    t["rate"] = t["rate"].map(pct)
    return t


def add_columns(df):
    df = df.copy()
    df["period"] = np.where(df[TIME_COL] >= POLICY_CHANGE, AFTER, BEFORE)
    df["size_band"] = np.where(df["claim_amount_inr"] < SMALL_CLAIM, "under Rs 2,000", "Rs 2,000 and above")
    df["partner_age"] = np.where(df["onboarded_date"] >= NEW_CUT, "new", "old")
    df["partner_age_alt"] = np.where(df["onboarded_date"] >= ALT_CUT, "new", "old")
    return df


# 1 ------------------------------------------------------------------------------------------
def item1(tr):
    section("1. Fraud rate by month and by source")
    print(rates(tr, tr[TIME_COL].dt.to_period("M").rename("month")).to_string())
    print()
    print(rates(tr, "source").to_string())
    by_period = tr.groupby("period")["is_fraud"].mean()
    crm_before = tr[(tr["source"] == "crm") & (tr["period"] == BEFORE)]["is_fraud"].mean()
    zoho = tr[tr["source"] == "legacy_zoho"]["is_fraud"].mean()
    print(f"\nCRM claims before 1 May 2026 only: {pct(crm_before)}  (Zoho: {pct(zoho)})")
    month = tr[TIME_COL].dt.to_period("M")
    last = tr[month == month.max()]["is_fraud"].mean()
    so_what(f"The monthly rate sits around 1% until April 2026, then rises ({pct(by_period[BEFORE])} before 1 May, "
            f"{pct(by_period[AFTER])} after) and is still climbing in the last month ({pct(last)}), so the test months may run "
            f"higher again. CRM looks worse than Zoho only because it contains May-June; like for like the two systems are close "
            f"({pct(crm_before)} vs {pct(zoho)}), so the undecided cases hidden in the Zoho zeros do not visibly distort the rate.")


# 2 ------------------------------------------------------------------------------------------
def item2(tr, te):
    section("2. Before vs after 1 May 2026, by claim size (policy: under Rs 2,000 auto-approved from 1 May 2026)")
    t = tr.groupby(["period", "size_band"])["is_fraud"].agg(claims="size", fraud="sum", rate="mean")
    t["fraud_rs"] = tr[tr["is_fraud"] == 1].groupby(["period", "size_band"])["claim_amount_inr"].sum()
    t["fraud_rs"] = t["fraud_rs"].fillna(0).round(0).astype(int)
    r = t["rate"].copy()
    t["rate"] = t["rate"].map(pct)
    print(t.to_string())

    print("\nshare of claims under Rs 2,000:")
    share = tr.groupby("period")["size_band"].apply(lambda s: (s == "under Rs 2,000").mean())
    test_share = (te["size_band"] == "under Rs 2,000").mean()
    for k in [BEFORE, AFTER]:
        print(f"  train, {k:<18} {pct(share[k])}")
    print(f"  test (Jul-Sep 2026)       {pct(test_share)}")

    months = tr.groupby("period")[TIME_COL].apply(lambda s: s.dt.to_period("M").nunique())
    rs_month = tr[tr["is_fraud"] == 1].groupby("period")["claim_amount_inr"].sum() / months
    n_month = tr.groupby("period")["is_fraud"].sum() / months
    print("\nfraud per month:")
    for k in [BEFORE, AFTER]:
        print(f"  {k:<18} {months[k]:>2} months  {n_month[k]:5.1f} fraud claims/month  Rs {rs_month[k]:,.0f}/month")

    f = t["fraud"]
    so_what(f"The fraud moved. Before the change small claims were the safe ones ({pct(r[(BEFORE, 'under Rs 2,000')])}) and large "
            f"claims the risky ones ({pct(r[(BEFORE, 'Rs 2,000 and above')])}). After it, {f[(AFTER, 'under Rs 2,000')]} of "
            f"{f[AFTER].sum()} frauds are small claims ({pct(r[(AFTER, 'under Rs 2,000')])}) and large claims are nearly clean. "
            f"The test set is {pct(test_share)} small claims, the same mix as May-June, so it sits in the new regime; a model fitted "
            f"only to pre-May data would learn the opposite pattern. In rupees fraud did not rise: Rs {rs_month[BEFORE]:,.0f} a month "
            f"before, Rs {rs_month[AFTER]:,.0f} after. More fraud claims ({n_month[BEFORE]:.1f} to {n_month[AFTER]:.1f} a month), "
            "each much smaller.")


# 3 ------------------------------------------------------------------------------------------
def item3(tr, partners):
    section("3. New vs old partners")
    print("partners onboarded on/after each cut-off (policy section 6: 'about 60 new service partners over the past year'):")
    rows = []
    for cut in pd.date_range("2025-06-01", "2025-12-01", freq="MS"):
        rows.append({"cut-off": cut.date(), "partners in partners.csv": int((partners["onboarded_date"] >= cut).sum())})
    cuts = pd.DataFrame(rows).set_index("cut-off")
    print(cuts.to_string())
    counts = new_outlet_counts(tr)
    print(f"Definition used everywhere for 'new outlet' counts: {counts['definition']}: {counts['new_outlets']}, of which "
          f"{counts['new_outlets_with_fraud']} have a confirmed fraud.")
    n_new, n_alt = cuts.loc[NEW_CUT.date(), "partners in partners.csv"], cuts.loc[ALT_CUT.date(), "partners in partners.csv"]
    # Months only: exact onboarding dates would identify a partner to anyone holding partners.csv.
    old_last = partners[partners["onboarded_date"] < NEW_CUT]["onboarded_date"].max().strftime("%B %Y")
    new_first = partners[partners["onboarded_date"] >= NEW_CUT]["onboarded_date"].min().strftime("%B %Y")
    print(f"no partner was onboarded between {old_last} and {new_first}")

    print("\nnew = onboarded on/after 2025-10-01")
    print(rates(tr, "partner_age").to_string())
    print(rates(tr, ["period", "partner_age"]).to_string())
    same = bool((tr["partner_age"] == tr["partner_age_alt"]).all())
    print(f"\nsecond cut (new = onboarded on/after 2025-11-01): "
          f"{'selects exactly the same claims, so the tables are identical' if same else 'differs:'}")
    if not same:
        print(rates(tr, "partner_age_alt").to_string())
        print(rates(tr, ["period", "partner_age_alt"]).to_string())

    r = tr.groupby(["period", "partner_age"])["is_fraud"].agg(["sum", "mean", "size"])
    so_what(f"The 1 Oct and 1 Nov cuts give {n_new} and {n_alt} partners: the same group, because nobody was onboarded between "
            f"{old_last} and {new_first}, and both match the policy's 'about 60' (only "
            f"{counts['new_outlets']} of them have a decided claim since 1 May 2026). New partners had "
            f"{int(r.loc[(BEFORE, 'new'), 'sum'])} frauds in {int(r.loc[(BEFORE, 'new'), 'size'])} claims before 1 May and "
            f"{int(r.loc[(AFTER, 'new'), 'sum'])} in {int(r.loc[(AFTER, 'new'), 'size'])} after ({pct(r.loc[(AFTER, 'new'), 'mean'])}), "
            f"against {pct(r.loc[(AFTER, 'old'), 'mean'])} for old partners after. Ritu is right about where the fraud has been since "
            f"May, but all {int(r.loc[(BEFORE, 'old'), 'sum'])} frauds before May came from old partners, and item 4 shows the new "
            "group is not guilty as a whole.")


# 4 ------------------------------------------------------------------------------------------
def item4(tr):
    section("4. Partner concentration")
    p = tr.groupby("partner_id").agg(claims=("is_fraud", "size"), fraud=("is_fraud", "sum"), group=("partner_age", "first"))
    total = int(p["fraud"].sum())

    def top10(tab, n_fraud, title):
        top = tab.sort_values(["fraud", "claims"], ascending=[False, True]).head(10).copy()
        top["rate"] = (top["fraud"] / top["claims"]).map(pct)
        top["cum_share_of_fraud"] = (top["fraud"].cumsum() / n_fraud).map(pct)
        print(f"\n{title}")
        print(top[["claims", "fraud", "rate", "group", "cum_share_of_fraud"]].to_string())
        return top["fraud"].sum() / n_fraud

    print(f"partners with at least one decided train claim: {len(p)}; with at least one fraud: {int((p['fraud'] > 0).sum())}")
    share_all = top10(p, total, f"top 10 partners by fraud count, all 15 months ({total} frauds):")

    after = tr[tr["period"] == AFTER]
    pa = after.groupby("partner_id").agg(claims=("is_fraud", "size"), fraud=("is_fraud", "sum"), group=("partner_age", "first"))
    total_after = int(pa["fraud"].sum())
    share_after = top10(pa, total_after, f"top 10 partners by fraud count, May-June 2026 only ({total_after} frauds):")
    after_new_fraud = int(after[after["partner_age"] == "new"]["is_fraud"].sum())
    n_after_fraud_partners = int((pa["fraud"] > 0).sum())

    counts = new_outlet_counts(tr)
    print(f"\n{counts['definition']}:")
    print(f"  {counts['new_outlets']} new outlets -> NO confirmed fraud: {counts['new_outlets_without_fraud']}, at least one: "
          f"{counts['new_outlets_with_fraud']}")
    print(f"  for contrast, established outlets with a decided claim since 1 May 2026: {counts['established_outlets']} -> at least one "
          f"confirmed fraud: {counts['established_outlets_with_fraud']}")

    print("\ndoes a partner's fraud record carry over the 1 May change?")
    before = tr[tr["period"] == BEFORE]
    fraud_before = set(before[before["is_fraud"] == 1]["partner_id"])
    fraud_after = set(after[after["is_fraud"] == 1]["partner_id"])
    carried = after[after["partner_id"].isin(fraud_before)]
    earlier = before[before["partner_id"].isin(fraud_after)]
    print(f"  partners with a fraud claim before 1 May: {len(fraud_before)}; their May-June record: {len(carried)} claims, "
          f"{int(carried['is_fraud'].sum())} fraud ({pct(carried['is_fraud'].mean())})")
    print(f"  partners with a fraud claim in May-June: {len(fraud_after)}; their record before 1 May: {len(earlier)} claims, "
          f"{int(earlier['is_fraud'].sum())} fraud")
    print(f"  partners with fraud in both periods: {len(fraud_before & fraud_after)}")

    zero, some, n = counts["new_outlets_without_fraud"], counts["new_outlets_with_fraud"], counts["new_outlets"]
    so_what(f"Both people are right. Fraud is concentrated in a few outlets (top 10 of {len(p)} partners hold {pct(share_all)} over "
            f"15 months; in May-June all {total_after} frauds come from {n_after_fraud_partners} partners, {after_new_fraud} of them "
            f"from new partners). But {zero} of the {n} new outlets (decided claims since 1 May) have no confirmed fraud; only {some} have, so a "
            f"'new partner' rule would hold up {zero} clean outlets to reach {some}. The catch for modelling: the outlets changed. "
            f"The {len(fraud_before)} partners with fraud before May produced {int(carried['is_fraud'].sum())} fraud in {len(carried)} "
            f"May-June claims, and the May-June fraud partners had {int(earlier['is_fraud'].sum())} fraud before May. A partner's "
            "pre-May fraud history does not predict May-June; only history built up inside the new regime can.")


# 5 ------------------------------------------------------------------------------------------
def item5(tr):
    section("5. Repeated serials (normalised serial, decided train claims only)")
    n_claims = tr.groupby("serial_norm")["claim_id"].size()
    repeated = n_claims[n_claims > 1]
    print(f"distinct serials: {len(n_claims)}; appearing in more than one claim: {len(repeated)} "
          f"(covering {int(repeated.sum())} of {len(tr)} claims); max claims on one serial: {int(n_claims.max())}")

    # "Seen before" = another claim with the same serial was first submitted strictly earlier.
    first_seen = tr.groupby("serial_norm")[TIME_COL].transform("min")
    tr = tr.assign(serial_seen_before=np.where(tr[TIME_COL] > first_seen, "seen before", "first time"))
    print()
    print(rates(tr, "serial_seen_before").to_string())
    print(rates(tr, ["period", "serial_seen_before"]).to_string())

    pairs = tr[["claim_id", "serial_norm", TIME_COL, "partner_id", "sku"]].merge(
        tr[["serial_norm", TIME_COL, "partner_id", "sku"]], on="serial_norm", suffixes=("", "_prev"))
    pairs = pairs[pairs[TIME_COL + "_prev"] < pairs[TIME_COL]]
    flags = pairs.assign(other_partner=pairs["partner_id"] != pairs["partner_id_prev"],
                         other_sku=pairs["sku"] != pairs["sku_prev"]).groupby("claim_id")[["other_partner", "other_sku"]].any()
    seen = tr[tr["serial_seen_before"] == "seen before"].join(flags, on="claim_id")
    print("\namong claims whose serial was seen before:")
    for col, text in [("other_partner", "earlier claim came from a different partner"), ("other_sku", "earlier claim was for a different SKU")]:
        g = seen.groupby(col)["is_fraud"].agg(["size", "sum", "mean"])
        for flag, row in g.iterrows():
            print(f"  {text}: {str(flag):<5} claims {int(row['size']):>5}  fraud {int(row['sum']):>3}  rate {pct(row['mean'])}")

    r = tr.groupby("serial_seen_before")["is_fraud"].mean()
    same_sku = seen[~seen["other_sku"]]
    so_what(f"Serials do not identify a unit in this data. {len(repeated)} of {len(n_claims)} serials appear in more than one claim, "
            f"and of the {len(seen)} repeat claims {int(seen['other_sku'].sum())} are for a different product and "
            f"{int(seen['other_partner'].sum())} come from a different partner than an earlier claim, which real serials could not do "
            f"at this rate. Fraud is {pct(r['seen before'])} on a seen-before serial against {pct(r['first time'])} on a first-time "
            f"one: no usable signal. The only hint is same serial and same product ({len(same_sku)} claims, "
            f"{int(same_sku['is_fraud'].sum())} fraud), too few to trust. Do not sell 'duplicate serial' to Kestrel as a rule.")


# 6 ------------------------------------------------------------------------------------------
def item6(tr):
    section("6. inspector_note: leaky words? (report only)")
    note = tr["inspector_note"].str.lower()
    has_note = note != ""
    print(f"claims with a note: {int(has_note.sum())} of {len(tr)}; distinct note texts: {note[has_note].nunique()}")
    print("\nnotes containing an obviously leaky word:")
    leaky_total = 0
    for stem in LEAKY_STEMS:
        hit = note.str.contains(stem, regex=False)
        leaky_total += int(hit.sum())
        print(f"  '{stem}': {int(hit.sum())} notes ({int(tr[hit]['is_fraud'].sum())} on fraud claims)")

    print("\nshare of claims with NO note, by label:")
    blank = tr.assign(no_note=~has_note).groupby("is_fraud")["no_note"].agg(["size", "mean"])
    for lab, row in blank.iterrows():
        print(f"  is_fraud={lab}: {pct(row['mean'])} of {int(row['size'])}")

    print("\nevery distinct note text:")
    t = rates(tr[has_note], "inspector_note").sort_values("claims", ascending=False)
    print(t.to_string())

    print("\ntop words (share of notes in that class containing the word):")
    tables = {}
    for lab, name in [(1, "fraud"), (0, "non-fraud")]:
        notes = note[has_note & (tr["is_fraud"] == lab)]
        c = Counter(w for n in notes for w in set(re.findall(r"[a-z]+", n)))
        tables[f"{name} ({len(notes)} notes)"] = pd.Series(c) / len(notes)
    words = pd.concat(tables, axis=1).fillna(0)  # the whole vocabulary: the notes use very few words
    words["difference"] = words.iloc[:, 0] - words.iloc[:, 1]
    words = words.rename_axis("word").reset_index().sort_values(["difference", "word"], ascending=[False, True]).set_index("word")  # ties in a fixed order
    print(words.map(pct).to_string())
    widest = words["difference"].abs().idxmax()

    note_rates = tr[has_note].groupby("inspector_note")["is_fraud"].mean()
    so_what(f"No leak. None of the leaky words appears ({leaky_total} hits); the notes are {note[has_note].nunique()} stock phrases whose "
            f"fraud rates all sit between {pct(note_rates.min())} and {pct(note_rates.max())}, and the widest gap in word use between "
            f"fraud and non-fraud notes is {abs(words.loc[widest, 'difference']) * 100:.1f} points ('{widest}'), on only "
            f"{int((has_note & (tr['is_fraud'] == 1)).sum())} fraud notes. The text carries little signal either way, which fits rule 4. "
            "Fraud claims more often have no note at all, but that follows from the May change (small claims stopped being "
            "inspected, see item 8) rather than from anything the inspector wrote.")


# 7 ------------------------------------------------------------------------------------------
def item7(tr):
    section("7. Baseline: predict every claim as 'not fraud'")
    rows = []
    for name, df in [("all training labels", tr), ("train period (before 1 May 2026)", tr[tr["period"] == BEFORE]),
                     ("May-June 2026", tr[tr["period"] == AFTER])]:
        rows.append({"set": name, "claims": len(df), "fraud missed": int(df["is_fraud"].sum()),
                     "accuracy": pct(1 - df["is_fraud"].mean()),
                     "fraud_rs_paid_out": int(df[df["is_fraud"] == 1]["claim_amount_inr"].sum())})
    t = pd.DataFrame(rows).set_index("set")
    print(t.to_string())
    so_what(f"A model that never flags anything scores {t.loc['all training labels', 'accuracy']} accuracy on all labels and "
            f"{t.loc['May-June 2026', 'accuracy']} on May-June, already above the board's 97% target while catching zero fraud and "
            f"letting Rs {t.loc['May-June 2026', 'fraud_rs_paid_out']:,} through in two months. Accuracy cannot tell a useless model "
            "from a good one; this is the number to show Ritu when replacing the KPI.")


# 8 ------------------------------------------------------------------------------------------
def item8(tr, te):
    section("8. Test set (Jul-Sep 2026) vs train from 1 May 2026 (May-June, decided claims only)")
    val = tr[tr["period"] == AFTER]
    train_partners = set(tr["partner_id"])
    train_serials = set(tr["serial_norm"])
    common_desc = tr["claim_description"].value_counts()
    common_desc = set(common_desc[common_desc >= 100].index)
    last_train = tr[TIME_COL].max()

    def profile(df, months):
        small = df["claim_amount_inr"] < SMALL_CLAIM
        insp = df["partner_inspected"] == "Y"
        q = df["claim_amount_inr"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        return {
            "claims": f"{len(df)}",
            "claims per month": f"{len(df) / months:.0f}",
            "partner_inspected = Y": pct(insp.mean()),
            "  inspected, claims under Rs 2,000": pct(insp[small].mean()),
            "  inspected, claims Rs 2,000 and above": pct(insp[~small].mean()),
            "photo_attached = Y": pct((df["photo_attached"] == "Y").mean()),
            "inspector_note blank": pct((df["inspector_note"] == "").mean()),
            "claims under Rs 2,000": pct(small.mean()),
            "amount Rs: p10": f"{q[0.1]:.0f}", "amount Rs: p25": f"{q[0.25]:.0f}", "amount Rs: median": f"{q[0.5]:.0f}",
            "amount Rs: p75": f"{q[0.75]:.0f}", "amount Rs: p90": f"{q[0.9]:.0f}",
            "amount Rs: mean": f"{df['claim_amount_inr'].mean():.0f}", "amount Rs: max": f"{df['claim_amount_inr'].max():.0f}",
            "claims from new partners (1 Oct 2025 cut)": pct((df["partner_age"] == "new").mean()),
            "distinct partners": f"{df['partner_id'].nunique()}",
            "customer_prior_claims >= 2": pct((df["customer_prior_claims"] >= 2).mean()),
            "days_since_purchase median": f"{df['days_since_purchase'].median():.0f}",
            "source = crm": pct((df["source"] == "crm").mean()),
            "claim_description not one of the common texts": f"{int((~df['claim_description'].isin(common_desc)).sum())}",
        }

    cmp = pd.DataFrame({"train May-Jun 2026": profile(val, 2), "test Jul-Sep 2026": profile(te, 3)})
    print(cmp.to_string())

    fam = pd.DataFrame({"train May-Jun": val["family"].value_counts(normalize=True), "test": te["family"].value_counts(normalize=True)})
    print(f"\nproduct family mix: largest difference in share = {pct((fam['train May-Jun'] - fam['test']).abs().max())}")

    print("\nthings the test set has that the model will have no history for:")
    no_hist = ~te["partner_id"].isin(train_partners)
    late = te["onboarded_date"] > last_train
    print(f"  test claims from partners with no decided train claim: {int(no_hist.sum())} of {len(te)} ({pct(no_hist.mean())}), "
          f"from {te[no_hist]['partner_id'].nunique()} partners")
    print(f"  of which partners onboarded after the last train claim ({last_train.date()}): "
          f"{te[late]['partner_id'].nunique()} partners, {int(late.sum())} claims")
    new_te = te[te["partner_age"] == "new"]
    print(f"  test claims from new partners (1 Oct 2025 cut): {len(new_te)}, of which from partners with no decided train claim: "
          f"{int((~new_te['partner_id'].isin(train_partners)).sum())}")
    print(f"  test claims whose serial appears in a decided train claim: {int(te['serial_norm'].isin(train_serials).sum())} ({pct(te['serial_norm'].isin(train_serials).mean())})")
    print(f"  test claim_ids repeated: {int(te['claim_id'].duplicated().sum())} (n_submissions is always {int(te['n_submissions'].max())})")
    print(f"  gap between last train claim and last test claim: {(te[TIME_COL].max() - last_train).days} days "
          "(labels for July-September fraud will not exist when those claims are scored)")

    insp_v, insp_t = (val["partner_inspected"] == "Y").mean(), (te["partner_inspected"] == "Y").mean()
    so_what(f"On the surface the test set looks like May-June: inspected {pct(insp_t)} vs {pct(insp_v)}, the same photo share and "
            "the same amount distribution, so a model validated on May-June is being tested on like-for-like claims. The difference "
            f"that can hurt is history: {int(no_hist.sum())} test claims ({pct(no_hist.mean())}) come from partners the model has "
            f"never seen a decided claim from, including {te[late]['partner_id'].nunique()} partners onboarded after training ends, and "
            "the fraud in May-June sat in exactly that kind of young outlet. Partner-history features will be blank where they matter "
            "most, so the model needs a sensible fallback for partners with little or no history.")


def main():
    tr = add_columns(pd.read_parquet(DATA / "clean_train.parquet"))
    te = add_columns(pd.read_parquet(DATA / "clean_test.parquet"))
    partners = pd.read_csv(DATA / "partners.csv", parse_dates=["onboarded_date"])
    print("Kestrel warranty claims - exploration of the cleaned training data (src/explore.py)")
    print(f"train: {len(tr)} decided claims, {int(tr['is_fraud'].sum())} fraud ({pct(tr['is_fraud'].mean())}), "
          f"{tr[TIME_COL].min().date()} to {tr[TIME_COL].max().date()}; time axis = {TIME_COL}")
    print(f"test: {len(te)} claims, {te[TIME_COL].min().date()} to {te[TIME_COL].max().date()} (used in items 2 and 8 only)")
    item1(tr)
    item2(tr, te)
    item3(tr, partners)
    item4(tr)
    item5(tr)
    item6(tr)
    item7(tr)
    item8(tr, te)


if __name__ == "__main__":
    with masked_report("explore"):
        main()
