"""Clean the Kestrel claims pack.

Reads data/*.csv, prints the inspections the cleaning rules are based on (a-e),
then writes data/clean_train.parquet, data/clean_test.parquet and
data/undecided.parquet.

Time axis: first_submitted_at (the first time a claim_id was submitted) is the timestamp
for all ordering, the train/validation split and history features. submitted_at is the
latest copy's timestamp and is kept for reference only. n_submissions is for reporting
only and must not be used as a model feature.

Run from the repo root:  python src/clean.py
"""
from pathlib import Path
import re

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
TIME_COL = "first_submitted_at"  # the only timestamp to order, split or build history on
VALID_START = pd.Timestamp("2026-05-01")  # CLAUDE.md rule 1: validation = May-June 2026
SERIAL_OK = re.compile(r"^KH\d{9}$")
TEXT_COLS = ["claim_description", "inspector_note"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 45)


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load_claims(name):
    # Everything as text first so copies of a claim can be compared exactly as exported.
    df = pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], format="%Y-%m-%d %H:%M")
    return df


def normalise_serial(s):
    """Uppercase and remove whitespace, dashes and dots. No O/0-style letter fixes (see section c)."""
    return s.str.upper().str.replace(r"[\s\-.]", "", regex=True)


def serial_shape(s):
    return s.str.replace(r"\d", "9", regex=True)


# ----------------------------------------------------------------------------------------------
# a) repeated claim_ids
# ----------------------------------------------------------------------------------------------
def inspect_duplicates(raw):
    section("a) Repeated claim_ids in train.csv")
    other = [c for c in raw.columns if c != "claim_id"]
    sizes = raw.groupby("claim_id").size()
    rep_ids = sizes[sizes > 1].index
    rep = raw[raw["claim_id"].isin(rep_ids)]
    print(f"claim_ids appearing more than once: {len(rep_ids)}  (rows involved: {len(rep)})")
    print(f"copies per repeated id: {sizes[sizes > 1].value_counts().sort_index().to_dict()}")
    print(f"rows that are identical in every column to another row: {int(raw.duplicated(keep=False).sum())}")

    differs = (rep.groupby("claim_id")[other].nunique() > 1).sum()
    print("\nnumber of repeated ids in which each column differs between copies:")
    print(differs.to_string())
    print(f"\nlabels change between copies: {int(differs['is_fraud'])} ids")
    print(f"timestamps change between copies: {int(differs['submitted_at'])} ids")

    g = rep.groupby("claim_id")["submitted_at"]
    gap = g.max() - g.min()
    gap_days = gap.dt.total_seconds() / 86400
    print("\ngap between earliest and latest copy, in days:")
    print(gap_days.describe().round(2).to_string())
    print(f"gaps that are an exact whole number of days (same clock time): {int((gap_days % 1 == 0).sum())} of {len(gap_days)}")
    part_day = gap_days % 1 != 0
    print(f"of the {int(part_day.sum())} part-day gaps, latest copy stamped at the very last train timestamp "
          f"({raw['submitted_at'].max()}): {int((g.max()[part_day] == raw['submitted_at'].max()).sum())}")
    print(f"ids whose latest timestamp is shared by two copies (no unique latest): {int((g.max() == g.min()).sum())}")
    in_order = rep.groupby("claim_id")["submitted_at"].apply(lambda s: s.is_monotonic_increasing)
    print(f"ids where the latest copy is also the last one in file order: {int(in_order.sum())} of {len(in_order)}")
    straddle = (g.min() < VALID_START) & (g.max() >= VALID_START)
    print(f"ids with one copy before {VALID_START.date()} and another on/after it (would straddle train/validation): {int(straddle.sum())}")

    # Prefer examples where something other than the timestamp changes, if any exist.
    non_ts = [c for c in other if c != "submitted_at"]
    odd_ids = rep.groupby("claim_id")[non_ts].nunique().gt(1).any(axis=1)
    odd_ids = list(odd_ids[odd_ids].index[:5])
    fill = [i for i in pd.Series(rep_ids).sample(5, random_state=0) if i not in odd_ids]
    examples = (odd_ids + fill)[:5]
    print(f"\n5 concrete examples (ids with a non-timestamp difference shown first: {len(odd_ids)} found):")
    for cid in examples:
        print(f"\n--- {cid} ---")
        print(rep[rep["claim_id"] == cid].sort_values("submitted_at").drop(columns="claim_id").to_string())


def dedupe(df):
    """One row per claim_id: the latest copy's columns, timed by its first submission (TIME_COL)."""
    df = df.sort_values(["claim_id", "submitted_at"], kind="stable")
    g = df.groupby("claim_id")
    df["n_submissions"] = g["claim_id"].transform("size")
    df["first_submitted_at"] = g["submitted_at"].transform("min")
    df = df.drop_duplicates("claim_id", keep="last")
    return df.sort_values([TIME_COL, "claim_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# b) labels
# ----------------------------------------------------------------------------------------------
def inspect_labels(raw, claims):
    section("b) is_fraud by source")
    bad = set(raw["is_fraud"].unique()) - {"0", "1", ""}
    print(f"is_fraud values other than 1 / 0 / blank: {sorted(bad) if bad else 'none'}")
    for name, df in [("raw rows", raw), ("one row per claim_id", claims)]:
        lab = df["is_fraud"].replace("", "blank")
        tab = pd.crosstab(df["source"], lab, margins=True).reindex(columns=["1", "0", "blank", "All"], fill_value=0)
        print(f"\n{name}:")
        print(tab.to_string())

    crm = claims[claims["source"] == "crm"]
    zoho = claims[claims["source"] == "legacy_zoho"]
    blank_share = (crm["is_fraud"] == "").mean()
    print(f"\nCRM claims still undecided (blank): {blank_share:.2%}")
    print(f"Zoho claims blank: {int((zoho['is_fraud'] == '').sum())} of {len(zoho)} "
          "-> impossible for real data; undecided Zoho cases were exported as 0 (email thread).")
    print(f"If Zoho had the CRM's undecided share, about {blank_share * len(zoho):.0f} of its "
          f"{int((zoho['is_fraud'] == '0').sum())} zeros would really be undecided (assumption, not a measurement).")
    late_zoho = (raw["source"] == "legacy_zoho") & (raw["submitted_at"] >= "2025-10-01")
    first_seen = raw.groupby("claim_id")["submitted_at"].transform("min")
    print(f"source vs 1 Oct 2025 cutover: legacy_zoho rows dated on/after it: {int(late_zoho.sum())} "
          f"(of which re-submissions of a claim first submitted before it: {int((late_zoho & (first_seen < '2025-10-01')).sum())}), "
          f"crm rows dated before it: {int(((raw['source'] == 'crm') & (raw['submitted_at'] < '2025-10-01')).sum())}")

    print("\nfraud rate by n_submissions (decided claims only):")
    dec = claims[claims["is_fraud"] != ""].assign(y=lambda d: d["is_fraud"].astype(int))
    print(dec.groupby("n_submissions")["y"].agg(claims="size", fraud="sum", rate="mean").round(4).to_string())


# ----------------------------------------------------------------------------------------------
# c) serials
# ----------------------------------------------------------------------------------------------
def inspect_serials(train_raw, test_raw):
    section("c) product_serial")
    both = pd.concat([train_raw["product_serial"], test_raw["product_serial"]], ignore_index=True)
    messy = both[~both.str.match(SERIAL_OK)]
    print(f"serials not already in the clean form KH + 9 digits: {len(messy)} of {len(both)} ({len(messy) / len(both):.1%})")
    print("\nshapes (every digit shown as 9, quotes show spaces):")
    shapes = serial_shape(both).map(repr).value_counts()
    print(shapes.to_string())

    print("\n20 messy examples (spread across the messy shapes):")
    per_shape = messy.groupby(serial_shape(messy), group_keys=False).apply(lambda s: s.sample(min(len(s), 20), random_state=0))
    take = -(-20 // max(serial_shape(messy).nunique(), 1))
    ex = per_shape.groupby(serial_shape(per_shape), group_keys=False).head(take).head(20)
    for v in ex:
        print(f"  {v!r:>18}  ->  {normalise_serial(pd.Series([v])).iloc[0]}")

    chars = pd.Series(list("".join(both))).value_counts()
    print("\nevery non-digit character found in serials:", {repr(k): int(v) for k, v in chars.items() if not k.isdigit()})

    norm = normalise_serial(both)
    still_bad = norm[~norm.str.match(SERIAL_OK)]
    letters_in_digits = norm.str[2:].str.contains(r"[A-Z]").sum()
    print(f"after uppercase + removing spaces/dashes/dots: {len(still_bad)} serials still not KH + 9 digits")
    print(f"letters inside the 9-digit part (O for 0, I for 1, ...): {int(letters_in_digits)} "
          "-> O/0-style fixes are applied only if this is non-zero; it is not applied here.")

    print("\ndistinct serials before -> after normalising:")
    for name, s in [("train rows", train_raw["product_serial"]), ("test rows", test_raw["product_serial"]), ("train + test", both)]:
        print(f"  {name:<13} {s.nunique():>6} -> {normalise_serial(s).nunique():>6}   (rows: {len(s)})")


# ----------------------------------------------------------------------------------------------
# d) / e) checks against the reference tables
# ----------------------------------------------------------------------------------------------
def inspect_onboarding(sets, partners):
    section("d) Claims submitted before the partner's onboarded_date (reported only, not fixed)")
    print(f"partners.csv onboarded_date range: {partners['onboarded_date'].min().date()} to {partners['onboarded_date'].max().date()}; "
          f"partners onboarded after the last train claim (2026-06-30): {int((partners['onboarded_date'] > '2026-06-30').sum())}")
    for name, df in sets:
        early = df[df[TIME_COL] < df["onboarded_date"]]
        days = (early["onboarded_date"] - early[TIME_COL]).dt.days
        line = f"{name}: {len(early)} of {len(df)} claims, from {early['partner_id'].nunique()} partners"
        if len(early):
            line += f"; days before onboarding min/median/max = {days.min()}/{days.median():.0f}/{days.max()}"
        if "is_fraud" in df.columns and len(early):
            line += f"; labels {early['is_fraud'].replace('', 'blank').value_counts().to_dict()}"
        print(line)


def inspect_amounts(sets):
    section("e) Claim amount vs list price, days_since_purchase vs warranty (counts only)")
    for name, df in sets:
        ratio = df["claim_amount_inr"] / df["list_price_inr"]
        out = df["days_since_purchase"] > df["warranty_months"] * 30
        print(f"{name} ({len(df)} claims):")
        print(f"  claim_amount_inr > list price: {int((ratio > 1).sum())};  > 1.5x: {int((ratio > 1.5).sum())};  "
              f"> 2x: {int((ratio > 2).sum())};  max ratio: {ratio.max():.2f}")
        print(f"  days_since_purchase > warranty_months*30: {int(out.sum())};  "
              f"max days by warranty_months: {df.groupby('warranty_months')['days_since_purchase'].max().to_dict()}")


# ----------------------------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------------------------
def finish(df, partners, products, name):
    """is_zoho, numeric types, normalised serial, joins. Prints the row count after each step."""
    df["is_zoho"] = (df["source"] == "legacy_zoho").astype("int8")
    print(f"{name}: after adding is_zoho: {len(df)} rows (is_zoho=1: {int(df['is_zoho'].sum())})")

    df["days_since_purchase"] = df["days_since_purchase"].astype(int)
    df["customer_prior_claims"] = df["customer_prior_claims"].astype(int)
    df["claim_amount_inr"] = df["claim_amount_inr"].astype(float)
    df["serial_norm"] = normalise_serial(df["product_serial"])

    df = df.merge(partners, on="partner_id", how="left", validate="many_to_one")
    print(f"{name}: after joining partners: {len(df)} rows (no partner match: {int(df['onboarded_date'].isna().sum())})")
    df = df.merge(products, on="sku", how="left", validate="many_to_one")
    print(f"{name}: after joining products: {len(df)} rows (no product match: {int(df['list_price_inr'].isna().sum())})")
    return df


def main():
    train_raw = load_claims("train.csv")
    test_raw = load_claims("test_unlabelled.csv")
    partners = pd.read_csv(DATA / "partners.csv", parse_dates=["onboarded_date"])
    products = pd.read_csv(DATA / "products.csv")
    print(f"loaded: train {len(train_raw)} rows, test {len(test_raw)} rows, partners {len(partners)}, products {len(products)}")
    print(f"timestamps parsed: train {train_raw['submitted_at'].min()} to {train_raw['submitted_at'].max()}, "
          f"test {test_raw['submitted_at'].min()} to {test_raw['submitted_at'].max()}")

    inspect_duplicates(train_raw)
    train = dedupe(train_raw)
    test = dedupe(test_raw)
    inspect_labels(train_raw, train)
    inspect_serials(train_raw, test_raw)

    section("Cleaning steps and row counts")
    print(f"train: raw {len(train_raw)} rows -> after keeping the latest copy per claim_id: {len(train)} rows")
    print(f"test: raw {len(test_raw)} rows -> after keeping the latest copy per claim_id: {len(test)} rows "
          f"(repeated ids in test: {int(test_raw['claim_id'].duplicated().sum())}, so n_submissions is always 1 there)")
    train = finish(train, partners, products, "train")
    test = finish(test, partners, products, "test")

    inspect_onboarding([("train", train), ("test", test)], partners)
    inspect_amounts([("train", train), ("test", test)])

    section("Split off undecided claims and write outputs")
    undecided = train[train["is_fraud"] == ""].drop(columns="is_fraud").reset_index(drop=True)
    train = train[train["is_fraud"] != ""].reset_index(drop=True)
    train["is_fraud"] = train["is_fraud"].astype("int8")
    print(f"train: after dropping blank is_fraud: {len(train)} rows (fraud {int(train['is_fraud'].sum())}, "
          f"not fraud {int((train['is_fraud'] == 0).sum())}); set aside as undecided: {len(undecided)} rows")

    moved = (train[TIME_COL] < VALID_START) & (train["submitted_at"] >= VALID_START)
    print()
    print(f"split on {TIME_COL} at {VALID_START.date()} (claims re-submitted across the date stay with "
          f"their first submission: {int(moved.sum())} decided claims):")
    period = (train[TIME_COL] >= VALID_START).map({False: "train period (before)", True: "validation period (on/after)"})
    print(train.groupby(period)["is_fraud"].agg(claims="size", fraud="sum", rate="mean").round(4).to_string())
    print(f"rows ordered by {TIME_COL}: {train[TIME_COL].is_monotonic_increasing and test[TIME_COL].is_monotonic_increasing}")

    train.to_parquet(DATA / "clean_train.parquet", index=False)
    test.to_parquet(DATA / "clean_test.parquet", index=False)
    undecided.to_parquet(DATA / "undecided.parquet", index=False)
    for f, df in [("clean_train.parquet", train), ("clean_test.parquet", test), ("undecided.parquet", undecided)]:
        print(f"wrote data/{f}: {len(df)} rows x {df.shape[1]} columns")
    print("columns:", list(train.columns))


if __name__ == "__main__":
    main()
