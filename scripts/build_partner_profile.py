"""Rebuild models/partner_profile.csv (real partner IDs, git-ignored) from the data pack in data/,
and models/partner_profile.example.csv (masked IDs, committed) from it.

If models/partner_profile.csv already exists, the rebuild must reproduce it exactly, byte for byte;
otherwise the script stops with an error and changes nothing.

Needs the data pack in data/ (train.csv, test_unlabelled.csv, partners.csv, products.csv). The cleaned
files are created first if they are missing. Run from the repo root:
    python scripts/build_partner_profile.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean import DATA  # noqa: E402


def main():
    needed = ["train.csv", "test_unlabelled.csv", "partners.csv", "products.csv"]
    missing = [f for f in needed if not (DATA / f).exists()]
    if missing:
        sys.exit(f"data/ is missing {missing}. Put the Kestrel data pack in data/ first. Without it the app runs on "
                 "models/partner_profile.example.csv (masked partner IDs).")
    if not all((DATA / f).exists() for f in ["clean_train.parquet", "clean_test.parquet", "undecided.parquet"]):
        print("cleaned files not found: running src/clean.py first")
        import clean
        clean.main()

    from partner_profile import EXAMPLE_PROFILE, REAL_PROFILE, build_partner_profile, csv_text, example_profile, write_profiles

    profile, overall = build_partner_profile()
    rebuilt = csv_text(profile)
    if REAL_PROFILE.exists():
        current = REAL_PROFILE.read_bytes().decode("utf-8")
        assert rebuilt == current, ("the rebuilt partner profile differs from models/partner_profile.csv. Nothing was written. "
                                    "Either the data pack changed or the file was edited by hand.")
        print(f"PASS: rebuilt from data/ and identical, byte for byte, to the existing models/partner_profile.csv "
              f"({len(profile)} partners, {len(rebuilt):,} characters).")
    else:
        print(f"models/partner_profile.csv did not exist; built it from data/ ({len(profile)} partners).")
    had_example = EXAMPLE_PROFILE.exists() and EXAMPLE_PROFILE.read_bytes().decode("utf-8")
    write_profiles(profile)
    example_text = csv_text(example_profile(profile))
    print(f"models/partner_profile.example.csv written with masked partner IDs"
          + (" (identical to the existing one)." if had_example == example_text else "."))
    print(f"partners with decided new-regime claims: {int((profile['decided_new_claims'] > 0).sum())}; with partner_rate_ratio > 1: "
          f"{int((profile['rate_ratio'] > 1).sum())}; overall new-regime fraud rate {overall:.4f}")


if __name__ == "__main__":
    main()
