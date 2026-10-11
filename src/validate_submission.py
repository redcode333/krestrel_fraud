"""Check predictions.csv against every requirement of the brief. Exits with an error on the first failure.

Run from the repo root:  python src/validate_submission.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
EXPECTED_ROWS = 2252


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)
    print(f"ok    {message}")


def main():
    check(PREDICTIONS.exists(), "predictions.csv exists in the repository root")
    pred = pd.read_csv(PREDICTIONS)
    check(list(pred.columns) == ["claim_id", "score"], "columns are exactly claim_id,score")
    check(len(pred) == EXPECTED_ROWS, f"exactly {EXPECTED_ROWS:,} rows (found {len(pred):,})")
    check(pred["claim_id"].is_unique, "one row per claim_id")
    check(pd.api.types.is_float_dtype(pred["score"]), "score is numeric")
    check(not pred["score"].isna().any(), "no NaN score")
    check(bool(np.isfinite(pred["score"]).all()), "no infinite score")
    check(bool(((pred["score"] > 0) & (pred["score"] < 1)).all()),
          f"every score is strictly between 0 and 1 (min {pred['score'].min():.6f}, max {pred['score'].max():.6f})")
    check(pred["score"].nunique() > 1, f"scores are not all the same ({pred['score'].nunique():,} distinct values)")

    check(SAMPLE.exists(), "data/sample_submission.csv is available to compare against")
    sample = pd.read_csv(SAMPLE)
    check(list(sample.columns) == list(pred.columns), "same columns as sample_submission.csv")
    check(set(pred["claim_id"]) == set(sample["claim_id"]), "same claim_ids as sample_submission.csv")
    check(list(pred["claim_id"]) == list(sample["claim_id"]), "same order as sample_submission.csv")
    test = pd.read_csv(ROOT / "data" / "test_unlabelled.csv")
    check(set(pred["claim_id"]) == set(test["claim_id"]), "one row for every row of test_unlabelled.csv")

    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines() if (ROOT / ".gitignore").exists() else []
    rules = [line.strip() for line in ignore if line.strip() and not line.strip().startswith("#")]
    hiding = [r for r in rules if not r.startswith("!") and r.lstrip("/") in ("predictions.csv", "*.csv", "*")]
    check(not hiding, "predictions.csv is not excluded by .gitignore")
    print(f"\nALL CHECKS PASSED: predictions.csv is valid ({len(pred):,} rows).")


if __name__ == "__main__":
    main()
