"""Shared test setup. The app tests need no data/ folder and pass on the example partner profile alone."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

HAS_DATA = all((ROOT / "data" / name).exists() for name in ["train.csv", "test_unlabelled.csv", "sample_submission.csv",
                                                             "clean_train.parquet", "clean_test.parquet", "undecided.parquet"])
