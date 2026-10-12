"""predictions.csv: the checks that need no data, and the full validation script when data/ is present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import HAS_DATA, ROOT


def test_predictions_file_has_the_required_shape():
    pred = pd.read_csv(ROOT / "predictions.csv")
    assert list(pred.columns) == ["claim_id", "score"]
    assert len(pred) == 2252 and pred["claim_id"].is_unique
    assert pred["score"].notna().all() and np.isfinite(pred["score"]).all()
    assert ((pred["score"] > 0) & (pred["score"] < 1)).all() and pred["score"].nunique() > 1


def test_predictions_file_is_not_git_ignored():
    rules = [line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert not [r for r in rules if r and not r.startswith(("#", "!")) and r.lstrip("/") in ("predictions.csv", "*.csv", "*")]


@pytest.mark.skipif(not HAS_DATA, reason="needs the Kestrel data pack in data/")
def test_validation_script_passes_against_the_sample_submission(capsys):
    import validate_submission

    validate_submission.main()
    assert "ALL CHECKS PASSED" in capsys.readouterr().out
