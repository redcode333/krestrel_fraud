"""Report output with partner IDs masked (ops-policy section 10).

Every report is written twice:
  reports/raw/<name>.txt   real partner IDs, git-ignored
  reports/<name>.txt       partner IDs replaced by P001, P002, ...; this one is committed

The mapping lives in data/partner_mask.csv (git-ignored). It is drawn at random ONCE, saved,
and read back on every later run, so a real ID always gets the same masked ID in every report,
and the masked IDs cannot be rebuilt from anything in the repository. If the mapping file is
deleted, a new one is drawn and check_mask_consistency() fails until every report is regenerated.

In the masked copy, any date on a line that names a partner is removed as well: an exact
onboarding date would identify the partner to anyone holding partners.csv.

Claim-level detail (single claims, their rows, their feature values) is printed with raw_only().
It shows on the console and in reports/raw/, and is dropped from the committed copy, which may
hold summaries only.

Run from the repo root to check every report against the mapping:  python src/reporting.py
"""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import io
import random
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RAW_REPORTS = REPORTS / "raw"
MASK_FILE = ROOT / "data" / "partner_mask.csv"
PARTNER_ID = re.compile(r"SP\d{4}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
RAW_ONLY = "[raw only] "


def raw_only(text=""):
    """Print claim-level detail that must not reach the committed copy of a report."""
    for line in str(text).split("\n"):
        print(RAW_ONLY + line)


@lru_cache(maxsize=None)
def partner_mask():
    """Real partner_id -> masked ID. Created on first use only; afterwards always read from the private file."""
    ids = sorted(pd.read_csv(ROOT / "data" / "partners.csv")["partner_id"])
    bad = [i for i in ids if not PARTNER_ID.fullmatch(i)]
    if bad:
        raise ValueError(f"partner IDs not in the form SP + 4 digits, masking would miss them: {bad[:5]}")
    if not MASK_FILE.exists():
        shuffled = list(ids)
        random.SystemRandom().shuffle(shuffled)  # no seed on purpose: a seed in the repo would make the mask reversible
        pd.DataFrame({"partner_id": shuffled, "masked_id": [f"P{i:03d}" for i in range(1, len(shuffled) + 1)]}).to_csv(
            MASK_FILE, index=False)
    m = pd.read_csv(MASK_FILE)
    assert m["partner_id"].is_unique, "partner_mask.csv: a real ID appears twice"
    assert m["masked_id"].is_unique, "partner_mask.csv: two real IDs share one masked ID"
    missing = set(ids) - set(m["partner_id"])
    assert not missing, f"partner_mask.csv has no entry for {sorted(missing)[:5]}; do not edit or extend it by hand"
    return dict(zip(m["partner_id"], m["masked_id"]))


def mask_partner_ids(text):
    mapping = partner_mask()

    def swap(match):
        pid = match.group(0)
        if pid not in mapping:
            raise ValueError(f"{pid} is not in the partner mask; refusing to write a report that could leak it")
        return mapping[pid].ljust(len(pid))  # same width as the real ID so printed tables stay aligned

    lines, dropped = [], 0

    def flush_dropped():
        nonlocal dropped
        if dropped:
            lines.append(f"[{dropped} lines of claim-level detail kept out of the committed copy (policy section 10); "
                         "they are in reports/raw/]")
            dropped = 0

    for line in text.split("\n"):
        if line.startswith(RAW_ONLY):
            dropped += 1
            continue
        flush_dropped()
        if PARTNER_ID.search(line):
            line = DATE.sub("[removed] ", line)  # same width as a date
        lines.append(PARTNER_ID.sub(swap, line))
    flush_dropped()
    return "\n".join(lines)


def check_mask_consistency():
    """Assert that the same real ID maps to the same masked ID in every report.

    Each committed report must be exactly its raw copy masked with the one current mapping.
    Since all of them are tied to the same mapping, they agree with each other. Returns the
    number of reports checked.
    """
    checked = 0
    for raw in sorted(list(RAW_REPORTS.glob("*.txt")) + list(RAW_REPORTS.glob("*.md"))):
        masked = REPORTS / raw.name
        assert masked.exists(), f"reports/raw/{raw.name} has no masked copy in reports/"
        masked_text = masked.read_text(encoding="utf-8")
        assert not PARTNER_ID.search(masked_text), f"reports/{raw.name} still contains a real partner ID"
        assert mask_partner_ids(raw.read_text(encoding="utf-8")) == masked_text, (
            f"reports/{raw.name} was not masked with the current data/partner_mask.csv: a real ID would map to a "
            "different masked ID than in other reports. Regenerate this report.")
        checked += 1
    return checked


def write_masked_file(filename, text):
    """Write one report file twice: reports/raw/<filename> as given, reports/<filename> with partner IDs masked."""
    RAW_REPORTS.mkdir(parents=True, exist_ok=True)
    (RAW_REPORTS / filename).write_text(text, encoding="utf-8")
    (REPORTS / filename).write_text(mask_partner_ids(text), encoding="utf-8")
    return check_mask_consistency()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def masked_report(name):
    """Everything printed inside the block goes to the console, the raw report and the masked report."""
    buffer = io.StringIO()
    stdout = sys.stdout
    sys.stdout = _Tee(stdout, buffer)
    try:
        yield
    finally:
        sys.stdout = stdout
    RAW_REPORTS.mkdir(parents=True, exist_ok=True)
    (RAW_REPORTS / f"{name}.txt").write_text(buffer.getvalue(), encoding="utf-8")
    (REPORTS / f"{name}.txt").write_text(mask_partner_ids(buffer.getvalue()), encoding="utf-8")
    n = check_mask_consistency()
    print(f"\nsaved: reports/{name}.txt (partner IDs masked, safe to commit) and reports/raw/{name}.txt (real IDs, git-ignored)")
    print(f"partner mask check passed: {n} reports all use the one mapping in data/partner_mask.csv")


if __name__ == "__main__":
    existed = MASK_FILE.exists()
    mapping = partner_mask()
    print(f"data/partner_mask.csv: {'reused existing file' if existed else 'created now'}, {len(mapping)} partners, "
          f"{len(set(mapping.values()))} distinct masked IDs")
    print(f"partner mask check passed: {check_mask_consistency()} reports all use this one mapping")
