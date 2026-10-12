"""Kestrel claim review service: one endpoint that scores a claim, and one page that calls it.

  GET  /         the page (one self-contained HTML file: inline CSS and JS, nothing loaded from outside)
  POST /score    ONE claim as JSON -> score, risk band, whether the review rule applies, reasons, notes
  GET  /health   which partner profile is loaded, and whether the model files are in place
  GET  /examples three example claims built from the loaded partner profile

All scoring, explanation and band logic lives in src/scoring.py; this file only translates HTTP.
No paid API, no key, no data/ folder needed. A bad record gets a polite 4xx that names the field.

Start it from the repository root:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from pathlib import Path
import json
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

import scoring  # noqa: E402

log = logging.getLogger("kestrel")
app = FastAPI(title="Kestrel claim review", version="1.0", redoc_url=None)
PAGE = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")
MAX_BODY_BYTES = 100_000
PROFILE_TEXT = {
    "real": "real partner profile (models/partner_profile.csv)",
    "example": "EXAMPLE partner profile with masked outlet IDs (models/partner_profile.example.csv); real partner IDs are not in it "
               "and are scored as unknown outlets",
}


def problem(status, message, fields=None):
    """A polite error: what is wrong, and which field to fix."""
    fields = list(fields or [])
    return JSONResponse(status_code=status, content={"error": message, "field": fields[0] if fields else None, "fields": fields})


def artifacts():
    """(model, constants, partners, source), or None with the reason when the model files are not in place."""
    try:
        return scoring.load_artifacts(), None
    except FileNotFoundError as error:
        return None, str(error)


def example_claims():
    """Three example claims whose partner IDs come from whichever profile is loaded."""
    loaded, _ = artifacts()
    if loaded is None:
        return []
    partners = list(loaded[2].values())
    flagged = [p for p in partners if p["rate_ratio"] > 1]
    ordinary = [p for p in partners if p["decided_new_claims"] > 0 and p["rate_ratio"] <= 1 and p["is_new_partner"] == 0]
    new = [p for p in partners if p["decided_new_claims"] == 0 and p["is_new_partner"] == 1]
    picks = [
        ("High-risk outlet", "An outlet with confirmed fraud above the average rate since 1 May 2026.",
         max(flagged, key=lambda p: (p["rate_ratio"], p["decided_new_claims"], p["partner_id"])) if flagged else None,
         {"sku": "KH-MG-02", "product_serial": "KH482915530", "days_since_purchase": 210, "claim_amount_inr": 1450,
          "photo_attached": "N", "customer_prior_claims": 1}),
        ("Ordinary claim", "An established outlet with a clean record.",
         max(ordinary, key=lambda p: (p["decided_new_claims"], p["partner_id"])) if ordinary else None,
         {"sku": "KH-WP-01", "product_serial": "KH305518842", "days_since_purchase": 150, "claim_amount_inr": 3200,
          "photo_attached": "Y", "customer_prior_claims": 0}),
        # Made up to resemble the highest-scoring real claims from outlets with no decided history (they score up to 0.552).
        # The outlet is chosen by its activity, not its ID, so the real and the example profile pick the same one.
        ("New outlet", "A recently onboarded outlet with no decided claims yet.",
         max(new, key=lambda p: (p["claims_last_30d"], p["claims_last_7d"], p["avg_amount_last_30d"], p["partner_id"])) if new else None,
         {"sku": "KH-RH-01", "product_serial": "kh-771204963", "days_since_purchase": 215, "claim_amount_inr": 1400,
          "photo_attached": "Y", "customer_prior_claims": 1}),
    ]
    return [{"label": label, "description": description,
             "claim": {"claim_id": f"EXAMPLE-{i + 1}", "submitted_at": "2026-10-05 11:20", "partner_id": partner["partner_id"], **fields}}
            for i, (label, description, partner, fields) in enumerate(picks) if partner is not None]


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, error: StarletteHTTPException):
    return problem(error.status_code, str(error.detail))


@app.get("/health")
def health():
    loaded, reason = artifacts()
    if loaded is None:
        return JSONResponse(status_code=503, content={"status": "model files missing", "detail": reason, "paid_api": False})
    model, constants, partners, source = loaded
    return {"status": "ok", "partner_profile": source, "partner_profile_note": PROFILE_TEXT[source], "partners_in_profile": len(partners),
            "model": constants["model"], "history_frozen_on": constants["history_frozen_on"],
            "partner_activity_as_of": constants["profile_date"], "watch_score_threshold": constants.get("watch_score_threshold"),
            "paid_api": False}


@app.get("/examples")
def examples():
    return example_claims()


@app.get("/", response_class=HTMLResponse)
def page():
    loaded, reason = artifacts()
    config = {"ready": loaded is not None, "problem": reason, "examples": example_claims(),
              "profile": loaded[3] if loaded else None, "profile_note": PROFILE_TEXT[loaded[3]] if loaded else None,
              "skus": sorted(loaded[1]["products"]) if loaded else [],
              "bands": scoring.BANDS, "history_frozen_on": loaded[1]["history_frozen_on"] if loaded else None,
              "activity_as_of": loaded[1]["profile_date"] if loaded else None}
    return PAGE.replace("__CONFIG__", json.dumps(config).replace("<", "\\u003c"))


@app.post("/score")
async def score(request: Request):
    """Score ONE claim. The body is a JSON object with the claim's fields."""
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return problem(413, f"The request is too large ({len(body):,} bytes). Send one claim as a small JSON object.")
    try:
        claim = json.loads(body)
    except Exception:
        return problem(400, "The request body is not valid JSON. Send one claim as a JSON object, for example "
                            '{"partner_id": "...", "sku": "KH-MG-02", "submitted_at": "2026-10-05 11:20", "claim_amount_inr": 1450, '
                            '"days_since_purchase": 210, "customer_prior_claims": 1, "photo_attached": "N"}.')
    if not isinstance(claim, dict):
        return problem(400, "Send ONE claim as a JSON object with named fields, not a list or a single value.")
    try:
        return scoring.assess(claim)
    except scoring.ClaimError as error:
        return problem(422, f"This claim cannot be scored: {error}.", error.fields)
    except FileNotFoundError as error:
        return problem(503, f"The service is not ready: {error}")
    except Exception:   # last resort: never a stack trace or a 500 for something in the record
        log.exception("unexpected failure while scoring a claim")
        return problem(400, "This claim could not be scored. Please check the field values; nothing was saved.")
