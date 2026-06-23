"""Provenance Guard — Flask application.

Milestone 3 scope: the app skeleton, the ``POST /submit`` route with input
validation, and Signal 1 (LLM classification) wired in. The pieces that arrive
in later milestones are present only as clearly-marked placeholders so the
response keeps the shape of the Section 3 contract without pretending to be
finished:

  * Signal 2 — stylometric heuristics       -> Milestone 4
  * Confidence scorer / verdict bands        -> Milestone 4
  * Transparency label generator             -> Milestone 5
  * POST /appeal, GET /content               -> Milestone 5
  * Flask-Limiter rate limiting              -> Milestone 5

The audit log (Section 11) IS live now: every /submit writes a structured
SQLite row and GET /log reads the most recent rows back for the demo view.

Input bounds (Section 9) ARE enforced now, because validation belongs to the
/submit route itself.
"""

from __future__ import annotations

import hashlib
import logging
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from audit_log import get_log, init_db, record_decision
from signals.llm_signal import classify_with_llm

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Input bounds (planning.md Section 9). Below MIN the stylometric signal is
# unreliable; above MAX we reject to protect against cost abuse / huge payloads.
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 20_000


def create_app() -> Flask:
    """Application factory. Lets tests build an isolated app instance."""
    app = Flask(__name__)

    # Ensure the audit log table exists before the first request (Section 11).
    init_db()

    @app.get("/health")
    def health():
        # groq_available is reported per-call in /submit; here we only confirm
        # the service is up. A real check would ping Groq — deferred for now.
        return jsonify({"status": "ok", "groq_available": None}), 200

    @app.post("/submit")
    def submit():
        # --- Input validation (Section 3 / Section 9) -> 400 on bad input ----
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400

        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."}), 400

        creator_id = body.get("creator_id")  # optional; passed through untouched
        too_short = len(text) < MIN_TEXT_LENGTH  # flagged, not rejected (Section 9)

        # --- Signal 1: LLM classification -----------------------------------
        llm = classify_with_llm(text)

        # --- Signals/scoring not yet built (Milestones 4-5) -----------------
        # Stub stylometric signal and scorer so the response keeps its shape.
        # These placeholder values are intentionally inert, NOT real readings.
        stylometric_stub = {"p_ai": None, "features": {}}

        # Both signals failing means we cannot honestly classify -> 503.
        # In M3 only the LLM signal exists, so "both fail" == LLM unavailable.
        if not llm.available:
            return jsonify({"error": "Classification temporarily unavailable.", "status": "unavailable"}), 503

        content_id = str(uuid.uuid4())
        status = "classified"

        # --- Audit log (Section 11) -----------------------------------------
        # Write a structured row for EVERY classified submission. We store the
        # content hash rather than the raw text (Section 11) so the log stays
        # queryable without retaining creator content verbatim. verdict /
        # confidence / combined_score are still None here; the M4 scorer fills
        # them, and record_decision already accepts them as optional.
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record_decision(
            content_id=content_id,
            status=status,
            creator_id=creator_id if isinstance(creator_id, str) else None,
            content_hash=content_hash,
            p_ai_llm=llm.p_ai,
            llm_rationale=llm.rationale,
            llm_available=llm.available,
        )

        response = {
            "content_id": content_id,
            "verdict": None,           # filled by confidence scorer (M4)
            "combined_score": None,    # filled by confidence scorer (M4)
            "confidence": None,        # filled by confidence scorer (M4)
            "label": {                 # filled by label generator (M5)
                "variant": None,
                "text": None,
            },
            "signals": {
                "llm": llm.to_dict(),
                "stylometric": stylometric_stub,
            },
            "status": status,
            "warnings": ["text_below_min_length"] if too_short else [],
        }
        return jsonify(response), 200

    @app.get("/log")
    def log():
        # Demo / grading visibility into the audit log (Section 11). No auth —
        # in a real system this would be access-controlled; here it just exposes
        # the most recent structured entries so the log can be shown.
        limit = request.args.get("limit", default=50, type=int)
        limit = max(1, min(limit, 500))  # keep the response bounded
        return jsonify({"entries": get_log(limit=limit)}), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
