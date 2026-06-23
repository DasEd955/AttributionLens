"""Provenance Guard — Flask application.

Milestone 3 scope: the app skeleton, the ``POST /submit`` route with input
validation, and Signal 1 (LLM classification) wired in. The pieces that arrive
in later milestones are present only as clearly-marked placeholders so the
response keeps the shape of the Section 3 contract without pretending to be
finished:

  * Signal 2 — stylometric heuristics       -> Milestone 4
  * Confidence scorer / verdict bands        -> Milestone 4
  * Transparency label generator             -> Milestone 5
  * POST /appeal, GET /content, audit log    -> Milestone 5
  * Flask-Limiter rate limiting              -> Milestone 5

Input bounds (Section 9) ARE enforced now, because validation belongs to the
/submit route itself.
"""

from __future__ import annotations

import logging
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

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
            "status": "classified",
            "warnings": ["text_below_min_length"] if too_short else [],
        }
        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
