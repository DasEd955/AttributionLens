"""app.py - Provenance Guard Flask application entry point; wires routes to signals and audit log.

Milestone 3 scope: the app skeleton, the POST /submit route with input
validation, and Signal 1 (LLM classification) wired in.

Milestone 4 scope (now live): Signal 2 (stylometric heuristics) runs alongside
Signal 1, and the confidence scorer (Section 5) combines both into a single
``combined_score``, a ``confidence``, and a ``verdict``. The /submit response
and the audit log now carry both individual signal scores and the combined
result.

Milestone 5 scope (now live): the transparency label generator (Section 7) maps
each verdict and confidence to one of the three reader-facing label variants,
and the /submit response carries the real label instead of a placeholder. The
appeals workflow (Section 8) is live: POST /appeal flips a contested decision to
``under_review``, logs the appeal, and confirms receipt, and GET /content reads
the full decision record (with any attached appeals) for a human reviewer.
Flask-Limiter rate limiting (Section 10) is not wired in this change.

The audit log (Section 11) is live: every /submit writes a structured SQLite row
and GET /log reads the most recent rows back for the demo view.

Input bounds (Section 9) are enforced now, because validation belongs to the
/submit route itself.
"""

from __future__ import annotations
import hashlib
import logging
import uuid
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from audit_log import (
    get_appeals,
    get_decision,
    get_log,
    init_db,
    record_appeal,
    record_decision,
    set_status,
)
from labels import generate_label
from scoring import score
from signals.llm_signal import classify_with_llm
from signals.stylometric_signal import analyze_stylometry

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Input bounds (planning.md Section 9). Below MIN the stylometric signal is
# unreliable; above MAX we reject to protect against cost abuse / huge payloads.
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 20_000


def create_app() -> Flask:
    """Build and return the configured Flask application instance.

    Calls init_db() to ensure the audit log table exists before any request
    is handled. Designed as a factory so tests can spin up isolated instances
    without sharing state with the module-level app object.

    Returns:
        Flask: A fully configured Flask application with /health, /submit,
               and /log routes registered.
    """
    app = Flask(__name__)

    # Ensure the audit log table exists before the first request (Section 11).
    init_db()

    @app.get("/health")
    def health():
        """Return a simple liveness check confirming the service is running.

        Returns:
            Response: JSON with ``status: "ok"`` and ``groq_available: null``
                      (Groq liveness check is deferred to a later milestone),
                      HTTP 200.
        """
        return jsonify({"status": "ok", "groq_available": None}), 200

    @app.post("/submit")
    def submit():
        """Accept a text submission, run both signals, score them, and write an audit row.

        Validates the request body against the input bounds defined in Section 9
        of planning.md, runs Signal 1 (LLM) and Signal 2 (stylometric), combines
        them with the confidence scorer (Section 5) into a ``combined_score``,
        ``confidence``, and ``verdict``, writes a structured audit log entry
        carrying both individual signal scores and the combined result, and
        returns the Section 3 response shape. The transparency label (Milestone
        5) is still a null placeholder.

        Signal 2 always runs (it is pure Python and cannot fail externally), so
        the system degrades gracefully to stylometry alone when the LLM signal is
        unavailable (Section 9). Only when BOTH signals are unavailable does the
        route return 503; in Milestone 4 the stylometric signal is always
        available, so 503 is effectively unreachable here and reserved for future
        failure modes.

        Returns:
            Response: JSON matching the Section 3 contract, HTTP 200 on success.
                      HTTP 400 on invalid input.
                      HTTP 503 only if every signal is unavailable.
        """
        # --- Input Validation (Section 3 / Section 9) -> 400 on bad input ----
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400

        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."}), 400

        creator_id = body.get("creator_id")  # Optional; passed through untouched
        too_short = len(text) < MIN_TEXT_LENGTH  # Flagged, not rejected (Section 9)

        # --- Signal 1: LLM Classification -----------------------------------
        llm = classify_with_llm(text)

        # --- Signal 2: Stylometric Heuristics (Section 4) -------------------
        # Pure Python; runs regardless of Groq availability. This is what lets
        # the system degrade to a single signal instead of failing (Section 9).
        style = analyze_stylometry(text)

        # --- Confidence Scorer (Section 5) ----------------------------------
        # Combine both signals into combined_score, confidence, and verdict.
        # When the LLM is down, the scorer caps confidence (Section 9) so the
        # lone, gameable structural signal cannot present as a confident verdict.
        scored = score(llm.p_ai, style.p_ai, llm_available=llm.available)

        # 503 is reserved for the case where NO signal is available. The
        # stylometric signal is always available, so this guards a future state.
        if not llm.available and style.p_ai is None:
            return jsonify({"error": "Classification temporarily unavailable.", "status": "unavailable"}), 503

        # --- Transparency Label Generator (Section 7) -----------------------
        # Map the verdict and confidence to one of the three fixed reader-facing
        # label variants. The label changes with the score; it is never the same
        # text regardless of the result.
        label = generate_label(scored.verdict, scored.confidence)

        content_id = str(uuid.uuid4())
        status = "classified"

        # --- Audit Log (Section 11) -----------------------------------------
        # Write a structured row for EVERY classified submission. We store the
        # content hash rather than the raw text (Section 11) so the log stays
        # queryable without retaining creator content verbatim. The row now
        # carries BOTH individual signal scores (p_ai_llm, p_ai_style) and the
        # combined result (combined_score, confidence, verdict) per Section 11.
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record_decision(
            content_id=content_id,
            status=status,
            creator_id=creator_id if isinstance(creator_id, str) else None,
            content_hash=content_hash,
            p_ai_llm=llm.p_ai,
            llm_rationale=llm.rationale,
            llm_available=llm.available,
            p_ai_style=style.p_ai,
            style_features=style.features,
            combined_score=scored.combined_p_ai,
            confidence=scored.confidence,
            verdict=scored.verdict,
            label_variant=label.variant,
        )

        response = {
            "content_id": content_id,
            "verdict": scored.verdict,
            "combined_score": scored.combined_p_ai,
            "confidence": scored.confidence,
            "label": label.to_dict(),  # Section 7 reader-facing label
            "signals": {
                "llm": llm.to_dict(),
                "stylometric": style.to_dict(),
            },
            "status": status,
            "warnings": ["text_below_min_length"] if too_short else [],
        }
        return jsonify(response), 200

    @app.get("/log")
    def log():
        """Return the most recent audit log entries for demo and grading visibility.

        Accepts an optional ``limit`` query parameter (1-500, default 50) to cap
        the result set. In a production system this endpoint would be
        access controlled; here it is intentionally open so the audit log can be
        inspected during grading.

        Returns:
            Response: JSON ``{"entries": [...]}`` with the most recent rows
                      newest-first, HTTP 200.
        """
        limit = request.args.get("limit", default=50, type=int)
        limit = max(1, min(limit, 500))  # Keep the response bounded
        return jsonify({"entries": get_log(limit=limit)}), 200

    @app.post("/appeal")
    def appeal():
        """Accept a creator's appeal, flip the decision to under_review, and log it.

        Implements the Section 8 appeals workflow. Validates the payload, looks
        up the contested decision, changes its status from ``classified`` to
        ``under_review``, writes an ``appeal`` row to the audit log linked by
        ``content_id``, and returns a confirmation. Reclassification is
        deliberately not automated (Section 8): a contested decision is escalated
        to a human, never silently overturned by the pipeline that made the call.

        The reasoning field is accepted under either the spec name ``reasoning``
        or the alias ``creator_reasoning`` so callers using either spelling work.

        Returns:
            Response: JSON ``{content_id, status, message, appeal_id}`` with HTTP
                      200 on success.
                      HTTP 400 on a missing content_id or empty reasoning.
                      HTTP 404 when the content_id is unknown.
        """
        # --- Validate Appeal Payload (Section 8) -> 400 on bad input --------
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        content_id = body.get("content_id")
        if not isinstance(content_id, str) or not content_id.strip():
            return jsonify({"error": "Field 'content_id' is required and must be a non-empty string."}), 400

        # Accept both the spec field 'reasoning' and the alias 'creator_reasoning'.
        reasoning = body.get("reasoning")
        if reasoning is None:
            reasoning = body.get("creator_reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            return jsonify({"error": "Field 'reasoning' is required and must be a non-empty string."}), 400

        content_id = content_id.strip()
        reasoning = reasoning.strip()
        creator_id = body.get("creator_id")

        # --- Lookup Content Record (Section 8) -> 404 if unknown ------------
        decision = get_decision(content_id)
        if decision is None:
            return jsonify({"error": "Unknown content_id."}), 404

        # --- Update status to under_review (Section 8) ----------------------
        set_status(content_id, "under_review")

        # --- Log the appeal alongside the original decision (Section 11) ----
        appeal_id = str(uuid.uuid4())
        record_appeal(
            appeal_id=appeal_id,
            content_id=content_id,
            reasoning=reasoning,
            creator_id=creator_id if isinstance(creator_id, str) else None,
        )

        return jsonify({
            "content_id": content_id,
            "status": "under_review",
            "message": "Your appeal was received and the content is now under review.",
            "appeal_id": appeal_id,
        }), 200

    @app.get("/content/<content_id>")
    def content(content_id):
        """Return the full stored decision record and any appeals for a human reviewer.

        Supporting endpoint from Section 3. A reviewer working the appeal queue
        reads this to see the original verdict, both raw signal scores, the LLM
        rationale, the combined score and confidence, the label variant, the
        current status, and the creator's appeal reasoning, all in one place,
        without re-running the pipeline (Section 8).

        Args:
            content_id (str): The UUID of the decision to fetch, from the URL.

        Returns:
            Response: JSON of the decision record with an ``appeals`` list,
                      HTTP 200. HTTP 404 when the content_id is unknown.
        """
        decision = get_decision(content_id)
        if decision is None:
            return jsonify({"error": "Unknown content_id."}), 404
        decision["appeals"] = get_appeals(content_id)
        return jsonify(decision), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
