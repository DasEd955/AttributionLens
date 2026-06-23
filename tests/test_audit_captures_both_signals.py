"""test_audit_captures_both_signals.py - Verifies the audit row records both signals plus the combined result.

Milestone 4 checkpoint requirement (planning.md Section 11): the audit log must
record each signal's individual score (p_ai_llm, p_ai_style) alongside the
combined confidence scorer output (combined_score, confidence, verdict). The
demo-facing GET /log view only surfaces a subset, so this test reads the raw
decisions row directly through sqlite3 to confirm every column was populated.
"""

import sqlite3
import app as app_module
from signals.llm_signal import LLMSignalResult

VALID_TEXT = (
    "This is a sufficiently long and reasonably varied piece of writing. It mixes "
    "a couple of short sentences with a longer one, so the structural signal has "
    "something real to measure rather than guessing on too little text."
)


def _stub_llm(monkeypatch, result):
    """Replace classify_with_llm on app_module with a lambda returning result.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        result (LLMSignalResult): The fixed signal result to return for any call.
    """
    monkeypatch.setattr(app_module, "classify_with_llm", lambda text: result)


def test_audit_row_records_both_signals_and_combined(client, audit_db, monkeypatch):
    """A scored submission writes both individual signal scores and the combined result.

    Drives one /submit call, then reads the raw decisions row to assert that
    p_ai_llm, p_ai_style, combined_score, confidence, and verdict are all
    populated and internally consistent with the JSON response.
    """
    _stub_llm(monkeypatch, LLMSignalResult(0.7, "looks AI", True))
    body = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "u1"}).get_json()
    content_id = body["content_id"]

    conn = sqlite3.connect(audit_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM decisions WHERE content_id = ?", (content_id,)
    ).fetchone()
    conn.close()

    # Signal 1 individual score.
    assert row["p_ai_llm"] == 0.7
    assert row["llm_available"] == 1
    # Signal 2 individual score and its feature payload.
    assert row["p_ai_style"] is not None
    assert row["style_features"] is not None
    # Combined confidence-scorer output, consistent with the response.
    assert row["combined_score"] == body["combined_score"]
    assert row["confidence"] == body["confidence"]
    assert row["verdict"] == body["verdict"]
