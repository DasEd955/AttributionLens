"""test_submit_route.py - Integration tests for the POST /submit and supporting routes.

Verifies the Section 3 API contract and Section 9 input bounds end-to-end
through the Flask test client. The LLM signal is monkeypatched on app_module
so these tests never reach the Groq API.

Coverage:
  HTTP 400 on malformed, missing, or out-of-bounds input.
  HTTP 200 response contains all Section 3 contract keys with Signal 1 populated.
  HTTP 503 when the only available signal is unavailable.
  Short text is flagged via the warnings list rather than rejected.
  Every classified submission writes one structured audit log entry.
  A 503 response writes no audit entry.
  GET /log respects the limit query parameter and returns the correct shape.
"""

from signals.llm_signal import LLMSignalResult
from tests.helpers import stub_llm

VALID_TEXT = "This is a sufficiently long piece of writing meant to clear the minimum length bound for the route."


def test_missing_text_returns_400(client):
    """POST /submit with no ``text`` field returns HTTP 400."""
    resp = client.post("/submit", json={"creator_id": "abc"})
    assert resp.status_code == 400


def test_empty_text_returns_400(client):
    """POST /submit with a whitespace-only ``text`` value returns HTTP 400."""
    resp = client.post("/submit", json={"text": "   "})
    assert resp.status_code == 400


def test_non_json_body_returns_400(client):
    """POST /submit with a non-JSON body returns HTTP 400."""
    resp = client.post("/submit", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_text_too_long_returns_400(client):
    """POST /submit with text exceeding MAX_TEXT_LENGTH returns HTTP 400."""
    resp = client.post("/submit", json={"text": "a" * 20_001})
    assert resp.status_code == 400


def test_valid_submission_returns_contract_shape(client, monkeypatch):
    """A valid submission returns HTTP 200 with all Section 3 contract keys present."""
    stub_llm(monkeypatch, LLMSignalResult(0.7, "looks AI", True))
    resp = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "u1"})
    assert resp.status_code == 200

    data = resp.get_json()
    # Contract keys from Section 3.
    for key in ("content_id", "verdict", "combined_score", "confidence", "label", "signals", "status"):
        assert key in data
    assert data["signals"]["llm"]["p_ai"] == 0.7
    assert data["signals"]["llm"]["available"] is True
    assert data["status"] == "classified"


def test_valid_submission_is_scored_with_both_signals(client, monkeypatch):
    """A valid submission populates the Milestone 4 scored fields and the stylometric block.

    Verifies that Signal 2 now runs (its block is no longer a null stub) and
    that the confidence scorer fills combined_score, confidence, and verdict.
    """
    stub_llm(monkeypatch, LLMSignalResult(0.7, "looks AI", True))
    data = client.post("/submit", json={"text": VALID_TEXT}).get_json()

    # Scorer output is present and in range (no longer null placeholders).
    assert data["verdict"] in ("likely_ai", "likely_human", "uncertain")
    assert 0.0 <= data["combined_score"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0

    # Signal 2 actually ran: a real probability and feature dict, not the stub.
    style = data["signals"]["stylometric"]
    assert isinstance(style["p_ai"], float)
    assert "subscores" in style["features"]


def test_short_text_is_flagged_not_rejected(client, monkeypatch):
    """Text shorter than MIN_TEXT_LENGTH is accepted (HTTP 200) but flagged in warnings."""
    stub_llm(monkeypatch, LLMSignalResult(0.5, "r", True))
    resp = client.post("/submit", json={"text": "Too short."})
    assert resp.status_code == 200
    assert "text_below_min_length" in resp.get_json()["warnings"]


def test_llm_unavailable_degrades_to_stylometry(client, monkeypatch):
    """When the LLM signal is down, /submit degrades to stylometry alone (Section 9).

    As of Milestone 4, Signal 2 always runs, so an unavailable LLM no longer
    produces a 503. The request succeeds, the response marks the LLM
    unavailable, and the scorer caps confidence so the lone structural signal
    cannot present a confident verdict.
    """
    stub_llm(monkeypatch, LLMSignalResult(0.5, "unavailable", False))
    resp = client.post("/submit", json={"text": VALID_TEXT})
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["signals"]["llm"]["available"] is False
    assert data["signals"]["stylometric"]["p_ai"] is not None
    # Degraded confidence cap (Section 9): cannot be a high-confidence call.
    assert data["confidence"] <= 0.5


def test_health_endpoint(client):
    """GET /health returns HTTP 200 with status "ok"."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


# --- Audit Log Integration (Section 11) ---------------------------------------


def test_submission_writes_audit_entry(client, monkeypatch):
    """A successful /submit call writes exactly one structured, fully scored entry.

    As of Milestone 4 the audit row carries the combined verdict and confidence
    (Section 11), so attribution and confidence are now populated, not null.
    """
    stub_llm(monkeypatch, LLMSignalResult(0.81, "looks AI", True))
    resp = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "u1"})
    body = resp.get_json()
    content_id = body["content_id"]

    entries = client.get("/log").get_json()["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["content_id"] == content_id
    assert entry["creator_id"] == "u1"
    assert entry["llm_score"] == 0.81
    assert entry["status"] == "classified"
    # M4: now scored. The log mirrors the response verdict and confidence.
    assert entry["attribution"] == body["verdict"]
    assert entry["confidence"] == body["confidence"]


def test_degraded_submission_still_writes_audit_entry(client, monkeypatch):
    """A degraded (LLM-down) submission is still a classified decision -> one audit row.

    Replaces the old M3 "503 writes nothing" test: since Signal 2 always runs,
    an unavailable LLM degrades to a logged, classified decision rather than 503.
    """
    stub_llm(monkeypatch, LLMSignalResult(0.5, "unavailable", False))
    client.post("/submit", json={"text": VALID_TEXT})
    entries = client.get("/log").get_json()["entries"]
    assert len(entries) == 1
    assert entries[0]["status"] == "classified"


def test_log_returns_at_least_three_entries(client, monkeypatch):
    """GET /log returns at least three structured entries after three submissions."""
    # The demo requires >= 3 structured entries visible in the log.
    stub_llm(monkeypatch, LLMSignalResult(0.4, "r", True))
    for _ in range(3):
        client.post("/submit", json={"text": VALID_TEXT})

    entries = client.get("/log").get_json()["entries"]
    assert len(entries) == 3
    for entry in entries:
        assert set(entry.keys()) == {
            "content_id", "creator_id", "timestamp",
            "attribution", "confidence", "llm_score", "status",
        }


def test_log_respects_limit_query_param(client, monkeypatch):
    """GET /log?limit=N returns exactly N entries when more than N exist."""
    stub_llm(monkeypatch, LLMSignalResult(0.4, "r", True))
    for _ in range(4):
        client.post("/submit", json={"text": VALID_TEXT})

    entries = client.get("/log?limit=2").get_json()["entries"]
    assert len(entries) == 2


def test_empty_log_endpoint(client):
    """GET /log returns HTTP 200 with an empty entries list when no submissions exist."""
    resp = client.get("/log")
    assert resp.status_code == 200
    assert resp.get_json() == {"entries": []}
