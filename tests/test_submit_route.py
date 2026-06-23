"""Unit tests for the POST /submit route (Milestone 3 scope).

Verify the Section 3 API contract and Section 9 input bounds:
  * 400 on malformed / missing / out-of-bounds input
  * 200 response carries the contract keys with Signal 1 populated
  * 503 when the (only) signal is unavailable
  * graceful-degrade path does not crash the route

The LLM signal is monkeypatched so these tests never hit the network.
"""

import app as app_module
from signals.llm_signal import LLMSignalResult

VALID_TEXT = "This is a sufficiently long piece of writing meant to clear the minimum length bound for the route."


def _stub_llm(monkeypatch, result):
    monkeypatch.setattr(app_module, "classify_with_llm", lambda text: result)


def test_missing_text_returns_400(client):
    resp = client.post("/submit", json={"creator_id": "abc"})
    assert resp.status_code == 400


def test_empty_text_returns_400(client):
    resp = client.post("/submit", json={"text": "   "})
    assert resp.status_code == 400


def test_non_json_body_returns_400(client):
    resp = client.post("/submit", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_text_too_long_returns_400(client):
    resp = client.post("/submit", json={"text": "a" * 20_001})
    assert resp.status_code == 400


def test_valid_submission_returns_contract_shape(client, monkeypatch):
    _stub_llm(monkeypatch, LLMSignalResult(0.7, "looks AI", True))
    resp = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "u1"})
    assert resp.status_code == 200

    data = resp.get_json()
    # Contract keys from Section 3.
    for key in ("content_id", "verdict", "combined_score", "confidence", "label", "signals", "status"):
        assert key in data
    assert data["signals"]["llm"]["p_ai"] == 0.7
    assert data["signals"]["llm"]["available"] is True
    assert data["status"] == "classified"


def test_short_text_is_flagged_not_rejected(client, monkeypatch):
    _stub_llm(monkeypatch, LLMSignalResult(0.5, "r", True))
    resp = client.post("/submit", json={"text": "Too short."})
    assert resp.status_code == 200
    assert "text_below_min_length" in resp.get_json()["warnings"]


def test_signal_unavailable_returns_503(client, monkeypatch):
    _stub_llm(monkeypatch, LLMSignalResult(0.5, "unavailable", False))
    resp = client.post("/submit", json={"text": VALID_TEXT})
    assert resp.status_code == 503


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"
