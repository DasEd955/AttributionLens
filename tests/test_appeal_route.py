"""test_appeal_route.py - Integration tests for POST /appeal and GET /content/<id>.

Verifies the Section 8 appeals workflow end-to-end through the Flask test
client: an appeal flips the contested decision to ``under_review``, the appeal
is logged alongside the original decision, and the /log and /content views
surface the creator's reasoning. The LLM signal is monkeypatched so these tests
never reach the Groq API.

Coverage:
  HTTP 400 on a missing content_id or empty/missing reasoning.
  HTTP 404 on an unknown content_id.
  HTTP 200 confirmation carrying content_id, status, message, and appeal_id.
  The appealed decision shows status under_review and appeal_reasoning in /log.
  The 'creator_reasoning' alias is accepted in place of 'reasoning'.
  GET /content returns the decision with its attached appeals.
"""

from signals.llm_signal import LLMSignalResult
from tests.helpers import stub_llm

VALID_TEXT = "This is a sufficiently long piece of writing meant to clear the minimum length bound for the route."
REASONING = "I wrote this myself from personal experience and my formal style is my own."


def _submit_and_get_content_id(client, monkeypatch):
    """Submit a valid piece of text and return its content_id.

    Args:
        client: The Flask test client fixture.
        monkeypatch: The pytest monkeypatch fixture, used to stub the LLM signal.

    Returns:
        str: The content_id from the /submit response.
    """
    stub_llm(monkeypatch, LLMSignalResult(0.5, "r", True))
    resp = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "u1"})
    return resp.get_json()["content_id"]


def test_appeal_missing_content_id_returns_400(client):
    """POST /appeal with no content_id returns HTTP 400."""
    resp = client.post("/appeal", json={"reasoning": REASONING})
    assert resp.status_code == 400


def test_appeal_missing_reasoning_returns_400(client, monkeypatch):
    """POST /appeal with no reasoning returns HTTP 400 (reasoning is required, Section 8)."""
    content_id = _submit_and_get_content_id(client, monkeypatch)
    resp = client.post("/appeal", json={"content_id": content_id})
    assert resp.status_code == 400


def test_appeal_unknown_content_id_returns_404(client):
    """POST /appeal with an unknown content_id returns HTTP 404."""
    resp = client.post("/appeal", json={"content_id": "does-not-exist", "reasoning": REASONING})
    assert resp.status_code == 404


def test_appeal_returns_confirmation_shape(client, monkeypatch):
    """A valid appeal returns HTTP 200 with the Section 3 confirmation contract keys."""
    content_id = _submit_and_get_content_id(client, monkeypatch)
    resp = client.post("/appeal", json={"content_id": content_id, "reasoning": REASONING})
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["content_id"] == content_id
    assert data["status"] == "under_review"
    assert isinstance(data["message"], str) and data["message"]
    assert isinstance(data["appeal_id"], str) and data["appeal_id"]


def test_appeal_flips_status_and_logs_reasoning(client, monkeypatch):
    """An appeal flips the decision to under_review and surfaces the reasoning in /log."""
    content_id = _submit_and_get_content_id(client, monkeypatch)
    client.post("/appeal", json={"content_id": content_id, "reasoning": REASONING})

    entries = client.get("/log").get_json()["entries"]
    entry = next(e for e in entries if e["content_id"] == content_id)
    assert entry["status"] == "under_review"
    assert entry["appeal_reasoning"] == REASONING


def test_appeal_accepts_creator_reasoning_alias(client, monkeypatch):
    """POST /appeal accepts the 'creator_reasoning' alias in place of 'reasoning'."""
    content_id = _submit_and_get_content_id(client, monkeypatch)
    resp = client.post(
        "/appeal",
        json={"content_id": content_id, "creator_reasoning": REASONING},
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "under_review"


def test_content_endpoint_returns_decision_with_appeals(client, monkeypatch):
    """GET /content/<id> returns the decision record with its attached appeals."""
    content_id = _submit_and_get_content_id(client, monkeypatch)
    client.post("/appeal", json={"content_id": content_id, "reasoning": REASONING})

    resp = client.get(f"/content/{content_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["content_id"] == content_id
    assert data["status"] == "under_review"
    assert len(data["appeals"]) == 1
    assert data["appeals"][0]["reasoning"] == REASONING


def test_content_endpoint_unknown_id_returns_404(client):
    """GET /content/<id> with an unknown content_id returns HTTP 404."""
    resp = client.get("/content/does-not-exist")
    assert resp.status_code == 404
