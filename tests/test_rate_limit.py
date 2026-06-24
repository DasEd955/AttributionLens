"""test_rate_limit.py - Integration tests for the Section 10 Flask-Limiter quotas.

These tests build a dedicated app with rate limiting enabled (the shared
``client`` fixture disables it so the rest of the suite can fire freely) and
confirm that a caller exceeding a per-endpoint quota is stopped with a 429 before
the route body runs. The LLM signal is stubbed so /submit never reaches Groq, and
each test points the audit log at its own temp database.

Coverage:
  /submit returns 200 up to the hourly quota then 429 once it is exceeded.
  The 429 response is structured JSON, not HTML.
  /appeal enforces its own (stricter) quota independently of /submit.
  Rate limiting is off by default in the shared test client.
"""

import app as app_module
from app import APPEAL_RATE_LIMIT, SUBMIT_RATE_LIMIT, create_app
from signals.llm_signal import LLMSignalResult

VALID_TEXT = "This is a sufficiently long piece of writing meant to clear the minimum length bound for the route."
REASONING = "I wrote this myself and believe the classification is mistaken about my work."


def _submit_quota_per_hour() -> int:
    """Return the hourly /submit quota parsed from SUBMIT_RATE_LIMIT.

    Reads the first clause of the limit string (for example "10 per hour") so the
    test stays in step with the configured number instead of hard-coding it.

    Returns:
        int: The number of /submit requests allowed per hour.
    """
    return int(SUBMIT_RATE_LIMIT.split(";")[0].split(" ")[0])


def _appeal_quota_per_hour() -> int:
    """Return the hourly /appeal quota parsed from APPEAL_RATE_LIMIT.

    Returns:
        int: The number of /appeal requests allowed per hour.
    """
    return int(APPEAL_RATE_LIMIT.split(";")[0].split(" ")[0])


def _limited_client(monkeypatch, audit_db):
    """Build a Flask test client with rate limiting enabled and the LLM stubbed.

    Args:
        monkeypatch: The pytest monkeypatch fixture, used to stub the LLM signal.
        audit_db: The audit_db fixture, ensuring writes land in a temp database.

    Returns:
        FlaskClient: A test client for an app with Section 10 limits enforced.
    """
    monkeypatch.setattr(app_module, "classify_with_llm", lambda text: LLMSignalResult(0.5, "r", True))
    app = create_app(enable_rate_limit=True)
    app.config.update(TESTING=True)
    return app.test_client()


def test_submit_returns_429_once_quota_exceeded(monkeypatch, audit_db):
    """/submit returns 200 up to the hourly quota, then 429 on the next request."""
    client = _limited_client(monkeypatch, audit_db)
    quota = _submit_quota_per_hour()

    for _ in range(quota):
        resp = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "ratelimit-test"})
        assert resp.status_code == 200

    over = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "ratelimit-test"})
    assert over.status_code == 429


def test_rate_limited_response_is_structured_json(monkeypatch, audit_db):
    """The 429 response is structured JSON carrying a rate_limited status."""
    client = _limited_client(monkeypatch, audit_db)
    quota = _submit_quota_per_hour()

    for _ in range(quota):
        client.post("/submit", json={"text": VALID_TEXT, "creator_id": "ratelimit-test"})

    over = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "ratelimit-test"})
    assert over.status_code == 429
    data = over.get_json()
    assert data is not None
    assert data["status"] == "rate_limited"
    assert "error" in data


def test_appeal_enforces_its_own_quota(monkeypatch, audit_db):
    """/appeal returns 429 once its (stricter) hourly quota is exceeded."""
    client = _limited_client(monkeypatch, audit_db)
    content_id = client.post(
        "/submit", json={"text": VALID_TEXT, "creator_id": "ratelimit-test"}
    ).get_json()["content_id"]

    quota = _appeal_quota_per_hour()
    for _ in range(quota):
        resp = client.post("/appeal", json={"content_id": content_id, "reasoning": REASONING})
        # 200 the first time, then 404-free repeats still count against the quota.
        assert resp.status_code in (200, 404)

    over = client.post("/appeal", json={"content_id": content_id, "reasoning": REASONING})
    assert over.status_code == 429


def test_rate_limit_disabled_in_shared_client(client, monkeypatch):
    """The shared test client has limiting off, so many submits all return 200."""
    from tests.helpers import stub_llm

    stub_llm(monkeypatch, LLMSignalResult(0.5, "r", True))
    for _ in range(_submit_quota_per_hour() + 5):
        resp = client.post("/submit", json={"text": VALID_TEXT, "creator_id": "u1"})
        assert resp.status_code == 200
