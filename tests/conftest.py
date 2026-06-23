"""Shared pytest fixtures and fake Groq client for unit tests.

Tests never hit the real Groq API. We inject a fake client that mimics the
small slice of the SDK surface we use:

    client.chat.completions.create(...).choices[0].message.content -> str
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make the repo root importable (so `import app`, `import signals...` work)
# regardless of where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeGroqClient:
    """Minimal stand-in for groq.Groq.

    ``reply`` is the raw string the fake model "returns". If ``raise_exc`` is
    set, ``create`` raises it instead — used to test graceful degradation.
    """

    def __init__(self, reply: str = '{"p_ai": 0.5, "rationale": "x"}', raise_exc: Exception | None = None):
        self._reply = reply
        self._raise_exc = raise_exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls = []  # records (model, messages) for assertions

    def _create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if self._raise_exc is not None:
            raise self._raise_exc
        message = SimpleNamespace(content=self._reply)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


@pytest.fixture
def fake_groq():
    return FakeGroqClient


@pytest.fixture
def audit_db(tmp_path, monkeypatch):
    """Point the audit log at a throwaway SQLite file for the duration of a test.

    Each test gets its own empty DB, so writes from one test never leak into
    another and the real ``audit_log.db`` is never touched.
    """
    db_file = tmp_path / "audit_test.db"
    monkeypatch.setenv("AUDIT_DB_PATH", str(db_file))
    return str(db_file)


@pytest.fixture
def client(audit_db):
    """Flask test client built from the application factory.

    Depends on ``audit_db`` so the app's audit log writes land in the
    per-test temp database (AUDIT_DB_PATH is set before create_app runs).
    """
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()
