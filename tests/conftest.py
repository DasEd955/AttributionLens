"""conftest.py - Shared pytest fixtures and a fake Groq client for all test modules.

Tests never hit the real Groq API. A FakeGroqClient is injected wherever the
signal code calls the real SDK. It mimics the exact surface used:

    client.chat.completions.create(...).choices[0].message.content -> str

Fixtures:
    fake_groq   The FakeGroqClient class itself (not an instance); tests call it with
                the reply string or exception they want the fake model to return.
    audit_db    Redirects AUDIT_DB_PATH to a tmp_path file for the duration of
                one test; each test gets an isolated, empty database.
    client      Flask test client built from create_app(), depends on audit_db so
                all route-level audit writes land in the per-test temp database.
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
    set, ``create`` raises it instead. Used to test graceful degradation.
    """

    def __init__(self, reply: str = '{"p_ai": 0.5, "rationale": "x"}', raise_exc: Exception | None = None):
        self._reply = reply
        self._raise_exc = raise_exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls = []  # Records (model, messages) for assertions

    def _create(self, model, messages, **kwargs):
        """Simulate chat.completions.create; raise or return the configured reply.

        Records each call in self.calls so tests can assert on the model and
        messages that were passed. If raise_exc was set, raises it instead of
        returning a response object, which exercises the graceful degradation path.

        Args:
            model (str): The model identifier passed by the caller.
            messages (list): The messages list passed by the caller.
            **kwargs: Any additional keyword arguments (e.g. temperature, response_format).

        Returns:
            SimpleNamespace: An object mimicking a Groq completion with
                             .choices[0].message.content set to self._reply.

        Raises:
            Exception: Whatever was passed as raise_exc, if set.
        """
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
    Rate limiting is disabled here (Section 10) so the suite can fire many
    requests without tripping a 429; the dedicated rate-limit test builds its
    own app with limiting enabled.
    """
    from app import create_app

    app = create_app(enable_rate_limit=False)
    app.config.update(TESTING=True)
    return app.test_client()
