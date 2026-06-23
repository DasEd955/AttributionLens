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
def client():
    """Flask test client built from the application factory."""
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()
