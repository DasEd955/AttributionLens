"""helpers.py - Shared test helper functions for the AttributionLens test suite.

Utilities here are used across multiple test modules. Keeping them in one place
avoids duplicate definitions and gives a single location to update if the
interfaces they wrap change.

Helpers:
  stub_llm   Monkeypatch classify_with_llm on app_module for a fixed result.
"""

import app as app_module


def stub_llm(monkeypatch, result):
    """Replace classify_with_llm on app_module with a stub returning result.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        result (LLMSignalResult): The fixed signal result to return for any call.
    """
    monkeypatch.setattr(app_module, "classify_with_llm", lambda text: result)
