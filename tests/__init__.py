"""tests - Pytest test suite for the Provenance Guard service.

All tests in this package run against throwaway fixtures (in-memory or
tmp_path SQLite databases, monkeypatched LLM calls) so they never touch the
real Groq API or the production audit_log.db. See conftest.py for shared
fixtures.
"""
