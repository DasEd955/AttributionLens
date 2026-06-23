"""test_audit_log.py - Unit tests for audit_log.py against a throwaway SQLite database.

Exercises the audit log module directly (not via the Flask route) using the
audit_db fixture from conftest, which redirects AUDIT_DB_PATH to a tmp_path
file so the production database is never touched.

Coverage:
  A recorded decision round-trips through get_log with the demo entry shape.
  Timestamp is ISO-8601 UTC with millisecond precision and a trailing Z.
  get_log returns entries newest first and honours the limit argument.
  Optional Milestone 4/5 fields (verdict, confidence) are stored and read back.
  style_features dict is serialized to JSON on the way in and back out.
  init_db is idempotent across multiple calls and across existing data.
"""

import json
import re

import audit_log


TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_record_and_read_back_minimal_entry(audit_db):
    """A minimal record_decision call produces an entry with the demo-facing shape.

    Confirms that content_id, creator_id, llm_score, and status round-trip
    correctly and that Milestone 4/5 fields (attribution, confidence) are null.
    """
    audit_log.init_db()
    audit_log.record_decision(
        content_id="c1",
        status="classified",
        creator_id="user-1",
        p_ai_llm=0.81,
    )

    entries = audit_log.get_log()
    assert len(entries) == 1
    entry = entries[0]
    # Demo entry shape from the task example.
    assert set(entry.keys()) == {
        "content_id", "creator_id", "timestamp",
        "attribution", "confidence", "llm_score", "status",
    }
    assert entry["content_id"] == "c1"
    assert entry["creator_id"] == "user-1"
    assert entry["llm_score"] == 0.81
    assert entry["status"] == "classified"
    # Not yet scored in Milestone 3 -> attribution/confidence are null.
    assert entry["attribution"] is None
    assert entry["confidence"] is None


def test_timestamp_is_iso_utc_millis_with_z(audit_db):
    """record_decision returns and stores a well-formed ISO-8601 UTC timestamp.

    Validates both the return value and the stored row against the regex
    YYYY-MM-DDTHH:MM:SS.mmmZ to confirm millisecond precision and the trailing Z.
    """
    audit_log.init_db()
    ts = audit_log.record_decision(content_id="c1", status="classified")
    assert TIMESTAMP_RE.match(ts), ts
    # And the stored row carries the same well-formed timestamp.
    assert TIMESTAMP_RE.match(audit_log.get_log()[0]["timestamp"])


def test_get_log_newest_first_and_limit(audit_db):
    """get_log returns entries ordered newest first and respects the limit argument.

    Inserts five rows and asserts they come back in reverse insertion order,
    then confirms that limit=2 returns only the two most recent.
    """
    audit_log.init_db()
    for i in range(5):
        audit_log.record_decision(content_id=f"c{i}", status="classified")

    all_entries = audit_log.get_log()
    assert [e["content_id"] for e in all_entries] == ["c4", "c3", "c2", "c1", "c0"]

    limited = audit_log.get_log(limit=2)
    assert len(limited) == 2
    assert limited[0]["content_id"] == "c4"


def test_optional_scoring_fields_round_trip(audit_db):
    """Milestone 4/5 scoring fields stored via record_decision are readable via get_log.

    Confirms that verdict maps to ``attribution`` and that confidence and
    llm_score survive the round-trip with their original numeric values.
    """
    audit_log.init_db()
    audit_log.record_decision(
        content_id="c1",
        status="classified",
        verdict="likely_ai",
        confidence=0.78,
        p_ai_llm=0.81,
    )
    entry = audit_log.get_log()[0]
    assert entry["attribution"] == "likely_ai"
    assert entry["confidence"] == 0.78
    assert entry["llm_score"] == 0.81


def test_style_features_serialized_as_json(audit_db):
    """style_features dict is stored as a JSON string in the decisions column.

    Reads the raw column value directly via sqlite3 (bypassing _row_to_entry)
    to confirm the dict was serialized, not stored as a Python repr.
    """
    audit_log.init_db()
    audit_log.record_decision(
        content_id="c1",
        status="classified",
        style_features={"ttr": 0.6, "burstiness": 0.2},
    )
    # Read the raw column to confirm it was stored as a JSON string.
    import sqlite3

    conn = sqlite3.connect(audit_db)
    raw = conn.execute(
        "SELECT style_features FROM decisions WHERE content_id = 'c1'"
    ).fetchone()[0]
    conn.close()
    assert json.loads(raw) == {"ttr": 0.6, "burstiness": 0.2}


def test_init_db_is_idempotent(audit_db):
    """init_db can be called multiple times without raising or losing existing rows."""
    audit_log.init_db()
    audit_log.init_db()  # Second call must not raise or drop data
    audit_log.record_decision(content_id="c1", status="classified")
    audit_log.init_db()  # Even after data exists
    assert len(audit_log.get_log()) == 1


def test_get_log_empty_when_no_entries(audit_db):
    """get_log returns an empty list when no decisions have been recorded."""
    audit_log.init_db()
    assert audit_log.get_log() == []
