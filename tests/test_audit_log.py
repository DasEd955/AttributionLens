"""Unit tests for the audit log module (planning.md Section 11, Milestone 3).

These exercise audit_log.py directly against a throwaway DB file (via the
``audit_db`` fixture in conftest, which sets AUDIT_DB_PATH):

  * a recorded decision round-trips through get_log with the demo entry shape
  * timestamp is ISO-8601 UTC with millisecond precision and a trailing Z
  * get_log returns newest first and honours the limit
  * optional Milestone-4/5 fields are stored and read back
  * style_features dict is serialized to JSON on the way in
"""

import json
import re

import audit_log


TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_record_and_read_back_minimal_entry(audit_db):
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
    audit_log.init_db()
    ts = audit_log.record_decision(content_id="c1", status="classified")
    assert TIMESTAMP_RE.match(ts), ts
    # And the stored row carries the same well-formed timestamp.
    assert TIMESTAMP_RE.match(audit_log.get_log()[0]["timestamp"])


def test_get_log_newest_first_and_limit(audit_db):
    audit_log.init_db()
    for i in range(5):
        audit_log.record_decision(content_id=f"c{i}", status="classified")

    all_entries = audit_log.get_log()
    assert [e["content_id"] for e in all_entries] == ["c4", "c3", "c2", "c1", "c0"]

    limited = audit_log.get_log(limit=2)
    assert len(limited) == 2
    assert limited[0]["content_id"] == "c4"


def test_optional_scoring_fields_round_trip(audit_db):
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
    audit_log.init_db()
    audit_log.init_db()  # second call must not raise or drop data
    audit_log.record_decision(content_id="c1", status="classified")
    audit_log.init_db()  # even after data exists
    assert len(audit_log.get_log()) == 1


def test_get_log_empty_when_no_entries(audit_db):
    audit_log.init_db()
    assert audit_log.get_log() == []
