"""audit_log.py - Audit log; structured, queryable SQLite record of every classification decision.

The store is SQLite (a ``decisions`` table; an ``appeals`` table arrives in
Milestone 5). SQLite gives durable, structured logging with no extra setup, and
unlike ``print()`` the entries are queryable for the ``GET /log`` demo view.

Milestone 3 scope: write one row per ``/submit`` call with at least timestamp,
content_id, attribution result, and the Signal-1 (LLM) score, and read the most
recent rows back out. The full Section 11 schema is created now so Milestones 4
and 5 only have to start *populating* the columns, not migrate the table.

The DB path is injected (env var ``AUDIT_DB_PATH``, default ``audit_log.db``) so
tests can point at a throwaway file and never touch the real log.
"""

from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_DB_PATH = "audit_log.db"

# Section 11 decisions table. Columns the later milestones fill (p_ai_style,
# combined_score, confidence, verdict, label_variant) are nullable now so a
# Milestone 3 row is valid without them.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    content_id    TEXT PRIMARY KEY,
    content_hash  TEXT,
    creator_id    TEXT,
    p_ai_llm      REAL,
    llm_rationale TEXT,
    llm_available INTEGER,
    p_ai_style    REAL,
    style_features TEXT,
    combined_score REAL,
    confidence    REAL,
    verdict       TEXT,
    label_variant TEXT,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def _db_path() -> str:
    """Return the SQLite database path from the environment, with a default fallback.

    Reads AUDIT_DB_PATH from the environment so tests can redirect writes to a
    throwaway file without touching the production database.

    Returns:
        str: Absolute or relative path to the SQLite database file.
    """
    return os.environ.get("AUDIT_DB_PATH", DEFAULT_DB_PATH)


def _connect(path: Optional[str] = None) -> sqlite3.Connection:
    """Open a SQLite connection with row_factory set to sqlite3.Row.

    Args:
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        sqlite3.Connection: An open connection whose rows support column-name access.
    """
    conn = sqlite3.connect(path or _db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Optional[str] = None) -> None:
    """Create the decisions table if it does not exist.

    Idempotent: safe to call multiple times and safe to call after rows have
    already been written. Uses CREATE TABLE IF NOT EXISTS so existing data is
    never dropped.

    Args:
        path (str, optional): Database file path. Defaults to _db_path().
    """
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with millisecond precision.

    Format: ``YYYY-MM-DDTHH:MM:SS.mmmZ``. Millisecond precision is used rather
    than microsecond so the timestamp is compact while still sortable and
    unambiguous for the audit log's ordering queries.

    Returns:
        str: UTC timestamp string, e.g. ``2025-04-01T14:32:10.123Z``.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def record_decision(
    *,
    content_id: str,
    status: str,
    creator_id: Optional[str] = None,
    content_hash: Optional[str] = None,
    p_ai_llm: Optional[float] = None,
    llm_rationale: Optional[str] = None,
    llm_available: Optional[bool] = None,
    p_ai_style: Optional[float] = None,
    style_features: Optional[dict] = None,
    combined_score: Optional[float] = None,
    confidence: Optional[float] = None,
    verdict: Optional[str] = None,
    label_variant: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Write one decision row to the audit log and return its created_at timestamp.

    Only content_id and status are required in Milestone 3; all other fields
    are optional so the same function signature serves the richer rows produced
    by the Milestone 4/5 scorer and label generator without any breaking change.
    style_features is serialized to a JSON string before storage.

    Args:
        content_id (str): UUID for this submission (primary key).
        status (str): Lifecycle state, e.g. "classified" or "unavailable".
        creator_id (str, optional): Opaque creator identifier passed in by the caller.
        content_hash (str, optional): SHA-256 hex digest of the submitted text.
        p_ai_llm (float, optional): LLM signal probability in [0, 1].
        llm_rationale (str, optional): Short explanation from the LLM signal.
        llm_available (bool, optional): Whether the LLM signal ran successfully.
        p_ai_style (float, optional): Stylometric signal probability (Milestone 4).
        style_features (dict, optional): Raw stylometric feature values (Milestone 4).
        combined_score (float, optional): Weighted combined probability (Milestone 4).
        confidence (float, optional): Confidence score in [0, 1] (Milestone 4).
        verdict (str, optional): Human-readable verdict string (Milestone 4).
        label_variant (str, optional): Transparency label variant identifier (Milestone 5).
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        str: The ISO-8601 UTC created_at timestamp written to the row.
    """
    created_at = _utc_now_iso()
    features_json = json.dumps(style_features) if style_features is not None else None
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO decisions (
                content_id, content_hash, creator_id, p_ai_llm, llm_rationale,
                llm_available, p_ai_style, style_features, combined_score,
                confidence, verdict, label_variant, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_id,
                content_hash,
                creator_id,
                p_ai_llm,
                llm_rationale,
                None if llm_available is None else int(llm_available),
                p_ai_style,
                features_json,
                combined_score,
                confidence,
                verdict,
                label_variant,
                status,
                created_at,
            ),
        )
    return created_at


def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a raw SQLite row into the demo-facing audit entry dict shape.

    Picks only the seven fields defined in the Section 11 demo entry contract.
    Milestone 4/5 fields (verdict, confidence) are included as null until the
    scorer and label generator populate them.

    Args:
        row (sqlite3.Row): A row from the decisions table.

    Returns:
        dict[str, Any]: A dict with keys content_id, creator_id, timestamp,
                        attribution, confidence, llm_score, status.
    """
    return {
        "content_id": row["content_id"],
        "creator_id": row["creator_id"],
        "timestamp": row["created_at"],
        # "attribution" is the verdict; null until the Milestone-4 scorer fills it.
        "attribution": row["verdict"],
        "confidence": row["confidence"],
        "llm_score": row["p_ai_llm"],
        "status": row["status"],
    }


def get_log(limit: int = 50, path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the most recent decision entries ordered newest first.

    Queries the decisions table sorted by created_at DESC (with rowid as a
    tiebreaker for same-millisecond inserts) and applies a row cap so the GET
    /log response stays bounded.

    Args:
        limit (int, optional): Maximum number of rows to return. Defaults to 50.
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        list[dict[str, Any]]: List of audit entry dicts in newest-first order.
                              Returns an empty list when no rows exist.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_entry(row) for row in rows]
