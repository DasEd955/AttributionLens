"""Audit log — structured, queryable record of every decision (planning.md §11).

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
# Milestone-3 row is valid without them.
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
    return os.environ.get("AUDIT_DB_PATH", DEFAULT_DB_PATH)


def _connect(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or _db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Optional[str] = None) -> None:
    """Create the decisions table if it does not exist. Idempotent."""
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)


def _utc_now_iso() -> str:
    """UTC timestamp like ``2025-04-01T14:32:10.123Z`` (millisecond precision)."""
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
    """Write one decision row and return its ISO-8601 ``created_at`` timestamp.

    Only ``content_id`` and ``status`` are required in Milestone 3; everything
    else is optional so the same function serves the richer rows of later
    milestones without a signature change.
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
    """Shape a stored row into the demo-facing audit entry (task example)."""
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
    """Return the most recent decision entries, newest first.

    ``limit`` caps how many rows come back so ``GET /log`` stays bounded.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_entry(row) for row in rows]
