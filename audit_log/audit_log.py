"""audit_log.py - Audit log; structured, queryable SQLite record of every classification decision.

The store is SQLite (a ``decisions`` table and, as of Milestone 5, an
``appeals`` table). SQLite gives durable, structured logging with no extra
setup, and unlike ``print()`` the entries are queryable for the ``GET /log``
demo view.

Milestone 3 scope: write one row per ``/submit`` call with at least timestamp,
content_id, attribution result, and the Signal-1 (LLM) score, and read the most
recent rows back out. The full Section 11 schema is created now so Milestones 4
and 5 only have to start *populating* the columns, not migrate the table.

Milestone 5 scope (now live): an ``appeals`` table holds the Section 11 appeal
records, ``record_appeal`` writes one per ``POST /appeal``, ``set_status`` flips
a decision to ``under_review``, and ``get_decision`` reads a single decision row
back for the appeal lookup and the ``GET /content`` reviewer view. The ``/log``
demo entry now surfaces the most recent appeal reasoning alongside the decision
so a contested entry is visible in one place.

The DB path is injected (env var ``AUDIT_DB_PATH``, default ``audit_log.db``) so
tests can point at a throwaway file and never touch the real log.
"""

from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "audit_log.db")

# Section 11 decisions table. Columns the later milestones fill (p_ai_style,
# combined_score, confidence, verdict, label_variant) are nullable now so a
# Milestone 3 row is valid without them.  Signal 3 (grounding) columns
# (p_grounding_human, grounding_features, grounding_factor) are also nullable
# for backwards compatibility with rows written before Signal 3 was added.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    content_id          TEXT PRIMARY KEY,
    content_hash        TEXT,
    creator_id          TEXT,
    p_ai_llm            REAL,
    llm_rationale       TEXT,
    llm_available       INTEGER,
    p_ai_style          REAL,
    style_features      TEXT,
    p_grounding_human   REAL,
    grounding_features  TEXT,
    grounding_factor    REAL,
    combined_score      REAL,
    confidence          REAL,
    verdict             TEXT,
    label_variant       TEXT,
    status              TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appeals (
    appeal_id   TEXT PRIMARY KEY,
    content_id  TEXT NOT NULL,
    creator_id  TEXT,
    reasoning   TEXT NOT NULL,
    created_at  TEXT NOT NULL
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
    """Create the decisions and appeals tables if they do not exist, and migrate
    any existing table to include columns added in later milestones.

    Idempotent: safe to call multiple times and safe to call after rows have
    already been written. Uses CREATE TABLE IF NOT EXISTS so existing data is
    never dropped. ALTER TABLE ADD COLUMN is a no-op when the column already
    exists (SQLite raises OperationalError which we swallow).

    Args:
        path (str, optional): Database file path. Defaults to _db_path().
    """
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
        # Migrate existing databases that predate Milestone 6 grounding columns.
        for col, typedef in (
            ("p_grounding_human", "REAL"),
            ("grounding_features", "TEXT"),
            ("grounding_factor", "REAL"),
        ):
            try:
                conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass  # Column already exists


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
    p_grounding_human: Optional[float] = None,
    grounding_features: Optional[dict] = None,
    grounding_factor: Optional[float] = None,
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
    style_features and grounding_features are serialized to JSON strings before
    storage. Signal 3 (grounding) fields were added in Milestone 6.

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
        p_grounding_human (float, optional): Grounding signal probability in [0, 1] (Milestone 6).
        grounding_features (dict, optional): Raw grounding feature counts and subscores (Milestone 6).
        grounding_factor (float, optional): Grounding confidence modifier in [0.85, 1.15] (Milestone 6).
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
    grounding_features_json = json.dumps(grounding_features) if grounding_features is not None else None
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO decisions (
                content_id, content_hash, creator_id, p_ai_llm, llm_rationale,
                llm_available, p_ai_style, style_features,
                p_grounding_human, grounding_features, grounding_factor,
                combined_score, confidence, verdict, label_variant, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                p_grounding_human,
                grounding_features_json,
                grounding_factor,
                combined_score,
                confidence,
                verdict,
                label_variant,
                status,
                created_at,
            ),
        )
    return created_at


def get_decision(content_id: str, path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return the full stored decision record for one content_id, or None if absent.

    Reads a single ``decisions`` row by primary key and returns every Section 11
    field as a dict, deserializing ``style_features`` from JSON and coercing the
    stored ``llm_available`` integer back to a bool. This is the lookup the
    /appeal endpoint uses to confirm a content_id exists, and the record a human
    reviewer reads via GET /content.

    Args:
        content_id (str): The UUID primary key of the decision to fetch.
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        dict[str, Any] | None: The decision record, or None when no row matches.
    """
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE content_id = ?",
            (content_id,),
        ).fetchone()
    if row is None:
        return None
    features = json.loads(row["style_features"]) if row["style_features"] is not None else None
    grounding_features = (
        json.loads(row["grounding_features"]) if row["grounding_features"] is not None else None
    )
    return {
        "content_id": row["content_id"],
        "content_hash": row["content_hash"],
        "creator_id": row["creator_id"],
        "p_ai_llm": row["p_ai_llm"],
        "llm_rationale": row["llm_rationale"],
        "llm_available": None if row["llm_available"] is None else bool(row["llm_available"]),
        "p_ai_style": row["p_ai_style"],
        "style_features": features,
        "p_grounding_human": row["p_grounding_human"],
        "grounding_features": grounding_features,
        "grounding_factor": row["grounding_factor"],
        "combined_score": row["combined_score"],
        "confidence": row["confidence"],
        "verdict": row["verdict"],
        "label_variant": row["label_variant"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def set_status(content_id: str, status: str, path: Optional[str] = None) -> bool:
    """Update the status of one decision row and report whether a row changed.

    Used by the appeals workflow (Section 8) to flip a contested decision from
    ``classified`` to ``under_review``. Does not touch any other field.

    Args:
        content_id (str): The UUID primary key of the decision to update.
        status (str): The new lifecycle status, e.g. "under_review".
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        bool: True if a matching row was updated, False if no row matched.
    """
    with _connect(path) as conn:
        cursor = conn.execute(
            "UPDATE decisions SET status = ? WHERE content_id = ?",
            (status, content_id),
        )
    return cursor.rowcount > 0


def record_appeal(
    *,
    appeal_id: str,
    content_id: str,
    reasoning: str,
    creator_id: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Write one appeal row to the audit log and return its created_at timestamp.

    Persists the Section 11 appeal record linked to the contested decision by
    ``content_id``. The reasoning is the creator's free text explanation and is
    required (the /appeal route rejects an empty one before reaching here).

    Args:
        appeal_id (str): UUID for this appeal (primary key).
        content_id (str): The contested decision's content_id (foreign key).
        reasoning (str): The creator's free-text explanation of the appeal.
        creator_id (str, optional): Opaque creator identifier, if provided.
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        str: The ISO-8601 UTC created_at timestamp written to the row.
    """
    created_at = _utc_now_iso()
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO appeals (appeal_id, content_id, creator_id, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (appeal_id, content_id, creator_id, reasoning, created_at),
        )
    return created_at


def get_appeals(content_id: str, path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return all appeals filed against one decision, newest first.

    Reads the ``appeals`` table for a content_id so the /content reviewer view
    (Section 8) can show the creator's appeal reasoning alongside the original
    decision. Returns an empty list when the decision has never been appealed.

    Args:
        content_id (str): The contested decision's content_id.
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        list[dict[str, Any]]: Appeal record dicts by newest first; empty if none.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM appeals WHERE content_id = ? ORDER BY created_at DESC, rowid DESC",
            (content_id,),
        ).fetchall()
    return [
        {
            "appeal_id": row["appeal_id"],
            "content_id": row["content_id"],
            "creator_id": row["creator_id"],
            "reasoning": row["reasoning"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _row_to_entry(row: sqlite3.Row, appeal_reasoning: Optional[str] = None) -> dict[str, Any]:
    """Convert a raw SQLite row into the demo-facing audit entry dict shape.

    Picks the fields defined in the Section 11 demo entry contract, plus the
    Milestone 5 ``appeal_reasoning`` so a contested entry shows its appeal text
    in the same /log row as the original decision. ``appeal_reasoning`` is null
    until the decision has been appealed; ``status`` flips to ``under_review``
    once it has.

    Args:
        row (sqlite3.Row): A row from the decisions table.
        appeal_reasoning (str, optional): The most recent appeal reasoning for
            this decision, or None when it has never been appealed.

    Returns:
        dict[str, Any]: A dict with keys content_id, creator_id, timestamp,
                        attribution, confidence, llm_score, style_score,
                        combined_score, status, appeal_filed, appeal_reasoning.
    """
    return {
        "content_id": row["content_id"],
        "creator_id": row["creator_id"],
        "timestamp": row["created_at"],
        # "attribution" is the verdict; null until the Milestone-4 scorer fills it.
        "attribution": row["verdict"],
        "confidence": row["confidence"],
        # All three individual signal scores are surfaced so the demo /log row
        # shows the LLM, stylometric, and grounding inputs that produced the result.
        "llm_score": row["p_ai_llm"],
        "style_score": row["p_ai_style"],
        "grounding_score": row["p_grounding_human"],
        "grounding_factor": row["grounding_factor"],
        "combined_score": row["combined_score"],
        "status": row["status"],
        # True once a POST /appeal has been filed against this decision (M5).
        "appeal_filed": appeal_reasoning is not None,
        # Null until a POST /appeal attaches reasoning to this decision (M5).
        "appeal_reasoning": appeal_reasoning,
    }


def _get_overview_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return verdict counts, total decisions, and appeal rate from the database.

    Queries the decisions table for per-verdict row counts and the total row
    count, then queries the appeals table for the total appeal count and the
    number of decisions currently under review. The appeal rate is the ratio of
    total appeals to total decisions.

    Args:
        conn (sqlite3.Connection): An open database connection with row_factory
            set to sqlite3.Row.

    Returns:
        dict[str, Any]: A dict with keys ``total`` (int), ``verdict_counts``
            (dict with likely_ai, likely_human, uncertain), ``appeal_rate``
            (float), and ``appeal_counts`` (dict with total and pending).
    """
    verdict_rows = conn.execute(
        """
        SELECT verdict, COUNT(*) AS cnt
        FROM decisions
        WHERE verdict IS NOT NULL
        GROUP BY verdict
        """
    ).fetchall()
    verdict_counts: dict[str, int] = {r["verdict"]: r["cnt"] for r in verdict_rows}

    total_row = conn.execute("SELECT COUNT(*) AS cnt FROM decisions").fetchone()
    total: int = total_row["cnt"] if total_row else 0

    appeal_total_row = conn.execute("SELECT COUNT(*) AS cnt FROM appeals").fetchone()
    appeal_total: int = appeal_total_row["cnt"] if appeal_total_row else 0

    # "pending" proxied by under_review; upheld status is not tracked yet.
    pending_row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM decisions WHERE status = 'under_review'"
    ).fetchone()
    pending: int = pending_row["cnt"] if pending_row else 0

    appeal_rate: float = round(appeal_total / total, 4) if total > 0 else 0.0

    return {
        "total": total,
        "verdict_counts": {
            "likely_ai": verdict_counts.get("likely_ai", 0),
            "likely_human": verdict_counts.get("likely_human", 0),
            "uncertain": verdict_counts.get("uncertain", 0),
        },
        "appeal_rate": appeal_rate,
        "appeal_counts": {
            "total": appeal_total,
            "pending": pending,
        },
    }


def _get_signal_disagreement_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return signal disagreement metrics for rows where both LLM and style signals ran.

    Computes per-row disagreement as ``abs(p_ai_llm - p_ai_style)`` and
    aggregates the mean disagreement and the fraction of rows that exceed the
    high disagreement threshold of 0.40.

    Args:
        conn (sqlite3.Connection): An open database connection with row_factory
            set to sqlite3.Row.

    Returns:
        dict[str, Any]: A dict with keys ``n`` (int, row count), ``avg_disagreement``
            (float), ``pct_high_disagreement`` (float), and
            ``high_disagreement_threshold`` (float, always 0.40).
    """
    rows = conn.execute(
        """
        SELECT ABS(p_ai_llm - p_ai_style) AS disagreement
        FROM decisions
        WHERE p_ai_llm IS NOT NULL AND p_ai_style IS NOT NULL
        """
    ).fetchall()
    disagreements = [r["disagreement"] for r in rows]
    n = len(disagreements)
    avg_disagreement: float = round(sum(disagreements) / n, 4) if n > 0 else 0.0
    high_count: int = sum(1 for d in disagreements if d > 0.40)
    pct_high: float = round(high_count / n, 4) if n > 0 else 0.0

    return {
        "n": n,
        "avg_disagreement": avg_disagreement,
        "pct_high_disagreement": pct_high,
        "high_disagreement_threshold": 0.40,
    }


def _get_grounding_influence_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return grounding influence metrics for rows where Signal 3 ran.

    Computes the per-row confidence delta between the final confidence score
    and the base confidence before the grounding modifier was applied.
    Base confidence is derived as ``decisiveness * agreement`` clamped to
    [0, 1], where ``decisiveness = 2 * abs(combined_score - 0.5)`` and
    ``agreement = 1 - abs(p_ai_llm - p_ai_style)``. Rows with a null
    grounding_factor predate Signal 3 and are excluded.

    Args:
        conn (sqlite3.Connection): An open database connection with row_factory
            set to sqlite3.Row.

    Returns:
        dict[str, Any]: A dict with keys ``n`` (int), ``avg_influence`` (mean
            absolute delta), ``avg_delta`` (signed mean delta), ``pct_boosted``,
            ``pct_reduced``, and ``pct_neutral`` (all floats).
    """
    rows = conn.execute(
        """
        SELECT
            confidence,
            grounding_factor,
            combined_score,
            p_ai_llm,
            p_ai_style
        FROM decisions
        WHERE grounding_factor IS NOT NULL
          AND combined_score IS NOT NULL
          AND p_ai_llm IS NOT NULL
          AND p_ai_style IS NOT NULL
        """
    ).fetchall()

    deltas: list[float] = []
    boosted = 0
    reduced = 0
    neutral = 0
    for row in rows:
        decisiveness = 2.0 * abs(row["combined_score"] - 0.5)
        agreement = 1.0 - abs(row["p_ai_llm"] - row["p_ai_style"])
        base_conf = min(1.0, max(0.0, decisiveness * agreement))
        delta = round(row["confidence"] - base_conf, 4)
        deltas.append(delta)
        if delta > 0.001:
            boosted += 1
        elif delta < -0.001:
            reduced += 1
        else:
            neutral += 1

    n = len(deltas)
    avg_influence: float = round(sum(abs(d) for d in deltas) / n, 4) if n > 0 else 0.0
    avg_delta: float = round(sum(deltas) / n, 4) if n > 0 else 0.0
    pct_boosted: float = round(boosted / n, 4) if n > 0 else 0.0
    pct_reduced: float = round(reduced / n, 4) if n > 0 else 0.0
    pct_neutral: float = round(neutral / n, 4) if n > 0 else 0.0

    return {
        "n": n,
        "avg_influence": avg_influence,
        "avg_delta": avg_delta,
        "pct_boosted": pct_boosted,
        "pct_reduced": pct_reduced,
        "pct_neutral": pct_neutral,
    }


def get_dashboard_stats(path: Optional[str] = None) -> dict[str, Any]:
    """Return aggregate metrics used by the analytics dashboard.

    Delegates to three focused helpers, each responsible for one analytics
    domain, then merges their results into the single response dict consumed
    by the dashboard API. The four metrics covered are:

    1. Detection pattern counts and appeal rate (via _get_overview_stats).
    2. Signal disagreement rate (via _get_signal_disagreement_stats).
    3. Grounding influence (via _get_grounding_influence_stats).

    All helpers share one open connection so the snapshot is consistent.

    Args:
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        dict[str, Any]: Aggregate stats with keys ``total``, ``verdict_counts``,
            ``appeal_rate``, ``appeal_counts``, ``signal_disagreement``, and
            ``grounding_influence``.
    """
    with _connect(path) as conn:
        overview = _get_overview_stats(conn)
        signal_disagreement = _get_signal_disagreement_stats(conn)
        grounding_influence = _get_grounding_influence_stats(conn)

    return {
        **overview,
        "signal_disagreement": signal_disagreement,
        "grounding_influence": grounding_influence,
    }


def get_dashboard_timeseries(days: int = 30, path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return per-day verdict counts for the verdict distribution bar chart.

    Groups the decisions table by calendar date (UTC) and verdict, returning
    one row per (date, verdict) pair. The frontend aggregates these into the
    stacked bar chart.

    Args:
        days (int, optional): How many calendar days back to include.
            Defaults to 30. Clamped to [1, 365].
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        list[dict[str, Any]]: Rows with keys ``date`` (YYYY-MM-DD string),
            ``likely_ai``, ``likely_human``, ``uncertain`` counts for that day.
            Ordered oldest to newest.
    """
    days = max(1, min(days, 365))
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT
                substr(created_at, 1, 10) AS date,
                verdict,
                COUNT(*) AS cnt
            FROM decisions
            WHERE verdict IS NOT NULL
              AND created_at >= datetime('now', ? || ' days')
            GROUP BY date, verdict
            ORDER BY date ASC
            """,
            (f"-{days}",),
        ).fetchall()

    # Pivot (date, verdict, cnt) into {date, likely_ai, likely_human, uncertain}
    day_map: dict[str, dict[str, int]] = {}
    for row in rows:
        d = row["date"]
        if d not in day_map:
            day_map[d] = {"date": d, "likely_ai": 0, "likely_human": 0, "uncertain": 0}
        v = row["verdict"]
        if v in day_map[d]:
            day_map[d][v] = row["cnt"]

    return list(day_map.values())


def get_scatter_points(limit: int = 500, path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return individual submission signal scores for the scatterplot.

    Each point carries the LLM score, stylometric score, and verdict so the
    frontend can plot p_ai_llm on the Y-axis, p_ai_style on the X-axis, and
    color each dot by verdict. Only rows where both signals ran are included.

    Args:
        limit (int, optional): Maximum rows to return. Defaults to 500.
            Clamped to [1, 2000] so the response stays renderable.
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        list[dict[str, Any]]: List of dicts with keys ``p_ai_llm``,
            ``p_ai_style``, ``verdict``, and ``content_id``, newest first.
    """
    limit = max(1, min(limit, 2000))
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT content_id, p_ai_llm, p_ai_style, verdict
            FROM decisions
            WHERE p_ai_llm IS NOT NULL AND p_ai_style IS NOT NULL
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "content_id": row["content_id"],
            "p_ai_llm": row["p_ai_llm"],
            "p_ai_style": row["p_ai_style"],
            "verdict": row["verdict"],
        }
        for row in rows
    ]


def get_log(limit: int = 50, path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the most recent decision entries ordered newest first.

    Queries the decisions table sorted by created_at DESC (with rowid as a
    tiebreaker for same-millisecond inserts) and applies a row cap so the GET
    /log response stays bounded. Each entry is joined to the most recent appeal
    reasoning for its content_id (Section 8) via a correlated subquery, so a
    contested decision shows both its ``under_review`` status and the creator's
    reasoning in one row.

    Args:
        limit (int, optional): Maximum number of rows to return. Defaults to 50.
        path (str, optional): Database file path. Defaults to _db_path().

    Returns:
        list[dict[str, Any]]: List of audit entry dicts in newest-first order.
                              Returns an empty list when no rows exist.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT d.*, (
                SELECT a.reasoning FROM appeals a
                WHERE a.content_id = d.content_id
                ORDER BY a.created_at DESC, a.rowid DESC
                LIMIT 1
            ) AS appeal_reasoning
            FROM decisions d
            ORDER BY d.created_at DESC, d.rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_entry(row, appeal_reasoning=row["appeal_reasoning"]) for row in rows]
