"""seed_dashboard.py - Populate the audit log with a balanced set of synthetic decisions.

Writes directly to SQLite, bypassing the Flask API and rate limiter entirely.
Target distribution: ~15% likely_ai, ~38% likely_human, ~47% uncertain
(16 AI + 40 human + 43 uncertain = 99 rows, matching the project's demo DB shape).

Usage:
    python scripts/seed_dashboard.py
    python scripts/seed_dashboard.py --db path/to/audit_log.db
    python scripts/seed_dashboard.py --clear   # wipe seed rows first, then re-seed
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

# Resolve DB path the same way audit_log.py does.
_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "audit_log", "audit_log.db")


def _utc_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _spread_timestamps(n: int, days_back: int = 3) -> list[str]:
    """Return n ISO timestamps spread evenly over the last `days_back` days."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    step = (now - start) / max(n, 1)
    return [_utc_iso(start + step * i) for i in range(n)]


# ---------------------------------------------------------------------------
# Seed Data: (p_ai_llm, p_ai_style, combined_score, confidence,
#             grounding_factor, p_grounding_human, verdict, label_variant,
#             creator_id_suffix)
# ---------------------------------------------------------------------------

_STYLE_AI = json.dumps({
    "num_sentences": 4, "num_words": 68, "type_token_ratio": 0.71,
    "mean_sentence_length": 17.0, "too_short": False,
    "subscores": {"burstiness": 0.05, "type_token_ratio": 0.82,
                  "punctuation": 0.28, "complexity": 0.19},
})
_STYLE_HUMAN = json.dumps({
    "num_sentences": 5, "num_words": 54, "type_token_ratio": 0.88,
    "mean_sentence_length": 10.8, "too_short": False,
    "subscores": {"burstiness": 0.91, "type_token_ratio": 0.12,
                  "punctuation": 0.65, "complexity": 0.48},
})
_STYLE_UNCERTAIN = json.dumps({
    "num_sentences": 4, "num_words": 58, "type_token_ratio": 0.79,
    "mean_sentence_length": 14.5, "too_short": False,
    "subscores": {"burstiness": 0.42, "type_token_ratio": 0.51,
                  "punctuation": 0.45, "complexity": 0.33},
})
_GROUNDING_LOW = json.dumps({"temporal_hits": 0, "spatial_hits": 0,
                              "sensory_hits": 0, "firsthand_hits": 0})
_GROUNDING_HIGH = json.dumps({"temporal_hits": 1, "spatial_hits": 1,
                               "sensory_hits": 2, "firsthand_hits": 1})
_GROUNDING_MID = json.dumps({"temporal_hits": 0, "spatial_hits": 1,
                              "sensory_hits": 1, "firsthand_hits": 0})
_LLM_RAT_AI = (
    "The text exhibits uniform sentence structure, templated transitional phrases, "
    "and lacks personal anecdotes or sensory specificity characteristic of AI-generated content."
)
_LLM_RAT_HUMAN = (
    "The text includes idiosyncratic expressions, specific personal details, and informal "
    "register consistent with human authorship."
)
_LLM_RAT_UNCERTAIN = (
    "The text contains some structured phrasing but also personal context; "
    "signals are mixed and confidence is low."
)

# 16 AI rows
_AI_ROWS = [
    (0.85, 0.78, 0.82, 0.76, 0.91, 0.18, "likely_ai", "high_confidence_ai", "ai"),
    (0.91, 0.82, 0.87, 0.81, 0.91, 0.13, "likely_ai", "high_confidence_ai", "ai"),
    (0.79, 0.71, 0.75, 0.69, 0.93, 0.25, "likely_ai", "high_confidence_ai", "ai"),
    (0.88, 0.85, 0.87, 0.83, 0.92, 0.13, "likely_ai", "high_confidence_ai", "ai"),
    (0.93, 0.89, 0.91, 0.87, 0.90, 0.09, "likely_ai", "high_confidence_ai", "ai"),
    (0.82, 0.76, 0.79, 0.73, 0.94, 0.21, "likely_ai", "high_confidence_ai", "ai"),
    (0.87, 0.80, 0.84, 0.78, 0.92, 0.16, "likely_ai", "high_confidence_ai", "ai"),
    (0.90, 0.84, 0.87, 0.82, 0.91, 0.13, "likely_ai", "high_confidence_ai", "ai"),
    (0.86, 0.77, 0.82, 0.75, 0.93, 0.18, "likely_ai", "high_confidence_ai", "ai"),
    (0.92, 0.88, 0.90, 0.86, 0.90, 0.10, "likely_ai", "high_confidence_ai", "ai"),
    (0.84, 0.79, 0.82, 0.77, 0.92, 0.18, "likely_ai", "high_confidence_ai", "ai"),
    (0.89, 0.83, 0.86, 0.80, 0.91, 0.14, "likely_ai", "high_confidence_ai", "ai"),
    (0.94, 0.90, 0.92, 0.88, 0.90, 0.08, "likely_ai", "high_confidence_ai", "ai"),
    (0.80, 0.73, 0.77, 0.70, 0.93, 0.23, "likely_ai", "high_confidence_ai", "ai"),
    (0.87, 0.81, 0.84, 0.79, 0.91, 0.16, "likely_ai", "high_confidence_ai", "ai"),
    (0.91, 0.86, 0.89, 0.84, 0.90, 0.11, "likely_ai", "high_confidence_ai", "ai"),
]

# 40 human rows
_HUMAN_ROWS = [
    (0.20, 0.30, 0.24, 0.44, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.25, 0.19, 0.51, 0.96, 0.38, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.35, 0.26, 0.40, 0.94, 0.32, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.20, 0.14, 0.58, 0.97, 0.45, "likely_human", "high_confidence_human", "human"),
    (0.25, 0.30, 0.27, 0.38, 0.93, 0.29, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.28, 0.20, 0.48, 0.95, 0.35, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.32, 0.25, 0.42, 0.94, 0.31, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.22, 0.15, 0.55, 0.97, 0.42, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.40, 0.28, 0.36, 0.94, 0.28, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.26, 0.19, 0.50, 0.96, 0.37, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.30, 0.24, 0.43, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.18, 0.13, 0.60, 0.97, 0.46, "likely_human", "high_confidence_human", "human"),
    (0.25, 0.33, 0.28, 0.37, 0.93, 0.28, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.27, 0.20, 0.49, 0.95, 0.36, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.31, 0.24, 0.43, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.21, 0.14, 0.57, 0.97, 0.43, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.38, 0.27, 0.38, 0.94, 0.29, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.25, 0.19, 0.51, 0.96, 0.38, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.30, 0.24, 0.44, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.20, 0.14, 0.58, 0.97, 0.45, "likely_human", "high_confidence_human", "human"),
    (0.25, 0.32, 0.28, 0.37, 0.93, 0.28, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.28, 0.20, 0.48, 0.95, 0.35, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.34, 0.26, 0.41, 0.94, 0.31, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.22, 0.15, 0.55, 0.97, 0.42, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.39, 0.28, 0.36, 0.94, 0.28, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.26, 0.19, 0.50, 0.96, 0.37, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.30, 0.24, 0.44, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.19, 0.13, 0.59, 0.97, 0.45, "likely_human", "high_confidence_human", "human"),
    (0.25, 0.31, 0.27, 0.38, 0.93, 0.29, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.27, 0.20, 0.49, 0.95, 0.36, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.33, 0.25, 0.42, 0.94, 0.31, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.21, 0.14, 0.57, 0.97, 0.43, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.37, 0.27, 0.38, 0.94, 0.29, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.25, 0.19, 0.51, 0.96, 0.38, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.30, 0.24, 0.43, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.20, 0.14, 0.58, 0.97, 0.45, "likely_human", "high_confidence_human", "human"),
    (0.25, 0.33, 0.28, 0.37, 0.93, 0.28, "likely_human", "high_confidence_human", "human"),
    (0.15, 0.28, 0.20, 0.48, 0.95, 0.35, "likely_human", "high_confidence_human", "human"),
    (0.20, 0.31, 0.24, 0.43, 0.94, 0.30, "likely_human", "high_confidence_human", "human"),
    (0.10, 0.22, 0.15, 0.55, 0.97, 0.42, "likely_human", "high_confidence_human", "human"),
]

# 43 uncertain rows
_UNCERTAIN_ROWS = [
    (0.55, 0.45, 0.51, 0.06, 0.97, 0.45, "uncertain", "uncertain", "unc"),
    (0.65, 0.40, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.50, 0.56, 0.12, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.45, 0.59, 0.18, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.48, 0.52, 0.05, 0.97, 0.44, "uncertain", "uncertain", "unc"),
    (0.65, 0.42, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.52, 0.57, 0.14, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.47, 0.60, 0.20, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.46, 0.51, 0.06, 0.97, 0.45, "uncertain", "uncertain", "unc"),
    (0.65, 0.41, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.51, 0.56, 0.12, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.46, 0.59, 0.18, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.47, 0.52, 0.05, 0.97, 0.44, "uncertain", "uncertain", "unc"),
    (0.65, 0.43, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.53, 0.57, 0.14, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.48, 0.60, 0.20, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.45, 0.51, 0.06, 0.97, 0.45, "uncertain", "uncertain", "unc"),
    (0.65, 0.40, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.50, 0.56, 0.12, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.45, 0.59, 0.18, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.48, 0.52, 0.05, 0.97, 0.44, "uncertain", "uncertain", "unc"),
    (0.65, 0.42, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.52, 0.57, 0.14, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.47, 0.60, 0.20, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.46, 0.51, 0.06, 0.97, 0.45, "uncertain", "uncertain", "unc"),
    (0.65, 0.41, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.51, 0.56, 0.12, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.46, 0.59, 0.18, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.47, 0.52, 0.05, 0.97, 0.44, "uncertain", "uncertain", "unc"),
    (0.65, 0.43, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.53, 0.57, 0.14, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.48, 0.60, 0.20, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.45, 0.51, 0.06, 0.97, 0.45, "uncertain", "uncertain", "unc"),
    (0.65, 0.40, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.50, 0.56, 0.12, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.45, 0.59, 0.18, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.48, 0.52, 0.05, 0.97, 0.44, "uncertain", "uncertain", "unc"),
    (0.65, 0.42, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.52, 0.57, 0.14, 0.94, 0.40, "uncertain", "uncertain", "unc"),
    (0.70, 0.47, 0.60, 0.20, 0.93, 0.30, "uncertain", "uncertain", "unc"),
    (0.55, 0.46, 0.51, 0.06, 0.97, 0.45, "uncertain", "uncertain", "unc"),
    (0.65, 0.41, 0.55, 0.10, 0.95, 0.35, "uncertain", "uncertain", "unc"),
    (0.60, 0.51, 0.56, 0.12, 0.94, 0.40, "uncertain", "uncertain", "unc"),
]

_RATIONALE = {
    "likely_ai": _LLM_RAT_AI,
    "likely_human": _LLM_RAT_HUMAN,
    "uncertain": _LLM_RAT_UNCERTAIN,
}
_STYLE_FEAT = {
    "likely_ai": _STYLE_AI,
    "likely_human": _STYLE_HUMAN,
    "uncertain": _STYLE_UNCERTAIN,
}
_GROUNDING_FEAT = {
    "likely_ai": _GROUNDING_LOW,
    "likely_human": _GROUNDING_HIGH,
    "uncertain": _GROUNDING_MID,
}


def seed(db_path: str, clear: bool = False) -> None:
    all_rows = _AI_ROWS + _HUMAN_ROWS + _UNCERTAIN_ROWS
    timestamps = _spread_timestamps(len(all_rows), days_back=3)

    conn = sqlite3.connect(db_path)
    try:
        if clear:
            conn.execute(
                "DELETE FROM decisions WHERE creator_id LIKE 'seed-%'"
            )
            conn.commit()
            print("Cleared previous seed rows.")

        inserted = 0
        for i, (row_data, ts) in enumerate(zip(all_rows, timestamps)):
            (p_llm, p_style, combined, confidence,
             grounding_factor, p_grounding_human,
             verdict, label_variant, suffix) = row_data

            content_id = str(uuid.uuid4())
            content_hash = uuid.uuid4().hex + uuid.uuid4().hex
            creator_id = f"seed-{suffix}-{i + 1}"

            conn.execute(
                """
                INSERT INTO decisions (
                    content_id, content_hash, creator_id,
                    p_ai_llm, llm_rationale, llm_available,
                    p_ai_style, style_features,
                    p_grounding_human, grounding_features, grounding_factor,
                    combined_score, confidence, verdict, label_variant,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_id, content_hash, creator_id,
                    p_llm, _RATIONALE[verdict], 1,
                    p_style, _STYLE_FEAT[verdict],
                    p_grounding_human, _GROUNDING_FEAT[verdict], grounding_factor,
                    combined, confidence, verdict, label_variant,
                    "classified", ts,
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    # Report final distribution
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT verdict, COUNT(*) FROM decisions WHERE verdict IS NOT NULL GROUP BY verdict"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    conn.close()

    counts = {v: c for v, c in rows}
    ai_count = counts.get("likely_ai", 0)
    print(f"Inserted {inserted} seed rows into {db_path}")
    print(f"Total decisions: {total}")
    print(f"  likely_ai:    {ai_count:3d}  ({ai_count/total*100:.1f}%)")
    print(f"  likely_human: {counts.get('likely_human', 0):3d}  ({counts.get('likely_human', 0)/total*100:.1f}%)")
    print(f"  uncertain:    {counts.get('uncertain', 0):3d}  ({counts.get('uncertain', 0)/total*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the AttributionLens audit log with balanced demo data.")
    parser.add_argument("--db", default=os.environ.get("AUDIT_DB_PATH", _DEFAULT_DB),
                        help="Path to audit_log.db (default: audit_log/audit_log.db)")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing seed rows before inserting")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: database not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    seed(args.db, clear=args.clear)


if __name__ == "__main__":
    main()
