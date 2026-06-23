"""audit_log - Audit logging package for the AttributionLens pipeline.

Re-exports the public interface from audit_log.audit_log so callers can import
directly from the package (``from audit_log import record_decision``) rather
than from the submodule.

Current exports:
  DEFAULT_DB_PATH   Default filesystem path for the SQLite audit database.
  init_db           Create the decisions table if it does not already exist.
  record_decision   Write one scored decision row to the audit database.
  get_log           Read the most recent N decision rows from the database.
"""

from audit_log.audit_log import DEFAULT_DB_PATH, init_db, record_decision, get_log
