"""audit_log - Audit logging package for the AttributionLens pipeline.

Re-exports the public interface from audit_log.audit_log so callers can import
directly from the package (``from audit_log import record_decision``) rather
than from the submodule.

Current exports:
  DEFAULT_DB_PATH   Default filesystem path for the SQLite audit database.
  init_db           Create the decisions and appeals tables if they do not exist.
  record_decision   Write one scored decision row to the audit database.
  get_log           Read the most recent N decision rows from the database.
  get_decision      Read one full decision record by content_id.
  set_status        Update the status of one decision row.
  record_appeal     Write one appeal row to the audit database.
  get_appeals       Read all appeals filed against one decision.
"""

from audit_log.audit_log import (
    DEFAULT_DB_PATH,
    init_db,
    record_decision,
    get_log,
    get_decision,
    set_status,
    record_appeal,
    get_appeals,
)
