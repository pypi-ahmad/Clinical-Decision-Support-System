"""SQLite persistence with WAL, size caps and audit log.

The DB path is configurable via ``MEDISCAN_DB_PATH`` so production deployments
can place the file outside the source tree (and ideally on an encrypted
filesystem or using SQLCipher — see README for migration notes).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from backend.logging_config import get_logger

_logger = get_logger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent / "medical_records.db"
DB_PATH: str = os.environ.get("MEDISCAN_DB_PATH", str(_DEFAULT_PATH))

# Hard cap on persisted JSON payload size. Keeps a malicious / buggy client
# from ballooning the DB.
MAX_JSON_BYTES = int(os.environ.get("MEDISCAN_MAX_RECORD_BYTES", 256 * 1024))

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Single connection guarded by a lock. sqlite3 in WAL mode tolerates many
# readers + one writer; for an in-process single-app workload this is fine
# and avoids the per-call reconnect fsync penalty.
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # --- Durability / concurrency ---------------------------------------
        # WAL lets many readers coexist with one writer (correct for a single
        # FastAPI process with a threadpool). synchronous=NORMAL is the
        # documented safe pairing with WAL on modern filesystems.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # Wait up to 5s before raising OperationalError on a locked DB — this
        # replaces the default "fail immediately" behaviour that would trip
        # under even mild concurrency from the FastAPI threadpool.
        conn.execute("PRAGMA busy_timeout=5000;")
        # --- Integrity -------------------------------------------------------
        conn.execute("PRAGMA foreign_keys=ON;")
        # Defence-in-depth against CVE-2020-{9327,13434}-style schema attacks
        # from untrusted extensions / attached DBs.
        conn.execute("PRAGMA trusted_schema=OFF;")
        # --- PHI hygiene -----------------------------------------------------
        # Zero freed pages so deleted MRNs / audit rows don't linger in the DB
        # file (does not protect against forensic recovery of the OS block
        # layer, but removes the trivial file-grep leak).
        conn.execute("PRAGMA secure_delete=ON;")
        # Keep temp b-trees (sort / hash join spills) in RAM so PHI never
        # hits the on-disk temp file.
        conn.execute("PRAGMA temp_store=MEMORY;")
        _conn = conn
    return _conn


def init_db() -> None:
    """Create tables if missing. Safe to call many times."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY,
                mrn TEXT NOT NULL,
                date TEXT,
                full_json TEXT NOT NULL,
                lineage TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        # Backfill the lineage column if upgrading from an earlier schema.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(records)").fetchall()}
        if "lineage" not in existing_cols:
            conn.execute("ALTER TABLE records ADD COLUMN lineage TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_mrn_date ON records (mrn, date DESC)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                mrn_hash TEXT,
                actor TEXT,
                correlation_id TEXT,
                payload TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_tasks (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                mrn_hash TEXT,
                correlation_id TEXT,
                confidence_score REAL,
                validation_errors TEXT,
                document_type TEXT,
                structured_json TEXT,
                reviewer TEXT,
                review_notes TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_review_tasks_status ON review_tasks (status, created_at DESC)")


def _sanitize_mrn(raw: Any) -> str:
    if raw is None:
        return "UNKNOWN"
    text = str(raw).strip()
    if not text:
        return "UNKNOWN"
    return text[:128]


def _sanitize_date(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text if _ISO_DATE_RE.match(text) else None


def save_record(data: dict, lineage: dict | None = None) -> int:
    """Persist a record. Returns the new row id.

    ``lineage`` is an optional dict captured by :mod:`backend.lineage` that is
    stored alongside the record for provenance / auditability.
    """
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    if len(blob.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Record payload exceeds the configured size limit.")

    mrn = _sanitize_mrn(data.get("patient", {}).get("mrn"))
    date = _sanitize_date(data.get("encounter", {}).get("date"))
    lineage_blob = json.dumps(lineage, separators=(",", ":")) if lineage else None

    with _lock:
        conn = _get_connection()
        cursor = conn.execute(
            "INSERT INTO records (mrn, date, full_json, lineage) VALUES (?, ?, ?, ?)",
            (mrn, date, blob, lineage_blob),
        )
        row_id = cursor.lastrowid
    _logger.info("record_saved", mrn_len=len(mrn), row_id=row_id, has_lineage=bool(lineage_blob))
    return int(row_id or 0)


def get_patient_history(mrn: str | None) -> dict | None:
    """Return the single latest prior record for ``mrn``, or ``None``."""
    if not mrn:
        return None
    sanitized = _sanitize_mrn(mrn)
    with _lock:
        conn = _get_connection()
        row = conn.execute(
            "SELECT full_json FROM records WHERE mrn = ? ORDER BY date DESC, id DESC LIMIT 1",
            (sanitized,),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["full_json"])
    except (ValueError, TypeError):
        _logger.warning("history_parse_failed", row_id=row.get("id"))
        return None


def record_audit_event(
    event_type: str,
    *,
    mrn_hash: str | None = None,
    actor: str | None = None,
    correlation_id: str | None = None,
    payload: dict | None = None,
) -> None:
    """Append an audit row. Never raises — audit failures must not block ops."""
    try:
        with _lock:
            conn = _get_connection()
            conn.execute(
                "INSERT INTO audit_events (event_type, mrn_hash, actor, correlation_id, payload)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    event_type,
                    mrn_hash,
                    actor,
                    correlation_id,
                    json.dumps(payload or {}, separators=(",", ":")),
                ),
            )
    except Exception as exc:  # pragma: no cover - best-effort logging path
        _logger.warning("audit_write_failed", reason=str(exc))


def enqueue_review_task(
    *,
    mrn_hash: str | None,
    correlation_id: str | None,
    confidence_score: float | None,
    validation_errors: list | None,
    document_type: str | None,
    structured_data: dict | None,
) -> int:
    """Create a human-review task; returns its row id."""
    with _lock:
        conn = _get_connection()
        cursor = conn.execute(
            """
            INSERT INTO review_tasks (
                status, mrn_hash, correlation_id, confidence_score,
                validation_errors, document_type, structured_json
            ) VALUES ('pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                mrn_hash,
                correlation_id,
                confidence_score,
                json.dumps(validation_errors or [], separators=(",", ":")),
                document_type,
                json.dumps(structured_data or {}, separators=(",", ":")),
            ),
        )
        return int(cursor.lastrowid or 0)


def list_pending_reviews(limit: int = 50) -> list[dict]:
    """Return pending review tasks (newest first)."""
    with _lock:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT id, mrn_hash, correlation_id, confidence_score, document_type, "
            "validation_errors, created_at FROM review_tasks "
            "WHERE status = 'pending' ORDER BY created_at DESC LIMIT ?",
            (int(max(1, min(limit, 500))),),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "mrn_hash": row["mrn_hash"],
            "correlation_id": row["correlation_id"],
            "confidence_score": row["confidence_score"],
            "document_type": row["document_type"],
            "validation_errors": json.loads(row["validation_errors"] or "[]"),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def resolve_review_task(task_id: int, *, approve: bool, reviewer: str | None, notes: str | None) -> bool:
    """Mark a review task as approved or rejected. Returns True if updated."""
    new_status = "approved" if approve else "rejected"
    with _lock:
        conn = _get_connection()
        cursor = conn.execute(
            "UPDATE review_tasks SET status = ?, reviewer = ?, review_notes = ?, "
            "updated_at = datetime('now') WHERE id = ? AND status = 'pending'",
            (new_status, reviewer, notes, int(task_id)),
        )
        return cursor.rowcount > 0
