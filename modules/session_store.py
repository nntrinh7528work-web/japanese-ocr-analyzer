"""SQLite-backed session persistence for the OCR Analyzer."""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import secrets
import sqlite3
import threading
import time
import uuid

_DB_PATH: str = str(
    pathlib.Path(__file__).resolve().parent.parent / "data" / "sessions.db"
)

_lock = threading.Lock()

_CREATE_TABLES_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    analysis_language TEXT DEFAULT 'japanese'
);

CREATE TABLE IF NOT EXISTS image_items (
    id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    original_image_bytes BLOB,
    processed_image_bytes BLOB,
    report_json TEXT,
    ocr_result_json TEXT,
    edited_text TEXT DEFAULT '',
    ocr_error TEXT,
    item_order INTEGER NOT NULL,
    PRIMARY KEY (id, session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_cache (
    session_id TEXT PRIMARY KEY,
    analysis_json TEXT NOT NULL,
    partial_json TEXT DEFAULT '[]',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_settings (
    session_id TEXT PRIMARY KEY,
    settings_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notion_workspace_config (
    config_id INTEGER PRIMARY KEY CHECK (config_id = 1),
    config_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notion_sync_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    external_id TEXT NOT NULL UNIQUE,
    source_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    completed_items INTEGER NOT NULL DEFAULT 0,
    total_items INTEGER NOT NULL DEFAULT 0,
    notion_page_id TEXT,
    notion_page_url TEXT,
    error TEXT,
    item_errors_json TEXT NOT NULL DEFAULT '[]',
    next_retry_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notion_sync_session_hash
ON notion_sync_runs(session_id, source_hash, updated_at);

CREATE INDEX IF NOT EXISTS idx_notion_sync_status
ON notion_sync_runs(status, next_retry_at, updated_at);

-- V5 document library.  The legacy session-wide tables stay in place so a
-- deployed app can migrate existing browser sessions without re-running OCR.
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'unknown',
    language_source TEXT NOT NULL DEFAULT 'auto',
    status TEXT NOT NULL DEFAULT 'draft',
    source_hash TEXT NOT NULL DEFAULT '',
    active_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_session_updated
ON documents(session_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS document_image_items (
    id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    name TEXT NOT NULL,
    original_image_bytes BLOB,
    processed_image_bytes BLOB,
    report_json TEXT,
    ocr_result_json TEXT,
    edited_text TEXT DEFAULT '',
    ocr_error TEXT,
    detected_language TEXT NOT NULL DEFAULT 'unknown',
    language_confidence TEXT NOT NULL DEFAULT 'low',
    language_source TEXT NOT NULL DEFAULT 'auto',
    language_override INTEGER NOT NULL DEFAULT 0,
    mismatch_status TEXT NOT NULL DEFAULT 'none',
    item_order INTEGER NOT NULL,
    PRIMARY KEY (id, document_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_items_order
ON document_image_items(document_id, item_order);

CREATE TABLE IF NOT EXISTS analysis_versions (
    version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    analysis_language TEXT NOT NULL,
    analysis_mode TEXT NOT NULL DEFAULT 'full_analysis',
    model_name TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    analysis_json TEXT,
    partial_json TEXT NOT NULL DEFAULT '[]',
    source_snapshot_json TEXT NOT NULL DEFAULT '[]',
    job_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(document_id, version_number),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analysis_versions_document
ON analysis_versions(document_id, version_number DESC);
"""


def _get_connection() -> sqlite3.Connection:
    """Open a database connection with WAL mode and foreign keys enabled.

    Creates the ``data/`` directory and all required tables if they do not
    already exist.
    """
    db_path = pathlib.Path(_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Streamlit and the detached workers are separate processes.  A schema
    # check can briefly contend with a Notion/job write, so do not let a normal
    # SQLite lock turn into a user-visible startup failure.
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(3):
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(_CREATE_TABLES_SQL)
            version_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(analysis_versions)").fetchall()
            }
            if "source_snapshot_json" not in version_columns:
                conn.execute(
                    "ALTER TABLE analysis_versions ADD COLUMN source_snapshot_json TEXT NOT NULL DEFAULT '[]'"
                )
                conn.commit()
            return conn
        except sqlite3.OperationalError as exc:
            conn.close()
            last_error = exc
            if "locked" not in str(exc).lower() or attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise last_error or sqlite3.OperationalError("Unable to open session database")


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --- public API -----------------------------------------------------------


def generate_session_id() -> str:
    """Generate an unguessable lowercase hexadecimal session ID."""
    return secrets.token_hex(8)


def create_session(
    session_id: str,
    analysis_language: str = "japanese",
) -> None:
    """Insert a new session row into the database.

    Args:
        session_id: Unique identifier for the session.
        analysis_language: Language used for analysis (default ``'japanese'``).
    """
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, analysis_language) "
                "VALUES (?, ?, ?, ?)",
                (session_id, now, now, analysis_language),
            )
            conn.commit()
        finally:
            conn.close()


def session_exists(session_id: str) -> bool:
    """Return ``True`` if a session with *session_id* exists."""
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


def save_image_items(session_id: str, items: list[dict]) -> None:
    """Persist a list of image-item dicts for the given session.

    All existing items for the session are deleted first so the stored list
    always mirrors the caller's list exactly.

    Each *item* dict must contain the keys ``id``, ``name``,
    ``original_image_bytes``, ``processed_image_bytes``, ``report``,
    ``ocr_result``, ``edited_text``, and ``ocr_error``.
    """
    with _lock:
        conn = _get_connection()
        try:
            existing_ids = {
                row[0]
                for row in conn.execute(
                    "SELECT id FROM image_items WHERE session_id = ?", (session_id,)
                ).fetchall()
            }
            current_ids = {item["id"] for item in items}
            removed_ids = existing_ids - current_ids
            if removed_ids:
                conn.executemany(
                    "DELETE FROM image_items WHERE session_id = ? AND id = ?",
                    [(session_id, item_id) for item_id in removed_ids],
                )
            for idx, item in enumerate(items):
                report_json = json.dumps(item.get("report")) if item.get("report") is not None else None
                ocr_result_json = (
                    json.dumps(item.get("ocr_result"))
                    if item.get("ocr_result") is not None
                    else None
                )
                if item["id"] in existing_ids:
                    # Image bytes never change after upload. Avoid rewriting large
                    # BLOBs every time OCR text or page order changes.
                    conn.execute(
                        "UPDATE image_items SET name = ?, report_json = ?, "
                        "ocr_result_json = ?, edited_text = ?, ocr_error = ?, item_order = ? "
                        "WHERE session_id = ? AND id = ?",
                        (
                            item["name"], report_json, ocr_result_json,
                            item.get("edited_text", ""), item.get("ocr_error"), idx,
                            session_id, item["id"],
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO image_items "
                        "(id, session_id, name, original_image_bytes, processed_image_bytes, "
                        "report_json, ocr_result_json, edited_text, ocr_error, item_order) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            item["id"], session_id, item["name"],
                            item.get("original_image_bytes"), item.get("processed_image_bytes"),
                            report_json, ocr_result_json, item.get("edited_text", ""),
                            item.get("ocr_error"), idx,
                        ),
                    )
            conn.commit()
        finally:
            conn.close()


def load_image_items(session_id: str) -> list[dict]:
    """Load all image items for a session, ordered by ``item_order``.

    JSON fields are deserialized back into Python dicts. Returns an empty
    list when the session does not exist or has no items.
    """
    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(
                "SELECT id, name, original_image_bytes, processed_image_bytes, "
                "report_json, ocr_result_json, edited_text, ocr_error "
                "FROM image_items WHERE session_id = ? ORDER BY item_order",
                (session_id,),
            ).fetchall()

            items: list[dict] = []
            for row in rows:
                (
                    item_id,
                    name,
                    original_bytes,
                    processed_bytes,
                    report_json,
                    ocr_result_json,
                    edited_text,
                    ocr_error,
                ) = row
                items.append(
                    {
                        "id": item_id,
                        "name": name,
                        "original_image_bytes": original_bytes,
                        "processed_image_bytes": processed_bytes,
                        "report": json.loads(report_json) if report_json is not None else None,
                        "ocr_result": (
                            json.loads(ocr_result_json)
                            if ocr_result_json is not None
                            else None
                        ),
                        "edited_text": edited_text,
                        "ocr_error": ocr_error,
                    }
                )
            return items
        finally:
            conn.close()


def save_analysis(
    session_id: str,
    analysis: dict | None,
    partial: list[dict] | None = None,
) -> None:
    """Save (or delete) the analysis result and partial page analyses.

    If *analysis* is ``None`` the row is deleted entirely.  Otherwise the
    row is inserted or replaced.
    """
    with _lock:
        conn = _get_connection()
        try:
            partial = partial if partial is not None else []
            if analysis is None and not partial:
                conn.execute(
                    "DELETE FROM analysis_cache WHERE session_id = ?",
                    (session_id,),
                )
            else:
                partial_json = json.dumps(partial)
                conn.execute(
                    "INSERT OR REPLACE INTO analysis_cache "
                    "(session_id, analysis_json, partial_json) VALUES (?, ?, ?)",
                    (session_id, json.dumps(analysis), partial_json),
                )
            conn.commit()
        finally:
            conn.close()


def load_analysis(session_id: str) -> tuple[dict | None, list[dict]]:
    """Load analysis and partial results for a session.

    Returns ``(None, [])`` when no cached analysis exists.
    """
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT analysis_json, partial_json FROM analysis_cache "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None, []
            analysis = json.loads(row[0])
            partial = json.loads(row[1]) if row[1] else []
            return analysis, partial
        finally:
            conn.close()


# --- Document library -----------------------------------------------------


def _document_title_from_items(items: list[dict]) -> str:
    if items:
        name = str(items[0].get("name") or "").strip()
        stem = pathlib.Path(name).stem.strip()
        if stem:
            return stem[:120]
    return "Bài mới"


def _source_hash(items: list[dict]) -> str:
    payload = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "text": str(item.get("edited_text") or "").strip(),
            "notes": (item.get("ocr_result") or {}).get("ocr_notes", []),
        }
        for item in items
        if str(item.get("edited_text") or "").strip()
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _decode_document(row: sqlite3.Row | tuple | None) -> dict | None:
    if row is None:
        return None
    return dict(row) if isinstance(row, sqlite3.Row) else None


def create_document(
    session_id: str,
    title: str = "Bài mới",
    language: str = "unknown",
    language_source: str = "auto",
) -> dict:
    """Create one independent OCR document inside an existing session."""
    document_id = str(uuid.uuid4())
    now = _utcnow_iso()
    title = str(title or "Bài mới").strip()[:120] or "Bài mới"
    language = language if language in {"japanese", "english", "unknown"} else "unknown"
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "INSERT INTO documents (document_id, session_id, title, language, language_source, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'draft', ?, ?)",
                (document_id, session_id, title, language, language_source, now, now),
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id))
            conn.commit()
            return _decode_document(
                conn.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
            ) or {}
        finally:
            conn.close()


def list_documents(session_id: str) -> list[dict]:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT d.*, COUNT(i.id) AS image_count, COUNT(v.version_id) AS version_count "
                "FROM documents d "
                "LEFT JOIN document_image_items i ON i.document_id=d.document_id "
                "LEFT JOIN analysis_versions v ON v.document_id=d.document_id "
                "WHERE d.session_id=? GROUP BY d.document_id ORDER BY d.updated_at DESC",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def get_document(document_id: str) -> dict | None:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_document(
                conn.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
            )
        finally:
            conn.close()


def rename_document(document_id: str, title: str) -> None:
    title = str(title or "Bài mới").strip()[:120] or "Bài mới"
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("UPDATE documents SET title=?, updated_at=? WHERE document_id=?", (title, _utcnow_iso(), document_id))
            conn.commit()
        finally:
            conn.close()


def update_document_language(document_id: str, language: str, source: str = "manual") -> None:
    if language not in {"japanese", "english", "unknown"}:
        raise ValueError("Unsupported document language")
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE documents SET language=?, language_source=?, updated_at=? WHERE document_id=?",
                (language, source, _utcnow_iso(), document_id),
            )
            conn.commit()
        finally:
            conn.close()


def save_document_items(document_id: str, items: list[dict]) -> None:
    """Persist images/OCR belonging to one document without touching others."""
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            existing = {
                row[0] for row in conn.execute(
                    "SELECT id FROM document_image_items WHERE document_id=?", (document_id,)
                ).fetchall()
            }
            current = {str(item["id"]) for item in items}
            for removed in existing - current:
                conn.execute("DELETE FROM document_image_items WHERE document_id=? AND id=?", (document_id, removed))
            for order, item in enumerate(items):
                report_json = json.dumps(item.get("report")) if item.get("report") is not None else None
                ocr_json = json.dumps(item.get("ocr_result")) if item.get("ocr_result") is not None else None
                values = (
                    item.get("name") or "Ảnh không tên", report_json, ocr_json,
                    item.get("edited_text", ""), item.get("ocr_error"),
                    item.get("detected_language", "unknown"), item.get("language_confidence", "low"),
                    item.get("language_source", "auto"), int(bool(item.get("language_override", False))),
                    item.get("mismatch_status", "none"), order, document_id, item["id"],
                )
                if item["id"] in existing:
                    conn.execute(
                        "UPDATE document_image_items SET name=?, report_json=?, ocr_result_json=?, edited_text=?, "
                        "ocr_error=?, detected_language=?, language_confidence=?, language_source=?, "
                        "language_override=?, mismatch_status=?, item_order=? WHERE document_id=? AND id=?",
                        values,
                    )
                else:
                    conn.execute(
                        "INSERT INTO document_image_items (id, document_id, name, original_image_bytes, processed_image_bytes, "
                        "report_json, ocr_result_json, edited_text, ocr_error, detected_language, language_confidence, "
                        "language_source, language_override, mismatch_status, item_order) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            item["id"], document_id, item.get("name") or "Ảnh không tên",
                            item.get("original_image_bytes"), item.get("processed_image_bytes"),
                            report_json, ocr_json, item.get("edited_text", ""), item.get("ocr_error"),
                            item.get("detected_language", "unknown"), item.get("language_confidence", "low"),
                            item.get("language_source", "auto"), int(bool(item.get("language_override", False))),
                            item.get("mismatch_status", "none"), order,
                        ),
                    )
            source_hash = _source_hash(items)
            conn.execute(
                "UPDATE documents SET source_hash=?, status='needs_analysis', updated_at=? "
                "WHERE document_id=?",
                (source_hash, now, document_id),
            )
            conn.commit()
        finally:
            conn.close()


def load_document_items(document_id: str) -> list[dict]:
    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(
                "SELECT id, name, original_image_bytes, processed_image_bytes, report_json, ocr_result_json, edited_text, "
                "ocr_error, detected_language, language_confidence, language_source, language_override, mismatch_status "
                "FROM document_image_items WHERE document_id=? ORDER BY item_order",
                (document_id,),
            ).fetchall()
            return [
                {
                    "id": row[0], "name": row[1], "original_image_bytes": row[2], "processed_image_bytes": row[3],
                    "report": json.loads(row[4]) if row[4] else None,
                    "ocr_result": json.loads(row[5]) if row[5] else None,
                    "edited_text": row[6] or "", "ocr_error": row[7], "detected_language": row[8] or "unknown",
                    "language_confidence": row[9] or "low", "language_source": row[10] or "auto",
                    "language_override": bool(row[11]), "mismatch_status": row[12] or "none",
                }
                for row in rows
            ]
        finally:
            conn.close()


def move_document_item_to_new_document(
    source_document_id: str, item_id: str, session_id: str,
) -> dict | None:
    """Move one OCR page, including its text, to a fresh independent lesson."""
    items = load_document_items(source_document_id)
    selected = next((item for item in items if item.get("id") == item_id), None)
    if not selected:
        return None
    document = create_document(session_id, _document_title_from_items([selected]))
    save_document_items(document["document_id"], [selected])
    detected = str(selected.get("detected_language") or "unknown")
    if detected in {"japanese", "english"}:
        update_document_language(document["document_id"], detected, "moved_item")
    save_document_items(source_document_id, [item for item in items if item.get("id") != item_id])
    return document


def create_analysis_version(
    document_id: str, source_hash: str, analysis_language: str, analysis_mode: str, model_name: str | None = None,
    *, status: str = "pending", job_id: str | None = None, source_items: list[dict] | None = None,
) -> dict:
    version_id = str(uuid.uuid4())
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            number = int(conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM analysis_versions WHERE document_id=?", (document_id,)
            ).fetchone()[0])
            conn.execute(
                "INSERT INTO analysis_versions (version_id, document_id, version_number, source_hash, analysis_language, "
                "analysis_mode, model_name, status, job_id, source_snapshot_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version_id, document_id, number, source_hash, analysis_language, analysis_mode, model_name, status, job_id,
                    json.dumps(_source_snapshot(source_items or []), ensure_ascii=False), now, now,
                ),
            )
            conn.execute("UPDATE documents SET status=?, updated_at=? WHERE document_id=?", (status, now, document_id))
            conn.commit()
            return dict(conn.execute("SELECT * FROM analysis_versions WHERE version_id=?", (version_id,)).fetchone())
        finally:
            conn.close()


def list_analysis_versions(document_id: str) -> list[dict]:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM analysis_versions WHERE document_id=? ORDER BY version_number DESC", (document_id,)
            ).fetchall()
            return [_decode_analysis_version(row) or {} for row in rows]
        finally:
            conn.close()


def _decode_analysis_version(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    data["analysis"] = json.loads(data.pop("analysis_json") or "null")
    data["partial"] = json.loads(data.pop("partial_json") or "[]")
    data["source_snapshot"] = json.loads(data.pop("source_snapshot_json") or "[]")
    return data


def _source_snapshot(items: list[dict]) -> list[dict]:
    """Keep the exact OCR source of a version without duplicating image BLOBs."""
    return [
        {
            "id": item.get("id"), "name": item.get("name"),
            "edited_text": item.get("edited_text", ""),
            "ocr_result": item.get("ocr_result"),
            "detected_language": item.get("detected_language", "unknown"),
        }
        for item in items
    ]


def get_analysis_version(version_id: str) -> dict | None:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_analysis_version(
                conn.execute("SELECT * FROM analysis_versions WHERE version_id=?", (version_id,)).fetchone()
            )
        finally:
            conn.close()


def save_analysis_version(
    version_id: str, analysis: dict | None = None, partial: list[dict] | None = None, *, status: str | None = None,
    job_id: str | None = None,
) -> None:
    fields, values = ["updated_at=?"], [_utcnow_iso()]
    if analysis is not None:
        fields.append("analysis_json=?")
        values.append(json.dumps(analysis, ensure_ascii=False))
    if partial is not None:
        fields.append("partial_json=?")
        values.append(json.dumps(partial, ensure_ascii=False))
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if job_id is not None:
        fields.append("job_id=?")
        values.append(job_id)
    values.append(version_id)
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(f"UPDATE analysis_versions SET {', '.join(fields)} WHERE version_id=?", values)
            conn.commit()
        finally:
            conn.close()


def activate_analysis_version(document_id: str, version_id: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE documents SET active_version_id=?, status='analyzed', updated_at=? WHERE document_id=?",
                (version_id, _utcnow_iso(), document_id),
            )
            conn.commit()
        finally:
            conn.close()


def get_document_workspace(document_id: str) -> dict | None:
    document = get_document(document_id)
    if not document:
        return None
    document["items"] = load_document_items(document_id)
    document["versions"] = list_analysis_versions(document_id)
    active_id = document.get("active_version_id")
    active = get_analysis_version(active_id) if active_id else None
    document["active_version"] = active
    return document


def migrate_legacy_session_to_documents(session_id: str) -> list[dict]:
    """Idempotently turn the previous single-document session into version 1."""
    if not session_id:
        return []
    # A browser can resume while a previous deployment is still cleaning up a
    # session row. Recreate the lightweight parent row before adding documents.
    if not session_exists(session_id):
        create_session(session_id)
    existing = list_documents(session_id)
    if existing:
        return existing
    items = load_image_items(session_id)
    analysis, partial = load_analysis(session_id)
    language = str((analysis or {}).get("analysis_language") or "unknown")
    if language not in {"japanese", "english"}:
        language = "unknown"
    document = create_document(session_id, _document_title_from_items(items), language, "legacy")
    if items:
        save_document_items(document["document_id"], items)
    if analysis is not None or partial:
        version = create_analysis_version(
            document["document_id"], _source_hash(items), language, str((analysis or {}).get("analysis_mode") or "full_analysis"),
            (analysis or {}).get("model_used"), status="done" if analysis else "running", source_items=items,
        )
        save_analysis_version(version["version_id"], analysis, partial, status="done" if analysis else "running")
        if analysis:
            activate_analysis_version(document["document_id"], version["version_id"])
    return list_documents(session_id)


def cleanup_old_sessions(max_age_hours: int = 24) -> int:
    """Delete sessions whose ``updated_at`` is older than *max_age_hours*.

    Related ``image_items`` and ``analysis_cache`` rows are removed
    automatically via ``ON DELETE CASCADE``.

    Returns the number of sessions deleted.
    """
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=max_age_hours)
    ).isoformat()

    with _lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


def update_session_timestamp(session_id: str) -> None:
    """Set the session's ``updated_at`` field to the current UTC time."""
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
        finally:
            conn.close()


def save_settings(session_id: str, settings: dict) -> None:
    """Persist user-facing session settings such as the JPY budget."""
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO session_settings (session_id, settings_json) VALUES (?, ?)",
                (session_id, json.dumps(settings)),
            )
            conn.commit()
        finally:
            conn.close()


def load_settings(session_id: str) -> dict:
    """Load persisted session settings."""
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT settings_json FROM session_settings WHERE session_id = ?", (session_id,)
            ).fetchone()
            return json.loads(row[0]) if row else {}
        finally:
            conn.close()


# --- Notion workspace and durable sync queue -----------------------------


def save_notion_workspace_config(config: dict) -> None:
    """Persist non-secret Notion resource IDs created during bootstrap."""
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO notion_workspace_config "
                "(config_id, config_json, updated_at) VALUES (1, ?, ?)",
                (json.dumps(config), now),
            )
            conn.commit()
        finally:
            conn.close()


def load_notion_workspace_config() -> dict:
    """Load locally cached Notion database/data-source IDs, never a token."""
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT config_json FROM notion_workspace_config WHERE config_id = 1"
            ).fetchone()
            return json.loads(row[0]) if row else {}
        finally:
            conn.close()


def _decode_notion_run(row: sqlite3.Row | tuple | None, columns: list[str] | None = None) -> dict | None:
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        data = dict(row)
    else:
        data = dict(zip(columns or [], row))
    data["payload"] = json.loads(data.pop("payload_json") or "{}")
    data["item_errors"] = json.loads(data.pop("item_errors_json") or "[]")
    return data


def ensure_notion_sync_run(
    session_id: str,
    external_id: str,
    source_hash: str,
    payload: dict,
    *,
    force: bool = False,
) -> dict:
    """Create or refresh one idempotent Notion sync run for an OCR source."""
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    stable_payload = dict(payload)
    stable_payload.pop("created_at", None)
    stable_json = json.dumps(stable_payload, ensure_ascii=False, sort_keys=True, default=str)
    payload_hash = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM notion_sync_runs WHERE external_id = ?", (external_id,)
            ).fetchone()
            if row is None:
                run_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO notion_sync_runs "
                    "(run_id, session_id, external_id, source_hash, payload_json, payload_hash, "
                    "status, total_items, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                    (
                        run_id,
                        session_id,
                        external_id,
                        source_hash,
                        payload_json,
                        payload_hash,
                        len(payload.get("learning_items") or []),
                        now,
                        now,
                    ),
                )
            else:
                changed = row["payload_hash"] != payload_hash
                should_reset = changed or (force and row["status"] not in ("queued", "running"))
                if should_reset:
                    conn.execute(
                        "UPDATE notion_sync_runs SET session_id=?, source_hash=?, payload_json=?, "
                        "payload_hash=?, status='pending', completed_items=0, total_items=?, "
                        "error=NULL, item_errors_json='[]', next_retry_at=NULL, updated_at=? "
                        "WHERE external_id=?",
                        (
                            session_id,
                            source_hash,
                            payload_json,
                            payload_hash,
                            len(payload.get("learning_items") or []),
                            now,
                            external_id,
                        ),
                    )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM notion_sync_runs WHERE external_id = ?", (external_id,)
            ).fetchone()
            return _decode_notion_run(result) or {}
        finally:
            conn.close()


def get_notion_sync_run(run_id: str) -> dict | None:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_notion_run(
                conn.execute("SELECT * FROM notion_sync_runs WHERE run_id = ?", (run_id,)).fetchone()
            )
        finally:
            conn.close()


def get_notion_sync_for_source(session_id: str, source_hash: str) -> dict | None:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_notion_run(
                conn.execute(
                    "SELECT * FROM notion_sync_runs WHERE session_id=? AND source_hash=? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (session_id, source_hash),
                ).fetchone()
            )
        finally:
            conn.close()


def get_notion_sync_for_external_id(external_id: str) -> dict | None:
    """Load a document-scoped Notion run without conflating identical images."""
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_notion_run(
                conn.execute(
                    "SELECT * FROM notion_sync_runs WHERE external_id=?", (external_id,)
                ).fetchone()
            )
        finally:
            conn.close()


def dispatch_notion_sync_run(run_id: str, stale_after_seconds: int = 120) -> bool:
    """Atomically mark a due run queued so only one worker is spawned."""
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now = now_dt.isoformat()
    stale = (now_dt - datetime.timedelta(seconds=stale_after_seconds)).isoformat()
    with _lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "UPDATE notion_sync_runs SET status='queued', updated_at=? WHERE run_id=? AND ("
                "status='pending' OR "
                "(status='retry' AND (next_retry_at IS NULL OR next_retry_at<=?)) OR "
                "(status IN ('queued','running') AND updated_at<?))",
                (now, run_id, now, stale),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
            conn.close()


def list_due_notion_sync_runs(limit: int = 3) -> list[dict]:
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM notion_sync_runs WHERE status='pending' OR "
                "(status='retry' AND (next_retry_at IS NULL OR next_retry_at<=?)) "
                "ORDER BY created_at LIMIT ?",
                (now, limit),
            ).fetchall()
            return [_decode_notion_run(row) or {} for row in rows]
        finally:
            conn.close()


def mark_notion_sync_running(run_id: str) -> bool:
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "UPDATE notion_sync_runs SET status='running', attempts=attempts+1, "
                "error=NULL, updated_at=? WHERE run_id=? AND status='queued'",
                (now, run_id),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
            conn.close()


def update_notion_sync_progress(
    run_id: str,
    completed_items: int,
    *,
    notion_page_id: str | None = None,
    notion_page_url: str | None = None,
    item_errors: list[dict] | None = None,
) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE notion_sync_runs SET completed_items=?, "
                "notion_page_id=COALESCE(?, notion_page_id), "
                "notion_page_url=COALESCE(?, notion_page_url), "
                "item_errors_json=COALESCE(?, item_errors_json), updated_at=? WHERE run_id=?",
                (
                    completed_items,
                    notion_page_id,
                    notion_page_url,
                    json.dumps(item_errors, ensure_ascii=False) if item_errors is not None else None,
                    _utcnow_iso(),
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def finish_notion_sync_run(
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    notion_page_id: str | None = None,
    notion_page_url: str | None = None,
    item_errors: list[dict] | None = None,
    next_retry_at: str | None = None,
) -> None:
    if status not in {"done", "partial", "failed", "retry"}:
        raise ValueError(f"Unsupported Notion sync status: {status}")
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE notion_sync_runs SET status=?, error=?, "
                "notion_page_id=COALESCE(?, notion_page_id), "
                "notion_page_url=COALESCE(?, notion_page_url), "
                "item_errors_json=COALESCE(?, item_errors_json), next_retry_at=?, updated_at=? "
                "WHERE run_id=?",
                (
                    status,
                    error,
                    notion_page_id,
                    notion_page_url,
                    json.dumps(item_errors, ensure_ascii=False) if item_errors is not None else None,
                    next_retry_at,
                    _utcnow_iso(),
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def retry_notion_sync_run(run_id: str) -> bool:
    with _lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "UPDATE notion_sync_runs SET status='pending', error=NULL, next_retry_at=NULL, "
                "updated_at=? WHERE run_id=? AND status NOT IN ('queued','running')",
                (_utcnow_iso(), run_id),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
            conn.close()
