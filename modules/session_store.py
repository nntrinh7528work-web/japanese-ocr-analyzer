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
    document_type TEXT NOT NULL DEFAULT 'image',
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

CREATE TABLE IF NOT EXISTS video_sources (
    source_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE,
    source_kind TEXT NOT NULL,
    source_url TEXT,
    video_id TEXT,
    file_name TEXT,
    mime_type TEXT,
    local_path TEXT,
    duration_seconds REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    raw_transcript_json TEXT NOT NULL DEFAULT '[]',
    clean_transcript_json TEXT NOT NULL DEFAULT '[]',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    transcript_provider TEXT,
    transcript_hash TEXT,
    ingest_usage_json TEXT NOT NULL DEFAULT '{}',
    translation_runs_json TEXT NOT NULL DEFAULT '[]',
    cost_estimate_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_sources_status
ON video_sources(status, updated_at);

CREATE TABLE IF NOT EXISTS video_cues (
    cue_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    speaker TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'unknown',
    source_text TEXT NOT NULL DEFAULT '',
    translation_vi TEXT NOT NULL DEFAULT '',
    transcript_provider TEXT NOT NULL DEFAULT '',
    translation_provider TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    warning TEXT NOT NULL DEFAULT '',
    original_source_text TEXT NOT NULL DEFAULT '',
    confidence TEXT NOT NULL DEFAULT 'unknown',
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    uncertainty_reason TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 0,
    source_window_index INTEGER,
    recheck_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    UNIQUE(source_id, ordinal),
    FOREIGN KEY (source_id) REFERENCES video_sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_cues_source_order
ON video_cues(source_id, ordinal);

CREATE TABLE IF NOT EXISTS video_segments (
    segment_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    title TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'unknown',
    original_text TEXT NOT NULL DEFAULT '',
    clean_text TEXT NOT NULL DEFAULT '',
    speakers_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    analysis_json TEXT,
    usage_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(source_id, ordinal),
    FOREIGN KEY (source_id) REFERENCES video_sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_segments_source_order
ON video_segments(source_id, ordinal);
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
            document_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "document_type" not in document_columns:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN document_type TEXT NOT NULL DEFAULT 'image'"
                )
                conn.commit()
            video_source_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(video_sources)").fetchall()
            }
            if "translation_runs_json" not in video_source_columns:
                conn.execute(
                    "ALTER TABLE video_sources ADD COLUMN translation_runs_json TEXT NOT NULL DEFAULT '[]'"
                )
                conn.commit()
            video_cue_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(video_cues)").fetchall()
            }
            cue_migrations = {
                "original_source_text": "TEXT NOT NULL DEFAULT ''",
                "confidence": "TEXT NOT NULL DEFAULT 'unknown'",
                "verification_status": "TEXT NOT NULL DEFAULT 'unverified'",
                "uncertainty_reason": "TEXT NOT NULL DEFAULT ''",
                "revision": "INTEGER NOT NULL DEFAULT 0",
                "source_window_index": "INTEGER",
                "recheck_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column, definition in cue_migrations.items():
                if column not in video_cue_columns:
                    conn.execute(f"ALTER TABLE video_cues ADD COLUMN {column} {definition}")
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
    document_type: str = "image",
) -> dict:
    """Create one independent OCR document inside an existing session."""
    document_id = str(uuid.uuid4())
    now = _utcnow_iso()
    title = str(title or "Bài mới").strip()[:120] or "Bài mới"
    language = language if language in {"japanese", "english", "unknown"} else "unknown"
    document_type = document_type if document_type in {"image", "video"} else "image"
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "INSERT INTO documents (document_id, session_id, title, document_type, language, language_source, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
                (document_id, session_id, title, document_type, language, language_source, now, now),
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
                "SELECT d.*, COUNT(DISTINCT i.id) AS image_count, COUNT(DISTINCT v.version_id) AS version_count, "
                "COUNT(DISTINCT vs.source_id) AS video_count "
                "FROM documents d "
                "LEFT JOIN document_image_items i ON i.document_id=d.document_id "
                "LEFT JOIN analysis_versions v ON v.document_id=d.document_id "
                "LEFT JOIN video_sources vs ON vs.document_id=d.document_id "
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


def update_document_source_hash(document_id: str, source_hash: str, status: str = "needs_analysis") -> None:
    """Update a non-image document source without going through image item persistence."""
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE documents SET source_hash=?, status=?, updated_at=? WHERE document_id=?",
                (source_hash, status, _utcnow_iso(), document_id),
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


def create_video_source(
    document_id: str,
    source_kind: str,
    *,
    source_url: str | None = None,
    video_id: str | None = None,
    file_name: str | None = None,
    mime_type: str | None = None,
    local_path: str | None = None,
    duration_seconds: float | None = None,
    metadata: dict | None = None,
    status: str = "pending",
) -> dict:
    """Create the single video source owned by a video document."""
    source_id = str(uuid.uuid4())
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "INSERT INTO video_sources (source_id, document_id, source_kind, source_url, video_id, "
                "file_name, mime_type, local_path, duration_seconds, status, metadata_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id, document_id, source_kind, source_url, video_id, file_name, mime_type,
                    local_path, duration_seconds, status,
                    json.dumps(metadata or {}, ensure_ascii=False), now, now,
                ),
            )
            conn.execute(
                "UPDATE documents SET document_type='video', status=?, updated_at=? WHERE document_id=?",
                (status, now, document_id),
            )
            conn.commit()
            return _decode_video_source(
                conn.execute("SELECT * FROM video_sources WHERE source_id=?", (source_id,)).fetchone()
            ) or {}
        finally:
            conn.close()


def _decode_video_source(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    for column, key, fallback in (
        ("metadata_json", "metadata", {}),
        ("raw_transcript_json", "raw_transcript", []),
        ("clean_transcript_json", "clean_transcript", []),
        ("warnings_json", "transcript_warnings", []),
        ("ingest_usage_json", "ingest_usage", {}),
        ("translation_runs_json", "translation_runs", []),
        ("cost_estimate_json", "cost_estimate", {}),
    ):
        data[key] = json.loads(data.pop(column) or json.dumps(fallback))
    return data


def get_video_source(source_id: str) -> dict | None:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_video_source(
                conn.execute("SELECT * FROM video_sources WHERE source_id=?", (source_id,)).fetchone()
            )
        finally:
            conn.close()


def get_document_video_source(document_id: str) -> dict | None:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return _decode_video_source(
                conn.execute("SELECT * FROM video_sources WHERE document_id=?", (document_id,)).fetchone()
            )
        finally:
            conn.close()


def update_video_source(source_id: str, **changes) -> None:
    """Update whitelisted video state fields and their structured JSON values."""
    mapping = {
        "status": "status", "duration_seconds": "duration_seconds", "transcript_provider": "transcript_provider",
        "transcript_hash": "transcript_hash", "error": "error", "local_path": "local_path",
        "metadata": "metadata_json", "raw_transcript": "raw_transcript_json",
        "clean_transcript": "clean_transcript_json", "transcript_warnings": "warnings_json",
        "ingest_usage": "ingest_usage_json", "cost_estimate": "cost_estimate_json",
        "translation_runs": "translation_runs_json",
    }
    fields, values = [], []
    for key, value in changes.items():
        column = mapping.get(key)
        if not column:
            continue
        if column.endswith("_json"):
            value = json.dumps(value, ensure_ascii=False)
        fields.append(f"{column}=?")
        values.append(value)
    if not fields:
        return
    now = _utcnow_iso()
    fields.append("updated_at=?")
    values.extend((now, source_id))
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(f"UPDATE video_sources SET {', '.join(fields)} WHERE source_id=?", values)
            row = conn.execute("SELECT document_id, status FROM video_sources WHERE source_id=?", (source_id,)).fetchone()
            if row:
                conn.execute("UPDATE documents SET status=?, updated_at=? WHERE document_id=?", (row[1], now, row[0]))
            conn.commit()
        finally:
            conn.close()


def replace_video_segments(source_id: str, segments: list[dict]) -> None:
    """Replace the chapter index after transcript cleanup, preserving source order."""
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("DELETE FROM video_segments WHERE source_id=?", (source_id,))
            for ordinal, segment in enumerate(segments, 1):
                conn.execute(
                    "INSERT INTO video_segments (segment_id, source_id, ordinal, start_seconds, end_seconds, title, "
                    "language, original_text, clean_text, speakers_json, status, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(segment.get("segment_id") or uuid.uuid4()), source_id, ordinal,
                        float(segment.get("start_seconds", 0) or 0), float(segment.get("end_seconds", 0) or 0),
                        str(segment.get("title") or f"Đoạn {ordinal}"), str(segment.get("language") or "unknown"),
                        str(segment.get("original_text") or ""), str(segment.get("clean_text") or ""),
                        json.dumps(segment.get("speakers") or [], ensure_ascii=False),
                        str(segment.get("status") or "pending"), now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def _stable_video_cue_id(source_id: str, ordinal: int, cue: dict) -> str:
    value = "|".join((
        source_id,
        str(ordinal),
        f"{float(cue.get('start_seconds', cue.get('start', 0)) or 0):.3f}",
        f"{float(cue.get('end_seconds', cue.get('end', 0)) or 0):.3f}",
        str(cue.get("source_text") or cue.get("text") or ""),
    ))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def replace_video_cues(source_id: str, cues: list[dict]) -> None:
    """Replace timestamped transcript cues while preserving their source order."""
    now = _utcnow_iso()
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("DELETE FROM video_cues WHERE source_id=?", (source_id,))
            for ordinal, cue in enumerate(cues, 1):
                start = float(cue.get("start_seconds", cue.get("start", 0)) or 0)
                end = max(start, float(cue.get("end_seconds", cue.get("end", start)) or start))
                source_text = str(cue.get("source_text") or cue.get("text") or "").strip()
                if not source_text:
                    continue
                translation = str(cue.get("translation_vi") or "").strip()
                conn.execute(
                    "INSERT INTO video_cues (cue_id, source_id, ordinal, start_seconds, end_seconds, speaker, "
                    "language, source_text, translation_vi, transcript_provider, translation_provider, status, "
                    "warning, original_source_text, confidence, verification_status, uncertainty_reason, revision, "
                    "source_window_index, recheck_json, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(cue.get("cue_id") or _stable_video_cue_id(source_id, ordinal, cue)),
                        source_id, ordinal, start, end, str(cue.get("speaker") or ""),
                        str(cue.get("language") or "unknown"), source_text, translation,
                        str(cue.get("transcript_provider") or ""),
                        str(cue.get("translation_provider") or ""),
                        str(cue.get("status") or ("translated" if translation else "translation_pending")),
                        str(cue.get("warning") or ""),
                        str(cue.get("original_source_text") or source_text),
                        str(cue.get("confidence") or "unknown"),
                        str(cue.get("verification_status") or "unverified"),
                        str(cue.get("uncertainty_reason") or ""),
                        int(cue.get("revision", 0) or 0),
                        int(cue.get("source_window_index", 0) or 0) or None,
                        json.dumps(cue.get("recheck") or {}, ensure_ascii=False), now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def list_video_cues(source_id: str) -> list[dict]:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM video_cues WHERE source_id=? ORDER BY ordinal", (source_id,)
                ).fetchall()
            ]
            for row in rows:
                try:
                    parsed = json.loads(row.pop("recheck_json", "{}") or "{}")
                except (TypeError, json.JSONDecodeError):
                    parsed = {}
                row["recheck"] = parsed if isinstance(parsed, dict) else {}
            return rows
        finally:
            conn.close()


def update_video_cue(cue_id: str, **changes) -> None:
    mapping = {
        "translation_vi": "translation_vi", "translation_provider": "translation_provider",
        "status": "status", "warning": "warning", "language": "language",
        "source_text": "source_text", "speaker": "speaker",
        "start_seconds": "start_seconds", "end_seconds": "end_seconds",
        "original_source_text": "original_source_text", "confidence": "confidence",
        "verification_status": "verification_status", "uncertainty_reason": "uncertainty_reason",
        "revision": "revision", "source_window_index": "source_window_index", "recheck": "recheck_json",
    }
    fields, values = [], []
    for key, value in changes.items():
        column = mapping.get(key)
        if column:
            fields.append(f"{column}=?")
            values.append(json.dumps(value, ensure_ascii=False) if column == "recheck_json" else value)
    if not fields:
        return
    fields.append("updated_at=?")
    values.extend((_utcnow_iso(), cue_id))
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(f"UPDATE video_cues SET {', '.join(fields)} WHERE cue_id=?", values)
            conn.commit()
        finally:
            conn.close()


def ensure_video_cues(source: dict | None) -> list[dict]:
    """Lazily expose timestamp rows saved before the cue schema was introduced."""
    if not source:
        return []
    source_id = str(source.get("source_id") or "")
    existing = list_video_cues(source_id)
    if existing:
        return existing
    rows = source.get("clean_transcript") or source.get("raw_transcript") or []
    if not rows:
        return []
    provider = str(source.get("transcript_provider") or "legacy_transcript")
    replace_video_cues(source_id, [
        {
            "start": row.get("start", 0), "end": row.get("end", 0),
            "text": row.get("text", ""), "speaker": row.get("speaker", ""),
            "language": row.get("language", "unknown"),
            "translation_vi": row.get("translation_vi", ""),
            "transcript_provider": row.get("transcript_provider") or provider,
            "translation_provider": row.get("translation_provider", ""),
            "warning": row.get("warning", ""),
            "original_source_text": row.get("original_source_text") or row.get("text", ""),
            "confidence": row.get("confidence", "unknown"),
            "verification_status": row.get("verification_status", "legacy"),
            "uncertainty_reason": row.get(
                "uncertainty_reason", "Transcript cũ chưa được xác minh bằng pipeline V2."
            ),
            "revision": row.get("revision", 0),
            "source_window_index": row.get("source_window_index", 0),
            "recheck": row.get("recheck") if isinstance(row.get("recheck"), dict) else {},
        }
        for row in rows if isinstance(row, dict)
    ])
    return list_video_cues(source_id)


def _decode_video_segment(row: sqlite3.Row) -> dict:
    data = dict(row)
    speakers = json.loads(data.pop("speakers_json") or "[]")
    analysis = json.loads(data.pop("analysis_json") or "null")
    usage = json.loads(data.pop("usage_json") or "{}")
    # Older interrupted video jobs can contain a JSON string rather than an
    # object. Normalize at the storage boundary so every consumer is safe.
    data["speakers"] = speakers if isinstance(speakers, list) else []
    data["analysis"] = analysis if isinstance(analysis, dict) else {}
    data["usage"] = usage if isinstance(usage, dict) else {}
    return data


def list_video_segments(source_id: str) -> list[dict]:
    with _lock:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        try:
            return [
                _decode_video_segment(row) for row in conn.execute(
                    "SELECT * FROM video_segments WHERE source_id=? ORDER BY ordinal", (source_id,)
                ).fetchall()
            ]
        finally:
            conn.close()


def update_video_segment(
    segment_id: str, *, analysis: dict | None = None, usage: dict | None = None,
    status: str | None = None, error: str | None = None, clean_text: str | None = None,
) -> None:
    fields, values = ["updated_at=?"], [_utcnow_iso()]
    for column, value in (("analysis_json", analysis), ("usage_json", usage)):
        if value is not None:
            fields.append(f"{column}=?")
            values.append(json.dumps(value, ensure_ascii=False))
    for column, value in (("status", status), ("error", error), ("clean_text", clean_text)):
        if value is not None:
            fields.append(f"{column}=?")
            values.append(value)
    values.append(segment_id)
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(f"UPDATE video_segments SET {', '.join(fields)} WHERE segment_id=?", values)
            conn.commit()
        finally:
            conn.close()


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
    document["video_source"] = get_document_video_source(document_id)
    document["video_segments"] = (
        list_video_segments(document["video_source"]["source_id"])
        if document["video_source"] else []
    )
    document["video_cues"] = ensure_video_cues(document["video_source"])
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
            local_paths = [
                str(row[0]) for row in conn.execute(
                    "SELECT vs.local_path FROM video_sources vs JOIN documents d ON d.document_id=vs.document_id "
                    "JOIN sessions s ON s.session_id=d.session_id WHERE s.updated_at < ? AND vs.local_path IS NOT NULL",
                    (cutoff,),
                ).fetchall() if row[0]
            ]
            cursor = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff,),
            )
            conn.commit()
            deleted = cursor.rowcount
        finally:
            conn.close()
    for local_path in local_paths:
        try:
            pathlib.Path(local_path).unlink(missing_ok=True)
        except OSError:
            pass
    return deleted


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
