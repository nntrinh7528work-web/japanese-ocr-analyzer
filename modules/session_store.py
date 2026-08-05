"""SQLite-backed session persistence for the OCR Analyzer."""

from __future__ import annotations

import datetime
import json
import pathlib
import secrets
import sqlite3
import threading

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
"""


def _get_connection() -> sqlite3.Connection:
    """Open a database connection with WAL mode and foreign keys enabled.

    Creates the ``data/`` directory and all required tables if they do not
    already exist.
    """
    db_path = pathlib.Path(_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(_CREATE_TABLES_SQL)
    return conn


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
                "INSERT INTO sessions (session_id, created_at, updated_at, analysis_language) "
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
