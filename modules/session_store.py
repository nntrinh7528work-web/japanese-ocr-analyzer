"""SQLite-backed session persistence for the OCR Analyzer."""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import secrets
import sqlite3
import threading
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
