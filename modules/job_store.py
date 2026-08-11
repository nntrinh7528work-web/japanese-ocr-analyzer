"""SQLite-backed job store for background analysis tasks."""

import sqlite3
import json
import uuid
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "jobs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            input_text TEXT,
            lang TEXT,
            result TEXT,
            error TEXT,
            session_id TEXT,
            partial_result TEXT,
            source_hash TEXT,
            document_id TEXT,
            version_id TEXT,
            job_kind TEXT,
            source_id TEXT,
            stage TEXT,
            checkpoint_json TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "session_id" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN session_id TEXT")
    if "partial_result" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN partial_result TEXT")
    if "source_hash" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN source_hash TEXT")
    if "document_id" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN document_id TEXT")
    if "version_id" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN version_id TEXT")
    for column in ("job_kind", "source_id", "stage", "checkpoint_json"):
        if column not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
    conn.commit()
    conn.close()


def create_job(
    input_text: str,
    lang: str,
    session_id: str | None = None,
    source_hash: str | None = None,
    document_id: str | None = None,
    version_id: str | None = None,
    job_kind: str | None = None,
    source_id: str | None = None,
    stage: str | None = None,
    checkpoint: dict | None = None,
) -> str:
    init_db()
    job_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute(
        "INSERT INTO jobs (job_id, status, input_text, lang, session_id, partial_result, "
        "source_hash, document_id, version_id, job_kind, source_id, stage, checkpoint_json, created_at, updated_at) "
        "VALUES (?, 'pending', ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job_id, input_text, lang, session_id, source_hash, document_id, version_id,
            job_kind, source_id, stage, json.dumps(checkpoint or {}), time.time(), time.time(),
        ),
    )
    conn.commit()
    conn.close()
    return job_id


def update_job(
    job_id: str,
    status: str,
    result: dict | None = None,
    error: str | None = None,
    partial_result: list[dict] | None = None,
    stage: str | None = None,
    checkpoint: dict | None = None,
):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute(
        "UPDATE jobs SET status=?, "
        "result=COALESCE(?, result), error=?, partial_result=COALESCE(?, partial_result), "
        "stage=COALESCE(?, stage), checkpoint_json=COALESCE(?, checkpoint_json), updated_at=? WHERE job_id=?",
        (
            status,
            json.dumps(result) if result is not None else None,
            error,
            json.dumps(partial_result) if partial_result is not None else None,
            stage,
            json.dumps(checkpoint) if checkpoint is not None else None,
            time.time(),
            job_id,
        ),
    )
    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    if data["result"]:
        data["result"] = json.loads(data["result"])
    data["partial_result"] = json.loads(data.get("partial_result") or "[]")
    data["checkpoint"] = json.loads(data.pop("checkpoint_json") or "{}")
    return data


def cleanup_old_jobs(max_age_hours: int = 48) -> int:
    """Remove stale jobs so the local queue database cannot grow forever."""
    cutoff = time.time() - max_age_hours * 3600
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


init_db()
