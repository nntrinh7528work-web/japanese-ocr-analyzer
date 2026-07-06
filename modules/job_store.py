"""SQLite-backed job store for background analysis tasks."""

import sqlite3
import json
import uuid
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "jobs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            input_text TEXT,
            lang TEXT,
            result TEXT,
            error TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)
    conn.commit()
    conn.close()


def create_job(input_text: str, lang: str) -> str:
    job_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO jobs (job_id, status, input_text, lang, created_at, updated_at) "
        "VALUES (?, 'pending', ?, ?, ?, ?)",
        (job_id, input_text, lang, time.time(), time.time()),
    )
    conn.commit()
    conn.close()
    return job_id


def update_job(job_id: str, status: str, result: dict | None = None, error: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE jobs SET status=?, result=?, error=?, updated_at=? WHERE job_id=?",
        (status, json.dumps(result) if result else None, error, time.time(), job_id),
    )
    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    if data["result"]:
        data["result"] = json.loads(data["result"])
    return data


init_db()
