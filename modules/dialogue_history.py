"""SQLite-backed dialogue history and full SuperMemo-2 (SM-2) Spaced Repetition system."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "dialogue_history.db"
_LOCK = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS practice_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                topic TEXT NOT NULL,
                language TEXT NOT NULL,
                level TEXT NOT NULL,
                situation TEXT,
                politeness_level TEXT,
                scenario_description TEXT,
                dialogue_json TEXT NOT NULL,
                summary TEXT,
                quiz_score INTEGER
            )
            """
        )
        try:
            conn.execute("ALTER TABLE practice_sessions ADD COLUMN scenario_description TEXT;")
        except Exception:
            pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sm2_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                reading TEXT,
                meaning TEXT NOT NULL,
                easiness_factor REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 0,
                repetitions INTEGER DEFAULT 0,
                next_review_date TEXT NOT NULL,
                last_reviewed TEXT
            )
            """
        )
    return conn


def save_dialogue_session(result: dict[str, Any], quiz_score: int | None = None) -> int:
    """Save practice session and add new vocab items to SM-2 card table."""
    conn = _get_connection()
    now_iso = datetime.now(timezone.utc).isoformat()

    with _LOCK, conn:
        cursor = conn.execute(
            """
            INSERT INTO practice_sessions (created_at, topic, language, level, situation, politeness_level, scenario_description, dialogue_json, summary, quiz_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                result.get("topic", "Hội thoại"),
                result.get("language", "Tiếng Nhật"),
                result.get("level", "Trung bình"),
                result.get("situation", "Thông thường"),
                result.get("politeness_level", "Lịch sự"),
                result.get("scenario_description", ""),
                json.dumps(result.get("dialogue", []), ensure_ascii=False),
                result.get("summary", ""),
                quiz_score,
            ),
        )
        session_id = cursor.lastrowid

        # Extract words from dialogue coverage check or summary
        words = list(result.get("coverage_check", {}).keys())
        today_str = datetime.now(timezone.utc).date().isoformat()

        for w in words:
            if not w:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO sm2_cards (word, reading, meaning, easiness_factor, interval, repetitions, next_review_date, last_reviewed)
                VALUES (?, ?, ?, 2.5, 0, 0, ?, ?)
                """,
                (w, "", f"Từ vựng/ngữ pháp trong bài '{result.get('topic')}'", today_str, now_iso),
            )

    return session_id


def get_practice_history(limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent practice sessions."""
    conn = _get_connection()
    with _LOCK:
        cursor = conn.execute(
            """
            SELECT id, created_at, topic, language, level, situation, politeness_level, scenario_description, quiz_score
            FROM practice_sessions
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    return [
        {
            "id": r[0],
            "created_at": r[1],
            "topic": r[2],
            "language": r[3],
            "level": r[4],
            "situation": r[5],
            "politeness_level": r[6],
            "scenario_description": r[7],
            "quiz_score": r[8],
        }
        for r in rows
    ]


def get_due_sm2_cards() -> list[dict[str, Any]]:
    """Fetch SM-2 cards that are due for review today or overdue."""
    conn = _get_connection()
    today_str = datetime.now(timezone.utc).date().isoformat()

    with _LOCK:
        cursor = conn.execute(
            """
            SELECT id, word, reading, meaning, easiness_factor, interval, repetitions, next_review_date
            FROM sm2_cards
            WHERE next_review_date <= ?
            ORDER BY next_review_date ASC
            """,
            (today_str,),
        )
        rows = cursor.fetchall()

    return [
        {
            "id": r[0],
            "word": r[1],
            "reading": r[2],
            "meaning": r[3],
            "easiness_factor": r[4],
            "interval": r[5],
            "repetitions": r[6],
            "next_review_date": r[7],
        }
        for r in rows
    ]


def update_sm2_card(card_id: int, quality_rating: int) -> dict[str, Any]:
    """
    Apply full SuperMemo-2 (SM-2) algorithm based on quality_rating (0-5).
    
    Quality rating scale:
    5 - Perfect response
    4 - Good response with hesitation
    3 - Correct response recalled with difficulty
    2 - Incorrect response, but easy to recall upon seeing
    1 - Incorrect response, familiar
    0 - Complete blackout
    """
    q = max(0, min(5, quality_rating))
    conn = _get_connection()

    with _LOCK, conn:
        cursor = conn.execute(
            "SELECT easiness_factor, interval, repetitions FROM sm2_cards WHERE id = ?",
            (card_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Card {card_id} not found.")

        ef, interval, reps = row[0], row[1], row[2]

        if q >= 3:
            if reps == 0:
                interval = 1
            elif reps == 1:
                interval = 6
            else:
                interval = max(1, int(round(interval * ef)))
            reps += 1
        else:
            reps = 0
            interval = 1

        # Calculate new Easiness Factor (EF)
        ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        if ef < 1.3:
            ef = 1.3

        next_date = (datetime.now(timezone.utc).date() + timedelta(days=interval)).isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            UPDATE sm2_cards
            SET easiness_factor = ?, interval = ?, repetitions = ?, next_review_date = ?, last_reviewed = ?
            WHERE id = ?
            """,
            (ef, interval, reps, next_date, now_iso, card_id),
        )

    return {
        "card_id": card_id,
        "new_ef": round(ef, 2),
        "new_interval": interval,
        "new_repetition": reps,
        "next_review_date": next_date,
    }


def get_streak_days() -> int:
    """Calculate practice streak in consecutive days."""
    conn = _get_connection()
    with _LOCK:
        cursor = conn.execute("SELECT DISTINCT date(created_at) FROM practice_sessions ORDER BY date(created_at) DESC")
        dates = [r[0] for r in cursor.fetchall()]

    if not dates:
        return 0

    today = datetime.now(timezone.utc).date()
    streak = 0
    check_date = today

    for d_str in dates:
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        if d == check_date:
            streak += 1
            check_date -= timedelta(days=1)
        elif d == today - timedelta(days=1) and streak == 0:
            streak += 1
            check_date = d - timedelta(days=1)
        elif d < check_date:
            break

    return streak
