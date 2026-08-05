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
            CREATE TABLE IF NOT EXISTS practice_sessions_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                topic TEXT NOT NULL,
                language TEXT NOT NULL,
                level TEXT NOT NULL,
                situation TEXT,
                politeness_level TEXT,
                scenario_description TEXT,
                result_json TEXT NOT NULL,
                summary TEXT,
                quiz_score INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sm2_cards_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                word TEXT NOT NULL,
                reading TEXT,
                meaning TEXT NOT NULL,
                easiness_factor REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 0,
                repetitions INTEGER DEFAULT 0,
                next_review_date TEXT NOT NULL,
                last_reviewed TEXT,
                UNIQUE(session_id, word)
            )
            """
        )
    return conn


def save_dialogue_session(
    result: dict[str, Any],
    quiz_score: int | None = None,
    session_id: str = "default",
) -> int:
    """Save practice session and add new vocab items to SM-2 card table."""
    conn = _get_connection()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        with _LOCK, conn:
            cursor = conn.execute(
                """
                INSERT INTO practice_sessions_v2
                (session_id, created_at, topic, language, level, situation,
                 politeness_level, scenario_description, result_json, summary, quiz_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now_iso,
                    result.get("topic", "Hội thoại"),
                    result.get("language", "Tiếng Nhật"),
                    result.get("level", "Trung bình"),
                    result.get("situation", "Thông thường"),
                    result.get("politeness_level", "Lịch sự"),
                    result.get("scenario_description", ""),
                    json.dumps(result, ensure_ascii=False),
                    result.get("summary", ""),
                    quiz_score,
                ),
            )
            history_id = cursor.lastrowid

            # Extract words from dialogue coverage check or summary.
            words = list(result.get("coverage_check", {}).keys())
            today_str = datetime.now(timezone.utc).date().isoformat()

            for w in words:
                if not w:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sm2_cards_v2
                    (session_id, word, reading, meaning, easiness_factor, interval,
                     repetitions, next_review_date, last_reviewed)
                    VALUES (?, ?, ?, ?, 2.5, 0, 0, ?, ?)
                    """,
                    (
                        session_id,
                        w,
                        "",
                        f"Từ vựng/ngữ pháp trong bài '{result.get('topic')}'",
                        today_str,
                        now_iso,
                    ),
                )
        return int(history_id)
    finally:
        conn.close()


def get_practice_history(limit: int = 20, session_id: str = "default") -> list[dict[str, Any]]:
    """Retrieve recent practice sessions."""
    conn = _get_connection()
    try:
        with _LOCK:
            cursor = conn.execute(
                """
                SELECT id, created_at, topic, language, level, situation,
                       politeness_level, scenario_description, quiz_score
                FROM practice_sessions_v2
                WHERE session_id = ? ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

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


def get_due_sm2_cards(session_id: str = "default") -> list[dict[str, Any]]:
    """Fetch SM-2 cards that are due for review today or overdue."""
    conn = _get_connection()
    today_str = datetime.now(timezone.utc).date().isoformat()

    try:
        with _LOCK:
            cursor = conn.execute(
                """
                SELECT id, word, reading, meaning, easiness_factor, interval,
                       repetitions, next_review_date
                FROM sm2_cards_v2
                WHERE session_id = ? AND next_review_date <= ?
                ORDER BY next_review_date ASC
                """,
                (session_id, today_str),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

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


def update_sm2_card(card_id: int, quality_rating: int, session_id: str = "default") -> dict[str, Any]:
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

    try:
        with _LOCK, conn:
            cursor = conn.execute(
                "SELECT easiness_factor, interval, repetitions FROM sm2_cards_v2 "
                "WHERE id = ? AND session_id = ?",
                (card_id, session_id),
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

            # Calculate new Easiness Factor (EF).
            ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            if ef < 1.3:
                ef = 1.3

            next_date = (datetime.now(timezone.utc).date() + timedelta(days=interval)).isoformat()
            now_iso = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                UPDATE sm2_cards_v2
                SET easiness_factor = ?, interval = ?, repetitions = ?,
                    next_review_date = ?, last_reviewed = ?
                WHERE id = ? AND session_id = ?
                """,
                (ef, interval, reps, next_date, now_iso, card_id, session_id),
            )
    finally:
        conn.close()

    return {
        "card_id": card_id,
        "new_ef": round(ef, 2),
        "new_interval": interval,
        "new_repetition": reps,
        "next_review_date": next_date,
    }


def get_streak_days(session_id: str = "default") -> int:
    """Calculate practice streak in consecutive days."""
    conn = _get_connection()
    try:
        with _LOCK:
            cursor = conn.execute(
                "SELECT DISTINCT date(created_at) FROM practice_sessions_v2 "
                "WHERE session_id = ? ORDER BY date(created_at) DESC",
                (session_id,),
            )
            dates = [r[0] for r in cursor.fetchall()]
    finally:
        conn.close()

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


def update_quiz_score(history_id: int, score: int, session_id: str = "default") -> None:
    """Persist the latest quiz score for a saved dialogue."""
    conn = _get_connection()
    try:
        with _LOCK, conn:
            conn.execute(
                "UPDATE practice_sessions_v2 SET quiz_score = ? WHERE id = ? AND session_id = ?",
                (max(0, min(100, int(score))), history_id, session_id),
            )
    finally:
        conn.close()


def load_dialogue_session(history_id: int, session_id: str = "default") -> dict[str, Any] | None:
    """Load one saved dialogue result owned by the current session."""
    conn = _get_connection()
    try:
        with _LOCK:
            row = conn.execute(
                "SELECT result_json FROM practice_sessions_v2 WHERE id = ? AND session_id = ?",
                (history_id, session_id),
            ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()
