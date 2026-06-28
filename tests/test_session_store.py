"""Tests for the SQLite-backed session store."""

from __future__ import annotations

import datetime

import pytest

from modules import session_store


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the session store at a fresh temporary database for every test."""
    import pathlib  # noqa: F811 – re-import inside fixture for type reference

    db_file = str(tmp_path / "test_sessions.db")
    monkeypatch.setattr(session_store, "_DB_PATH", db_file)


class TestGenerateSessionId:
    """Tests for ``generate_session_id``."""

    def test_generate_session_id(self) -> None:
        """Generated ID is 6 lowercase-alphanumeric characters."""
        sid = session_store.generate_session_id()
        assert len(sid) == 6
        assert sid.isalnum()
        assert sid == sid.lower()

    def test_generate_session_id_uniqueness(self) -> None:
        """Two consecutive IDs should (almost certainly) differ."""
        ids = {session_store.generate_session_id() for _ in range(50)}
        # With 36^6 ≈ 2 billion possibilities, 50 draws should all be unique.
        assert len(ids) == 50


class TestCreateAndCheckSession:
    """Tests for ``create_session`` and ``session_exists``."""

    def test_create_and_check_session(self) -> None:
        """A freshly created session should be reported as existing."""
        sid = "abc123"
        assert not session_store.session_exists(sid)
        session_store.create_session(sid)
        assert session_store.session_exists(sid)

    def test_create_session_with_language(self) -> None:
        """Creating a session with a custom language should not raise."""
        session_store.create_session("lang01", analysis_language="english")
        assert session_store.session_exists("lang01")


class TestImageItems:
    """Tests for ``save_image_items`` and ``load_image_items``."""

    def test_save_and_load_image_items(self) -> None:
        """Round-trip save and load preserves all fields for two items."""
        sid = "img001"
        session_store.create_session(sid)

        items = [
            {
                "id": "item-a",
                "name": "page1.png",
                "original_image_bytes": b"\x89PNG_original",
                "processed_image_bytes": b"\x89PNG_processed",
                "report": {"score": 95, "details": ["ok"]},
                "ocr_result": {"text": "こんにちは"},
                "edited_text": "hello",
                "ocr_error": None,
            },
            {
                "id": "item-b",
                "name": "page2.jpg",
                "original_image_bytes": b"\xff\xd8\xff",
                "processed_image_bytes": b"\xff\xd8\xfe",
                "report": {"score": 80},
                "ocr_result": None,
                "edited_text": "",
                "ocr_error": "timeout",
            },
        ]

        session_store.save_image_items(sid, items)
        loaded = session_store.load_image_items(sid)

        assert len(loaded) == 2

        # First item
        assert loaded[0]["id"] == "item-a"
        assert loaded[0]["name"] == "page1.png"
        assert loaded[0]["original_image_bytes"] == b"\x89PNG_original"
        assert loaded[0]["processed_image_bytes"] == b"\x89PNG_processed"
        assert loaded[0]["report"] == {"score": 95, "details": ["ok"]}
        assert loaded[0]["ocr_result"] == {"text": "こんにちは"}
        assert loaded[0]["edited_text"] == "hello"
        assert loaded[0]["ocr_error"] is None

        # Second item
        assert loaded[1]["id"] == "item-b"
        assert loaded[1]["name"] == "page2.jpg"
        assert loaded[1]["report"] == {"score": 80}
        assert loaded[1]["ocr_result"] is None
        assert loaded[1]["ocr_error"] == "timeout"

    def test_save_overwrites_previous_items(self) -> None:
        """Saving a new item list replaces the old one entirely."""
        sid = "img002"
        session_store.create_session(sid)

        session_store.save_image_items(
            sid,
            [
                {
                    "id": "old",
                    "name": "old.png",
                    "original_image_bytes": b"",
                    "processed_image_bytes": b"",
                    "report": {},
                    "ocr_result": None,
                    "edited_text": "",
                    "ocr_error": None,
                }
            ],
        )
        session_store.save_image_items(
            sid,
            [
                {
                    "id": "new",
                    "name": "new.png",
                    "original_image_bytes": b"x",
                    "processed_image_bytes": b"y",
                    "report": {"v": 1},
                    "ocr_result": None,
                    "edited_text": "",
                    "ocr_error": None,
                }
            ],
        )

        loaded = session_store.load_image_items(sid)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "new"


class TestAnalysisCache:
    """Tests for ``save_analysis`` and ``load_analysis``."""

    def test_save_and_load_analysis(self) -> None:
        """Round-trip save and load preserves analysis and partial data."""
        sid = "anl001"
        session_store.create_session(sid)

        analysis = {"summary": "all good", "pages": [1, 2]}
        partial = [{"page": 1, "text": "aaa"}, {"page": 2, "text": "bbb"}]

        session_store.save_analysis(sid, analysis, partial)
        loaded_analysis, loaded_partial = session_store.load_analysis(sid)

        assert loaded_analysis == analysis
        assert loaded_partial == partial

    def test_save_none_deletes_analysis(self) -> None:
        """Passing ``None`` for analysis removes the cached row."""
        sid = "anl002"
        session_store.create_session(sid)

        session_store.save_analysis(sid, {"data": 1})
        session_store.save_analysis(sid, None)

        result, partial = session_store.load_analysis(sid)
        assert result is None
        assert partial == []


class TestCleanup:
    """Tests for ``cleanup_old_sessions``."""

    def test_cleanup_old_sessions(self) -> None:
        """Only sessions older than the threshold are deleted."""
        sid_old = "old001"
        sid_new = "new001"

        session_store.create_session(sid_old)
        session_store.create_session(sid_new)

        # Manually backdate the old session by 25 hours
        old_ts = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=25)
        ).isoformat()

        import sqlite3

        conn = sqlite3.connect(session_store._DB_PATH)
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (old_ts, sid_old),
        )
        conn.commit()
        conn.close()

        deleted = session_store.cleanup_old_sessions(max_age_hours=24)

        assert deleted == 1
        assert not session_store.session_exists(sid_old)
        assert session_store.session_exists(sid_new)


class TestNonexistentSession:
    """Tests for loading data from sessions that don't exist."""

    def test_load_nonexistent_session(self) -> None:
        """Loading items and analysis for a missing session returns defaults."""
        items = session_store.load_image_items("no_such_session")
        assert items == []

        analysis, partial = session_store.load_analysis("no_such_session")
        assert analysis is None
        assert partial == []
