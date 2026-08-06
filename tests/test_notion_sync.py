from __future__ import annotations

import datetime as dt

import pytest

from modules import notion_sync, session_store


@pytest.fixture(autouse=True)
def _temp_session_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_DB_PATH", str(tmp_path / "sessions.db"))


def _analysis(language="japanese"):
    page = {
        "page_index": 1,
        "page_name": "lesson.png",
        "source_label": "Trang 1: lesson.png",
        "source_text": "雨なので、予定を変更した。",
        "summary": "Thay đổi kế hoạch vì trời mưa.",
        "full_markdown": "# Phân tích\n\nNội dung chi tiết",
        "vocabulary_important": [
            {
                "word": "変更",
                "reading": "へんこう",
                "meaning": "thay đổi",
                "example": "予定を変更した。",
                "example_hiragana": "よていをへんこうした。",
                "example_translation": "Đã thay đổi kế hoạch.",
                "jlpt": "N3",
            }
        ],
        "kanji_analysis": [{"kanji": "雨", "kunyomi": "あめ", "meaning": "mưa", "jlpt": "N5"}],
        "connectors": [{"phrase": "ので", "meaning": "vì", "example": "雨なので", "jlpt": "N4"}],
        "grammar_points": [{"name": "～ので", "nuance": "nêu lý do khách quan", "formation": "V + ので"}],
        "sentence_breakdowns": [
            {
                "sentence_id": "p1-s1",
                "original": "雨なので、予定を変更した。",
                "reading": "あめなので、よていをへんこうした。",
                "structure_summary": "Nguyên nhân + kết quả",
                "translations": {"natural": "Vì trời mưa nên tôi đã đổi kế hoạch."},
            }
        ],
    }
    return {
        **page,
        "analysis_language": language,
        "model_used": "gemini-3.5-flash",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "page_analyses": [page],
    }


def _items():
    return [
        {
            "id": "one",
            "name": "lesson.png",
            "edited_text": "雨なので、予定を変更した。",
            "ocr_result": {"ocr_notes": [], "usage": {"input_tokens": 3, "output_tokens": 4}},
        }
    ]


def test_build_payload_extracts_high_value_rows_and_full_markdown():
    payload = notion_sync.build_notion_sync_payload(
        "session-a",
        _items(),
        _analysis(),
        billing_tier="free",
        created_at=dt.datetime(2026, 8, 6, tzinfo=dt.timezone.utc),
    )

    assert payload["external_id"].startswith("analysis:")
    assert payload["page_count"] == 1
    assert payload["total_tokens"] == 37
    assert "Phân tích" in payload["markdown"]
    assert "Trang 1: lesson.png" in payload["markdown"]
    assert {row["type"] for row in payload["learning_items"]} == {
        "Từ khó", "Kanji", "Từ nối", "Ngữ pháp", "Câu dài"
    }
    sentence = next(row for row in payload["learning_items"] if row["type"] == "Câu dài")
    assert sentence["sentence_id"] == "p1-s1"
    assert sentence["translation_vi"].startswith("Vì trời mưa")


def test_extract_english_rows_includes_collocations_and_discourse_markers():
    analysis = _analysis("english")
    page = analysis["page_analyses"][0]
    page["kanji_analysis"] = []
    page["connectors"] = []
    page["phrasal_collocations"] = [{"phrase": "carry out", "meaning": "thực hiện"}]
    page["discourse_markers"] = [{"phrase": "however", "function": "tương phản"}]

    rows = notion_sync.extract_learning_items(analysis)

    assert "Cụm từ" in {row["type"] for row in rows}
    assert any(row["title"] == "however" and row["type"] == "Từ nối" for row in rows)
    assert not any(row["type"] == "Kanji" for row in rows)


def test_split_markdown_respects_chunk_limit_and_preserves_text():
    content = "\n".join(f"line-{index}" for index in range(100))
    chunks = notion_sync.split_markdown(content, max_chars=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert "line-99" in chunks[-1]


class _Response:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.content = b"{}"

    def json(self):
        return self._body


class _HTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_client_retries_429_using_retry_after():
    http = _HTTP(
        [
            _Response(429, {"code": "rate_limited", "message": "slow"}, {"Retry-After": "2"}),
            _Response(200, {"ok": True}),
        ]
    )
    sleeps = []
    client = notion_sync.NotionClient(
        "secret-token",
        http=http,
        sleep=sleeps.append,
        monotonic=lambda: 10,
        max_attempts=2,
    )

    assert client.request("GET", "/users/me") == {"ok": True}
    assert 2 in sleeps
    assert len(http.calls) == 2
    assert http.calls[0][2]["headers"]["Notion-Version"] == "2026-03-11"


def test_learning_views_use_dynamic_today_filter():
    class Client:
        def __init__(self):
            self.created = []

        def request(self, method, path, payload=None):
            if method == "GET":
                return {"results": []}
            self.created.append(payload)
            return {"id": f"view-{len(self.created)}"}

    client = Client()
    names = notion_sync._create_learning_views(client, "database", "source")

    assert names == ["Ôn hôm nay", "Mục mới", "Theo loại", "Theo bài", "Đã nhớ"]
    today_filter = client.created[0]["filter"]["and"][0]
    assert today_filter == {"property": "Ngày ôn tiếp", "date": {"on_or_before": "today"}}
    assert "Đến hạn" not in notion_sync._item_schema()


def test_durable_queue_is_idempotent_and_can_retry():
    session_store.create_session("session-a")
    payload = {"external_id": "analysis:hash", "learning_items": [{"title": "A"}]}

    first = session_store.ensure_notion_sync_run("session-a", "analysis:hash", "hash", payload)
    second = session_store.ensure_notion_sync_run("session-a", "analysis:hash", "hash", payload)

    assert second["run_id"] == first["run_id"]
    assert session_store.dispatch_notion_sync_run(first["run_id"])
    assert not session_store.dispatch_notion_sync_run(first["run_id"])
    assert session_store.mark_notion_sync_running(first["run_id"])
    session_store.finish_notion_sync_run(first["run_id"], "failed", error="permission")
    assert session_store.retry_notion_sync_run(first["run_id"])
    assert session_store.get_notion_sync_run(first["run_id"])["status"] == "pending"


def test_changed_payload_resets_completed_run_without_creating_duplicate():
    session_store.create_session("session-a")
    first = session_store.ensure_notion_sync_run(
        "session-a", "analysis:hash", "hash", {"learning_items": []}
    )
    session_store.finish_notion_sync_run(first["run_id"], "done", notion_page_id="page-1")

    changed = session_store.ensure_notion_sync_run(
        "session-a", "analysis:hash", "hash", {"learning_items": [{"title": "new"}]}
    )

    assert changed["run_id"] == first["run_id"]
    assert changed["status"] == "pending"
    assert changed["notion_page_id"] == "page-1"


def test_execute_sync_keeps_lesson_when_one_learning_item_fails(monkeypatch):
    session_store.create_session("session-a")
    payload = {
        "external_id": "analysis:hash",
        "learning_items": [
            {"external_id": "learn:1", "title": "ok"},
            {"external_id": "learn:2", "title": "bad"},
        ],
    }
    run = session_store.ensure_notion_sync_run("session-a", "analysis:hash", "hash", payload)
    monkeypatch.setattr(
        notion_sync,
        "get_notion_settings",
        lambda: notion_sync.NotionSettings("token", "parent", None, None, None, None),
    )
    monkeypatch.setattr(
        notion_sync,
        "ensure_notion_workspace",
        lambda client, settings: {"lessons_data_source_id": "lessons", "items_data_source_id": "items"},
    )
    monkeypatch.setattr(
        notion_sync,
        "_upsert_lesson",
        lambda client, data_source_id, value: {"id": "lesson-page", "url": "https://notion.so/lesson"},
    )

    def _item(client, data_source_id, item, lesson_page_id):
        if item["title"] == "bad":
            raise notion_sync.NotionAPIError("invalid item", 400, "validation_error")
        return {"id": "item-page"}

    monkeypatch.setattr(notion_sync, "_upsert_learning_item", _item)

    result = notion_sync.execute_notion_sync(run, client=object())

    assert result["page_id"] == "lesson-page"
    assert len(result["item_errors"]) == 1
    saved = session_store.get_notion_sync_run(run["run_id"])
    assert saved["completed_items"] == 2
    assert saved["notion_page_url"].endswith("/lesson")
