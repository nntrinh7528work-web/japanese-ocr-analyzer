from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from modules import notion_migration, notion_sync, session_store


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


def test_build_payload_extracts_all_rows_and_structured_notion_markdown():
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
    assert "Đối chiếu OCR và giáo viên hướng dẫn dịch" in payload["markdown"]
    assert "Trang 1: lesson.png" in payload["markdown"]
    assert "Từ vựng" in payload["markdown"]
    assert payload["render_coverage"]["complete"] is True
    assert payload["unrendered_field_count"] == 0
    assert payload["layout_version"] == "4.0"
    assert {row["type"] for row in payload["learning_items"]} == {
        "Từ vựng", "Kanji", "Từ nối", "Ngữ pháp", "Câu"
    }
    sentence = next(row for row in payload["learning_items"] if row["type"] == "Câu")
    assert sentence["sentence_id"] == "p1-s1"
    assert sentence["natural_translation"].startswith("Vì trời mưa")
    raw = json.loads(payload["raw_json"])
    assert raw["analysis"] == _analysis()
    assert raw["sources"][0]["edited_text"] == "雨なので、予定を変更した。"
    assert payload["analysis_hash"] == hashlib.sha256(payload["raw_json"].encode()).hexdigest()
    assert payload["columns"]["long_sentence_count"] == 1
    assert payload["columns"]["grammar_count"] == 1


def test_changed_analysis_creates_a_new_immutable_external_id():
    first = notion_sync.build_notion_sync_payload("session-a", _items(), _analysis())
    changed = _analysis()
    changed["page_analyses"][0]["summary"] = "Một kết quả phân tích mới."
    second = notion_sync.build_notion_sync_payload("session-a", _items(), changed)

    assert first["source_hash"] == second["source_hash"]
    assert first["analysis_hash"] != second["analysis_hash"]
    assert first["external_id"] != second["external_id"]


def test_incomplete_page_analysis_is_marked_partial_and_reports_missing_page():
    items = _items() + [
        {
            "id": "two",
            "name": "lesson-2.png",
            "edited_text": "二ページ目。",
            "ocr_result": {"ocr_notes": [], "usage": {}},
        }
    ]

    payload = notion_sync.build_notion_sync_payload("session-a", items, _analysis())
    properties = notion_sync._lesson_properties(payload)

    assert payload["page_count"] == 2
    assert payload["analyzed_page_count"] == 1
    assert payload["missing_page_indices"] == [2]
    assert payload["sync_status"] == "Một phần"
    assert "Trang chưa có kết quả: 2" in payload["markdown"]
    assert properties["Số trang"]["number"] == 2
    assert properties["Số trang đã phân tích"]["number"] == 1
    assert properties["Trạng thái"]["select"]["name"] == "Một phần"


def test_kanji_properties_keep_on_and_kun_readings_separate():
    item = {
        "title": "**響**",
        "external_id": "concept:kanji:one",
        "type": "Kanji",
        "language": "japanese",
        "meaning_vi": "**vang vọng**",
        "onyomi": "キョウ",
        "kunyomi": "ひび.く",
        "source_json": '{"kanji":"**響**"}',
        "source_checksum": "checksum",
    }

    properties = notion_sync._entity_properties(item, "kanji", "lesson", None, {}, {})

    assert properties["Kanji"]["title"][0]["text"]["content"] == "響"
    assert properties["Nghĩa tiếng Việt"]["rich_text"][0]["text"]["content"] == "vang vọng"
    assert properties["Âm On"]["rich_text"][0]["text"]["content"] == "キョウ"
    assert properties["Âm Kun"]["rich_text"][0]["text"]["content"] == "ひび.く"
    assert "**響**" in properties["Dữ liệu nguồn"]["rich_text"][0]["text"]["content"]


def test_extracts_all_vocabulary_and_sentence_patterns_with_study_fields():
    analysis = _analysis()
    page = analysis["page_analyses"][0]
    page["vocabulary_all"] = [
        {"word": "予定", "reading": "よてい", "type": "danh từ", "meaning": "kế hoạch"}
    ]
    page["sentence_patterns"] = [
        {"pattern": "理由 + ので + 結果", "components": "nguyên nhân + kết quả", "function": "nêu lý do"}
    ]

    rows = notion_sync.extract_learning_items(analysis)

    vocab = next(row for row in rows if row["title"] == "予定")
    pattern = next(row for row in rows if row["type"] == "Mẫu câu")
    assert vocab["part_of_speech"] == "danh từ"
    assert pattern["formation"] == "nguyên nhân + kết quả"
    assert pattern["nuance"] == "nêu lý do"
    assert vocab["source_checksum"]


def test_detailed_vocabulary_does_not_double_count_the_same_occurrence():
    analysis = _analysis()
    page = analysis["page_analyses"][0]
    page["vocabulary_all"] = [{"word": "変更", "reading": "へんこう", "meaning": "thay đổi"}]

    row = next(item for item in notion_sync.extract_learning_items(analysis) if item["title"] == "変更")

    assert row["important"] is True
    assert row["occurrences_in_analysis"] == 1


def test_v4_extracts_every_kanji_vocabulary_shape_and_links_concepts():
    analysis = _analysis()
    page = analysis["page_analyses"][0]
    page["source_text"] = "対応と応答について対話する。"
    page["vocabulary_all"] = [
        {"word": "対応", "reading": "たいおう", "meaning": "đối ứng", "jlpt": "N3"}
    ]
    page["kanji_analysis"] = [
        {
            "kanji": "対", "onyomi": "タイ", "kunyomi": "む.かう",
            "meaning": "đối", "vocab": "対応、対話", "example": page["source_text"],
        },
        {
            "kanji": "応", "onyomi": ["オウ"], "kunyomi": "こた.える",
            "meaning": "ứng", "vocab": ["対応", {"word": "応答", "reading": "おうとう", "meaning": "phản hồi"}],
            "example": page["source_text"],
        },
    ]

    entities = notion_sync.extract_notion_entities(analysis, "analysis:one")
    vocabulary = {row["title"]: row for row in entities["vocabulary"]}
    kanji = {row["title"]: row for row in entities["kanji"]}

    assert set(vocabulary) >= {"対応", "対話", "応答"}
    assert "Từ vựng Kanji" in vocabulary["対応"]["groups"]
    assert vocabulary["対応"]["reading"] == "たいおう"
    assert vocabulary["対応"]["missing_details"] is False
    assert vocabulary["応答"]["reading"] == "おうとう"
    assert vocabulary["応答"]["meaning_vi"] == "phản hồi"
    assert vocabulary["応答"]["missing_details"] is False
    assert len(vocabulary["対応"]["related_kanji_external_ids"]) == 2
    assert kanji["対"]["onyomi"] == "タイ"
    assert kanji["対"]["kunyomi"] == "む.かう"
    assert kanji["応"]["related_vocabulary_external_ids"]


def test_v3_durable_payload_is_upgraded_from_raw_json_without_gemini():
    payload = notion_sync.build_notion_sync_payload("session-a", _items(), _analysis())
    legacy = {key: value for key, value in payload.items() if key not in {
        "sentences", "vocabulary", "kanji", "language_items"
    }}

    upgraded = notion_sync._upgrade_payload_v4(legacy)

    assert upgraded["sentences"]
    assert upgraded["vocabulary"]
    assert upgraded["kanji"]
    assert upgraded["language_items"]


def test_kanji_vocabulary_uses_details_from_a_later_page_before_building_id():
    analysis = _analysis()
    first = analysis["page_analyses"][0]
    first["vocabulary_important"] = []
    first["kanji_analysis"] = [{"kanji": "応", "vocab": "応答", "meaning": "ứng"}]
    second = {
        **first,
        "page_index": 2,
        "source_text": "応答を待つ。",
        "vocabulary_all": [{"word": "応答", "reading": "おうとう", "meaning": "phản hồi"}],
        "kanji_analysis": [],
        "sentence_breakdowns": [],
    }
    analysis["page_analyses"] = [first, second]

    entities = notion_sync.extract_notion_entities(analysis, "analysis:one")
    matches = [row for row in entities["vocabulary"] if row["title"] == "応答"]

    assert len(matches) == 1
    assert matches[0]["reading"] == "おうとう"
    assert matches[0]["meaning_vi"] == "phản hồi"
    assert "Từ vựng Kanji" in matches[0]["groups"]


def test_language_pattern_keeps_the_display_marker():
    entities = notion_sync.extract_notion_entities(_analysis(), "analysis:one")
    grammar = next(row for row in entities["language_items"] if row["type"] == "Ngữ pháp")

    assert grammar["title"] == "～ので"


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


def test_v4_concepts_are_global_but_sentences_remain_lesson_scoped():
    analysis = _analysis()
    analysis["page_analyses"].append({
        **analysis["page_analyses"][0],
        "page_index": 2,
        "source_label": "Trang 2",
    })

    first = notion_sync.extract_notion_entities(analysis, "analysis:first")
    second = notion_sync.extract_notion_entities(analysis, "analysis:second")

    assert first["kanji"][0]["external_id"] == second["kanji"][0]["external_id"]
    assert first["sentences"][0]["external_id"] != second["sentences"][0]["external_id"]


def test_renderer_preserves_unknown_structured_fields_in_supplemental_section():
    analysis = _analysis()
    analysis["page_analyses"][0]["grammar_points"][0]["usage"] = "Dùng khi nêu nguyên nhân khách quan"
    analysis["page_analyses"][0]["new_prompt_section"] = {
        "special_note": "Không được làm mất nội dung này",
        "nested": [{"value": "Chi tiết lồng"}],
    }

    payload = notion_sync.build_notion_sync_payload("session-a", _items(), analysis)

    assert "Dữ liệu bổ sung" in payload["markdown"]
    assert "Không được làm mất nội dung này" in payload["markdown"]
    assert "Chi tiết lồng" in payload["markdown"]
    assert "Dùng khi nêu nguyên nhân khách quan" in payload["markdown"]
    assert payload["unrendered_field_count"] == 0


def test_renderer_splits_large_ocr_and_large_tables_on_semantic_boundaries():
    analysis = _analysis()
    page = analysis["page_analyses"][0]
    page["source_text"] = "長い文。\n" * 25_000
    page["vocabulary_all"] = [
        {"num": index, "word": f"単語{index}", "reading": "たんご", "meaning": "từ vựng"}
        for index in range(101)
    ]
    items = _items()
    items[0]["edited_text"] = page["source_text"]

    payload = notion_sync.build_notion_sync_payload("session-a", items, analysis)

    ocr_sections = [section for section in payload["markdown_sections"] if "OCR gốc đã được duyệt" in section]
    assert len(ocr_sections) > 1
    assert all(section.count("<details") == section.count("</details>") == 1 for section in ocr_sections)
    assert payload["markdown"].count('<table fit-page-width="true"') >= 3
    assert payload["unrendered_field_count"] == 0


def test_lesson_properties_are_metadata_only_and_report_render_coverage():
    payload = notion_sync.build_notion_sync_payload("session-a", _items(), _analysis())
    properties = notion_sync._lesson_properties(payload)

    assert "OCR gốc" not in properties
    assert "Ngữ pháp" not in properties
    assert properties["Đủ nội dung Notion"]["checkbox"] is True
    assert properties["Số trường chưa hiển thị"]["number"] == 0
    assert properties["Phiên bản bố cục"]["rich_text"][0]["text"]["content"] == "4.0"


def test_study_item_page_body_keeps_every_source_field():
    source = {
        "name": "～ので",
        "meaning": "vì",
        "usage": "Nêu nguyên nhân khách quan",
        "example_1": "雨なので、行かない。",
        "example_1_hiragana": "あめなので、いかない。",
        "comparison": {"with": "～から", "difference": "khách quan hơn"},
    }
    item = {
        "title": "～ので",
        "type": "Ngữ pháp",
        "page_index": 1,
        "source_order": 2,
        "meaning_vi": "vì",
        "source_json": json.dumps(source, ensure_ascii=False),
        "source_checksum": "checksum",
    }

    markdown = notion_sync.render_notion_item_markdown(item)

    for value in ("Nêu nguyên nhân khách quan", "雨なので、行かない。", "あめなので、いかない。", "khách quan hơn"):
        assert value in markdown


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


def test_client_uploads_raw_json_as_a_notion_file():
    http = _HTTP(
        [
            _Response(200, {"id": "upload-1", "status": "pending"}),
            _Response(200, {"id": "upload-1", "status": "uploaded"}),
        ]
    )
    client = notion_sync.NotionClient(
        "secret-token", http=http, sleep=lambda _: None, monotonic=lambda: 10
    )

    result = client.upload_file("analysis.json", b'{"ok":true}', "application/json")

    assert result["status"] == "uploaded"
    assert http.calls[1][2]["files"]["file"][0] == "analysis.json"
    assert "Content-Type" not in http.calls[1][2]["headers"]


def test_schema_upgrade_adds_columns_and_preserves_existing_select_options():
    class Client:
        def __init__(self):
            self.patch = None

        def request(self, method, path, payload=None):
            if method == "GET":
                return {
                    "properties": {
                        "Tên": {"type": "title", "title": {}},
                        "Loại": {
                            "type": "select",
                            "select": {"options": [{"id": "old", "name": "Từ khó", "color": "blue"}]},
                        },
                    }
                }
            self.patch = payload
            return {}

    client = Client()
    notion_sync._ensure_data_source_schema(client, "items", notion_sync._item_schema())

    properties = client.patch["properties"]
    assert "Thiếu chi tiết" in properties
    assert "Nhóm" in properties


def test_text_limit_uses_notion_utf16_length_for_emoji():
    value = "a" * 1998 + "😞" + "b"

    content = notion_sync._text(value)[0]["text"]["content"]

    assert content == "a" * 1998 + "😞"
    assert len(content.encode("utf-16-le")) // 2 == 2000


def test_bootstrapped_database_is_rediscovered_after_local_cache_reset():
    class Client:
        def request(self, method, path, payload=None):
            if path.startswith("/blocks/hub/children"):
                return {
                    "results": [{
                        "id": "sentence-db", "type": "child_database",
                        "child_database": {"title": "Câu & bản dịch"},
                    }],
                    "has_more": False,
                }
            if path == "/databases/sentence-db":
                return {"id": "sentence-db", "data_sources": [{"id": "sentence-source"}]}
            raise AssertionError(path)

    assert notion_sync._find_child_database(
        Client(), "hub", {"Câu & bản dịch"}
    ) == ("sentence-db", "sentence-source")


def test_duplicate_v4_database_cleanup_archives_only_noncanonical(monkeypatch):
    discovered = {
        "Câu & bản dịch": [("sentence-db", "sentence-source"), ("sentence-copy", "sentence-copy-source")],
        "Kanji": [("kanji-copy", "kanji-copy-source"), ("kanji-db", "kanji-source")],
        "Ngữ pháp & liên kết": [("language-db", "language-source")],
    }
    monkeypatch.setattr(
        notion_migration,
        "_find_child_databases",
        lambda client, parent_id, titles: discovered[next(iter(titles))],
    )

    class Client:
        def __init__(self):
            self.calls = []

        def request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            return {}

    client = Client()
    archived = notion_migration._archive_duplicate_v4_databases(
        client,
        "hub",
        {
            "sentences_database_id": "sentence-db",
            "kanji_database_id": "kanji-db",
            "language_database_id": "language-db",
        },
    )

    assert archived == ["sentence-copy", "kanji-copy"]
    assert client.calls == [
        ("PATCH", "/databases/sentence-copy", {"in_trash": True}),
        ("PATCH", "/databases/kanji-copy", {"in_trash": True}),
    ]


def test_workspace_signature_changes_when_a_database_changes():
    first = {f"{key}_data_source_id": key for key in ("lessons", "items", "sentences", "kanji", "language")}
    second = dict(first, kanji_data_source_id="different-kanji")

    assert notion_migration._workspace_signature(first) == notion_migration._workspace_signature(dict(first))
    assert notion_migration._workspace_signature(first) != notion_migration._workspace_signature(second)


def test_occurrence_count_is_idempotent_for_the_same_lesson_relation():
    item = {
        "external_id": "concept:vocabulary:one", "title": "対応",
        "type": "Từ vựng", "language": "japanese", "groups": ["Từ trong bài"],
        "occurrences_in_analysis": 2, "source_json": "[]", "source_checksum": "sum",
    }
    existing = {
        "properties": {
            "Bài phân tích": {"relation": [{"id": "lesson-one"}]},
            "Số lần xuất hiện": {"number": 3},
        }
    }

    same = notion_sync._entity_properties(item, "vocabulary", "lesson-one", existing, {}, {})
    another = notion_sync._entity_properties(item, "vocabulary", "lesson-two", existing, {}, {})

    assert same["Số lần xuất hiện"]["number"] == 3
    assert another["Số lần xuất hiện"]["number"] == 5


def test_upsert_lesson_attaches_the_exact_raw_json_archive():
    class Client:
        def __init__(self):
            self.calls = []
            self.uploaded = None

        def request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            if path.endswith("/query"):
                return {"results": []}
            if method == "POST" and path == "/pages":
                return {"id": "lesson-1", "url": "https://notion.so/lesson-1"}
            if method == "PATCH" and path == "/pages/lesson-1" and payload.get("properties"):
                return {"id": "lesson-1", "url": "https://notion.so/lesson-1"}
            return {}

        def upload_file(self, filename, content, content_type):
            self.uploaded = (filename, content, content_type)
            return {"id": "upload-1", "status": "uploaded"}

    payload = notion_sync.build_notion_sync_payload("session-a", _items(), _analysis())
    client = Client()

    page = notion_sync._upsert_lesson(client, "lessons", payload)

    assert page["id"] == "lesson-1"
    assert client.uploaded[1] == payload["raw_json"].encode("utf-8")
    attachment = client.calls[-1][2]["properties"]["Bản JSON gốc"]["files"][0]
    assert attachment["file_upload"]["id"] == "upload-1"


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

    assert names == [
        "Tất cả", "Ôn hôm nay", "Mục mới", "Đang học", "Đã nhớ", "Theo bài",
        "Từ khó", "Từ vựng Kanji", "Tiếng Nhật", "Tiếng Anh", "Theo cấp độ",
    ]
    today_filter = client.created[1]["filter"]["and"][0]
    assert today_filter == {"property": "Ngày ôn tiếp", "date": {"on_or_before": "today"}}
    assert "Đến hạn" not in notion_sync._item_schema()


def test_existing_learning_view_is_updated_instead_of_duplicated():
    class Client:
        def __init__(self):
            self.calls = []

        def request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            if path.startswith("/views?database_id"):
                return {"results": [{"id": "view-all"}]}
            if path == "/views/view-all":
                return {"id": "view-all", "name": "Tất cả"}
            if path.startswith("/data_sources/") and method == "GET":
                return {"properties": {}}
            return {"id": "created"}

    client = Client()
    notion_sync._create_learning_views(client, "database", "source")

    assert any(method == "PATCH" and path == "/views/view-all" for method, path, _ in client.calls)
    assert not any(
        method == "POST" and (payload or {}).get("name") == "Tất cả"
        for method, _, payload in client.calls
    )


def test_migration_preflight_keeps_lesson_without_readable_archive(monkeypatch):
    lesson = {"id": "lesson-1", "properties": {"Tên bài": {"title": []}}}

    class Client:
        def request(self, method, path, payload=None):
            if path.endswith("/lessons/query"):
                return {"results": [lesson], "has_more": False}
            if path.endswith("/items/query"):
                return {"results": [], "has_more": False}
            raise AssertionError(f"Unexpected mutation during preflight: {method} {path}")

    monkeypatch.setattr(
        notion_migration,
        "ensure_notion_workspace",
        lambda client, settings: {
            "lessons_data_source_id": "lessons",
            "items_data_source_id": "items",
            "lessons_database_id": "lesson-db",
        },
    )
    settings = notion_sync.NotionSettings("token", "parent", "lesson-db", "lessons", "item-db", "items")

    result = notion_migration.rebuild_notion_workspace_v3(Client(), settings, confirm=False)

    assert result["status"] == "dry_run"
    assert result["readable_lesson_count"] == 0
    assert result["unreadable_lessons"][0]["page_id"] == "lesson-1"
    assert result["would_remove_obsolete_columns"] is False


def test_confirmed_migration_backs_up_before_archive_and_restores_study_state(monkeypatch):
    archive = {
        "sources": _items(),
        "analysis": _analysis(),
        "source_hash": "source",
        "schema_version": "2.0",
    }
    lesson = {
        "id": "old-lesson",
        "properties": {
            "External ID": {"rich_text": [{"plain_text": "analysis:old"}]},
            "Ngày phân tích": {"date": {"start": "2026-08-01T00:00:00+00:00"}},
            "Bản JSON gốc": {"files": [{"type": "file", "file": {"url": "https://files/archive.json"}}]},
        },
    }
    item = {
        "id": "old-item",
        "properties": {
            "Tên": {"title": [{"plain_text": "変更"}]},
            "Loại": {"select": {"name": "Từ vựng"}},
            "Ngôn ngữ": {"select": {"name": "Tiếng Nhật"}},
            "Cách đọc": {"rich_text": [{"plain_text": "へんこう"}]},
            "Trạng thái": {"select": {"name": "Đang học"}},
            "Ngày ôn tiếp": {"date": {"start": "2026-08-10"}},
            "Số lần ôn": {"number": 3},
            "Bài phân tích": {"relation": [{"id": "old-lesson"}]},
        },
    }

    class Response:
        status_code = 200
        content = json.dumps(archive, ensure_ascii=False).encode("utf-8")

    class HTTP:
        def request(self, method, url, **kwargs):
            assert method == "GET"
            return Response()

    class Client:
        http = HTTP()

        def __init__(self):
            self.calls = []

        def request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            if path.endswith("/lessons/query"):
                return {"results": [lesson], "has_more": False}
            if path.endswith("/items/query"):
                return {"results": [item], "has_more": False}
            if method == "POST" and path == "/pages":
                return {"id": "backup-page"}
            return {"id": path.rsplit("/", 1)[-1]}

        def upload_file(self, filename, content, content_type):
            self.calls.append(("UPLOAD", filename, content_type))
            return {"id": "backup-upload"}

    client = Client()
    monkeypatch.setattr(
        notion_migration,
        "ensure_notion_workspace",
        lambda client, settings: {
            "lessons_data_source_id": "lessons",
            "items_data_source_id": "items",
            "lessons_database_id": "lesson-db",
            "items_database_id": "item-db",
        },
    )
    monkeypatch.setattr(notion_migration, "_remove_obsolete_columns", lambda *args: None)
    monkeypatch.setattr(notion_migration, "_remove_obsolete_views", lambda *args: None)
    def _new_lesson(client, data_source_id, payload):
        client.calls.append(("UPSERT_LESSON", payload["external_id"], None))
        return {"id": "new-lesson"}

    monkeypatch.setattr(notion_migration, "_upsert_lesson", _new_lesson)

    created_items = []

    def _new_item(client, data_source_id, value, lesson_page_id):
        created_items.append(value)
        return {"id": f"new-item-{len(created_items)}"}

    monkeypatch.setattr(notion_migration, "_upsert_learning_item", _new_item)
    settings = notion_sync.NotionSettings("token", "parent", "lesson-db", "lessons", "item-db", "items")

    result = notion_migration.rebuild_notion_workspace_v3(client, settings, confirm=True)

    backup_index = next(index for index, call in enumerate(client.calls) if call[0] == "UPLOAD")
    rebuild_index = next(index for index, call in enumerate(client.calls) if call[0] == "UPSERT_LESSON")
    assert backup_index < rebuild_index
    assert result["status"] == "complete"
    assert result["rebuilt_lessons"] == 1
    assert result["rebuilt_items"] == len(created_items)
    restored = [
        call for call in client.calls
        if call[0] == "PATCH" and str(call[1]).startswith("/pages/new-item-")
        and (call[2] or {}).get("properties", {}).get("Trạng thái")
    ]
    assert restored
    assert restored[0][2]["properties"]["Trạng thái"]["select"]["name"] == "Đang học"
    assert restored[0][2]["properties"]["Số lần ôn"]["number"] == 3
    assert not any(
        call[0] == "PATCH" and call[1] == "/pages/old-lesson" and (call[2] or {}).get("in_trash")
        for call in client.calls
    )
    assert any(
        call[0] == "PATCH" and call[1] == "/pages/old-item" and (call[2] or {}).get("in_trash")
        for call in client.calls
    )


def test_v4_migration_backs_up_splits_databases_and_preserves_study_state(monkeypatch):
    archive = {
        "sources": _items(),
        "analysis": _analysis(),
        "source_hash": "source",
        "schema_version": "2.0",
    }
    lesson = {
        "id": "old-lesson",
        "properties": {
            "External ID": {"rich_text": [{"plain_text": "analysis:old"}]},
            "Ngày phân tích": {"date": {"start": "2026-08-01T00:00:00+00:00"}},
            "Bản JSON gốc": {"files": [{"type": "file", "file": {"url": "https://files/archive.json"}}]},
        },
    }
    old_item = {
        "id": "old-vocab",
        "properties": {
            "Tên": {"title": [{"plain_text": "変更"}]},
            "External ID": {"rich_text": [{"plain_text": "learn:legacy"}]},
            "Loại": {"select": {"name": "Từ vựng"}},
            "Ngôn ngữ": {"select": {"name": "Tiếng Nhật"}},
            "Cách đọc": {"rich_text": [{"plain_text": "へんこう"}]},
            "Trạng thái": {"select": {"name": "Đang học"}},
            "Ngày ôn tiếp": {"date": {"start": "2026-08-10"}},
            "Lần ôn gần nhất": {"date": {"start": "2026-08-08"}},
            "Số lần ôn": {"number": 4},
            "Bài phân tích": {"relation": [{"id": "old-lesson"}]},
        },
    }

    class Response:
        status_code = 200
        content = json.dumps(archive, ensure_ascii=False).encode("utf-8")

    class HTTP:
        def request(self, method, url, **kwargs):
            return Response()

    class Client:
        http = HTTP()

        def __init__(self):
            self.calls = []

        def request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            if path.endswith("/lessons/query"):
                return {"results": [lesson], "has_more": False}
            if path.endswith("/items/query"):
                return {"results": [old_item], "has_more": False}
            return {"id": path.rsplit("/", 1)[-1]}

    workspace = {
        "lessons_database_id": "lesson-db", "lessons_data_source_id": "lessons",
        "items_database_id": "item-db", "items_data_source_id": "items",
        "sentences_database_id": "sentence-db", "sentences_data_source_id": "sentences",
        "kanji_database_id": "kanji-db", "kanji_data_source_id": "kanji",
        "language_database_id": "language-db", "language_data_source_id": "language",
    }
    client = Client()
    monkeypatch.setattr(notion_migration, "ensure_notion_workspace", lambda *args: workspace)
    monkeypatch.setattr(
        notion_migration, "_create_backup_page",
        lambda *args: client.calls.append(("BACKUP", "backup", None)) or {"id": "backup-page"},
    )
    monkeypatch.setattr(notion_migration, "_rename_vocabulary_database", lambda *args: None)
    monkeypatch.setattr(notion_migration, "_remove_v4_obsolete_columns", lambda *args: None)
    monkeypatch.setattr(notion_migration, "_append_hub_v4_links", lambda *args: None)
    monkeypatch.setattr(
        notion_migration, "_upsert_lesson",
        lambda *args: client.calls.append(("UPSERT_LESSON", "lesson", None)) or {"id": "new-lesson"},
    )
    synced = []
    monkeypatch.setattr(
        notion_migration, "_sync_payload_entities",
        lambda client, workspace, payload, lesson_id: synced.append(payload) or [],
    )
    monkeypatch.setattr(
        notion_migration, "_query_external_id",
        lambda client, source, external: {"id": f"new-{source}-{external[-6:]}"},
    )
    restored = []
    monkeypatch.setattr(
        notion_migration, "_restore_study_state",
        lambda client, page_id, state: restored.append((page_id, state)),
    )
    settings = notion_sync.NotionSettings("token", "parent", "lesson-db", "lessons", "item-db", "items")

    result = notion_migration.rebuild_notion_workspace_v4(client, settings, confirm=True)

    assert result["status"] == "complete"
    assert synced and synced[0]["sentences"] and synced[0]["kanji"]
    assert synced[0]["vocabulary"] and synced[0]["language_items"]
    assert next(index for index, call in enumerate(client.calls) if call[0] == "BACKUP") < next(
        index for index, call in enumerate(client.calls) if call[0] == "UPSERT_LESSON"
    )
    assert any(state and state["status"] == "Đang học" and state["review_count"] == 4 for _, state in restored)
    assert any(
        call[0] == "PATCH" and call[1] == "/pages/old-vocab" and call[2].get("in_trash")
        for call in client.calls
    )


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

    class Client:
        def request(self, method, path, payload=None):
            return {"id": "lesson-page", "url": "https://notion.so/lesson"}

    result = notion_sync.execute_notion_sync(run, client=Client())

    assert result["page_id"] == "lesson-page"
    assert len(result["item_errors"]) == 1
    saved = session_store.get_notion_sync_run(run["run_id"])
    assert saved["completed_items"] == 2
    assert saved["notion_page_url"].endswith("/lesson")
