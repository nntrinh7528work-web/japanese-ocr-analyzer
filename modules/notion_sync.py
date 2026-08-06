"""Notion workspace bootstrap, payload mapping, and idempotent synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import re
import time
from typing import Any, Callable
from urllib.parse import urlencode

import requests

from config import (
    NOTION_ITEMS_DATABASE_ID,
    NOTION_ITEMS_DATA_SOURCE_ID,
    NOTION_LESSONS_DATABASE_ID,
    NOTION_LESSONS_DATA_SOURCE_ID,
    NOTION_PARENT_PAGE_ID,
    NOTION_TOKEN,
    PUBLIC_APP_URL,
)
from modules import session_store
from modules.cost_estimator import estimate_cost, estimate_run_costs, sum_costs
from modules.job_workflow import items_source_hash
from modules.sentence_analyzer import analysis_markdown

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"
MAX_MARKDOWN_CHARS = 120_000


class NotionAPIError(RuntimeError):
    """A sanitized Notion API failure with retry/auth metadata."""

    def __init__(self, message: str, status_code: int = 0, code: str = "", retry_after: float = 0):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500 or self.status_code == 0

    @property
    def authorization_error(self) -> bool:
        return self.status_code in (401, 403)


@dataclass(frozen=True)
class NotionSettings:
    token: str | None
    parent_page_id: str | None
    lessons_database_id: str | None
    lessons_data_source_id: str | None
    items_database_id: str | None
    items_data_source_id: str | None

    @property
    def configured(self) -> bool:
        has_workspace = bool(
            self.parent_page_id
            or (self.lessons_data_source_id and self.items_data_source_id)
        )
        return bool(self.token and has_workspace)


def get_notion_settings() -> NotionSettings:
    """Combine deployment secrets with non-secret resource IDs cached in SQLite."""
    local = session_store.load_notion_workspace_config()
    return NotionSettings(
        token=NOTION_TOKEN,
        parent_page_id=NOTION_PARENT_PAGE_ID,
        lessons_database_id=NOTION_LESSONS_DATABASE_ID or local.get("lessons_database_id"),
        lessons_data_source_id=NOTION_LESSONS_DATA_SOURCE_ID or local.get("lessons_data_source_id"),
        items_database_id=NOTION_ITEMS_DATABASE_ID or local.get("items_database_id"),
        items_data_source_id=NOTION_ITEMS_DATA_SOURCE_ID or local.get("items_data_source_id"),
    )


def notion_connection_state() -> dict[str, Any]:
    settings = get_notion_settings()
    if not settings.token:
        return {"configured": False, "label": "Chưa có NOTION_TOKEN"}
    if not settings.parent_page_id and not (
        settings.lessons_data_source_id and settings.items_data_source_id
    ):
        return {"configured": False, "label": "Thiếu trang cha hoặc database ID"}
    if settings.lessons_data_source_id and settings.items_data_source_id:
        return {"configured": True, "label": "Đã kết nối", "workspace_ready": True}
    return {"configured": True, "label": "Sẵn sàng tạo bảng", "workspace_ready": False}


class NotionClient:
    """Small REST client with throttling and Notion-aware retries."""

    def __init__(
        self,
        token: str,
        *,
        http: requests.Session | Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_attempts: int = 5,
    ) -> None:
        self.http = http or requests.Session()
        self.sleep = sleep
        self.monotonic = monotonic
        self.max_attempts = max_attempts
        self._last_request_at = 0.0
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = path if path.startswith("http") else NOTION_API_BASE + path
        last_error: NotionAPIError | None = None
        for attempt in range(self.max_attempts):
            wait = 0.36 - (self.monotonic() - self._last_request_at)
            if wait > 0:
                self.sleep(wait)
            try:
                response = self.http.request(
                    method,
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=45,
                )
                self._last_request_at = self.monotonic()
            except requests.RequestException as exc:
                last_error = NotionAPIError(f"Không thể kết nối Notion: {exc}")
                if attempt + 1 < self.max_attempts:
                    self.sleep(min(2 ** attempt, 8))
                    continue
                raise last_error from exc

            if 200 <= response.status_code < 300:
                return response.json() if response.content else {}

            try:
                body = response.json()
            except Exception:
                body = {}
            retry_after = float(response.headers.get("Retry-After", 0) or 0)
            message = str(body.get("message") or f"Notion HTTP {response.status_code}")
            last_error = NotionAPIError(
                message,
                status_code=response.status_code,
                code=str(body.get("code") or ""),
                retry_after=retry_after,
            )
            if last_error.retryable and attempt + 1 < self.max_attempts:
                self.sleep(retry_after or min(2 ** attempt, 8))
                continue
            raise last_error
        raise last_error or NotionAPIError("Notion request failed")


def _text(content: Any, limit: int = 2000) -> list[dict]:
    value = str(content or "").strip()[:limit]
    return [{"type": "text", "text": {"content": value}}] if value else []


def _select_options(names: list[str]) -> dict:
    colors = ["blue", "green", "purple", "orange", "pink", "yellow", "gray"]
    return {"options": [{"name": name, "color": colors[index % len(colors)]} for index, name in enumerate(names)]}


def _lesson_schema() -> dict:
    return {
        "Tên bài": {"title": {}},
        "External ID": {"rich_text": {}},
        "Ngôn ngữ": {"select": _select_options(["Tiếng Nhật", "Tiếng Anh"])},
        "Ngày phân tích": {"date": {}},
        "Số trang": {"number": {"format": "number"}},
        "Nguồn file": {"rich_text": {}},
        "Tóm tắt": {"rich_text": {}},
        "Model": {"rich_text": {}},
        "Tổng token": {"number": {"format": "number_with_commas"}},
        "Chi phí JPY": {"number": {"format": "yen"}},
        "OCR hash": {"rich_text": {}},
        "Trạng thái": {"select": _select_options(["Hoàn tất", "Một phần"])},
        "App URL": {"url": {}},
        "Đồng bộ lúc": {"date": {}},
    }


def _item_schema() -> dict:
    return {
        "Tên": {"title": {}},
        "External ID": {"rich_text": {}},
        "Loại": {"select": _select_options(["Từ khó", "Kanji", "Từ nối", "Ngữ pháp", "Câu dài", "Cụm từ"])},
        "Ngôn ngữ": {"select": _select_options(["Tiếng Nhật", "Tiếng Anh"])},
        "Cách đọc": {"rich_text": {}},
        "Nghĩa tiếng Việt": {"rich_text": {}},
        "Ví dụ": {"rich_text": {}},
        "Hiragana ví dụ": {"rich_text": {}},
        "Bản dịch": {"rich_text": {}},
        "ID câu nguồn": {"rich_text": {}},
        "Trang": {"number": {"format": "number"}},
        "Mức độ": {"select": _select_options(["N5", "N4", "N3", "N2", "N1", "A1", "A2", "B1", "B2", "C1", "C2", "Câu khó"])},
        "Trạng thái": {"select": _select_options(["Mới", "Đang học", "Đã nhớ"])},
        "Ngày ôn tiếp": {"date": {}},
        "Lần ôn gần nhất": {"date": {}},
        "Số lần ôn": {"number": {"format": "number"}},
        "Số lần xuất hiện": {"number": {"format": "number"}},
    }


def _database_data_source_id(client: NotionClient, database: dict) -> str:
    sources = database.get("data_sources") or []
    if sources:
        return str(sources[0].get("id") or "")
    refreshed = client.request("GET", f"/databases/{database['id']}")
    sources = refreshed.get("data_sources") or []
    if not sources:
        raise NotionAPIError("Notion không trả về data source của bảng vừa tạo.")
    return str(sources[0]["id"])


def _create_database(client: NotionClient, parent_page_id: str, title: str, schema: dict) -> tuple[str, str]:
    database = client.request(
        "POST",
        "/databases",
        {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "title": _text(title),
            "is_inline": False,
            "initial_data_source": {"title": _text(title), "properties": schema},
        },
    )
    return str(database["id"]), _database_data_source_id(client, database)


def _database_id_for_source(client: NotionClient, data_source_id: str) -> str:
    source = client.request("GET", f"/data_sources/{data_source_id}")
    return str((source.get("parent") or {}).get("database_id") or "")


def _create_learning_views(
    client: NotionClient,
    database_id: str,
    data_source_id: str,
    existing_names: list[str] | None = None,
) -> list[str]:
    created_names = list(existing_names or [])
    if not created_names:
        listed = client.request("GET", f"/views?database_id={database_id}")
        for entry in listed.get("results") or []:
            view_id = entry.get("id") or (entry.get("view") or {}).get("id")
            if not view_id:
                continue
            view = client.request("GET", f"/views/{view_id}")
            if view.get("name"):
                created_names.append(str(view["name"]))
    views = [
        (
            "Ôn hôm nay",
            {
                "and": [
                    {"property": "Ngày ôn tiếp", "date": {"on_or_before": "today"}},
                    {"property": "Trạng thái", "select": {"does_not_equal": "Đã nhớ"}},
                ]
            },
            [{"property": "Ngày ôn tiếp", "direction": "ascending"}],
        ),
        ("Mục mới", {"property": "Trạng thái", "select": {"equals": "Mới"}}, []),
        ("Theo loại", None, [{"property": "Loại", "direction": "ascending"}]),
        ("Theo bài", None, [{"property": "Trang", "direction": "ascending"}]),
        ("Đã nhớ", {"property": "Trạng thái", "select": {"equals": "Đã nhớ"}}, []),
    ]
    for name, view_filter, sorts in views:
        if name in created_names:
            continue
        payload: dict[str, Any] = {
            "database_id": database_id,
            "data_source_id": data_source_id,
            "name": name,
            "type": "table",
        }
        if view_filter:
            payload["filter"] = view_filter
        if sorts:
            payload["sorts"] = sorts
        client.request("POST", "/views", payload)
        created_names.append(name)
        cached = session_store.load_notion_workspace_config()
        cached["view_names"] = created_names
        session_store.save_notion_workspace_config(cached)
    return created_names


def ensure_notion_workspace(client: NotionClient, settings: NotionSettings | None = None) -> dict:
    """Return database IDs, bootstrapping both linked tables when needed."""
    settings = settings or get_notion_settings()
    if not settings.configured:
        raise NotionAPIError("Chưa cấu hình NOTION_TOKEN và NOTION_PARENT_PAGE_ID.", 401, "not_configured")

    lesson_db = settings.lessons_database_id
    lesson_ds = settings.lessons_data_source_id
    item_db = settings.items_database_id
    item_ds = settings.items_data_source_id
    local = session_store.load_notion_workspace_config()

    if lesson_ds and not lesson_db:
        lesson_db = _database_id_for_source(client, lesson_ds)
    if item_ds and not item_db:
        item_db = _database_id_for_source(client, item_ds)

    if not lesson_ds or not item_ds:
        if not settings.parent_page_id:
            raise NotionAPIError("Thiếu NOTION_PARENT_PAGE_ID để tạo hai bảng Notion.", 400, "missing_parent")
        lesson_db, lesson_ds = _create_database(
            client, settings.parent_page_id, "Bài phân tích", _lesson_schema()
        )
        item_db, item_ds = _create_database(
            client, settings.parent_page_id, "Mục cần học", _item_schema()
        )
        session_store.save_notion_workspace_config(
            {
                "lessons_database_id": lesson_db,
                "lessons_data_source_id": lesson_ds,
                "items_database_id": item_db,
                "items_data_source_id": item_ds,
                "relation_created": False,
                "view_names": [],
            }
        )
    item_source = client.request("GET", f"/data_sources/{item_ds}")
    relation = (item_source.get("properties") or {}).get("Bài phân tích") or {}
    if relation.get("type") != "relation" and "relation" not in relation:
        client.request(
            "PATCH",
            f"/data_sources/{item_ds}",
            {
                "properties": {
                    "Bài phân tích": {
                        "relation": {
                            "data_source_id": lesson_ds,
                            "dual_property": {"synced_property_name": "Mục cần học"},
                        }
                    }
                }
            },
        )
    local = session_store.load_notion_workspace_config()
    local["relation_created"] = True
    session_store.save_notion_workspace_config(local)

    local = session_store.load_notion_workspace_config()
    view_names = _create_learning_views(
        client,
        str(item_db),
        str(item_ds),
        existing_names=list(local.get("view_names") or []),
    )

    workspace = {
        "lessons_database_id": lesson_db,
        "lessons_data_source_id": lesson_ds,
        "items_database_id": item_db,
        "items_data_source_id": item_ds,
        "relation_created": True,
        "view_names": view_names,
        "views_created": len(view_names) >= 5,
    }
    session_store.save_notion_workspace_config(workspace)
    return workspace


def _first(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            if isinstance(value, list):
                return ", ".join(str(item) for item in value if item)
            return str(value).strip()
    return ""


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _learning_external_id(language: str, item_type: str, title: str, reading: str = "") -> str:
    raw = "|".join((language, item_type, _normalize_key(title), _normalize_key(reading)))
    return "learn:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_learning_items(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the selected high-value study rows from structured analysis data."""
    language = str(analysis.get("analysis_language") or "japanese")
    pages = analysis.get("page_analyses") or [analysis]
    result: dict[str, dict[str, Any]] = {}

    def add(item_type: str, page_index: int, row: dict, *, title_keys: tuple[str, ...], **fields: tuple[str, ...]) -> None:
        title = _first(row, *title_keys)
        if not title:
            return
        reading = _first(row, *fields.get("reading", ()))
        external_id = _learning_external_id(language, item_type, title, reading)
        item = {
            "external_id": external_id,
            "title": title,
            "type": item_type,
            "language": language,
            "reading": reading,
            "meaning_vi": _first(row, *fields.get("meaning", ())),
            "example": _first(row, *fields.get("example", ())),
            "example_reading": _first(row, *fields.get("example_reading", ())),
            "translation_vi": _first(row, *fields.get("translation", ())),
            "sentence_id": _first(row, *fields.get("sentence_id", ())),
            "page_index": page_index,
            "difficulty": _first(row, *fields.get("difficulty", ())),
        }
        result.setdefault(external_id, item)

    for page_number, page in enumerate(pages, 1):
        page_index = int(page.get("page_index", page_number) or page_number)
        for row in page.get("vocabulary_important") or []:
            add(
                "Từ khó", page_index, row, title_keys=("word", "phrase"),
                reading=("reading", "hiragana"), meaning=("meaning", "meaning_vi"),
                example=("example",), example_reading=("example_hiragana", "example_reading"),
                translation=("example_translation", "translation"), difficulty=("jlpt", "cefr", "difficulty"),
            )
        if language == "japanese":
            for row in page.get("kanji_analysis") or []:
                add(
                    "Kanji", page_index, row, title_keys=("kanji", "phrase"),
                    reading=("reading", "onyomi", "kunyomi"), meaning=("meaning", "meaning_vi"),
                    example=("example", "vocab"), translation=("translation",), difficulty=("jlpt", "difficulty"),
                )
            marker_rows = page.get("connectors") or []
        else:
            for row in page.get("phrasal_collocations") or []:
                add(
                    "Cụm từ", page_index, row, title_keys=("phrase", "word"),
                    meaning=("meaning", "meaning_vi", "explanation"), example=("example",),
                    translation=("example_translation", "translation"), difficulty=("cefr", "difficulty"),
                )
            marker_rows = page.get("discourse_markers") or page.get("connectors") or []
        for row in marker_rows:
            add(
                "Từ nối", page_index, row, title_keys=("phrase", "marker", "word"),
                reading=("reading",), meaning=("meaning", "meaning_vi", "function", "role"),
                example=("example",), translation=("translation",), difficulty=("jlpt", "cefr", "difficulty"),
            )
        for row in page.get("grammar_points") or []:
            add(
                "Ngữ pháp", page_index, row, title_keys=("name", "pattern"),
                meaning=("explanation", "nuance", "meaning", "meaning_vi"),
                example=("example", "formation"), translation=("translation", "example_translation"),
                difficulty=("jlpt", "cefr", "difficulty"),
            )
        for row in page.get("sentence_breakdowns") or []:
            translations = row.get("translations") or {}
            normalized = dict(row)
            normalized["natural_translation"] = translations.get("natural") or row.get("simplified_vi")
            add(
                "Câu dài", page_index, normalized, title_keys=("original",),
                reading=("reading",), meaning=("structure_summary",),
                translation=("natural_translation",), sentence_id=("sentence_id",),
                difficulty=("difficulty",),
            )
            result[_learning_external_id(language, "Câu dài", _first(row, "original"), _first(row, "reading"))]["difficulty"] = "Câu khó"
    return list(result.values())


def _cost_snapshot(
    items: list[dict[str, Any]],
    analysis: dict[str, Any],
    billing_tier: str,
    usd_to_jpy: float,
) -> dict[str, Any]:
    model = str(analysis.get("model_used") or "gemini-3.5-flash")
    ocr_costs = [
        estimate_cost(
            (item.get("ocr_result") or {}).get("usage"),
            (item.get("ocr_result") or {}).get("model_used") or model,
            billing_tier,
        )
        for item in items
        if item.get("ocr_result")
    ]
    main = estimate_cost(analysis.get("usage"), model, billing_tier)
    guidance = estimate_run_costs(
        analysis.get("translation_guidance_runs"),
        analysis.get("translation_guidance_usage"),
        analysis.get("translation_guidance_model") or model,
        billing_tier,
    )
    sentence = estimate_run_costs(
        analysis.get("sentence_analysis_runs"),
        analysis.get("sentence_analysis_usage"),
        analysis.get("sentence_analysis_model") or model,
        billing_tier,
    )
    total = sum_costs([*ocr_costs, main, guidance, sentence])
    total["total_cost_jpy"] = float(total["total_cost_usd"]) * float(usd_to_jpy)
    return total


def build_notion_sync_payload(
    session_id: str,
    items: list[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    billing_tier: str = "free",
    usd_to_jpy: float = 155,
    created_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build a secret-free durable payload for a Notion sync run."""
    source_hash = items_source_hash(items)
    external_id = f"analysis:{source_hash}"
    created = created_at or dt.datetime.now(dt.timezone.utc)
    language = str(analysis.get("analysis_language") or "japanese")
    pages = analysis.get("page_analyses") or [analysis]
    names = [str(item.get("name") or "") for item in items if item.get("name")]
    first_name = names[0].rsplit(".", 1)[0] if names else "Bài phân tích"
    title = first_name if len(names) <= 1 else f"{first_name} và {len(names) - 1} trang khác"
    summary = str(analysis.get("summary") or "").strip()
    if not summary and pages:
        summary = " ".join(str(page.get("summary") or "").strip() for page in pages if page.get("summary"))
    cost = _cost_snapshot(items, analysis, billing_tier, usd_to_jpy)
    app_url = PUBLIC_APP_URL.rstrip("/") + "/?" + urlencode({"session": session_id})
    markdown = "\n\n".join(
        part for part in (
            f"# {title}",
            f"> Nguồn OCR: {', '.join(names) or 'Không rõ'}",
            analysis_markdown(analysis).strip(),
        ) if part
    )
    return {
        "external_id": external_id,
        "source_hash": source_hash,
        "session_id": session_id,
        "title": title,
        "language": language,
        "created_at": created.isoformat(),
        "page_count": len(pages),
        "source_names": names,
        "summary": summary,
        "model": str(analysis.get("model_used") or ""),
        "total_tokens": int(cost.get("input_tokens", 0)) + int(cost.get("output_tokens", 0)),
        "cost_jpy": float(cost.get("total_cost_jpy", 0)),
        "app_url": app_url,
        "markdown": markdown,
        "learning_items": extract_learning_items(analysis),
    }


def split_markdown(content: str, max_chars: int = MAX_MARKDOWN_CHARS) -> list[str]:
    """Split large Markdown on line boundaries under Notion request limits."""
    if len(content) <= max_chars:
        return [content]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in content.splitlines(keepends=True):
        while len(line) > max_chars:
            if current:
                chunks.append("".join(current).strip())
                current, size = [], 0
            chunks.append(line[:max_chars].strip())
            line = line[max_chars:]
        if current and size + len(line) > max_chars:
            chunks.append("".join(current).strip())
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _query_external_id(client: NotionClient, data_source_id: str, external_id: str) -> dict | None:
    result = client.request(
        "POST",
        f"/data_sources/{data_source_id}/query",
        {
            "filter": {"property": "External ID", "rich_text": {"equals": external_id}},
            "page_size": 1,
        },
    )
    rows = result.get("results") or []
    return rows[0] if rows else None


def _lesson_properties(payload: dict) -> dict:
    return {
        "Tên bài": {"title": _text(payload["title"], 300)},
        "External ID": {"rich_text": _text(payload["external_id"])},
        "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if payload["language"] == "japanese" else "Tiếng Anh"}},
        "Ngày phân tích": {"date": {"start": payload["created_at"]}},
        "Số trang": {"number": payload["page_count"]},
        "Nguồn file": {"rich_text": _text(", ".join(payload.get("source_names") or []))},
        "Tóm tắt": {"rich_text": _text(payload.get("summary"))},
        "Model": {"rich_text": _text(payload.get("model"))},
        "Tổng token": {"number": payload.get("total_tokens", 0)},
        "Chi phí JPY": {"number": round(float(payload.get("cost_jpy", 0)), 4)},
        "OCR hash": {"rich_text": _text(payload["source_hash"])},
        "Trạng thái": {"select": {"name": "Hoàn tất"}},
        "App URL": {"url": payload.get("app_url") or None},
        "Đồng bộ lúc": {"date": {"start": dt.datetime.now(dt.timezone.utc).isoformat()}},
    }


def _write_lesson_markdown(client: NotionClient, page_id: str, markdown: str) -> None:
    chunks = split_markdown(markdown)
    first = chunks[0] if chunks else "Không có nội dung phân tích."
    client.request(
        "PATCH",
        f"/pages/{page_id}/markdown",
        {"type": "replace_content", "replace_content": {"new_str": first}},
    )
    for chunk in chunks[1:]:
        client.request(
            "PATCH",
            f"/pages/{page_id}/markdown",
            {"type": "insert_content", "insert_content": {"content": "\n\n" + chunk, "position": {"type": "end"}}},
        )


def _upsert_lesson(client: NotionClient, data_source_id: str, payload: dict) -> dict:
    existing = _query_external_id(client, data_source_id, payload["external_id"])
    properties = _lesson_properties(payload)
    if existing:
        page = client.request("PATCH", f"/pages/{existing['id']}", {"properties": properties})
    else:
        page = client.request(
            "POST",
            "/pages",
            {"parent": {"type": "data_source_id", "data_source_id": data_source_id}, "properties": properties},
        )
    _write_lesson_markdown(client, str(page["id"]), str(payload.get("markdown") or ""))
    return page


def _property_number(page: dict, name: str) -> float:
    return float((((page.get("properties") or {}).get(name) or {}).get("number")) or 0)


def _property_relations(page: dict, name: str) -> list[dict]:
    return list((((page.get("properties") or {}).get(name) or {}).get("relation")) or [])


def _safe_difficulty(value: str) -> str | None:
    match = re.search(r"\b(N[1-5]|[ABC][12])\b", value.upper())
    if match:
        return match.group(1)
    return "Câu khó" if value == "Câu khó" else None


def _item_properties(item: dict, lesson_page_id: str, existing: dict | None) -> dict:
    existing_relations = _property_relations(existing or {}, "Bài phân tích")
    relation_ids = {row.get("id") for row in existing_relations}
    relation_added = lesson_page_id not in relation_ids
    relations = existing_relations + ([{"id": lesson_page_id}] if relation_added else [])
    occurrence = _property_number(existing or {}, "Số lần xuất hiện") + (1 if relation_added else 0)
    difficulty = _safe_difficulty(str(item.get("difficulty") or ""))
    properties = {
        "Tên": {"title": _text(item.get("title"), 300)},
        "External ID": {"rich_text": _text(item.get("external_id"))},
        "Loại": {"select": {"name": item.get("type")}},
        "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if item.get("language") == "japanese" else "Tiếng Anh"}},
        "Cách đọc": {"rich_text": _text(item.get("reading"))},
        "Nghĩa tiếng Việt": {"rich_text": _text(item.get("meaning_vi"))},
        "Ví dụ": {"rich_text": _text(item.get("example"))},
        "Hiragana ví dụ": {"rich_text": _text(item.get("example_reading"))},
        "Bản dịch": {"rich_text": _text(item.get("translation_vi"))},
        "ID câu nguồn": {"rich_text": _text(item.get("sentence_id"))},
        "Trang": {"number": int(item.get("page_index") or 0)},
        "Bài phân tích": {"relation": relations[:100]},
        "Số lần xuất hiện": {"number": occurrence or 1},
    }
    if difficulty:
        properties["Mức độ"] = {"select": {"name": difficulty}}
    if not existing:
        properties.update(
            {
                "Trạng thái": {"select": {"name": "Mới"}},
                "Ngày ôn tiếp": {"date": {"start": dt.date.today().isoformat()}},
                "Số lần ôn": {"number": 0},
            }
        )
    return properties


def _upsert_learning_item(client: NotionClient, data_source_id: str, item: dict, lesson_page_id: str) -> dict:
    existing = _query_external_id(client, data_source_id, item["external_id"])
    properties = _item_properties(item, lesson_page_id, existing)
    if existing:
        return client.request("PATCH", f"/pages/{existing['id']}", {"properties": properties})
    return client.request(
        "POST",
        "/pages",
        {"parent": {"type": "data_source_id", "data_source_id": data_source_id}, "properties": properties},
    )


def execute_notion_sync(run: dict, client: NotionClient | None = None) -> dict:
    """Execute one durable run; item-level failures are isolated and returned."""
    settings = get_notion_settings()
    if not settings.configured or not settings.token:
        raise NotionAPIError("Notion chưa được cấu hình trong Streamlit Secrets.", 401, "not_configured")
    client = client or NotionClient(settings.token)
    workspace = ensure_notion_workspace(client, settings)
    payload = run["payload"]
    lesson = _upsert_lesson(client, workspace["lessons_data_source_id"], payload)
    page_id = str(lesson["id"])
    page_url = str(lesson.get("url") or "")
    errors: list[dict] = []
    for index, item in enumerate(payload.get("learning_items") or [], 1):
        try:
            _upsert_learning_item(client, workspace["items_data_source_id"], item, page_id)
        except NotionAPIError as exc:
            if exc.authorization_error or exc.retryable:
                raise
            errors.append({"external_id": item.get("external_id"), "title": item.get("title"), "error": str(exc)})
        session_store.update_notion_sync_progress(
            run["run_id"],
            index,
            notion_page_id=page_id,
            notion_page_url=page_url,
            item_errors=errors,
        )
    return {"page_id": page_id, "page_url": page_url, "item_errors": errors}


def enqueue_analysis_sync(
    session_id: str,
    items: list[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    billing_tier: str = "free",
    usd_to_jpy: float = 155,
    force: bool = False,
) -> dict:
    payload = build_notion_sync_payload(
        session_id,
        items,
        analysis,
        billing_tier=billing_tier,
        usd_to_jpy=usd_to_jpy,
    )
    return session_store.ensure_notion_sync_run(
        session_id,
        payload["external_id"],
        payload["source_hash"],
        payload,
        force=force,
    )


def retry_at(attempts: int) -> str:
    delay_minutes = min(2 ** max(attempts, 0), 60)
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=delay_minutes)).isoformat()
