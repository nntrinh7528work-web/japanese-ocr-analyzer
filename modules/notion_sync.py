"""Notion workspace bootstrap, payload mapping, and idempotent synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
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
RAW_ARCHIVE_SCHEMA_VERSION = "2.0"
NOTION_SCHEMA_VERSION = 2
DIRECT_UPLOAD_LIMIT = 20 * 1024 * 1024
MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024


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

    def _send_file_part(
        self,
        upload_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        part_number: int | None = None,
    ) -> dict:
        """Send one multipart body while preserving the REST client's retry policy."""
        url = f"{NOTION_API_BASE}/file_uploads/{upload_id}/send"
        headers = {key: value for key, value in self.headers.items() if key != "Content-Type"}
        last_error: NotionAPIError | None = None
        for attempt in range(self.max_attempts):
            wait = 0.36 - (self.monotonic() - self._last_request_at)
            if wait > 0:
                self.sleep(wait)
            try:
                response = self.http.request(
                    "POST",
                    url,
                    headers=headers,
                    files={"file": (filename, content, content_type)},
                    data={"part_number": str(part_number)} if part_number is not None else None,
                    timeout=90,
                )
                self._last_request_at = self.monotonic()
            except requests.RequestException as exc:
                last_error = NotionAPIError(f"Không thể tải file JSON lên Notion: {exc}")
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
            last_error = NotionAPIError(
                str(body.get("message") or f"Notion HTTP {response.status_code}"),
                status_code=response.status_code,
                code=str(body.get("code") or ""),
                retry_after=retry_after,
            )
            if last_error.retryable and attempt + 1 < self.max_attempts:
                self.sleep(retry_after or min(2 ** attempt, 8))
                continue
            raise last_error
        raise last_error or NotionAPIError("Notion file upload failed")

    def upload_file(self, filename: str, content: bytes, content_type: str) -> dict:
        """Upload a file through Notion's single- or multi-part upload API."""
        if len(content) <= DIRECT_UPLOAD_LIMIT:
            upload = self.request(
                "POST",
                "/file_uploads",
                {"mode": "single_part", "filename": filename, "content_type": content_type},
            )
            return self._send_file_part(str(upload["id"]), filename, content, content_type)

        part_count = math.ceil(len(content) / MULTIPART_CHUNK_SIZE)
        upload = self.request(
            "POST",
            "/file_uploads",
            {
                "mode": "multi_part",
                "filename": filename,
                "content_type": content_type,
                "number_of_parts": part_count,
            },
        )
        upload_id = str(upload["id"])
        for index in range(part_count):
            start = index * MULTIPART_CHUNK_SIZE
            self._send_file_part(
                upload_id,
                filename,
                content[start:start + MULTIPART_CHUNK_SIZE],
                content_type,
                part_number=index + 1,
            )
        return self.request("POST", f"/file_uploads/{upload_id}/complete", {})


def _text(content: Any, limit: int = 2000) -> list[dict]:
    value = str(content or "").strip()[:limit]
    return [{"type": "text", "text": {"content": value}}] if value else []


def _plain_preview(content: Any) -> str:
    """Remove Markdown decoration from searchable Notion property previews."""
    value = str(content or "").strip()
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = re.sub(r"\\([\\`*_\[\]{}()#+\-.!~>|])", r"\1", value)
    return re.sub(r"[ \t]+", " ", value).strip()


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
        "Số trang đã phân tích": {"number": {"format": "number"}},
        "Nguồn file": {"rich_text": {}},
        "Tóm tắt": {"rich_text": {}},
        "Model": {"rich_text": {}},
        "Tổng token": {"number": {"format": "number_with_commas"}},
        "Chi phí JPY": {"number": {"format": "yen"}},
        "OCR hash": {"rich_text": {}},
        "Trạng thái": {"select": _select_options(["Hoàn tất", "Một phần"])},
        "App URL": {"url": {}},
        "Đồng bộ lúc": {"date": {}},
        "Analysis hash": {"rich_text": {}},
        "Bản JSON gốc": {"files": {}},
        "Phiên bản dữ liệu": {"rich_text": {}},
        "OCR gốc": {"rich_text": {}},
        "Hướng dẫn dịch": {"rich_text": {}},
        "Dịch tự nhiên": {"rich_text": {}},
        "Từ vựng": {"rich_text": {}},
        "Kanji / Cụm từ": {"rich_text": {}},
        "Từ nối": {"rich_text": {}},
        "Ngữ pháp": {"rich_text": {}},
        "Mẫu câu": {"rich_text": {}},
        "Câu dài": {"rich_text": {}},
        "Cảnh báo OCR": {"rich_text": {}},
        "Số câu": {"number": {"format": "number"}},
        "Số từ vựng": {"number": {"format": "number"}},
        "Số kanji / cụm từ": {"number": {"format": "number"}},
        "Số từ nối": {"number": {"format": "number"}},
        "Số ngữ pháp": {"number": {"format": "number"}},
        "Số mẫu câu": {"number": {"format": "number"}},
        "Số câu dài": {"number": {"format": "number"}},
        "Token OCR": {"number": {"format": "number_with_commas"}},
        "Token phân tích": {"number": {"format": "number_with_commas"}},
        "Token hướng dẫn": {"number": {"format": "number_with_commas"}},
        "Token câu dài": {"number": {"format": "number_with_commas"}},
        "Chi phí OCR JPY": {"number": {"format": "yen"}},
        "Chi phí phân tích JPY": {"number": {"format": "yen"}},
        "Chi phí hướng dẫn JPY": {"number": {"format": "yen"}},
        "Chi phí câu dài JPY": {"number": {"format": "yen"}},
    }


def _item_schema() -> dict:
    return {
        "Tên": {"title": {}},
        "External ID": {"rich_text": {}},
        "Loại": {"select": _select_options(["Từ vựng", "Từ khó", "Kanji", "Từ nối", "Ngữ pháp", "Mẫu câu", "Câu dài", "Cụm từ"])},
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
        "Quan trọng": {"checkbox": {}},
        "Từ loại": {"rich_text": {}},
        "Từ gốc": {"rich_text": {}},
        "Công thức / Cấu tạo": {"rich_text": {}},
        "Sắc thái / Chức năng": {"rich_text": {}},
        "So sánh": {"rich_text": {}},
        "Vai trò / Liên kết": {"rich_text": {}},
        "Dịch theo cụm": {"rich_text": {}},
        "Dịch sát": {"rich_text": {}},
        "Dịch tự nhiên": {"rich_text": {}},
        "Điểm phức tạp": {"number": {"format": "number"}},
        "Dữ liệu nguồn": {"rich_text": {}},
        "JSON checksum": {"rich_text": {}},
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


def _ensure_data_source_schema(client: NotionClient, data_source_id: str, desired: dict) -> None:
    """Add missing columns/options without deleting or renaming user data."""
    source = client.request("GET", f"/data_sources/{data_source_id}")
    existing = source.get("properties") or {}
    updates: dict[str, Any] = {}
    for name, schema in desired.items():
        current = existing.get(name)
        if not current:
            updates[name] = schema
            continue
        if "select" not in schema or current.get("type") != "select":
            continue
        current_options = (current.get("select") or {}).get("options") or []
        current_names = {str(option.get("name") or "") for option in current_options}
        desired_options = (schema.get("select") or {}).get("options") or []
        missing = [option for option in desired_options if option.get("name") not in current_names]
        if missing:
            preserved = [
                ({"id": option["id"]} if option.get("id") else {"name": option.get("name")})
                for option in current_options
            ]
            updates[name] = {"select": {"options": preserved + missing}}
    if updates:
        client.request("PATCH", f"/data_sources/{data_source_id}", {"properties": updates})


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
    if int(local.get("schema_version") or 0) < NOTION_SCHEMA_VERSION:
        _ensure_data_source_schema(client, str(lesson_ds), _lesson_schema())
        _ensure_data_source_schema(client, str(item_ds), _item_schema())
        local = session_store.load_notion_workspace_config()
        local["schema_version"] = NOTION_SCHEMA_VERSION
        session_store.save_notion_workspace_config(local)
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
        "schema_version": NOTION_SCHEMA_VERSION,
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
    """Extract every study row while retaining the original structured record."""
    language = str(analysis.get("analysis_language") or "japanese")
    pages = analysis.get("page_analyses") or [analysis]
    result: dict[str, dict[str, Any]] = {}

    def add(
        item_type: str,
        page_index: int,
        row: dict,
        *,
        title_keys: tuple[str, ...],
        important: bool = False,
        count_occurrence: bool = True,
        **fields: tuple[str, ...],
    ) -> None:
        title = _first(row, *title_keys)
        if not title:
            return
        reading = _first(row, *fields.get("reading", ()))
        external_id = _learning_external_id(language, item_type, title, reading)
        source_json = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
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
            "important": important,
            "part_of_speech": _first(row, *fields.get("part_of_speech", ())),
            "base_form": _first(row, *fields.get("base_form", ())),
            "formation": _first(row, *fields.get("formation", ())),
            "nuance": _first(row, *fields.get("nuance", ())),
            "comparison": _first(row, *fields.get("comparison", ())),
            "linked_parts": _first(row, *fields.get("linked_parts", ())),
            "chunked_translation": _first(row, *fields.get("chunked_translation", ())),
            "literal_translation": _first(row, *fields.get("literal_translation", ())),
            "natural_translation": _first(row, *fields.get("natural_translation", ())),
            "complexity_score": float(row.get("complexity_score") or 0),
            "source_json": source_json,
            "source_checksum": hashlib.sha256(source_json.encode("utf-8")).hexdigest(),
            "occurrences_in_analysis": 1,
        }
        existing = result.get(external_id)
        if existing:
            if count_occurrence:
                existing["occurrences_in_analysis"] = int(existing.get("occurrences_in_analysis") or 1) + 1
            existing["important"] = bool(existing.get("important") or important)
            for key, value in item.items():
                if value not in (None, "", [], 0, 0.0) and existing.get(key) in (None, "", [], 0, 0.0):
                    existing[key] = value
            if len(source_json) > len(str(existing.get("source_json") or "")):
                existing["source_json"] = source_json
                existing["source_checksum"] = item["source_checksum"]
            return
        result[external_id] = item

    for page_number, page in enumerate(pages, 1):
        page_index = int(page.get("page_index", page_number) or page_number)
        for row in page.get("vocabulary_all") or []:
            add(
                "Từ vựng", page_index, row, title_keys=("word", "phrase"),
                reading=("reading", "hiragana"), meaning=("meaning", "meaning_vi"),
                example=("example",), example_reading=("example_hiragana", "example_reading"),
                translation=("example_translation", "translation"), difficulty=("jlpt", "cefr", "difficulty"),
                part_of_speech=("part_of_speech", "type"), base_form=("base_form",),
                nuance=("nuance", "usage", "note"), comparison=("comparison",),
            )
        for row in page.get("vocabulary_important") or []:
            add(
                "Từ vựng", page_index, row, title_keys=("word", "phrase"),
                important=True, count_occurrence=False,
                reading=("reading", "hiragana"), meaning=("meaning", "meaning_vi"),
                example=("example",), example_reading=("example_hiragana", "example_reading"),
                translation=("example_translation", "translation"), difficulty=("jlpt", "cefr", "difficulty"),
                part_of_speech=("part_of_speech", "type"), base_form=("base_form",),
                formation=("formation", "structure"), nuance=("nuance", "usage", "note"),
                comparison=("comparison",),
            )
        if language == "japanese":
            for row in page.get("kanji_analysis") or []:
                add(
                    "Kanji", page_index, row, title_keys=("kanji", "phrase"),
                    reading=("reading", "onyomi", "kunyomi"), meaning=("meaning", "meaning_vi"),
                    example=("example", "vocab"), translation=("translation",), difficulty=("jlpt", "difficulty"),
                    nuance=("role",),
                )
            marker_rows = page.get("connectors") or []
        else:
            for row in page.get("phrasal_collocations") or []:
                add(
                    "Cụm từ", page_index, row, title_keys=("phrase", "word"),
                    meaning=("meaning", "meaning_vi", "explanation"), example=("example",),
                    translation=("example_translation", "translation"), difficulty=("cefr", "difficulty"),
                    part_of_speech=("type",), nuance=("note", "usage"),
                )
            marker_rows = page.get("discourse_markers") or page.get("connectors") or []
        for row in marker_rows:
            add(
                "Từ nối", page_index, row, title_keys=("phrase", "marker", "word"),
                reading=("reading",), meaning=("meaning", "meaning_vi", "function", "role"),
                example=("example",), translation=("translation",), difficulty=("jlpt", "cefr", "difficulty"),
                part_of_speech=("type",), formation=("structure",),
                nuance=("role", "function", "register", "usage"), linked_parts=("linked_parts",),
            )
        for row in page.get("grammar_points") or []:
            add(
                "Ngữ pháp", page_index, row, title_keys=("name", "pattern"),
                meaning=("explanation", "nuance", "meaning", "meaning_vi"),
                example=("example", "formation"), translation=("translation", "example_translation"),
                difficulty=("jlpt", "cefr", "difficulty"),
                formation=("formation", "structure", "rule"),
                nuance=("nuance", "usage", "explanation"),
                comparison=("comparison", "note", "mistake"),
            )
        for row in page.get("sentence_patterns") or []:
            add(
                "Mẫu câu", page_index, row, title_keys=("pattern", "name"),
                meaning=("explanation", "function", "meaning", "meaning_vi"),
                example=("example",), translation=("translation", "example_translation"),
                difficulty=("jlpt", "cefr", "difficulty", "level"),
                formation=("components", "structure", "formation"),
                nuance=("function", "usage", "explanation"), comparison=("comparison", "note"),
            )
        for row in page.get("sentence_breakdowns") or []:
            translations = row.get("translations") or {}
            normalized = dict(row)
            normalized["natural_translation"] = translations.get("natural") or row.get("simplified_vi")
            normalized["chunked_translation"] = translations.get("chunked")
            normalized["literal_translation"] = translations.get("literal")
            add(
                "Câu dài", page_index, normalized, title_keys=("original",),
                reading=("reading",), meaning=("structure_summary",),
                translation=("natural_translation",), sentence_id=("sentence_id",),
                difficulty=("difficulty",),
                formation=("structure_summary",), natural_translation=("natural_translation",),
                chunked_translation=("chunked_translation",), literal_translation=("literal_translation",),
                nuance=("simplified_vi",),
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
    ocr = sum_costs(ocr_costs)
    total = sum_costs([ocr, main, guidance, sentence])
    total["total_cost_jpy"] = float(total["total_cost_usd"]) * float(usd_to_jpy)
    breakdown = {
        "ocr": ocr,
        "analysis": main,
        "guidance": guidance,
        "sentence": sentence,
    }
    for value in breakdown.values():
        value["total_cost_jpy"] = float(value.get("total_cost_usd", 0)) * float(usd_to_jpy)
        value["total_tokens"] = int(value.get("input_tokens", 0)) + int(value.get("output_tokens", 0))
    total["breakdown"] = breakdown
    return total


def _json_ready(value: Any) -> Any:
    """Return JSON-safe data while omitting image bytes from the archive."""
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
            if not str(key).endswith("_bytes")
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bytes):
        return {"omitted_binary_bytes": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _raw_analysis_archive(
    items: list[dict[str, Any]], analysis: dict[str, Any], source_hash: str
) -> tuple[str, str]:
    """Serialize the original structured result deterministically for integrity checks."""
    archive = {
        "schema_version": RAW_ARCHIVE_SCHEMA_VERSION,
        "source_hash": source_hash,
        "sources": [_json_ready(item) for item in items],
        "analysis": _json_ready(analysis),
    }
    raw_json = json.dumps(archive, ensure_ascii=False, indent=2, sort_keys=True)
    return raw_json, hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def _row_preview(row: dict, title_keys: tuple[str, ...], detail_keys: tuple[str, ...]) -> str:
    title = _plain_preview(_first(row, *title_keys))
    detail = _plain_preview(_first(row, *detail_keys))
    if title and detail:
        return f"{title}: {detail}"
    return title or detail


def _analysis_column_snapshot(
    items: list[dict[str, Any]], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Build searchable Notion-column previews; full fidelity remains in page/JSON."""
    pages = analysis.get("page_analyses") or [analysis]
    sections: dict[str, list[str]] = {
        "ocr": [], "guidance": [], "natural": [], "vocabulary": [], "script": [],
        "markers": [], "grammar": [], "patterns": [], "sentences": [], "warnings": [],
    }
    counts = {
        "sentence_count": 0, "vocabulary_count": 0, "script_count": 0,
        "marker_count": 0, "grammar_count": 0, "pattern_count": 0,
        "long_sentence_count": 0,
    }
    for fallback_index, page in enumerate(pages, 1):
        page_index = int(page.get("page_index", fallback_index) or fallback_index)
        prefix = f"P{page_index}"
        source_text = str(page.get("source_text") or page.get("confirmed_text") or "").strip()
        if not source_text and fallback_index <= len(items):
            source_text = str(items[fallback_index - 1].get("edited_text") or "").strip()
        if source_text:
            sections["ocr"].append(f"{prefix}: {source_text}")

        catalog = page.get("sentence_catalog") or []
        guidance_rows = page.get("translation_guidance") or []
        counts["sentence_count"] += len(catalog) or len(guidance_rows)
        for row in guidance_rows:
            translations = row.get("translations") or {}
            natural = _plain_preview(translations.get("natural"))
            key_points = row.get("key_points") or []
            points = "; ".join(
                _plain_preview(_first(point, "explanation_vi", "label", "source"))
                for point in key_points
            )
            sentence_id = str(row.get("sentence_id") or prefix)
            if natural or points:
                sections["guidance"].append(f"{sentence_id}: {natural}" + (f" | {points}" if points else ""))
            if natural:
                sections["natural"].append(f"{sentence_id}: {natural}")
            warning = _plain_preview(row.get("ocr_warning"))
            if warning:
                sections["warnings"].append(f"{sentence_id}: {warning}")

        vocab = page.get("vocabulary_all") or []
        counts["vocabulary_count"] += len(vocab)
        sections["vocabulary"].extend(
            f"{prefix}: {_row_preview(row, ('word', 'phrase'), ('meaning', 'meaning_vi'))}"
            for row in vocab
            if _row_preview(row, ("word", "phrase"), ("meaning", "meaning_vi"))
        )
        script_rows = (
            page.get("kanji_analysis") or []
            if str(analysis.get("analysis_language") or "japanese") == "japanese"
            else page.get("phrasal_collocations") or []
        )
        counts["script_count"] += len(script_rows)
        sections["script"].extend(
            f"{prefix}: {_row_preview(row, ('kanji', 'phrase'), ('meaning', 'meaning_vi', 'explanation'))}"
            for row in script_rows
            if _row_preview(row, ("kanji", "phrase"), ("meaning", "meaning_vi", "explanation"))
        )
        marker_rows = page.get("connectors") or page.get("discourse_markers") or []
        counts["marker_count"] += len(marker_rows)
        sections["markers"].extend(
            f"{prefix}: {_row_preview(row, ('phrase', 'marker'), ('meaning', 'function', 'role'))}"
            for row in marker_rows
            if _row_preview(row, ("phrase", "marker"), ("meaning", "function", "role"))
        )
        grammar_rows = page.get("grammar_points") or []
        counts["grammar_count"] += len(grammar_rows)
        sections["grammar"].extend(
            f"{prefix}: {_row_preview(row, ('name', 'pattern'), ('explanation', 'nuance', 'meaning', 'rule'))}"
            for row in grammar_rows
            if _row_preview(row, ("name", "pattern"), ("explanation", "nuance", "meaning", "rule"))
        )
        pattern_rows = page.get("sentence_patterns") or []
        counts["pattern_count"] += len(pattern_rows)
        sections["patterns"].extend(
            f"{prefix}: {_row_preview(row, ('pattern', 'name'), ('explanation', 'function', 'meaning'))}"
            for row in pattern_rows
            if _row_preview(row, ("pattern", "name"), ("explanation", "function", "meaning"))
        )
        breakdown_rows = page.get("sentence_breakdowns") or []
        counts["long_sentence_count"] += len(breakdown_rows)
        for row in breakdown_rows:
            natural = _plain_preview((row.get("translations") or {}).get("natural"))
            sections["sentences"].append(
                f"{row.get('sentence_id') or prefix}: {_first(row, 'original')}"
                + (f" → {natural}" if natural else "")
            )
        for warning in page.get("ocr_corrections") or []:
            sections["warnings"].append(f"{prefix}: {_plain_preview(warning)}")
    return {**{key: "\n".join(values) for key, values in sections.items()}, **counts}


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
    created = created_at or dt.datetime.now(dt.timezone.utc)
    language = str(analysis.get("analysis_language") or "japanese")
    pages = analysis.get("page_analyses") or [analysis]
    source_page_count = len(items) or len(pages)
    analyzed_page_count = len(pages)
    analyzed_page_indices = {
        int(page.get("page_index", index) or index)
        for index, page in enumerate(pages, 1)
    }
    missing_page_indices = [
        index for index in range(1, source_page_count + 1)
        if index not in analyzed_page_indices
    ]
    sync_status = (
        "Hoàn tất"
        if source_page_count > 0 and analyzed_page_count == source_page_count and not missing_page_indices
        else "Một phần"
    )
    names = [str(item.get("name") or "") for item in items if item.get("name")]
    first_name = names[0].rsplit(".", 1)[0] if names else "Bài phân tích"
    title = first_name if len(names) <= 1 else f"{first_name} và {len(names) - 1} trang khác"
    summary = str(analysis.get("summary") or "").strip()
    if not summary and pages:
        summary = " ".join(str(page.get("summary") or "").strip() for page in pages if page.get("summary"))
    cost = _cost_snapshot(items, analysis, billing_tier, usd_to_jpy)
    raw_json, analysis_hash = _raw_analysis_archive(items, analysis, source_hash)
    external_id = f"analysis:{source_hash}:{analysis_hash[:16]}"
    columns = _analysis_column_snapshot(items, analysis)
    app_url = PUBLIC_APP_URL.rstrip("/") + "/?" + urlencode({"session": session_id})
    status_note = ""
    if sync_status == "Một phần":
        missing_label = ", ".join(str(index) for index in missing_page_indices) or "không xác định"
        status_note = (
            "> ⚠️ Đồng bộ một phần: tài liệu có "
            f"{source_page_count} trang nguồn nhưng mới có kết quả của {analyzed_page_count} trang. "
            f"Trang chưa có kết quả: {missing_label}."
        )
    markdown = "\n\n".join(
        part for part in (
            f"# {title}",
            f"> Nguồn OCR: {', '.join(names) or 'Không rõ'}",
            status_note,
            (
                f"> Bản JSON gốc được đính kèm với SHA-256 `{analysis_hash}`. "
                "Các cột chỉ là bản tóm lược để tìm kiếm; nội dung gốc không bị diễn giải lại."
            ),
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
        "page_count": source_page_count,
        "analyzed_page_count": analyzed_page_count,
        "missing_page_indices": missing_page_indices,
        "sync_status": sync_status,
        "source_names": names,
        "summary": summary,
        "model": str(analysis.get("model_used") or ""),
        "total_tokens": int(cost.get("input_tokens", 0)) + int(cost.get("output_tokens", 0)),
        "cost_jpy": float(cost.get("total_cost_jpy", 0)),
        "cost_breakdown": cost.get("breakdown") or {},
        "app_url": app_url,
        "markdown": markdown,
        "analysis_hash": analysis_hash,
        "raw_json": raw_json,
        "raw_json_filename": f"analysis-{analysis_hash[:16]}.json",
        "archive_schema_version": RAW_ARCHIVE_SCHEMA_VERSION,
        "columns": columns,
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
    columns = payload.get("columns") or {}
    costs = payload.get("cost_breakdown") or {}

    def tokens(name: str) -> int:
        value = costs.get(name) or {}
        return int(value.get("total_tokens") or 0)

    def jpy(name: str) -> float:
        value = costs.get(name) or {}
        return round(float(value.get("total_cost_jpy") or 0), 4)

    return {
        "Tên bài": {"title": _text(payload["title"], 300)},
        "External ID": {"rich_text": _text(payload["external_id"])},
        "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if payload["language"] == "japanese" else "Tiếng Anh"}},
        "Ngày phân tích": {"date": {"start": payload["created_at"]}},
        "Số trang": {"number": payload["page_count"]},
        "Số trang đã phân tích": {"number": payload.get("analyzed_page_count", payload["page_count"])},
        "Nguồn file": {"rich_text": _text(", ".join(payload.get("source_names") or []))},
        "Tóm tắt": {"rich_text": _text(payload.get("summary"))},
        "Model": {"rich_text": _text(payload.get("model"))},
        "Tổng token": {"number": payload.get("total_tokens", 0)},
        "Chi phí JPY": {"number": round(float(payload.get("cost_jpy", 0)), 4)},
        "OCR hash": {"rich_text": _text(payload["source_hash"])},
        "Trạng thái": {"select": {"name": payload.get("sync_status") or "Hoàn tất"}},
        "App URL": {"url": payload.get("app_url") or None},
        "Đồng bộ lúc": {"date": {"start": dt.datetime.now(dt.timezone.utc).isoformat()}},
        "Analysis hash": {"rich_text": _text(payload.get("analysis_hash"))},
        "Phiên bản dữ liệu": {"rich_text": _text(payload.get("archive_schema_version"))},
        "OCR gốc": {"rich_text": _text(columns.get("ocr"))},
        "Hướng dẫn dịch": {"rich_text": _text(columns.get("guidance"))},
        "Dịch tự nhiên": {"rich_text": _text(columns.get("natural"))},
        "Từ vựng": {"rich_text": _text(columns.get("vocabulary"))},
        "Kanji / Cụm từ": {"rich_text": _text(columns.get("script"))},
        "Từ nối": {"rich_text": _text(columns.get("markers"))},
        "Ngữ pháp": {"rich_text": _text(columns.get("grammar"))},
        "Mẫu câu": {"rich_text": _text(columns.get("patterns"))},
        "Câu dài": {"rich_text": _text(columns.get("sentences"))},
        "Cảnh báo OCR": {"rich_text": _text(columns.get("warnings"))},
        "Số câu": {"number": int(columns.get("sentence_count") or 0)},
        "Số từ vựng": {"number": int(columns.get("vocabulary_count") or 0)},
        "Số kanji / cụm từ": {"number": int(columns.get("script_count") or 0)},
        "Số từ nối": {"number": int(columns.get("marker_count") or 0)},
        "Số ngữ pháp": {"number": int(columns.get("grammar_count") or 0)},
        "Số mẫu câu": {"number": int(columns.get("pattern_count") or 0)},
        "Số câu dài": {"number": int(columns.get("long_sentence_count") or 0)},
        "Token OCR": {"number": tokens("ocr")},
        "Token phân tích": {"number": tokens("analysis")},
        "Token hướng dẫn": {"number": tokens("guidance")},
        "Token câu dài": {"number": tokens("sentence")},
        "Chi phí OCR JPY": {"number": jpy("ocr")},
        "Chi phí phân tích JPY": {"number": jpy("analysis")},
        "Chi phí hướng dẫn JPY": {"number": jpy("guidance")},
        "Chi phí câu dài JPY": {"number": jpy("sentence")},
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
    existing_properties = (existing or {}).get("properties") or {}
    existing_hash = "".join(
        str(value.get("plain_text") or "")
        for value in (existing_properties.get("Analysis hash") or {}).get("rich_text") or []
    )
    existing_files = (existing_properties.get("Bản JSON gốc") or {}).get("files") or []
    if existing_hash != payload.get("analysis_hash") or not existing_files:
        raw_bytes = str(payload.get("raw_json") or "{}").encode("utf-8")
        filename = str(payload.get("raw_json_filename") or "analysis.json")
        upload = client.upload_file(filename, raw_bytes, "application/json")
        page = client.request(
            "PATCH",
            f"/pages/{page['id']}",
            {
                "properties": {
                    "Bản JSON gốc": {
                        "files": [
                            {
                                "name": filename,
                                "type": "file_upload",
                                "file_upload": {"id": str(upload["id"])},
                            }
                        ]
                    }
                }
            },
        )
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
    occurrence_delta = int(item.get("occurrences_in_analysis") or 1) if relation_added else 0
    occurrence = _property_number(existing or {}, "Số lần xuất hiện") + occurrence_delta
    difficulty = _safe_difficulty(str(item.get("difficulty") or ""))
    properties = {
        "Tên": {"title": _text(_plain_preview(item.get("title")), 300)},
        "External ID": {"rich_text": _text(item.get("external_id"))},
        "Loại": {"select": {"name": item.get("type")}},
        "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if item.get("language") == "japanese" else "Tiếng Anh"}},
        "Cách đọc": {"rich_text": _text(_plain_preview(item.get("reading")))},
        "Nghĩa tiếng Việt": {"rich_text": _text(_plain_preview(item.get("meaning_vi")))},
        "Ví dụ": {"rich_text": _text(_plain_preview(item.get("example")))},
        "Hiragana ví dụ": {"rich_text": _text(_plain_preview(item.get("example_reading")))},
        "Bản dịch": {"rich_text": _text(_plain_preview(item.get("translation_vi")))},
        "ID câu nguồn": {"rich_text": _text(item.get("sentence_id"))},
        "Trang": {"number": int(item.get("page_index") or 0)},
        "Bài phân tích": {"relation": relations[:100]},
        "Số lần xuất hiện": {"number": occurrence or 1},
        "Quan trọng": {"checkbox": bool(item.get("important"))},
        "Từ loại": {"rich_text": _text(_plain_preview(item.get("part_of_speech")))},
        "Từ gốc": {"rich_text": _text(_plain_preview(item.get("base_form")))},
        "Công thức / Cấu tạo": {"rich_text": _text(_plain_preview(item.get("formation")))},
        "Sắc thái / Chức năng": {"rich_text": _text(_plain_preview(item.get("nuance")))},
        "So sánh": {"rich_text": _text(_plain_preview(item.get("comparison")))},
        "Vai trò / Liên kết": {"rich_text": _text(_plain_preview(item.get("linked_parts")))},
        "Dịch theo cụm": {"rich_text": _text(_plain_preview(item.get("chunked_translation")))},
        "Dịch sát": {"rich_text": _text(_plain_preview(item.get("literal_translation")))},
        "Dịch tự nhiên": {
            "rich_text": _text(_plain_preview(item.get("natural_translation") or item.get("translation_vi")))
        },
        "Điểm phức tạp": {"number": float(item.get("complexity_score") or 0)},
        "Dữ liệu nguồn": {"rich_text": _text(item.get("source_json"))},
        "JSON checksum": {"rich_text": _text(item.get("source_checksum"))},
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
    if errors:
        client.request(
            "PATCH",
            f"/pages/{page_id}",
            {"properties": {"Trạng thái": {"select": {"name": "Một phần"}}}},
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
