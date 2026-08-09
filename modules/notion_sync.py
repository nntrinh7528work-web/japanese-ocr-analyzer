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
    NOTION_KANJI_DATABASE_ID,
    NOTION_KANJI_DATA_SOURCE_ID,
    NOTION_LANGUAGE_DATABASE_ID,
    NOTION_LANGUAGE_DATA_SOURCE_ID,
    NOTION_LESSONS_DATABASE_ID,
    NOTION_LESSONS_DATA_SOURCE_ID,
    NOTION_PARENT_PAGE_ID,
    NOTION_SENTENCES_DATABASE_ID,
    NOTION_SENTENCES_DATA_SOURCE_ID,
    NOTION_TOKEN,
    PUBLIC_APP_URL,
)
from modules import session_store
from modules.cost_estimator import estimate_cost, estimate_run_costs, sum_costs
from modules.job_workflow import items_source_hash
from modules.notion_renderer import (
    NOTION_LAYOUT_VERSION,
    render_notion_item_markdown,
    render_notion_lesson_markdown,
)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"
MAX_MARKDOWN_CHARS = 120_000
RAW_ARCHIVE_SCHEMA_VERSION = "3.0"
NOTION_SCHEMA_VERSION = 4
NOTION_VIEWS_VERSION = 4
DIRECT_UPLOAD_LIMIT = 20 * 1024 * 1024
MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024
LEARNING_VIEW_NAMES = [
    "Tất cả", "Từ vựng", "Từ khó", "Kanji", "Cụm từ", "Từ nối",
    "Ngữ pháp", "Mẫu câu", "Câu dài", "Ôn hôm nay", "Theo bài", "Đã nhớ",
]


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
    sentences_database_id: str | None = None
    sentences_data_source_id: str | None = None
    kanji_database_id: str | None = None
    kanji_data_source_id: str | None = None
    language_database_id: str | None = None
    language_data_source_id: str | None = None

    @property
    def configured(self) -> bool:
        has_workspace = bool(self.parent_page_id or all((
            self.lessons_data_source_id, self.items_data_source_id,
            self.sentences_data_source_id, self.kanji_data_source_id,
            self.language_data_source_id,
        )))
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
        sentences_database_id=NOTION_SENTENCES_DATABASE_ID or local.get("sentences_database_id"),
        sentences_data_source_id=NOTION_SENTENCES_DATA_SOURCE_ID or local.get("sentences_data_source_id"),
        kanji_database_id=NOTION_KANJI_DATABASE_ID or local.get("kanji_database_id"),
        kanji_data_source_id=NOTION_KANJI_DATA_SOURCE_ID or local.get("kanji_data_source_id"),
        language_database_id=NOTION_LANGUAGE_DATABASE_ID or local.get("language_database_id"),
        language_data_source_id=NOTION_LANGUAGE_DATA_SOURCE_ID or local.get("language_data_source_id"),
    )


def notion_connection_state() -> dict[str, Any]:
    settings = get_notion_settings()
    if not settings.token:
        return {"configured": False, "label": "Chưa có NOTION_TOKEN"}
    if not settings.configured:
        return {"configured": False, "label": "Thiếu trang cha hoặc database ID"}
    migration = session_store.load_notion_workspace_config().get("migration_v4") or {}
    migration_status = str(migration.get("status") or "")
    if migration_status in {"starting", "backed_up", "running"}:
        return {"configured": True, "label": "Đang nâng cấp bố cục v4", "workspace_ready": False}
    if migration_status == "retry":
        return {
            "configured": True,
            "label": "Nâng cấp v4 sẽ tự thử lại",
            "workspace_ready": False,
            "migration_error": str(migration.get("error") or ""),
        }
    if all((settings.lessons_data_source_id, settings.items_data_source_id,
            settings.sentences_data_source_id, settings.kanji_data_source_id,
            settings.language_data_source_id)):
        label = "Đã kết nối · bố cục v4" if migration_status in {"complete", "partial", "not_needed"} else "Đã kết nối"
        return {"configured": True, "label": label, "workspace_ready": True}
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
    value = str(content or "").strip()
    if len(value.encode("utf-16-le")) // 2 > limit:
        used_units = 0
        prefix: list[str] = []
        for character in value:
            character_units = 2 if ord(character) > 0xFFFF else 1
            if used_units + character_units > limit:
                break
            prefix.append(character)
            used_units += character_units
        value = "".join(prefix)
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
        "Phiên bản bố cục": {"rich_text": {}},
        "Đủ nội dung Notion": {"checkbox": {}},
        "Số trường chưa hiển thị": {"number": {"format": "number"}},
        "Lỗi đồng bộ": {"rich_text": {}},
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
    """Vocabulary database schema (the legacy items database is reused in v4)."""
    return {
        "Tên": {"title": {}},
        "External ID": {"rich_text": {}},
        "Nhóm": {"multi_select": _select_options(["Từ trong bài", "Từ khó", "Từ vựng Kanji", "Cụm từ"])},
        "Ngôn ngữ": {"select": _select_options(["Tiếng Nhật", "Tiếng Anh"])},
        "Cách đọc": {"rich_text": {}},
        "Nghĩa tiếng Việt": {"rich_text": {}},
        "Ví dụ": {"rich_text": {}},
        "Hiragana ví dụ": {"rich_text": {}},
        "Bản dịch": {"rich_text": {}},
        "Mức độ": {"select": _select_options(["N5", "N4", "N3", "N2", "N1", "A1", "A2", "B1", "B2", "C1", "C2", "Câu khó"])},
        "Trạng thái": {"select": _select_options(["Mới", "Đang học", "Đã nhớ"])},
        "Ngày ôn tiếp": {"date": {}},
        "Lần ôn gần nhất": {"date": {}},
        "Số lần ôn": {"number": {"format": "number"}},
        "Số lần xuất hiện": {"number": {"format": "number"}},
        "Thiếu chi tiết": {"checkbox": {}},
        "Từ loại": {"rich_text": {}},
        "Từ gốc": {"rich_text": {}},
        "Sắc thái / Chức năng": {"rich_text": {}},
        "So sánh": {"rich_text": {}},
        "Dữ liệu nguồn": {"rich_text": {}},
        "JSON checksum": {"rich_text": {}},
    }


def _sentence_schema() -> dict:
    return {
        "Câu": {"title": {}},
        "External ID": {"rich_text": {}},
        "ID câu": {"rich_text": {}},
        "Ngôn ngữ": {"select": _select_options(["Tiếng Nhật", "Tiếng Anh"])},
        "Trang": {"number": {"format": "number"}},
        "Thứ tự câu": {"number": {"format": "number"}},
        "Nguyên văn": {"rich_text": {}},
        "Hiragana": {"rich_text": {}},
        "Dịch tự nhiên": {"rich_text": {}},
        "Câu khó": {"checkbox": {}},
        "Điểm phức tạp": {"number": {"format": "number"}},
        "Cảnh báo OCR": {"rich_text": {}},
        "Trạng thái": {"select": _select_options(["Mới", "Đang học", "Đã nhớ"])},
        "Ngày ôn tiếp": {"date": {}},
        "Lần ôn gần nhất": {"date": {}},
        "Số lần ôn": {"number": {"format": "number"}},
        "Dữ liệu nguồn": {"rich_text": {}},
        "JSON checksum": {"rich_text": {}},
    }


def _kanji_schema() -> dict:
    return {
        "Kanji": {"title": {}},
        "External ID": {"rich_text": {}},
        "Âm On": {"rich_text": {}},
        "Âm Kun": {"rich_text": {}},
        "Nghĩa tiếng Việt": {"rich_text": {}},
        "JLPT": {"select": _select_options(["N5", "N4", "N3", "N2", "N1"])},
        "Vai trò trong bài": {"rich_text": {}},
        "Ví dụ": {"rich_text": {}},
        "Số lần xuất hiện": {"number": {"format": "number"}},
        "Trạng thái": {"select": _select_options(["Mới", "Đang học", "Đã nhớ"])},
        "Ngày ôn tiếp": {"date": {}},
        "Lần ôn gần nhất": {"date": {}},
        "Số lần ôn": {"number": {"format": "number"}},
        "Dữ liệu nguồn": {"rich_text": {}},
        "JSON checksum": {"rich_text": {}},
    }


def _language_schema() -> dict:
    return {
        "Tên": {"title": {}},
        "External ID": {"rich_text": {}},
        "Loại": {"select": _select_options(["Ngữ pháp", "Từ nối", "Mẫu câu"])},
        "Ngôn ngữ": {"select": _select_options(["Tiếng Nhật", "Tiếng Anh"])},
        "Cấu trúc": {"rich_text": {}},
        "Nghĩa / Chức năng": {"rich_text": {}},
        "Sắc thái": {"rich_text": {}},
        "So sánh": {"rich_text": {}},
        "Ví dụ": {"rich_text": {}},
        "Mức độ": {"select": _select_options(["N5", "N4", "N3", "N2", "N1", "A1", "A2", "B1", "B2", "C1", "C2"])},
        "Số lần xuất hiện": {"number": {"format": "number"}},
        "Trạng thái": {"select": _select_options(["Mới", "Đang học", "Đã nhớ"])},
        "Ngày ôn tiếp": {"date": {}},
        "Lần ôn gần nhất": {"date": {}},
        "Số lần ôn": {"number": {"format": "number"}},
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


def _find_child_databases(
    client: NotionClient, parent_page_id: str, titles: set[str]
) -> list[tuple[str, str]]:
    """Rediscover bootstrapped databases when the app's local SQLite is recreated."""
    cursor = ""
    matches: list[tuple[str, str]] = []
    while True:
        path = f"/blocks/{parent_page_id}/children?page_size=100"
        if cursor:
            path += "&start_cursor=" + cursor
        response = client.request("GET", path)
        for block in response.get("results") or []:
            if block.get("type") != "child_database":
                continue
            title = str((block.get("child_database") or {}).get("title") or "")
            if title in titles:
                database_id = str(block.get("id") or "")
                database = client.request("GET", f"/databases/{database_id}")
                matches.append((database_id, _database_data_source_id(client, database)))
        if not response.get("has_more"):
            return matches
        cursor = str(response.get("next_cursor") or "")
        if not cursor:
            return matches


def _find_child_database(
    client: NotionClient, parent_page_id: str, titles: set[str]
) -> tuple[str, str] | None:
    matches = _find_child_databases(client, parent_page_id, titles)
    return matches[0] if matches else None


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
        option_type = "select" if "select" in schema else "multi_select" if "multi_select" in schema else ""
        if not option_type or current.get("type") != option_type:
            continue
        current_options = (current.get(option_type) or {}).get("options") or []
        current_names = {str(option.get("name") or "") for option in current_options}
        desired_options = (schema.get(option_type) or {}).get("options") or []
        missing = [option for option in desired_options if option.get("name") not in current_names]
        if missing:
            preserved = [
                ({"id": option["id"]} if option.get("id") else {"name": option.get("name")})
                for option in current_options
            ]
            updates[name] = {option_type: {"options": preserved + missing}}
    if updates:
        client.request("PATCH", f"/data_sources/{data_source_id}", {"properties": updates})


def _create_learning_views(
    client: NotionClient,
    database_id: str,
    data_source_id: str,
    existing_names: list[str] | None = None,
) -> list[str]:
    del existing_names
    return _create_database_views(client, database_id, data_source_id, "vocabulary")


def _create_database_views(
    client: NotionClient,
    database_id: str,
    data_source_id: str,
    kind: str,
) -> list[str]:
    """Create compact, mobile-friendly views for one v4 study database."""
    existing_views: dict[str, str] = {}
    listed = client.request("GET", f"/views?database_id={database_id}")
    for entry in listed.get("results") or []:
        view_id = entry.get("id") or (entry.get("view") or {}).get("id")
        if not view_id:
            continue
        view = client.request("GET", f"/views/{view_id}")
        if view.get("name"):
            existing_views[str(view["name"])] = str(view_id)
    source = client.request("GET", f"/data_sources/{data_source_id}")
    property_ids = {
        name: str(value.get("id") or "")
        for name, value in (source.get("properties") or {}).items()
        if value.get("id")
    }
    visible_by_kind = {
        "vocabulary": {"Tên", "Cách đọc", "Nghĩa tiếng Việt", "Nhóm", "Mức độ", "Trạng thái", "Ngày ôn tiếp"},
        "sentence": {"Câu", "Trang", "Thứ tự câu", "Dịch tự nhiên", "Câu khó", "Trạng thái", "Ngày ôn tiếp"},
        "kanji": {"Kanji", "Âm On", "Âm Kun", "Nghĩa tiếng Việt", "JLPT", "Trạng thái", "Ngày ôn tiếp"},
        "language": {"Tên", "Loại", "Nghĩa / Chức năng", "Mức độ", "Trạng thái", "Ngày ôn tiếp"},
    }
    visible_names = visible_by_kind[kind]
    configuration = None
    if property_ids:
        configuration = {
            "type": "table",
            "properties": [
                {
                    "property_id": property_id,
                    "visible": name in visible_names,
                    "wrap": name in {"Tên", "Câu", "Cách đọc", "Nghĩa tiếng Việt", "Dịch tự nhiên", "Nghĩa / Chức năng"},
                }
                for name, property_id in property_ids.items()
            ],
            "wrap_cells": True,
            "frozen_column_index": 1,
        }
    source_sorts = ([{"property": "Trang", "direction": "ascending"}, {"property": "Thứ tự câu", "direction": "ascending"}]
                    if kind == "sentence" else [])
    common = [
        ("Tất cả", None, source_sorts),
        ("Ôn hôm nay", {"and": [
            {"property": "Ngày ôn tiếp", "date": {"on_or_before": "today"}},
            {"property": "Trạng thái", "select": {"does_not_equal": "Đã nhớ"}},
        ]}, [{"property": "Ngày ôn tiếp", "direction": "ascending"}]),
        ("Mục mới", {"property": "Trạng thái", "select": {"equals": "Mới"}}, source_sorts),
        ("Đang học", {"property": "Trạng thái", "select": {"equals": "Đang học"}}, source_sorts),
        ("Đã nhớ", {"property": "Trạng thái", "select": {"equals": "Đã nhớ"}}, source_sorts),
        ("Theo bài", None, source_sorts),
    ]
    extras = {
        "vocabulary": [
            ("Từ khó", {"property": "Nhóm", "multi_select": {"contains": "Từ khó"}}, []),
            ("Từ vựng Kanji", {"property": "Nhóm", "multi_select": {"contains": "Từ vựng Kanji"}}, []),
            ("Tiếng Nhật", {"property": "Ngôn ngữ", "select": {"equals": "Tiếng Nhật"}}, []),
            ("Tiếng Anh", {"property": "Ngôn ngữ", "select": {"equals": "Tiếng Anh"}}, []),
            ("Theo cấp độ", None, [{"property": "Mức độ", "direction": "ascending"}]),
        ],
        "sentence": [
            ("Câu khó", {"property": "Câu khó", "checkbox": {"equals": True}}, source_sorts),
            ("Giải mã câu dài", {"property": "Câu khó", "checkbox": {"equals": True}}, source_sorts),
        ],
        "kanji": [("Theo JLPT", None, [{"property": "JLPT", "direction": "ascending"}])],
        "language": [
            ("Ngữ pháp", {"property": "Loại", "select": {"equals": "Ngữ pháp"}}, []),
            ("Từ nối", {"property": "Loại", "select": {"equals": "Từ nối"}}, []),
            ("Mẫu câu", {"property": "Loại", "select": {"equals": "Mẫu câu"}}, []),
            ("Theo cấp độ", None, [{"property": "Mức độ", "direction": "ascending"}]),
        ],
    }
    views = common + extras[kind]
    created_names: list[str] = []
    for name, view_filter, sorts in views:
        if name in existing_views:
            update_payload: dict[str, Any] = {"filter": view_filter, "sorts": sorts or None}
            if configuration:
                update_payload["configuration"] = configuration
            client.request("PATCH", f"/views/{existing_views[name]}", update_payload)
        else:
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
            if configuration:
                payload["configuration"] = configuration
            client.request("POST", "/views", payload)
        created_names.append(name)
    return created_names


def ensure_notion_workspace(client: NotionClient, settings: NotionSettings | None = None) -> dict:
    """Return the five v4 databases, creating missing study databases in place."""
    settings = settings or get_notion_settings()
    if not settings.configured:
        raise NotionAPIError("Chưa cấu hình NOTION_TOKEN và NOTION_PARENT_PAGE_ID.", 401, "not_configured")
    local = session_store.load_notion_workspace_config()

    resources = {
        "lessons": [settings.lessons_database_id, settings.lessons_data_source_id, "Bài phân tích", _lesson_schema()],
        "items": [settings.items_database_id, settings.items_data_source_id, "Từ vựng", _item_schema()],
        "sentences": [settings.sentences_database_id or local.get("sentences_database_id"), settings.sentences_data_source_id or local.get("sentences_data_source_id"), "Câu & bản dịch", _sentence_schema()],
        "kanji": [settings.kanji_database_id or local.get("kanji_database_id"), settings.kanji_data_source_id or local.get("kanji_data_source_id"), "Kanji", _kanji_schema()],
        "language": [settings.language_database_id or local.get("language_database_id"), settings.language_data_source_id or local.get("language_data_source_id"), "Ngữ pháp & liên kết", _language_schema()],
    }
    for key, value in resources.items():
        database_id, data_source_id, title, schema = value
        if data_source_id and not database_id:
            database_id = _database_id_for_source(client, str(data_source_id))
        if database_id and not data_source_id:
            database = client.request("GET", f"/databases/{database_id}")
            data_source_id = _database_data_source_id(client, database)
        if not data_source_id and settings.parent_page_id:
            aliases = {str(title)} | ({"Mục cần học"} if key == "items" else set())
            discovered = _find_child_database(client, settings.parent_page_id, aliases)
            if discovered:
                database_id, data_source_id = discovered
        if not data_source_id:
            if not settings.parent_page_id:
                raise NotionAPIError(
                    f"Thiếu NOTION_PARENT_PAGE_ID để tạo bảng {title}.", 400, "missing_parent"
                )
            database_id, data_source_id = _create_database(
                client, settings.parent_page_id, str(title), schema
            )
            checkpoint = session_store.load_notion_workspace_config()
            checkpoint[f"{key}_database_id"] = str(database_id)
            checkpoint[f"{key}_data_source_id"] = str(data_source_id)
            session_store.save_notion_workspace_config(checkpoint)
        resources[key][0] = str(database_id)
        resources[key][1] = str(data_source_id)
        _ensure_data_source_schema(client, str(data_source_id), schema)

    lesson_ds = str(resources["lessons"][1])
    item_ds = str(resources["items"][1])
    sentence_ds = str(resources["sentences"][1])
    kanji_ds = str(resources["kanji"][1])
    language_ds = str(resources["language"][1])

    def ensure_relation(source_id: str, name: str, target_id: str, synced_name: str) -> None:
        source = client.request("GET", f"/data_sources/{source_id}")
        relation = (source.get("properties") or {}).get(name) or {}
        if relation.get("type") == "relation" or "relation" in relation:
            return
        client.request("PATCH", f"/data_sources/{source_id}", {"properties": {name: {
            "relation": {"data_source_id": target_id, "dual_property": {"synced_property_name": synced_name}}
        }}})

    ensure_relation(item_ds, "Bài phân tích", lesson_ds, "Từ vựng")
    ensure_relation(sentence_ds, "Bài phân tích", lesson_ds, "Câu & bản dịch")
    ensure_relation(kanji_ds, "Bài phân tích", lesson_ds, "Kanji")
    ensure_relation(language_ds, "Bài phân tích", lesson_ds, "Ngữ pháp & liên kết")
    ensure_relation(item_ds, "Câu nguồn", sentence_ds, "Từ vựng")
    ensure_relation(kanji_ds, "Câu nguồn", sentence_ds, "Kanji")
    ensure_relation(language_ds, "Câu nguồn", sentence_ds, "Ngữ pháp & liên kết")
    ensure_relation(item_ds, "Kanji", kanji_ds, "Từ vựng liên quan")

    view_names = dict(local.get("view_names_v4") or {})
    if int(local.get("views_version") or 0) < NOTION_VIEWS_VERSION:
        for key, kind in (("items", "vocabulary"), ("sentences", "sentence"), ("kanji", "kanji"), ("language", "language")):
            view_names[key] = _create_database_views(
                client, str(resources[key][0]), str(resources[key][1]), kind
            )

    workspace = {**session_store.load_notion_workspace_config()}
    for key, value in resources.items():
        workspace[f"{key}_database_id"] = value[0]
        workspace[f"{key}_data_source_id"] = value[1]
    workspace.update({
        "relation_created": True,
        "view_names_v4": view_names,
        "views_created": all(view_names.get(key) for key in ("items", "sentences", "kanji", "language")),
        "schema_version": NOTION_SCHEMA_VERSION,
        "views_version": NOTION_VIEWS_VERSION,
    })
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


def _learning_external_id(
    lesson_external_id: str,
    language: str,
    item_type: str,
    page_index: int,
    title: str,
    reading: str = "",
) -> str:
    raw = "|".join(
        (
            lesson_external_id,
            language,
            item_type,
            str(page_index),
            _normalize_key(title),
            _normalize_key(reading),
        )
    )
    return "learn:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_learning_items(
    analysis: dict[str, Any], lesson_external_id: str = "analysis"
) -> list[dict[str, Any]]:
    """Extract every study row while retaining the original structured record."""
    language = str(analysis.get("analysis_language") or "japanese")
    pages = analysis.get("page_analyses") or [analysis]
    result: dict[str, dict[str, Any]] = {}
    source_orders: dict[tuple[int, str], int] = {}

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
        external_id = _learning_external_id(
            lesson_external_id, language, item_type, page_index, title, reading
        )
        existing = result.get(external_id)
        order_key = (page_index, item_type)
        if existing:
            source_order = int(existing.get("source_order") or 0)
        else:
            source_order = source_orders.get(order_key, 0) + 1
            source_orders[order_key] = source_order
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
            "source_order": source_order,
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
                reading=("reading", "hiragana"),
                meaning=("meaning", "meaning_vi", "vn_meaning", "definition"),
                example=("example", "example_text", "example_1", "example_2"),
                example_reading=(
                    "example_hiragana", "example_text_hiragana", "example_1_hiragana",
                    "example_2_hiragana", "example_reading",
                ),
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
            result[
                _learning_external_id(
                    lesson_external_id,
                    language,
                    "Câu dài",
                    page_index,
                    _first(row, "original"),
                    _first(row, "reading"),
                )
            ]["difficulty"] = "Câu khó"
    return list(result.values())


def _clean_term(value: Any) -> str:
    text = _plain_preview(value)
    return re.sub(r"^[~〜～]+|[~〜～]+$", "", text).strip()


def _concept_external_id(kind: str, language: str, title: str, reading: str = "") -> str:
    raw = "|".join((kind, language, _normalize_key(title), _normalize_key(reading)))
    return f"concept:{kind}:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _split_kanji_vocabulary(value: Any) -> list[str]:
    """Normalize Gemini's string/list/object Kanji vocabulary cell without inventing data."""
    values: list[str] = []
    if isinstance(value, dict):
        for key in ("word", "vocab", "vocabulary", "term", "name"):
            if value.get(key):
                values.extend(_split_kanji_vocabulary(value[key]))
        if not values:
            for nested in value.values():
                values.extend(_split_kanji_vocabulary(nested))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            values.extend(_split_kanji_vocabulary(nested))
    elif value not in (None, ""):
        text = _plain_preview(value)
        values.extend(re.split(r"\s*(?:、|,|，|;|；|/|／|\n|\|)\s*", text))
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = _clean_term(item).strip("[](){} ")
        key = _normalize_key(cleaned)
        if cleaned and cleaned not in {"-", "Không có", "None", "null"} and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _kanji_vocabulary_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        title = _first(value, "word", "vocab", "vocabulary", "term", "name")
        if title and not isinstance(value.get("vocab"), (list, dict)):
            records = []
            for split_title in _split_kanji_vocabulary(title):
                record = dict(value)
                record["word"] = split_title
                records.append(record)
            return records
        records: list[dict[str, Any]] = []
        for nested in value.values():
            records.extend(_kanji_vocabulary_records(nested))
        return records
    if isinstance(value, (list, tuple, set)):
        records = []
        for nested in value:
            records.extend(_kanji_vocabulary_records(nested))
        return records
    return [{"word": title} for title in _split_kanji_vocabulary(value)]


def extract_notion_entities(
    analysis: dict[str, Any], lesson_external_id: str
) -> dict[str, list[dict[str, Any]]]:
    """Map one complete analysis to the five-database v4 model."""
    from modules.sentence_analyzer import split_sentences

    language = str(analysis.get("analysis_language") or "japanese")
    pages = analysis.get("page_analyses") or [analysis]
    sentences: list[dict[str, Any]] = []
    sentence_by_page: dict[int, list[dict[str, Any]]] = {}

    for page_number, page in enumerate(pages, 1):
        page_index = int(page.get("page_index", page_number) or page_number)
        catalog = list(page.get("sentence_catalog") or [])
        if not catalog:
            catalog = split_sentences(
                str(page.get("source_text") or page.get("confirmed_text") or ""), language, page_index
            )
        guidance = {str(row.get("sentence_id") or ""): row for row in page.get("translation_guidance") or []}
        breakdowns = {str(row.get("sentence_id") or ""): row for row in page.get("sentence_breakdowns") or []}
        page_sentences: list[dict[str, Any]] = []
        for fallback_ordinal, catalog_row in enumerate(catalog, 1):
            sentence_id = str(catalog_row.get("sentence_id") or f"p{page_index}-s{fallback_ordinal}")
            ordinal = int(catalog_row.get("ordinal", fallback_ordinal) or fallback_ordinal)
            guide = guidance.get(sentence_id) or {}
            breakdown = breakdowns.get(sentence_id) or {}
            original = str(catalog_row.get("original") or guide.get("original") or breakdown.get("original") or "").strip()
            if not original:
                continue
            translations = guide.get("translations") if isinstance(guide.get("translations"), dict) else {}
            deep_translations = breakdown.get("translations") if isinstance(breakdown.get("translations"), dict) else {}
            natural = str(translations.get("natural") or deep_translations.get("natural") or breakdown.get("simplified_vi") or "")
            source = {"catalog": catalog_row, "guidance": guide, "sentence_breakdown": breakdown}
            source_json = json.dumps(source, ensure_ascii=False, sort_keys=True, default=str)
            raw_id = "|".join((lesson_external_id, str(page_index), sentence_id, original))
            entity = {
                "external_id": "sentence:" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest(),
                "type": "Câu",
                "language": language,
                "title": f"Câu {page_index}.{ordinal} · {original[:80]}",
                "sentence_id": sentence_id,
                "page_index": page_index,
                "source_order": ordinal,
                "original": original,
                "reading": str(guide.get("reading") or breakdown.get("reading") or ""),
                "natural_translation": natural,
                "ocr_warning": str(guide.get("ocr_warning") or ""),
                "complexity_score": float(catalog_row.get("complexity_score") or breakdown.get("complexity_score") or 0),
                "is_complex": bool(breakdown or catalog_row.get("analyzed") or catalog_row.get("auto_selected")),
                "source_json": source_json,
                "source_checksum": hashlib.sha256(source_json.encode("utf-8")).hexdigest(),
            }
            sentences.append(entity)
            page_sentences.append(entity)
        sentence_by_page[page_index] = page_sentences

    def matching_sentence_ids(page_index: int, row: dict[str, Any], title: str) -> list[str]:
        explicit = str(row.get("sentence_id") or "")
        page_rows = sentence_by_page.get(page_index) or []
        if explicit:
            matches = [item["external_id"] for item in page_rows if item["sentence_id"] == explicit]
            if matches:
                return matches
        candidates = [
            _plain_preview(row.get(key))
            for key in ("example", "example_text", "example_1", "example_2", "original")
            if row.get(key)
        ]
        term = _clean_term(title)
        matches: list[str] = []
        for sentence in page_rows:
            original = sentence["original"]
            exact_example = any(value and (value in original or original in value) for value in candidates)
            if language == "english":
                term_match = bool(term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", original, re.I))
            else:
                term_match = bool(term and term in original)
            if exact_example or term_match:
                matches.append(sentence["external_id"])
        return matches

    collections: dict[str, dict[str, dict[str, Any]]] = {
        "vocabulary": {}, "kanji": {}, "language_items": {}
    }

    def add_concept(kind: str, entity: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
        bucket = collections[kind]
        existing = bucket.get(entity["external_id"])
        if not existing:
            entity["source_records"] = [source_row]
            bucket[entity["external_id"]] = entity
            return entity
        for key in (
            "groups", "sentence_external_ids", "related_kanji_external_ids",
            "related_vocabulary_external_ids",
        ):
            values = list(existing.get(key) or [])
            for value in entity.get(key) or []:
                if value not in values:
                    values.append(value)
            existing[key] = values
        existing["occurrences_in_analysis"] = int(existing.get("occurrences_in_analysis") or 1) + 1
        existing["missing_details"] = bool(existing.get("missing_details") and entity.get("missing_details"))
        existing["source_records"].append(source_row)
        for key, value in entity.items():
            if value not in (None, "", [], 0, 0.0, False) and existing.get(key) in (None, "", [], 0, 0.0, False):
                existing[key] = value
        return existing

    vocabulary_source_index: dict[str, dict[str, Any]] = {}
    for page in pages:
        indexed_rows = [
            *(page.get("vocabulary_all") or []),
            *(page.get("vocabulary_important") or []),
            *((page.get("phrasal_collocations") or []) if language == "english" else []),
        ]
        for row in indexed_rows:
            title = _clean_term(_first(row, "word", "phrase"))
            if not title:
                continue
            key = _normalize_key(title)
            existing = vocabulary_source_index.get(key)
            if not existing or len(json.dumps(row, ensure_ascii=False, default=str)) > len(
                json.dumps(existing, ensure_ascii=False, default=str)
            ):
                vocabulary_source_index[key] = row

    known_vocabulary: dict[str, dict[str, Any]] = {}
    for page_number, page in enumerate(pages, 1):
        page_index = int(page.get("page_index", page_number) or page_number)
        vocab_rows: list[tuple[dict[str, Any], list[str], bool]] = []
        vocab_rows.extend((row, ["Từ trong bài"], False) for row in page.get("vocabulary_all") or [])
        vocab_rows.extend((row, ["Từ trong bài", "Từ khó"], False) for row in page.get("vocabulary_important") or [])
        if language == "english":
            vocab_rows.extend((row, ["Cụm từ"], False) for row in page.get("phrasal_collocations") or [])
        for row, groups, missing in vocab_rows:
            title = _clean_term(_first(row, "word", "phrase"))
            if not title:
                continue
            indexed = vocabulary_source_index.get(_normalize_key(title)) or {}
            reading = _first(row, "reading", "hiragana") or _first(indexed, "reading", "hiragana")
            canonical = _clean_term(_first(row, "base_form") or title)
            external_id = _concept_external_id("vocabulary", language, canonical, reading)
            entity = {
                "external_id": external_id, "type": "Từ vựng", "language": language,
                "title": title, "reading": reading,
                "meaning_vi": _first(row, "meaning", "meaning_vi", "vn_meaning", "definition"),
                "example": _first(row, "example", "example_text", "example_1"),
                "example_reading": _first(row, "example_hiragana", "example_text_hiragana", "example_1_hiragana"),
                "translation_vi": _first(row, "example_translation", "translation"),
                "difficulty": _first(row, "jlpt", "cefr", "difficulty", "level"),
                "part_of_speech": _first(row, "part_of_speech", "type"),
                "base_form": _first(row, "base_form"),
                "nuance": _first(row, "nuance", "usage", "note"),
                "comparison": _first(row, "comparison", "mistake"),
                "groups": groups,
                "missing_details": missing,
                "sentence_external_ids": matching_sentence_ids(page_index, row, title),
                "related_kanji_external_ids": [],
                "page_index": page_index,
                "source_order": int(row.get("num") or 0),
                "occurrences_in_analysis": 1,
            }
            added = add_concept("vocabulary", entity, row)
            known_vocabulary[_normalize_key(title)] = added

        if language == "japanese":
            for order, row in enumerate(page.get("kanji_analysis") or [], 1):
                title = _clean_term(_first(row, "kanji", "phrase"))
                if not title:
                    continue
                kanji_id = _concept_external_id("kanji", language, title)
                related_vocab_ids: list[str] = []
                for vocab_record in _kanji_vocabulary_records(row.get("vocab")):
                    word = _clean_term(_first(vocab_record, "word"))
                    if not word:
                        continue
                    source_detail = vocabulary_source_index.get(_normalize_key(word)) or {}
                    known = known_vocabulary.get(_normalize_key(word))
                    reading = (
                        _first(vocab_record, "reading", "hiragana")
                        or str((known or {}).get("reading") or "")
                        or _first(source_detail, "reading", "hiragana")
                    )
                    vocab_id = _concept_external_id("vocabulary", language, word, reading)
                    vocab_entity = {
                        "external_id": vocab_id, "type": "Từ vựng", "language": language,
                        "title": word, "reading": reading,
                        "meaning_vi": _first(vocab_record, "meaning", "meaning_vi", "vn_meaning") or str((known or {}).get("meaning_vi") or "") or _first(source_detail, "meaning", "meaning_vi", "vn_meaning", "definition"),
                        "example": _first(vocab_record, "example") or _first(row, "example"),
                        "example_reading": _first(vocab_record, "example_hiragana", "example_reading"),
                        "translation_vi": _first(vocab_record, "translation", "example_translation"),
                        "difficulty": _first(vocab_record, "jlpt", "difficulty") or str((known or {}).get("difficulty") or "") or _first(source_detail, "jlpt", "cefr", "difficulty"),
                        "part_of_speech": _first(vocab_record, "part_of_speech", "type") or str((known or {}).get("part_of_speech") or "") or _first(source_detail, "part_of_speech", "type"),
                        "base_form": _first(vocab_record, "base_form") or str((known or {}).get("base_form") or "") or _first(source_detail, "base_form"),
                        "nuance": _first(vocab_record, "nuance", "usage", "note") or str((known or {}).get("nuance") or ""),
                        "comparison": _first(vocab_record, "comparison"),
                        "groups": ["Từ vựng Kanji"],
                        "missing_details": not bool(known or source_detail or reading or _first(vocab_record, "meaning", "meaning_vi", "vn_meaning")),
                        "sentence_external_ids": matching_sentence_ids(page_index, row, word),
                        "related_kanji_external_ids": [kanji_id],
                        "page_index": page_index, "source_order": order,
                        "occurrences_in_analysis": 1,
                    }
                    add_concept("vocabulary", vocab_entity, {"kanji": title, "vocab": vocab_record, "source": row})
                    related_vocab_ids.append(vocab_id)
                kanji_entity = {
                    "external_id": kanji_id, "type": "Kanji", "language": language,
                    "title": title, "onyomi": _first(row, "onyomi"), "kunyomi": _first(row, "kunyomi"),
                    "meaning_vi": _first(row, "meaning", "meaning_vi"),
                    "difficulty": _first(row, "jlpt", "difficulty"),
                    "nuance": _first(row, "role"), "example": _first(row, "example"),
                    "sentence_external_ids": matching_sentence_ids(page_index, row, title),
                    "related_vocabulary_external_ids": related_vocab_ids,
                    "page_index": page_index, "source_order": order, "occurrences_in_analysis": 1,
                }
                add_concept("kanji", kanji_entity, row)

        marker_rows = (page.get("connectors") or []) if language == "japanese" else (page.get("discourse_markers") or page.get("connectors") or [])
        language_rows = [
            *(("Từ nối", row, ("phrase", "marker", "word")) for row in marker_rows),
            *(("Ngữ pháp", row, ("name", "pattern")) for row in page.get("grammar_points") or []),
            *(("Mẫu câu", row, ("pattern", "name")) for row in page.get("sentence_patterns") or []),
        ]
        for order, (item_type, row, title_keys) in enumerate(language_rows, 1):
            title = _plain_preview(_first(row, *title_keys)).strip()
            if not title:
                continue
            entity = {
                "external_id": _concept_external_id("language", language, f"{item_type}:{_clean_term(title)}"),
                "type": item_type, "language": language, "title": title,
                "meaning_vi": _first(row, "meaning", "meaning_vi", "explanation", "function", "role", "nuance"),
                "formation": _first(row, "formation", "structure", "rule", "components"),
                "nuance": _first(row, "nuance", "usage", "role", "function", "register", "explanation"),
                "comparison": _first(row, "comparison", "note", "mistake"),
                "example": _first(row, "example", "formation"),
                "difficulty": _first(row, "jlpt", "cefr", "difficulty", "level"),
                "sentence_external_ids": matching_sentence_ids(page_index, row, title),
                "page_index": page_index, "source_order": order, "occurrences_in_analysis": 1,
            }
            add_concept("language_items", entity, row)

    for bucket in collections.values():
        for entity in bucket.values():
            source_json = json.dumps(entity.pop("source_records", []), ensure_ascii=False, sort_keys=True, default=str)
            entity["source_json"] = source_json
            entity["source_checksum"] = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
    return {
        "sentences": sentences,
        "vocabulary": list(collections["vocabulary"].values()),
        "kanji": list(collections["kanji"].values()),
        "language_items": list(collections["language_items"].values()),
    }


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


def refresh_notion_render(
    payload: dict[str, Any],
    items: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild page content and coverage after payload metadata changes."""
    rendered = render_notion_lesson_markdown(items, analysis, payload)
    payload.update(
        {
            "markdown": rendered["markdown"],
            "markdown_sections": rendered["sections"],
            "layout_version": rendered["layout_version"],
            "render_coverage": {
                "complete": rendered["coverage_complete"],
                "rendered_field_paths": rendered["rendered_field_paths"],
                "unrendered_field_paths": rendered["unrendered_field_paths"],
            },
            "unrendered_field_count": rendered["unrendered_field_count"],
            "section_manifest": rendered["section_manifest"],
        }
    )
    return payload


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
    entities = extract_notion_entities(analysis, external_id)
    columns.update({
        "sentence_count": len(entities["sentences"]),
        "vocabulary_count": len(entities["vocabulary"]),
        "script_count": len(entities["kanji"]) if language == "japanese" else len(
            [item for item in entities["vocabulary"] if "Cụm từ" in item.get("groups", [])]
        ),
        "marker_count": len([item for item in entities["language_items"] if item.get("type") == "Từ nối"]),
        "grammar_count": len([item for item in entities["language_items"] if item.get("type") == "Ngữ pháp"]),
        "pattern_count": len([item for item in entities["language_items"] if item.get("type") == "Mẫu câu"]),
        "long_sentence_count": len([item for item in entities["sentences"] if item.get("is_complex")]),
    })
    payload = {
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
        "analysis_hash": analysis_hash,
        "raw_json": raw_json,
        "raw_json_filename": f"analysis-{analysis_hash[:16]}.json",
        "archive_schema_version": RAW_ARCHIVE_SCHEMA_VERSION,
        "columns": columns,
        **entities,
        # Retained for queue/debug compatibility; v4 routes each collection separately.
        "learning_items": [
            *entities["vocabulary"], *entities["kanji"],
            *entities["language_items"], *entities["sentences"],
        ],
    }
    return refresh_notion_render(payload, items, analysis)


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
        "Phiên bản bố cục": {"rich_text": _text(payload.get("layout_version") or NOTION_LAYOUT_VERSION)},
        "Đủ nội dung Notion": {"checkbox": bool((payload.get("render_coverage") or {}).get("complete"))},
        "Số trường chưa hiển thị": {"number": int(payload.get("unrendered_field_count") or 0)},
        "Lỗi đồng bộ": {"rich_text": []},
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


def _write_lesson_markdown(
    client: NotionClient,
    page_id: str,
    markdown: str,
    sections: list[str] | None = None,
) -> None:
    chunks: list[str] = []
    if sections:
        current = ""
        for section in sections:
            candidate = section if not current else current + "\n\n" + section
            if current and len(candidate) > MAX_MARKDOWN_CHARS:
                chunks.extend(split_markdown(current))
                current = section
            else:
                current = candidate
        if current:
            chunks.extend(split_markdown(current))
    else:
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
    _write_lesson_markdown(
        client,
        str(page["id"]),
        str(payload.get("markdown") or ""),
        list(payload.get("markdown_sections") or []),
    )
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
    return _entity_properties(item, "vocabulary", lesson_page_id, existing, {}, {})


def _merged_relations(existing: dict | None, name: str, page_ids: list[str]) -> list[dict]:
    relations = _property_relations(existing or {}, name)
    known = {str(row.get("id") or "") for row in relations}
    for page_id in page_ids:
        if page_id and page_id not in known:
            known.add(page_id)
            relations.append({"id": page_id})
    return relations[:100]


def _entity_properties(
    item: dict,
    kind: str,
    lesson_page_id: str,
    existing: dict | None,
    sentence_pages: dict[str, str],
    kanji_pages: dict[str, str],
) -> dict:
    existing_lessons = _property_relations(existing or {}, "Bài phân tích")
    relation_added = lesson_page_id not in {str(row.get("id") or "") for row in existing_lessons}
    lesson_relations = _merged_relations(existing, "Bài phân tích", [lesson_page_id])
    occurrence_delta = int(item.get("occurrences_in_analysis") or 1) if relation_added else 0
    occurrence = _property_number(existing or {}, "Số lần xuất hiện") + occurrence_delta
    difficulty = _safe_difficulty(str(item.get("difficulty") or ""))
    sentence_ids = [sentence_pages[value] for value in item.get("sentence_external_ids") or [] if value in sentence_pages]
    common = {
        "External ID": {"rich_text": _text(item.get("external_id"))},
        "Bài phân tích": {"relation": lesson_relations},
        "Dữ liệu nguồn": {"rich_text": _text(item.get("source_json"))},
        "JSON checksum": {"rich_text": _text(item.get("source_checksum"))},
    }
    if kind == "vocabulary":
        properties = {**common,
            "Tên": {"title": _text(_plain_preview(item.get("title")), 300)},
            "Nhóm": {"multi_select": [{"name": value} for value in item.get("groups") or ["Từ trong bài"]]},
            "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if item.get("language") == "japanese" else "Tiếng Anh"}},
            "Cách đọc": {"rich_text": _text(_plain_preview(item.get("reading")))},
            "Nghĩa tiếng Việt": {"rich_text": _text(_plain_preview(item.get("meaning_vi")))},
            "Ví dụ": {"rich_text": _text(_plain_preview(item.get("example")))},
            "Hiragana ví dụ": {"rich_text": _text(_plain_preview(item.get("example_reading")))},
            "Bản dịch": {"rich_text": _text(_plain_preview(item.get("translation_vi")))},
            "Câu nguồn": {"relation": _merged_relations(existing, "Câu nguồn", sentence_ids)},
            "Kanji": {"relation": _merged_relations(existing, "Kanji", [kanji_pages[value] for value in item.get("related_kanji_external_ids") or [] if value in kanji_pages])},
            "Số lần xuất hiện": {"number": occurrence or 1},
            "Thiếu chi tiết": {"checkbox": bool(item.get("missing_details"))},
            "Từ loại": {"rich_text": _text(_plain_preview(item.get("part_of_speech")))},
            "Từ gốc": {"rich_text": _text(_plain_preview(item.get("base_form")))},
            "Sắc thái / Chức năng": {"rich_text": _text(_plain_preview(item.get("nuance")))},
            "So sánh": {"rich_text": _text(_plain_preview(item.get("comparison")))},
        }
        if difficulty:
            properties["Mức độ"] = {"select": {"name": difficulty}}
    elif kind == "sentence":
        properties = {**common,
            "Câu": {"title": _text(_plain_preview(item.get("title")), 300)},
            "ID câu": {"rich_text": _text(item.get("sentence_id"))},
            "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if item.get("language") == "japanese" else "Tiếng Anh"}},
            "Trang": {"number": int(item.get("page_index") or 0)},
            "Thứ tự câu": {"number": int(item.get("source_order") or 0)},
            "Nguyên văn": {"rich_text": _text(_plain_preview(item.get("original")))},
            "Hiragana": {"rich_text": _text(_plain_preview(item.get("reading")))},
            "Dịch tự nhiên": {"rich_text": _text(_plain_preview(item.get("natural_translation")))},
            "Câu khó": {"checkbox": bool(item.get("is_complex"))},
            "Điểm phức tạp": {"number": float(item.get("complexity_score") or 0)},
            "Cảnh báo OCR": {"rich_text": _text(_plain_preview(item.get("ocr_warning")))},
        }
    elif kind == "kanji":
        properties = {**common,
            "Kanji": {"title": _text(_plain_preview(item.get("title")), 300)},
            "Âm On": {"rich_text": _text(_plain_preview(item.get("onyomi")))},
            "Âm Kun": {"rich_text": _text(_plain_preview(item.get("kunyomi")))},
            "Nghĩa tiếng Việt": {"rich_text": _text(_plain_preview(item.get("meaning_vi")))},
            "Vai trò trong bài": {"rich_text": _text(_plain_preview(item.get("nuance")))},
            "Ví dụ": {"rich_text": _text(_plain_preview(item.get("example")))},
            "Câu nguồn": {"relation": _merged_relations(existing, "Câu nguồn", sentence_ids)},
            "Số lần xuất hiện": {"number": occurrence or 1},
        }
        if difficulty and difficulty.startswith("N"):
            properties["JLPT"] = {"select": {"name": difficulty}}
    else:
        properties = {**common,
            "Tên": {"title": _text(_plain_preview(item.get("title")), 300)},
            "Loại": {"select": {"name": item.get("type")}},
            "Ngôn ngữ": {"select": {"name": "Tiếng Nhật" if item.get("language") == "japanese" else "Tiếng Anh"}},
            "Cấu trúc": {"rich_text": _text(_plain_preview(item.get("formation")))},
            "Nghĩa / Chức năng": {"rich_text": _text(_plain_preview(item.get("meaning_vi")))},
            "Sắc thái": {"rich_text": _text(_plain_preview(item.get("nuance")))},
            "So sánh": {"rich_text": _text(_plain_preview(item.get("comparison")))},
            "Ví dụ": {"rich_text": _text(_plain_preview(item.get("example")))},
            "Câu nguồn": {"relation": _merged_relations(existing, "Câu nguồn", sentence_ids)},
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
    return _upsert_entity(client, data_source_id, item, "vocabulary", lesson_page_id, {}, {})


def _upsert_entity(
    client: NotionClient,
    data_source_id: str,
    item: dict,
    kind: str,
    lesson_page_id: str,
    sentence_pages: dict[str, str],
    kanji_pages: dict[str, str],
) -> dict:
    existing = _query_external_id(client, data_source_id, item["external_id"])
    properties = _entity_properties(
        item, kind, lesson_page_id, existing, sentence_pages, kanji_pages
    )
    if existing:
        page = client.request("PATCH", f"/pages/{existing['id']}", {"properties": properties})
    else:
        page = client.request(
            "POST",
            "/pages",
            {"parent": {"type": "data_source_id", "data_source_id": data_source_id}, "properties": properties},
        )
    _write_lesson_markdown(client, str(page["id"]), render_notion_item_markdown(item))
    return page


def execute_notion_sync(run: dict, client: NotionClient | None = None) -> dict:
    """Execute one durable run; item-level failures are isolated and returned."""
    settings = get_notion_settings()
    if not settings.configured or not settings.token:
        raise NotionAPIError("Notion chưa được cấu hình trong Streamlit Secrets.", 401, "not_configured")
    client = client or NotionClient(settings.token)
    workspace = ensure_notion_workspace(client, settings)
    if workspace.get("lessons_database_id") and workspace.get("items_database_id"):
        from modules.notion_migration import migrate_notion_workspace_v4_if_needed

        migration = migrate_notion_workspace_v4_if_needed(client, settings, workspace)
        if migration.get("status") == "running":
            raise NotionAPIError(
                "Notion đang được nâng cấp lên bố cục v4; job sẽ tự thử lại.",
                503,
                "migration_running",
            )
    payload = _upgrade_payload_v4(run["payload"])
    lesson = _upsert_lesson(client, workspace["lessons_data_source_id"], payload)
    page_id = str(lesson["id"])
    page_url = str(lesson.get("url") or "")
    if not workspace.get("sentences_data_source_id"):
        errors = []
        for index, item in enumerate(payload.get("learning_items") or [], 1):
            try:
                _upsert_learning_item(client, workspace["items_data_source_id"], item, page_id)
            except NotionAPIError as exc:
                if exc.authorization_error or exc.retryable:
                    raise
                errors.append({"external_id": item.get("external_id"), "title": item.get("title"), "error": str(exc)})
            session_store.update_notion_sync_progress(
                run["run_id"], index, notion_page_id=page_id,
                notion_page_url=page_url, item_errors=errors,
            )
    else:
        errors = _sync_payload_entities(
            client,
            workspace,
            payload,
            page_id,
            progress=lambda index, values: session_store.update_notion_sync_progress(
                run["run_id"], index, notion_page_id=page_id,
                notion_page_url=page_url, item_errors=values,
            ),
        )
    if errors:
        client.request(
            "PATCH",
            f"/pages/{page_id}",
            {
                "properties": {
                    "Trạng thái": {"select": {"name": "Một phần"}},
                    "Lỗi đồng bộ": {
                        "rich_text": _text(
                            f"{len(errors)} mục cần học chưa đồng bộ được. Hãy thử lại từ ứng dụng."
                        )
                    },
                }
            },
        )
    return {"page_id": page_id, "page_url": page_url, "item_errors": errors}


def _upgrade_payload_v4(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade durable v3 queue payloads after deployment without another Gemini call."""
    if all(key in payload for key in ("sentences", "vocabulary", "kanji", "language_items")):
        return payload
    try:
        archive = json.loads(str(payload.get("raw_json") or "{}"))
        analysis = archive.get("analysis")
        if isinstance(analysis, dict):
            upgraded = dict(payload)
            entities = extract_notion_entities(analysis, str(payload.get("external_id") or "analysis"))
            upgraded.update(entities)
            upgraded["learning_items"] = [
                *entities["vocabulary"], *entities["kanji"],
                *entities["language_items"], *entities["sentences"],
            ]
            return upgraded
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return payload


def _sync_payload_entities(
    client: NotionClient,
    workspace: dict[str, Any],
    payload: dict[str, Any],
    lesson_page_id: str,
    progress: Callable[[int, list[dict]], None] | None = None,
) -> list[dict]:
    """Upsert v4 entities in dependency order while isolating validation failures."""
    errors: list[dict] = []
    sentence_pages: dict[str, str] = {}
    kanji_pages: dict[str, str] = {}
    index = 0

    def sync_collection(key: str, kind: str, data_source_key: str) -> dict[str, str]:
        nonlocal index
        result: dict[str, str] = {}
        for item in payload.get(key) or []:
            index += 1
            try:
                page = _upsert_entity(
                    client, str(workspace[data_source_key]), item, kind,
                    lesson_page_id, sentence_pages, kanji_pages,
                )
                result[str(item.get("external_id") or "")] = str(page.get("id") or "")
            except NotionAPIError as exc:
                if exc.authorization_error or exc.retryable:
                    raise
                errors.append({
                    "collection": key, "external_id": item.get("external_id"),
                    "title": item.get("title"), "error": str(exc),
                })
            if progress:
                progress(index, errors)
        return result

    sentence_pages.update(sync_collection("sentences", "sentence", "sentences_data_source_id"))
    kanji_pages.update(sync_collection("kanji", "kanji", "kanji_data_source_id"))
    sync_collection("vocabulary", "vocabulary", "items_data_source_id")
    sync_collection("language_items", "language", "language_data_source_id")
    return errors


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
