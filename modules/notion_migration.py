"""Restartable migration from preview-heavy Notion pages to layout v3."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

from modules import session_store
from modules.notion_sync import (
    NOTION_SCHEMA_VERSION,
    NotionAPIError,
    NotionClient,
    NotionSettings,
    _text,
    _upsert_learning_item,
    _upsert_lesson,
    build_notion_sync_payload,
    ensure_notion_workspace,
    extract_learning_items,
    refresh_notion_render,
)
from modules.notion_renderer import NOTION_LAYOUT_VERSION


OBSOLETE_LESSON_COLUMNS = (
    "OCR gốc",
    "Hướng dẫn dịch",
    "Dịch tự nhiên",
    "Từ vựng",
    "Kanji / Cụm từ",
    "Từ nối",
    "Ngữ pháp",
    "Mẫu câu",
    "Câu dài",
    "Cảnh báo OCR",
)


def _query_all(client: NotionClient, data_source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        body: dict[str, Any] = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        response = client.request("POST", f"/data_sources/{data_source_id}/query", body)
        rows.extend(response.get("results") or [])
        if not response.get("has_more"):
            return rows
        cursor = str(response.get("next_cursor") or "")
        if not cursor:
            return rows


def _property_plain(page: dict[str, Any], name: str) -> str:
    prop = ((page.get("properties") or {}).get(name) or {})
    values = prop.get("title") or prop.get("rich_text") or []
    return "".join(str(value.get("plain_text") or "") for value in values).strip()


def _property_select(page: dict[str, Any], name: str) -> str:
    return str(((((page.get("properties") or {}).get(name) or {}).get("select")) or {}).get("name") or "")


def _property_date(page: dict[str, Any], name: str) -> str:
    return str(((((page.get("properties") or {}).get(name) or {}).get("date")) or {}).get("start") or "")


def _property_number(page: dict[str, Any], name: str) -> float:
    return float((((page.get("properties") or {}).get(name) or {}).get("number")) or 0)


def _property_relations(page: dict[str, Any], name: str) -> set[str]:
    relations = (((page.get("properties") or {}).get(name) or {}).get("relation")) or []
    return {str(value.get("id") or "") for value in relations if value.get("id")}


def _archive_url(page: dict[str, Any]) -> str:
    files = ((((page.get("properties") or {}).get("Bản JSON gốc") or {}).get("files")) or [])
    if not files:
        return ""
    entry = files[0]
    file_type = str(entry.get("type") or "file")
    return str(((entry.get(file_type) or {}).get("url")) or "")


def _download_json(client: NotionClient, url: str) -> dict[str, Any]:
    if not url:
        raise ValueError("Bài không có file JSON gốc.")
    response = client.http.request("GET", url, timeout=90)
    if not 200 <= response.status_code < 300:
        raise ValueError(f"Không tải được JSON gốc (HTTP {response.status_code}).")
    try:
        return json.loads(response.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("File JSON gốc không đọc được.") from exc


def _learning_fingerprint_from_page(page: dict[str, Any]) -> str:
    parts = (
        _property_select(page, "Ngôn ngữ"),
        _property_select(page, "Loại"),
        _property_plain(page, "Tên"),
        _property_plain(page, "Cách đọc"),
    )
    normalized = "|".join(" ".join(part.lower().split()) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _learning_fingerprint_from_item(item: dict[str, Any]) -> str:
    language = "Tiếng Nhật" if item.get("language") == "japanese" else "Tiếng Anh"
    parts = (language, str(item.get("type") or ""), str(item.get("title") or ""), str(item.get("reading") or ""))
    normalized = "|".join(" ".join(part.lower().split()) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _study_state(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _property_select(page, "Trạng thái") or "Mới",
        "next_review": _property_date(page, "Ngày ôn tiếp"),
        "last_review": _property_date(page, "Lần ôn gần nhất"),
        "review_count": int(_property_number(page, "Số lần ôn")),
    }


def _resolve_backup_parent(client: NotionClient, settings: NotionSettings, workspace: dict) -> str:
    if settings.parent_page_id:
        return str(settings.parent_page_id)
    database_id = str(workspace.get("lessons_database_id") or "")
    if database_id:
        database = client.request("GET", f"/databases/{database_id}")
        parent = database.get("parent") or {}
        if parent.get("type") == "page_id" and parent.get("page_id"):
            return str(parent["page_id"])
    raise NotionAPIError("Không tìm thấy trang Study Hub để lưu backup; migration đã dừng.", 400, "missing_backup_parent")


def _create_backup_page(
    client: NotionClient,
    parent_page_id: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    content = json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")
    upload = client.upload_file(f"notion-v2-backup-{stamp}.json", content, "application/json")
    page = client.request(
        "POST",
        "/pages",
        {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "properties": {"title": {"title": _text(f"Backup Notion trước nâng cấp {stamp}")}},
            "markdown": (
                "# Backup trước khi nâng cấp Notion\n\n"
                "> File này chứa JSON bài phân tích và trạng thái ôn tập trước khi dựng lại bố cục."
            ),
        },
    )
    client.request(
        "PATCH",
        f"/blocks/{page['id']}/children",
        {
            "children": [
                {
                    "object": "block",
                    "type": "file",
                    "file": {
                        "type": "file_upload",
                        "file_upload": {"id": str(upload["id"])},
                        "caption": _text("Bản sao dữ liệu trước migration"),
                    },
                }
            ]
        },
    )
    return page


def _remove_obsolete_columns(client: NotionClient, lessons_data_source_id: str) -> None:
    source = client.request("GET", f"/data_sources/{lessons_data_source_id}")
    existing = source.get("properties") or {}
    removable = {name: None for name in OBSOLETE_LESSON_COLUMNS if name in existing}
    if removable:
        client.request("PATCH", f"/data_sources/{lessons_data_source_id}", {"properties": removable})


def _remove_obsolete_views(client: NotionClient, items_database_id: str) -> None:
    obsolete = {"Mục mới", "Theo loại"}
    listed = client.request("GET", f"/views?database_id={items_database_id}")
    for entry in listed.get("results") or []:
        view_id = entry.get("id") or (entry.get("view") or {}).get("id")
        if not view_id:
            continue
        view = client.request("GET", f"/views/{view_id}")
        if str(view.get("name") or "") in obsolete:
            client.request("DELETE", f"/views/{view_id}")


def _restore_study_state(client: NotionClient, page_id: str, state: dict[str, Any] | None) -> None:
    if not state:
        return
    properties: dict[str, Any] = {
        "Trạng thái": {"select": {"name": state.get("status") or "Mới"}},
        "Số lần ôn": {"number": int(state.get("review_count") or 0)},
    }
    if state.get("next_review"):
        properties["Ngày ôn tiếp"] = {"date": {"start": state["next_review"]}}
    if state.get("last_review"):
        properties["Lần ôn gần nhất"] = {"date": {"start": state["last_review"]}}
    client.request("PATCH", f"/pages/{page_id}", {"properties": properties})


def rebuild_notion_workspace_v3(
    client: NotionClient,
    settings: NotionSettings,
    *,
    confirm: bool = False,
) -> dict[str, Any]:
    """Back up, archive, and rebuild existing rows with the structured v3 layout."""
    workspace = ensure_notion_workspace(client, settings)
    lesson_rows = _query_all(client, str(workspace["lessons_data_source_id"]))
    item_rows = _query_all(client, str(workspace["items_data_source_id"]))
    archives: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for page in lesson_rows:
        try:
            archive = _download_json(client, _archive_url(page))
            if not isinstance(archive.get("analysis"), dict) or not isinstance(archive.get("sources"), list):
                raise ValueError("JSON không có sources/analysis hợp lệ.")
            archives.append(
                {
                    "page_id": str(page["id"]),
                    "external_id": _property_plain(page, "External ID"),
                    "created_at": _property_date(page, "Ngày phân tích"),
                    "metrics": {
                        "model": _property_plain(page, "Model"),
                        "total_tokens": int(_property_number(page, "Tổng token")),
                        "cost_jpy": _property_number(page, "Chi phí JPY"),
                        "breakdown": {
                            "ocr": {
                                "total_tokens": int(_property_number(page, "Token OCR")),
                                "total_cost_jpy": _property_number(page, "Chi phí OCR JPY"),
                            },
                            "analysis": {
                                "total_tokens": int(_property_number(page, "Token phân tích")),
                                "total_cost_jpy": _property_number(page, "Chi phí phân tích JPY"),
                            },
                            "guidance": {
                                "total_tokens": int(_property_number(page, "Token hướng dẫn")),
                                "total_cost_jpy": _property_number(page, "Chi phí hướng dẫn JPY"),
                            },
                            "sentence": {
                                "total_tokens": int(_property_number(page, "Token câu dài")),
                                "total_cost_jpy": _property_number(page, "Chi phí câu dài JPY"),
                            },
                        },
                    },
                    "archive": archive,
                }
            )
        except (ValueError, KeyError) as exc:
            unreadable.append({"page_id": str(page.get("id") or ""), "error": str(exc)})

    study_states: dict[str, dict[str, Any]] = {}
    for page in item_rows:
        study_states[_learning_fingerprint_from_page(page)] = _study_state(page)
    bundle = {
        "migration": "notion-layout-v3",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lessons": archives,
        "unreadable_lessons": unreadable,
        "learning_items": [
            {"page_id": str(page.get("id") or ""), "properties": page.get("properties") or {}}
            for page in item_rows
        ],
    }
    summary = {
        "lesson_count": len(lesson_rows),
        "readable_lesson_count": len(archives),
        "unreadable_lessons": unreadable,
        "item_count": len(item_rows),
        "would_remove_obsolete_columns": not unreadable,
    }
    if not confirm:
        return {**summary, "status": "dry_run"}

    parent_id = _resolve_backup_parent(client, settings, workspace)
    backup_page = _create_backup_page(client, parent_id, bundle)
    readable_ids = {entry["page_id"] for entry in archives}
    config = session_store.load_notion_workspace_config()
    config["migration_v3"] = {
        "status": "backed_up",
        "backup_page_id": str(backup_page.get("id") or ""),
        "completed_lessons": 0,
        "total_lessons": len(archives),
    }
    session_store.save_notion_workspace_config(config)

    rebuilt_lessons = 0
    rebuilt_items = 0
    replacement_item_ids: set[str] = set()
    for archive_entry in archives:
        archive = archive_entry["archive"]
        created_at = None
        if archive_entry.get("created_at"):
            try:
                created_at = dt.datetime.fromisoformat(str(archive_entry["created_at"]).replace("Z", "+00:00"))
            except ValueError:
                created_at = None
        payload = build_notion_sync_payload(
            "migration-v3",
            list(archive["sources"]),
            dict(archive["analysis"]),
            created_at=created_at,
        )
        metrics = archive_entry.get("metrics") or {}
        if metrics.get("model"):
            payload["model"] = metrics["model"]
        if metrics.get("total_tokens"):
            payload["total_tokens"] = metrics["total_tokens"]
        if metrics.get("cost_jpy"):
            payload["cost_jpy"] = metrics["cost_jpy"]
        if metrics.get("breakdown"):
            payload["cost_breakdown"] = metrics["breakdown"]
        if archive_entry.get("external_id"):
            payload["external_id"] = archive_entry["external_id"]
            payload["learning_items"] = extract_learning_items(
                dict(archive["analysis"]), payload["external_id"]
            )
        refresh_notion_render(payload, list(archive["sources"]), dict(archive["analysis"]))
        lesson = _upsert_lesson(client, str(workspace["lessons_data_source_id"]), payload)
        for item in payload.get("learning_items") or []:
            replacement_item_ids.add(str(item.get("external_id") or ""))
            item_page = _upsert_learning_item(
                client,
                str(workspace["items_data_source_id"]),
                item,
                str(lesson["id"]),
            )
            _restore_study_state(
                client,
                str(item_page["id"]),
                study_states.get(_learning_fingerprint_from_item(item)),
            )
            rebuilt_items += 1
        rebuilt_lessons += 1
        config = session_store.load_notion_workspace_config()
        config["migration_v3"] = {
            **dict(config.get("migration_v3") or {}),
            "status": "running",
            "completed_lessons": rebuilt_lessons,
        }
        session_store.save_notion_workspace_config(config)

    # Old v2 study rows are removed only after every replacement row is durable.
    for page in item_rows:
        relations = _property_relations(page, "Bài phân tích")
        external_id = _property_plain(page, "External ID")
        if (
            relations
            and relations.issubset(readable_ids)
            and external_id not in replacement_item_ids
        ):
            client.request("PATCH", f"/pages/{page['id']}", {"in_trash": True})

    if not unreadable:
        _remove_obsolete_columns(client, str(workspace["lessons_data_source_id"]))
        _remove_obsolete_views(client, str(workspace["items_database_id"]))

    config = session_store.load_notion_workspace_config()
    config["schema_version"] = NOTION_SCHEMA_VERSION
    config["migration_v3"] = {
        **dict(config.get("migration_v3") or {}),
        "status": "complete" if not unreadable else "partial",
        "completed_lessons": rebuilt_lessons,
        "rebuilt_items": rebuilt_items,
        "unreadable_lessons": unreadable,
    }
    session_store.save_notion_workspace_config(config)
    return {
        **summary,
        "status": "complete" if not unreadable else "partial",
        "backup_page_id": str(backup_page.get("id") or ""),
        "rebuilt_lessons": rebuilt_lessons,
        "rebuilt_items": rebuilt_items,
    }


def migrate_notion_workspace_v3_if_needed(
    client: NotionClient,
    settings: NotionSettings,
    workspace: dict[str, Any],
) -> dict[str, Any]:
    """Run the approved v3 rebuild once when active legacy lessons are detected."""
    config = session_store.load_notion_workspace_config()
    status = str((config.get("migration_v3") or {}).get("status") or "")
    if status in {"complete", "partial", "not_needed"}:
        return {"status": status}
    rows = _query_all(client, str(workspace["lessons_data_source_id"]))
    legacy_rows = [
        page for page in rows
        if _property_plain(page, "Phiên bản bố cục") != NOTION_LAYOUT_VERSION
    ]
    if not legacy_rows:
        config["migration_v3"] = {
            "status": "not_needed",
            "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        session_store.save_notion_workspace_config(config)
        return {"status": "not_needed"}
    return rebuild_notion_workspace_v3(client, settings, confirm=True)
