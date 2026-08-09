"""Restartable migrations for preview-heavy Notion study workspaces."""

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
    _query_external_id,
    _sync_payload_entities,
    _upsert_learning_item,
    _upsert_lesson,
    _find_child_databases,
    build_notion_sync_payload,
    ensure_notion_workspace,
    extract_learning_items,
    extract_notion_entities,
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


OBSOLETE_VOCABULARY_COLUMNS = (
    "Loại", "ID câu nguồn", "Trang", "Thứ tự nguồn", "Quan trọng",
    "Công thức / Cấu tạo", "Vai trò / Liên kết", "Dịch theo cụm",
    "Dịch sát", "Dịch tự nhiên", "Điểm phức tạp",
)


def _state_category(item_type: str) -> str:
    if item_type in {"Từ vựng", "Từ khó", "Cụm từ"}:
        return "vocabulary"
    if item_type == "Kanji":
        return "kanji"
    if item_type in {"Ngữ pháp", "Từ nối", "Mẫu câu"}:
        return f"language:{item_type}"
    if item_type in {"Câu", "Câu dài"}:
        return "sentence"
    return item_type.lower()


def _v4_state_key(language: str, category: str, title: str, reading: str = "") -> str:
    normalized = "|".join(" ".join(value.lower().split()) for value in (language, category, title, reading))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _v4_state_key_from_page(page: dict[str, Any]) -> str:
    item_type = _property_select(page, "Loại") or "Từ vựng"
    return _v4_state_key(
        _property_select(page, "Ngôn ngữ"), _state_category(item_type),
        _property_plain(page, "Tên"), _property_plain(page, "Cách đọc"),
    )


def _v4_state_key_from_entity(kind: str, entity: dict[str, Any]) -> str:
    language = "Tiếng Nhật" if entity.get("language") == "japanese" else "Tiếng Anh"
    category = f"language:{entity.get('type')}" if kind == "language" else kind
    title = str(entity.get("original") or entity.get("title") or "")
    return _v4_state_key(language, category, title, str(entity.get("reading") or ""))


def _merge_study_state(current: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if not current:
        return dict(incoming)
    rank = {"Mới": 0, "Đang học": 1, "Đã nhớ": 2}
    statuses = [str(current.get("status") or "Mới"), str(incoming.get("status") or "Mới")]
    next_dates = [value for value in (current.get("next_review"), incoming.get("next_review")) if value]
    last_dates = [value for value in (current.get("last_review"), incoming.get("last_review")) if value]
    return {
        "status": max(statuses, key=lambda value: rank.get(value, 0)),
        "review_count": max(int(current.get("review_count") or 0), int(incoming.get("review_count") or 0)),
        "next_review": min(next_dates) if next_dates else "",
        "last_review": max(last_dates) if last_dates else "",
    }


def _remove_v4_obsolete_columns(client: NotionClient, workspace: dict[str, Any]) -> None:
    _remove_obsolete_columns(client, str(workspace["lessons_data_source_id"]))
    source_id = str(workspace["items_data_source_id"])
    source = client.request("GET", f"/data_sources/{source_id}")
    existing = source.get("properties") or {}
    removable = {name: None for name in OBSOLETE_VOCABULARY_COLUMNS if name in existing}
    if removable:
        client.request("PATCH", f"/data_sources/{source_id}", {"properties": removable})


def _rename_vocabulary_database(client: NotionClient, workspace: dict[str, Any]) -> None:
    client.request("PATCH", f"/databases/{workspace['items_database_id']}", {"title": _text("Từ vựng")})
    client.request("PATCH", f"/data_sources/{workspace['items_data_source_id']}", {"name": "Từ vựng"})
    lesson_source = client.request("GET", f"/data_sources/{workspace['lessons_data_source_id']}")
    properties = lesson_source.get("properties") or {}
    if "Mục cần học" in properties and "Từ vựng" not in properties:
        client.request(
            "PATCH", f"/data_sources/{workspace['lessons_data_source_id']}",
            {"properties": {"Mục cần học": {"name": "Từ vựng"}}},
        )


def _append_hub_v4_links(client: NotionClient, parent_page_id: str, workspace: dict[str, Any]) -> None:
    links = []
    for key, label in (
        ("lessons", "Bài phân tích"), ("sentences", "Câu & bản dịch"),
        ("items", "Từ vựng"), ("kanji", "Kanji"),
        ("language", "Ngữ pháp & liên kết"),
    ):
        database_id = str(workspace.get(f"{key}_database_id") or "").replace("-", "")
        if database_id:
            links.append(f"- [{label}](https://www.notion.so/{database_id})")
    client.request(
        "PATCH", f"/pages/{parent_page_id}/markdown",
        {"type": "insert_content", "insert_content": {
            "content": "\n\n# Không gian học v4\n" + "\n".join(links) +
                       "\n\nBắt đầu ở view **Ôn hôm nay** trong từng bảng.",
            "position": {"type": "end"},
        }},
    )


def _workspace_signature(workspace: dict[str, Any]) -> str:
    source_ids = [
        str(workspace.get(f"{key}_data_source_id") or "")
        for key in ("lessons", "items", "sentences", "kanji", "language")
    ]
    return hashlib.sha256("|".join(source_ids).encode("utf-8")).hexdigest()


def _archive_duplicate_v4_databases(
    client: NotionClient, parent_page_id: str, workspace: dict[str, Any]
) -> list[str]:
    """Archive only non-canonical v4 databases after the canonical rebuild succeeds."""
    archived: list[str] = []
    for key, title in (
        ("sentences", "Câu & bản dịch"),
        ("kanji", "Kanji"),
        ("language", "Ngữ pháp & liên kết"),
    ):
        canonical_id = str(workspace.get(f"{key}_database_id") or "")
        for database_id, _ in _find_child_databases(client, parent_page_id, {title}):
            if database_id and database_id != canonical_id:
                client.request("PATCH", f"/databases/{database_id}", {"in_trash": True})
                archived.append(database_id)
    return archived


def rebuild_notion_workspace_v4(
    client: NotionClient,
    settings: NotionSettings,
    *,
    confirm: bool = False,
) -> dict[str, Any]:
    """Back up and rebuild the existing workspace into five v4 databases."""
    workspace = ensure_notion_workspace(client, settings)
    lesson_rows = _query_all(client, str(workspace["lessons_data_source_id"]))
    old_item_rows = _query_all(client, str(workspace["items_data_source_id"]))
    archives: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for page in lesson_rows:
        try:
            archive = _download_json(client, _archive_url(page))
            if not isinstance(archive.get("analysis"), dict) or not isinstance(archive.get("sources"), list):
                raise ValueError("JSON không có sources/analysis hợp lệ.")
            archives.append({
                "page_id": str(page["id"]),
                "external_id": _property_plain(page, "External ID"),
                "created_at": _property_date(page, "Ngày phân tích"),
                "archive": archive,
            })
        except (ValueError, KeyError) as exc:
            unreadable.append({"page_id": str(page.get("id") or ""), "error": str(exc)})

    study_states: dict[str, dict[str, Any]] = {}
    for page in old_item_rows:
        key = _v4_state_key_from_page(page)
        study_states[key] = _merge_study_state(study_states.get(key), _study_state(page))
    bundle = {
        "migration": "notion-layout-v4",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lessons": archives,
        "unreadable_lessons": unreadable,
        "learning_items": [
            {"page_id": str(page.get("id") or ""), "properties": page.get("properties") or {}}
            for page in old_item_rows
        ],
    }
    summary = {
        "lesson_count": len(lesson_rows), "readable_lesson_count": len(archives),
        "unreadable_lessons": unreadable, "item_count": len(old_item_rows),
        "would_remove_obsolete_columns": not unreadable,
    }
    if not confirm:
        return {**summary, "status": "dry_run"}
    if not archives and not unreadable:
        return {**summary, "status": "not_needed", "rebuilt_lessons": 0, "rebuilt_items": 0}

    config = session_store.load_notion_workspace_config()
    migration = dict(config.get("migration_v4") or {})
    workspace_signature = _workspace_signature(workspace)
    checkpoint_matches = migration.get("workspace_signature") == workspace_signature
    backup_page_id = str(migration.get("backup_page_id") or "")
    if not backup_page_id:
        parent_id = _resolve_backup_parent(client, settings, workspace)
        backup = _create_backup_page(client, parent_id, bundle)
        backup_page_id = str(backup.get("id") or "")
    config["migration_v4"] = {
        **migration, "status": "backed_up", "backup_page_id": backup_page_id,
        "completed_lessons": int(migration.get("completed_lessons") or 0),
        "total_lessons": len(archives),
        "lock_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workspace_signature": workspace_signature,
    }
    session_store.save_notion_workspace_config(config)
    _rename_vocabulary_database(client, workspace)

    migration = dict(session_store.load_notion_workspace_config().get("migration_v4") or {})
    completed_page_ids = set(migration.get("completed_page_ids") or []) if checkpoint_matches else set()
    rebuilt_lessons = len(completed_page_ids)
    rebuilt_items = int(migration.get("rebuilt_items") or 0) if checkpoint_matches else 0
    replacement_vocab_ids: set[str] = set()
    for archive_entry in archives:
        archive = archive_entry["archive"]
        created_at = None
        if archive_entry.get("created_at"):
            try:
                created_at = dt.datetime.fromisoformat(str(archive_entry["created_at"]).replace("Z", "+00:00"))
            except ValueError:
                pass
        payload = build_notion_sync_payload(
            "migration-v4", list(archive["sources"]), dict(archive["analysis"]), created_at=created_at
        )
        if archive_entry.get("external_id"):
            payload["external_id"] = archive_entry["external_id"]
            entities = extract_notion_entities(dict(archive["analysis"]), payload["external_id"])
            payload.update(entities)
            payload["learning_items"] = [
                *entities["vocabulary"], *entities["kanji"],
                *entities["language_items"], *entities["sentences"],
            ]
        replacement_vocab_ids.update(
            str(entity.get("external_id") or "") for entity in payload.get("vocabulary") or []
        )
        if archive_entry["page_id"] in completed_page_ids:
            continue
        refresh_notion_render(payload, list(archive["sources"]), dict(archive["analysis"]))
        lesson = _upsert_lesson(client, str(workspace["lessons_data_source_id"]), payload)
        errors = _sync_payload_entities(client, workspace, payload, str(lesson["id"]))
        if errors:
            raise NotionAPIError(f"Migration v4 còn {len(errors)} mục lỗi; dữ liệu cũ chưa bị archive.", 400, "migration_item_error")

        for kind, key, source_key in (
            ("sentence", "sentences", "sentences_data_source_id"),
            ("kanji", "kanji", "kanji_data_source_id"),
            ("vocabulary", "vocabulary", "items_data_source_id"),
            ("language", "language_items", "language_data_source_id"),
        ):
            for entity in payload.get(key) or []:
                page = _query_external_id(client, str(workspace[source_key]), str(entity["external_id"]))
                if page:
                    _restore_study_state(
                        client, str(page["id"]),
                        study_states.get(_v4_state_key_from_entity(kind, entity)),
                    )
                rebuilt_items += 1
        rebuilt_lessons += 1
        completed_page_ids.add(archive_entry["page_id"])
        config = session_store.load_notion_workspace_config()
        config["migration_v4"] = {
            **dict(config.get("migration_v4") or {}),
            "status": "running", "completed_lessons": rebuilt_lessons,
            "completed_page_ids": sorted(completed_page_ids),
            "rebuilt_items": rebuilt_items,
            "lock_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        session_store.save_notion_workspace_config(config)

    readable_ids = {entry["page_id"] for entry in archives}
    for page in old_item_rows:
        relations = _property_relations(page, "Bài phân tích")
        external_id = _property_plain(page, "External ID")
        if relations and relations.issubset(readable_ids) and external_id not in replacement_vocab_ids:
            client.request("PATCH", f"/pages/{page['id']}", {"in_trash": True})
    if not unreadable:
        _remove_v4_obsolete_columns(client, workspace)
        if settings.parent_page_id:
            archived_duplicates = _archive_duplicate_v4_databases(
                client, settings.parent_page_id, workspace
            )
        else:
            archived_duplicates = []
    else:
        archived_duplicates = []
    config = session_store.load_notion_workspace_config()
    if settings.parent_page_id and not (config.get("migration_v4") or {}).get("hub_links_created"):
        _append_hub_v4_links(client, settings.parent_page_id, workspace)
        config = session_store.load_notion_workspace_config()
        config.setdefault("migration_v4", {})["hub_links_created"] = True
    final_status = "complete" if not unreadable else "partial"
    config["schema_version"] = NOTION_SCHEMA_VERSION
    config["migration_v4"] = {
        **dict(config.get("migration_v4") or {}), "status": final_status,
        "completed_lessons": rebuilt_lessons, "rebuilt_items": rebuilt_items,
        "unreadable_lessons": unreadable,
        "archived_duplicate_databases": archived_duplicates,
        "workspace_signature": workspace_signature,
        "error": "",
    }
    session_store.save_notion_workspace_config(config)
    return {
        **summary, "status": final_status, "backup_page_id": backup_page_id,
        "rebuilt_lessons": rebuilt_lessons, "rebuilt_items": rebuilt_items,
        "archived_duplicate_databases": archived_duplicates,
    }


def migrate_notion_workspace_v4_if_needed(
    client: NotionClient,
    settings: NotionSettings,
    workspace: dict[str, Any],
) -> dict[str, Any]:
    config = session_store.load_notion_workspace_config()
    status = str((config.get("migration_v4") or {}).get("status") or "")
    if status in {"complete", "partial", "not_needed"}:
        return {"status": status}
    if status in {"starting", "backed_up", "running"}:
        lock_at = str((config.get("migration_v4") or {}).get("lock_at") or "")
        try:
            locked = dt.datetime.fromisoformat(lock_at.replace("Z", "+00:00"))
        except ValueError:
            locked = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        if dt.datetime.now(dt.timezone.utc) - locked < dt.timedelta(minutes=10):
            return {"status": "running"}
    rows = _query_all(client, str(workspace["lessons_data_source_id"]))
    if not rows:
        config["migration_v4"] = {"status": "not_needed", "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        session_store.save_notion_workspace_config(config)
        return {"status": "not_needed"}
    config["migration_v4"] = {
        **dict(config.get("migration_v4") or {}),
        "status": "starting", "lock_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    session_store.save_notion_workspace_config(config)
    return rebuild_notion_workspace_v4(client, settings, confirm=True)
