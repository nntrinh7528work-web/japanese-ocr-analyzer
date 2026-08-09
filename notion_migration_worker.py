"""Standalone, lock-aware Notion v4 migration worker."""

from __future__ import annotations

import datetime as dt
import traceback

from modules import session_store
from modules.notion_migration import migrate_notion_workspace_v4_if_needed
from modules.notion_sync import NotionClient, ensure_notion_workspace, get_notion_settings


def run_migration() -> None:
    settings = get_notion_settings()
    if not settings.configured or not settings.token:
        return
    try:
        client = NotionClient(settings.token)
        workspace = ensure_notion_workspace(client, settings)
        migrate_notion_workspace_v4_if_needed(client, settings, workspace)
    except Exception as exc:
        traceback.print_exc()
        config = session_store.load_notion_workspace_config()
        migration = dict(config.get("migration_v4") or {})
        migration.update({
            "status": "retry",
            "error": str(exc),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        config["migration_v4"] = migration
        session_store.save_notion_workspace_config(config)


if __name__ == "__main__":
    run_migration()
