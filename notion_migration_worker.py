"""Standalone, lock-aware Notion v4 migration worker."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import traceback
import uuid

from modules import session_store
from modules.notion_migration import migrate_notion_workspace_v4_if_needed
from modules.notion_sync import NotionClient, ensure_notion_workspace, get_notion_settings


_LOCK_PATH = Path(__file__).with_name(".notion_v4_migration.lock")
_LOCK_STALE_AFTER = dt.timedelta(minutes=30)


def _claim_worker_lock() -> tuple[int, str] | None:
    """Claim one process-wide migration slot before touching the Notion workspace."""
    owner = uuid.uuid4().hex
    for _ in range(2):
        try:
            descriptor = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                modified = dt.datetime.fromtimestamp(_LOCK_PATH.stat().st_mtime, dt.timezone.utc)
            except FileNotFoundError:
                continue
            if dt.datetime.now(dt.timezone.utc) - modified <= _LOCK_STALE_AFTER:
                return None
            try:
                _LOCK_PATH.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(descriptor, owner.encode("ascii"))
        return descriptor, owner
    return None


def _release_worker_lock(descriptor: int, owner: str) -> None:
    os.close(descriptor)
    try:
        if _LOCK_PATH.read_text(encoding="ascii") == owner:
            _LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def run_migration() -> None:
    claimed = _claim_worker_lock()
    if claimed is None:
        return
    descriptor, owner = claimed
    try:
        settings = get_notion_settings()
        if not settings.configured or not settings.token:
            return
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
    finally:
        _release_worker_lock(descriptor, owner)


if __name__ == "__main__":
    run_migration()
