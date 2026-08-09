from __future__ import annotations

import notion_migration_worker


def test_migration_worker_lock_allows_only_one_process(tmp_path, monkeypatch):
    monkeypatch.setattr(notion_migration_worker, "_LOCK_PATH", tmp_path / "migration.lock")

    first = notion_migration_worker._claim_worker_lock()
    assert first is not None
    assert notion_migration_worker._claim_worker_lock() is None

    notion_migration_worker._release_worker_lock(*first)
    second = notion_migration_worker._claim_worker_lock()
    assert second is not None
    notion_migration_worker._release_worker_lock(*second)
