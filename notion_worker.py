"""Standalone worker for durable Notion synchronization runs."""

from __future__ import annotations

import sys
import traceback

from modules import session_store
from modules.notion_sync import NotionAPIError, execute_notion_sync, retry_at


def run_notion_job(run_id: str) -> None:
    if not session_store.mark_notion_sync_running(run_id):
        return
    run = session_store.get_notion_sync_run(run_id)
    if not run:
        return
    try:
        result = execute_notion_sync(run)
        item_errors = result.get("item_errors") or []
        session_store.finish_notion_sync_run(
            run_id,
            "partial" if item_errors else "done",
            error=(f"{len(item_errors)} mục chưa đồng bộ được." if item_errors else None),
            notion_page_id=result.get("page_id"),
            notion_page_url=result.get("page_url"),
            item_errors=item_errors,
        )
    except NotionAPIError as exc:
        current = session_store.get_notion_sync_run(run_id) or run
        status = "retry" if exc.retryable else "failed"
        session_store.finish_notion_sync_run(
            run_id,
            status,
            error=str(exc),
            next_retry_at=retry_at(int(current.get("attempts", 1))) if status == "retry" else None,
        )
    except Exception as exc:
        traceback.print_exc()
        current = session_store.get_notion_sync_run(run_id) or run
        session_store.finish_notion_sync_run(
            run_id,
            "retry",
            error=str(exc),
            next_retry_at=retry_at(int(current.get("attempts", 1))),
        )


if __name__ == "__main__":
    run_notion_job(sys.argv[1])
