"""Pure state transitions for background analysis jobs."""

from __future__ import annotations

from collections.abc import MutableMapping
import hashlib
import json
from typing import Any


def sync_job_state(
    state: MutableMapping[str, Any],
    job_id: str,
    job: dict[str, Any],
    session_id: str,
    current_source_hash: str | None = None,
) -> tuple[str, bool]:
    """Apply owned job progress/result and return ``(status, changed)``."""
    if job.get("session_id") not in (None, session_id):
        return "foreign", False
    if job.get("source_hash") and current_source_hash != job["source_hash"]:
        return "stale", False

    state["current_job_id"] = job_id
    status = str(job.get("status") or "pending")
    changed = False
    partial = list(job.get("partial_result") or [])

    if partial and status in ("running", "failed"):
        if state.get("partial_page_analyses") != partial:
            state["partial_page_analyses"] = partial
            changed = True

    if status == "done" and state.get("applied_job_id") != job_id:
        state["analysis"] = job.get("result")
        state["partial_page_analyses"] = []
        state["applied_job_id"] = job_id
        changed = True

    return status, changed


def items_source_hash(items: list[dict[str, Any]]) -> str:
    """Hash ordered OCR sources so stale background results are rejected."""
    payload = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "text": str(item.get("edited_text") or "").strip(),
            "notes": (item.get("ocr_result") or {}).get("ocr_notes", []),
        }
        for item in items
        if str(item.get("edited_text") or "").strip()
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
