"""Pure state transitions for background analysis jobs."""

from __future__ import annotations

from collections.abc import MutableMapping
import hashlib
import json
from typing import Any

from modules.sentence_analyzer import merge_manual_breakdown
from modules.translation_guidance import merge_guidance_job


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
        result = job.get("result") or {}
        is_sentence_job = str(job.get("lang") or "").startswith("sentence_") or result.get("job_kind") == "sentence_deep_dive"
        is_guidance_job = str(job.get("lang") or "").startswith("guidance_") or result.get("job_kind") == "translation_guidance"
        if is_guidance_job:
            merged, merged_changed = merge_guidance_job(state.get("analysis"), result, job_id)
            if merged_changed:
                state["analysis"] = merged
                changed = True
        elif is_sentence_job:
            merged, merged_changed = merge_manual_breakdown(state.get("analysis"), result, job_id)
            if merged_changed:
                state["analysis"] = merged
                changed = True
        else:
            state["analysis"] = result
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
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
