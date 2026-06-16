"""Build downloadable analysis artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any


def safe_export_stem(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "japanese_analysis"


def default_export_stem(items: list[dict[str, Any]], created_at: datetime | None = None) -> str:
    """Build a useful default filename stem from the first item and timestamp."""
    timestamp = (created_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if not items:
        return f"japanese_analysis_{timestamp}"
    first_name = str(items[0].get("name") or "analysis").rsplit(".", 1)[0]
    return safe_export_stem(f"{first_name}_{len(items)}_items_{timestamp}")


def markdown_bytes(analysis: dict[str, Any]) -> bytes:
    """Return the full analysis Markdown as UTF-8 bytes."""
    content = str(analysis.get("full_markdown") or "").strip()
    if not content:
        raise ValueError("Không có nội dung Markdown để lưu.")
    return content.encode("utf-8")


def analysis_json_bytes(
    items: list[dict[str, Any]],
    analysis: dict[str, Any],
    session_cost: dict[str, Any],
    billing_tier: str,
    usd_to_vnd: float,
    budget: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> bytes:
    """Return a compact JSON archive of the session without embedding image bytes."""
    payload = {
        "created_at": (created_at or datetime.now()).isoformat(timespec="seconds"),
        "billing_tier": billing_tier,
        "usd_to_vnd": usd_to_vnd,
        "session_cost": session_cost,
        "budget": budget or {},
        "sources": [
            {
                "name": item.get("name", ""),
                "ocr_text": item.get("edited_text", ""),
                "ocr_notes": (item.get("ocr_result") or {}).get("ocr_notes", []),
                "confidence": (item.get("ocr_result") or {}).get("confidence"),
                "text_direction": (item.get("ocr_result") or {}).get("text_direction"),
            }
            for item in items
        ],
        "analysis": analysis,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
