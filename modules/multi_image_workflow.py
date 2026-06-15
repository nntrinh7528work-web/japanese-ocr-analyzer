"""State helpers for multi-image OCR and combined analysis."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from modules.image_processor import process_image


def image_id(data: bytes) -> str:
    """Return a stable identifier for an uploaded image."""
    return hashlib.sha256(data).hexdigest()


def create_image_item(data: bytes, name: str) -> dict[str, Any]:
    """Preprocess an image and create its independent workflow state."""
    result = process_image(data)
    return {
        "id": image_id(data),
        "name": name,
        "original_image_bytes": result["original_image_bytes"],
        "processed_image_bytes": result["processed_image_bytes"],
        "report": result["report"],
        "ocr_result": None,
        "edited_text": "",
        "ocr_error": None,
    }


def add_image_items(
    current_items: list[dict[str, Any]],
    sources: Iterable[tuple[str, bytes]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Add valid, non-duplicate images and return items, added names, and errors."""
    items = list(current_items)
    known_ids = {item["id"] for item in items}
    added: list[str] = []
    errors: list[str] = []
    for name, data in sources:
        source_id = image_id(data)
        if source_id in known_ids:
            continue
        try:
            item = create_image_item(data, name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        items.append(item)
        known_ids.add(source_id)
        added.append(name)
    return items, added, errors


def combined_text(items: list[dict[str, Any]]) -> str:
    """Combine edited OCR text in image order with clear page boundaries."""
    sections = []
    for index, item in enumerate(items, 1):
        text = item.get("edited_text", "").strip()
        if text:
            sections.append(f"=== ẢNH {index}: {item['name']} ===\n{text}")
    return "\n\n".join(sections)


def combined_notes(items: list[dict[str, Any]]) -> list[str]:
    """Combine OCR notes while preserving the source image."""
    notes = []
    for index, item in enumerate(items, 1):
        result = item.get("ocr_result") or {}
        notes.extend(f"Ảnh {index} ({item['name']}): {note}" for note in result.get("ocr_notes", []))
    return notes


def move_image_item(items: list[dict[str, Any]], item_id: str, direction: int) -> list[dict[str, Any]]:
    """Move an image one position up (-1) or down (+1)."""
    reordered = list(items)
    index = next((index for index, item in enumerate(reordered) if item["id"] == item_id), None)
    if index is None:
        return reordered
    target = index + direction
    if target < 0 or target >= len(reordered):
        return reordered
    reordered[index], reordered[target] = reordered[target], reordered[index]
    return reordered
