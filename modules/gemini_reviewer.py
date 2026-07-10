"""Gemini-based independent review of DeepSeek analysis output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REVIEW_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "gemini_review_prompt.txt"

_REQUIRED_KEYS = {"verdict", "confidence", "issues", "missing_items", "review_note_vi"}

_FALLBACK_REVIEW: dict[str, Any] = {
    "verdict": "review_unavailable",
    "confidence": 0.0,
    "issues": [],
    "missing_items": [],
    "review_note_vi": "Không thể hoàn tất bước kiểm tra Gemini.",
}


def _init_review_model():
    """Initialise a Gemini model using the project's existing SDK pattern."""
    import google.generativeai as genai

    from config import GEMINI_API_KEY, GEMINI_REVIEW_MODEL

    if not GEMINI_API_KEY:
        raise ValueError("Thiếu GEMINI_API_KEY.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_REVIEW_MODEL)


def _validate_review(data: dict) -> None:
    """Raise ValueError if required keys are missing."""
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Thiếu key review: {missing}")


def review_deepseek_analysis(
    source_text: str,
    deepseek_analysis: dict,
    language: str,
) -> dict:
    """Review DeepSeek analysis against the original source text using Gemini.

    Args:
        source_text: The original OCR text (ground truth).
        deepseek_analysis: The DeepSeek analysis dict to review.
        language: ``"ja"`` or ``"en"``.

    Returns:
        A review dict with ``verdict``, ``confidence``, ``issues``,
        ``missing_items``, and ``review_note_vi``.
        On failure, returns a dict with ``verdict="review_unavailable"``.
    """
    if not source_text or not source_text.strip():
        return dict(_FALLBACK_REVIEW)
    if language not in ("ja", "en"):
        return dict(_FALLBACK_REVIEW)

    try:
        model = _init_review_model()
    except Exception:
        return dict(_FALLBACK_REVIEW)

    analysis_json_str = json.dumps(deepseek_analysis, ensure_ascii=False, indent=2)

    prompt_template = REVIEW_PROMPT_PATH.read_text(encoding="utf-8")
    built_prompt = (
        prompt_template
        .replace("{source_text}", source_text.strip())
        .replace("{deepseek_analysis_json}", analysis_json_str)
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = model.generate_content(
                built_prompt,
                generation_config={
                    "temperature": 0.15,
                    "max_output_tokens": 2000,
                },
            )

            content = (response.text or "").strip()
            # Strip markdown code fences if present.
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first and last fence lines.
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            if not content:
                raise ValueError("Gemini trả về nội dung rỗng.")

            data = json.loads(content)
            _validate_review(data)
            return data

        except Exception as exc:
            last_error = exc
            if attempt == 0:
                continue

    # After all retries failed, return fallback instead of raising.
    return dict(_FALLBACK_REVIEW)
