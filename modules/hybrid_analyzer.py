"""Hybrid analyzer combining Himotoki offline dictionary with Gemini LLM enrichment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from modules.himotoki_analyzer import analyze_text_with_himotoki
from modules.text_analyzer import _init_model, parse_analysis_response, _usage

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "hybrid_analysis_prompt.txt"


def run_hybrid_analysis(text: str) -> dict[str, Any]:
    """Run morphological analysis using Himotoki, then enrich with Gemini.
    
    Args:
        text: Japanese text to analyze.
        
    Returns:
        Structured dict compatible with the app's analysis schema.
    """
    if not text or not text.strip():
        return {
            "summary": "Không có văn bản để phân tích.",
            "vocabulary_all": [],
            "vocabulary_important": [],
            "kanji_analysis": [],
            "grammar_points": [],
            "full_markdown": "Không có văn bản.",
        }

    # Step 1: Analyze offline with Himotoki
    himotoki_data = analyze_text_with_himotoki(text)
    
    # Step 2: Build the prompt using Himotoki's parsed structures as raw JSON
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    
    # Keep the prompt inputs compact to save token usage
    vocab_raw = json.dumps(himotoki_data["vocabulary_all"], ensure_ascii=False)
    kanji_raw = json.dumps(himotoki_data["kanji_analysis"], ensure_ascii=False)
    grammar_raw = json.dumps(himotoki_data["grammar_points"], ensure_ascii=False)
    
    built_prompt = (
        prompt_template.replace("{source_text}", text.strip())
        .replace("{vocab_raw_json}", vocab_raw)
        .replace("{kanji_raw_json}", kanji_raw)
        .replace("{grammar_raw_json}", grammar_raw)
    )

    # Step 3: Send to Gemini for translation, JLPT levels, and grammatical explanations
    model = _init_model()
    try:
        response = model.generate_content(
            built_prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 16384,
            },
        )
        
        if not response.text or not response.text.strip():
            raise ValueError("Gemini không trả về nội dung phân tích.")
            
        parsed = parse_analysis_response(response.text, "japanese")
        parsed["confirmed_text"] = text
        parsed["usage"] = _usage(response)
        return parsed

    except Exception as exc:
        logger.error(f"Lỗi phân tích Hybrid: {exc}")
        # Fallback to pure Himotoki offline data if Gemini fails
        himotoki_data["summary"] = f"Phân tích bằng Himotoki Offline (Gemini lỗi: {exc})"
        return himotoki_data


def run_page_analyses_hybrid(
    pages: list[dict[str, Any]],
    progress_callback: Any = None,
    page_done_callback: Any = None,
) -> dict[str, Any]:
    """Run Hybrid analysis page-by-page."""
    page_analyses = []
    for index, page in enumerate(pages, 1):
        if progress_callback:
            progress_callback(index - 1, len(pages), page.get("page_name", ""))
            
        # Run hybrid analysis on single page text
        page_analysis = run_hybrid_analysis(page["text"])
        page_analysis["page_index"] = page["page_index"]
        page_analysis["page_name"] = page["page_name"]
        page_analysis["source_label"] = f"Trang {page['page_index']}: {page['page_name']}"
        
        if page_done_callback:
            page_done_callback(page_analysis)
            
        page_analyses.append(page_analysis)
        
    if progress_callback:
        progress_callback(len(pages), len(pages), "Hoàn thành")
        
    from modules.text_analyzer import merge_page_analyses
    return merge_page_analyses(page_analyses, "japanese")
