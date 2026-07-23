"""Himotoki-backed Japanese text analyzer module."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


import re

def _split_text(text: str, max_len: int = 90) -> list[str]:
    """Split text into chunks of length <= max_len without breaking punctuation."""
    lines = text.split("\n")
    chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if len(line) <= max_len:
            chunks.append(line)
            continue
            
        sentences = re.split(r"([。、？！\.\?!])", line)
        current = ""
        for part in sentences:
            if not part:
                continue
            if len(current) + len(part) <= max_len:
                current += part
            else:
                if current:
                    chunks.append(current)
                if len(part) <= max_len:
                    current = part
                else:
                    # Force character split if a single part exceeds max_len
                    for i in range(0, len(part), max_len):
                        chunks.append(part[i:i+max_len])
                    current = ""
        if current:
            chunks.append(current)
    return chunks


def analyze_text_with_himotoki(text: str) -> dict[str, Any]:
    """Analyze Japanese text using Himotoki offline library.
    
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

    try:
        import himotoki
        
        # Split text into small chunks to avoid TextTooLongError (100 char limit)
        clean_text = text.strip()
        chunks = _split_text(clean_text)
        
        results = []
        last_index = 0
        
        for chunk in chunks:
            analysis_paths = himotoki.analyze(chunk)
            if not analysis_paths:
                continue
            chunk_results, _ = analysis_paths[0]
            # Find the position of this chunk in the original text to compute correct offsets
            idx = clean_text.find(chunk, last_index)
            if idx == -1:
                idx = last_index
                
            for item in chunk_results:
                item.start += idx
                item.end += idx
                results.append(item)
                
            last_index = idx + len(chunk)
        
        vocab_all = []
        vocab_important = []
        kanji_list = []
        grammar_list = []
        
        for item in results:
            item_text = getattr(item, "text", "")
            kana = getattr(item, "kana", "")
            source_text = getattr(item, "source_text", None) or item_text
            meanings = getattr(item, "meanings", []) or []
            meaning_str = "; ".join(meanings) if meanings else ""
            pos = getattr(item, "pos", "") or ""
            conj_type = getattr(item, "conj_type", None)
            
            # Vocabulary entry
            vocab_entry = {
                "word": item_text,
                "reading": kana,
                "dictionary_form": source_text,
                "part_of_speech": pos,
                "meaning": meaning_str,
            }
            vocab_all.append(vocab_entry)
            
            # If word has meanings or conjugations, add to important/grammar
            if meanings:
                vocab_important.append(vocab_entry)
                
            if conj_type:
                grammar_list.append({
                    "pattern": f"{item_text} ({conj_type})",
                    "meaning": f"Chia theo thể {conj_type} của động từ gốc {source_text}",
                    "example": item_text,
                })
                
            # Extract Kanji
            for char in item_text:
                if "\u4e00" <= char <= "\u9fff":
                    if not any(k["kanji"] == char for k in kanji_list):
                        kanji_list.append({
                            "kanji": char,
                            "onyomi": "",
                            "kunyomi": "",
                            "meaning": f"Hán tự xuất hiện trong từ '{item_text}'",
                        })
                        
        markdown_summary = f"### Phân tích bằng Himotoki (Offline)\n\n"
        markdown_summary += f"**Số từ phân tích được:** {len(results)}\n\n"
        markdown_summary += "| Từ | Hiragana | Từ gốc | Loại từ | Ý nghĩa |\n"
        markdown_summary += "|---|---|---|---|---|\n"
        for v in vocab_all:
            markdown_summary += f"| {v['word']} | {v['reading']} | {v['dictionary_form']} | {v['part_of_speech']} | {v['meaning']} |\n"

        return {
            "confirmed_text": text,
            "analysis_language": "japanese",
            "summary": f"Đã phân tích {len(results)} từ/cụm từ bằng thư viện Himotoki offline.",
            "vocabulary_all": vocab_all,
            "vocabulary_important": vocab_important,
            "kanji_analysis": kanji_list,
            "grammar_points": grammar_list,
            "full_markdown": markdown_summary,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    except Exception as exc:
        logger.error(f"Lỗi khi chạy Himotoki analyzer: {exc}")
        return {
            "confirmed_text": text,
            "analysis_language": "japanese",
            "summary": f"Lỗi phân tích bằng Himotoki: {exc}",
            "vocabulary_all": [],
            "vocabulary_important": [],
            "kanji_analysis": [],
            "grammar_points": [],
            "full_markdown": f"❌ Lỗi: {exc}",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


