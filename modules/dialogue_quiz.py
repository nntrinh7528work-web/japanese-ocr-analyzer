"""Pure code-based Mini Quiz generator built from dialogue data without API calls."""

from __future__ import annotations

import random
import re
from typing import Any


def generate_pure_quiz(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Generate 4 types of quiz items (Cloze, MCQ, Sentence Reorder, Translation) from dialogue result."""
    dialogue = result.get("dialogue", [])
    summary = result.get("summary", "")

    cloze_questions = []
    mcq_questions = []
    reorder_questions = []
    translate_questions = []

    # 1. Cloze Questions (Fill-in-the-blank)
    for turn_idx, turn in enumerate(dialogue):
        highlights = turn.get("highlights", [])
        text = turn.get("text", "")
        for word in highlights:
            if word in text and len(word) > 0:
                masked_text = text.replace(word, "【 _______ 】", 1)
                cloze_questions.append({
                    "id": f"cloze_{turn_idx}_{word}",
                    "sentence": masked_text,
                    "target_word": word,
                    "text_vi": turn.get("text_vi", ""),
                    "options": _generate_distractors(word, result),
                })

    # 2. MCQ Questions (Vocab & Meaning from summary/notes)
    # Parse vocabulary lines from summary or notes like "予約 (yoyaku): đặt trước"
    vocab_lines = re.findall(r"([一-龯ぁ-んァ-ヶa-zA-Z0-9〜\-\s]+)\s*\((.*?)\)?\s*:\s*(.+)", summary)
    for v_idx, match in enumerate(vocab_lines):
        word = match[0].strip()
        reading = match[1].strip()
        meaning = match[2].strip()
        if word and meaning:
            mcq_questions.append({
                "id": f"mcq_{v_idx}",
                "question": f"Từ vựng/cấu trúc `{word}` ({reading}) có nghĩa là gì?",
                "correct_answer": meaning,
                "options": _generate_mcq_options(meaning),
            })

    # 3. Sentence Reorder Questions
    for turn_idx, turn in enumerate(dialogue[:4]): # Take up to 4 turns
        text = turn.get("text", "")
        # Break sentence into 3-5 word chunks
        tokens = [t for t in re.split(r"(| |、|。|\?|！)", text) if t.strip()]
        if len(tokens) >= 3:
            shuffled = tokens.copy()
            random.seed(turn_idx + 42)
            random.shuffle(shuffled)
            reorder_questions.append({
                "id": f"reorder_{turn_idx}",
                "original": text,
                "text_vi": turn.get("text_vi", ""),
                "shuffled_tokens": shuffled,
            })

    # 4. Translation Questions (VI -> JP)
    for turn_idx, turn in enumerate(dialogue):
        if turn.get("text_vi") and turn.get("text"):
            translate_questions.append({
                "id": f"trans_{turn_idx}",
                "prompt_vi": turn["text_vi"],
                "correct_jp": turn["text"],
                "text_hira": turn.get("text_hira", ""),
            })

    return {
        "cloze": cloze_questions,
        "mcq": mcq_questions,
        "reorder": reorder_questions,
        "translate": translate_questions[:5], # Limit to 5
    }


def _generate_distractors(correct_word: str, result: dict[str, Any]) -> list[str]:
    """Generate options for cloze test including correct word and distractors."""
    # Gather other words used in dialogue
    all_highlights = []
    for t in result.get("dialogue", []):
        all_highlights.extend(t.get("highlights", []))

    distractors = [w for w in set(all_highlights) if w != correct_word]
    defaults = ["予約", "確認", "相談", "案内", "注文", "準備", "連絡"]
    for d in defaults:
        if d not in distractors and d != correct_word:
            distractors.append(d)

    random.seed(len(correct_word))
    selected = random.sample(distractors, min(3, len(distractors)))
    options = [correct_word] + selected
    random.shuffle(options)
    return options


def _generate_mcq_options(correct_meaning: str) -> list[str]:
    """Generate 4 MCQ distractors for meaning check."""
    distractor_pool = [
        "Đặt trước chỗ ngồi hoặc sản phẩm",
        "Nhận sự giúp đỡ/ân huệ từ ai đó",
        "Xác nhận lại thông tin với khách hàng",
        "Hủy đơn hàng hoặc dịch vụ đã đăng ký",
        "Cảm ơn và chào hỏi khi giao tiếp",
        "Yêu cầu trợ giúp trong tình huống khẩn cấp",
        "Hỏi về khoảng thời gian hoặc địa điểm",
    ]
    other_options = [d for d in distractor_pool if d.lower() not in correct_meaning.lower()]
    selected = random.sample(other_options, min(3, len(other_options)))
    options = [correct_meaning] + selected
    random.shuffle(options)
    return options
