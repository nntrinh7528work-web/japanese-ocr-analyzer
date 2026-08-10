"""Sentence splitting, complexity ranking, and Gemini-backed deep analysis."""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any


ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "candidate_tokens": 0,
    "thinking_tokens": 0,
}

_JA_OPEN = {"「": "」", "『": "』", "（": "）", "(": ")", "［": "］", "【": "】", "“": "”"}
_JA_CLOSE = set(_JA_OPEN.values())
_EN_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "a.m.", "p.m.", "fig.", "no.", "inc.", "ltd.", "u.s.", "u.k.",
}

_JA_CLAUSE_PATTERNS = (
    r"(?:ので|のに|ながら|けれども|けれど|ため(?:に)?|ところ|ものの|にもかかわらず|一方(?:で)?|の(?:で|に))",
    r"(?:なら|たら|れば|と)(?:、|\s)",
    r"(?:しかし|そして|それで|そのため|一方で|つまり|したがって|ところが|また|なお)",
    r"(?:こと|もの|という|よう)(?:を|が|は|に|で|だ|です|になる)",
    r"(?:させる|させられる|られる|れる|ことができる|得る|うる|てしまう|ておく|ている)",
    r"(?:たり|し)(?:、|て|たり)",
)
_JA_CONJUNCTIVE_GA_RE = re.compile(r"(?:[ぁ-んァ-ン一-龯々])(?:だ|です|だった|ました|ない|た|る|れる|られる|ている|たい)?が(?:、|\s)")
_EN_CLAUSE_RE = re.compile(
    r"\b(?:although|though|even though|because|since|while|whereas|if|unless|when|whenever|"
    r"before|after|until|once|so that|in order that|which|who|whom|whose|where|however|"
    r"therefore|moreover|nevertheless|yet|but|nor)\b",
    re.IGNORECASE,
)
_EN_NOUN_CLAUSE_RE = re.compile(
    r"\b(?:think|know|say|believe|show|suggest|argue|find|report|claim|mean|ensure|prove)\s+that\b",
    re.IGNORECASE,
)
_EN_RELATIVE_THAT_RE = re.compile(
    r"\b(?:the|a|an|this|these|those|my|your|his|her|our|their)\s+[A-Za-z][\w'-]*\s+that\s+(?:I|you|he|she|we|they|[A-Za-z][\w'-]*)\b",
    re.IGNORECASE,
)
_EN_COMPLEX_RE = re.compile(
    r"\b(?:to\s+\w+|\w+ing\b|\w+ed\b|has been|have been|had been|will have|"
    r"must|should|could|would|might|there is|there are|it is|what\s+\w+|not only|either\s+.+?\s+or)\b",
    re.IGNORECASE,
)
_JA_CHAR_RE = re.compile(r"[ぁ-んァ-ン一-龯々〆ヶ]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
BREAKDOWN_VERSION = "2.0"


def _language(value: str | None) -> str:
    return "japanese" if value == "japanese" else "english"


def detect_sentence_language(sentence: str, fallback_language: str = "english") -> tuple[str, str, str]:
    """Detect Japanese/English locally without sending source text to another API."""
    source = str(sentence or "")
    japanese_chars = len(_JA_CHAR_RE.findall(source))
    latin_chars = sum(len(word) for word in _LATIN_WORD_RE.findall(source))
    if japanese_chars:
        # A Japanese sentence often includes English product names or acronyms.
        # Kana/Kanji are the reliable signal for the grammar prompt to use.
        confidence = "high" if japanese_chars >= max(2, latin_chars // 4) else "medium"
        return "japanese", confidence, "auto"
    if latin_chars:
        return "english", "high" if latin_chars >= 3 else "medium", "auto"
    return _language(fallback_language), "low", "sidebar_fallback"


def merge_usage(*values: dict[str, Any] | None) -> dict[str, int]:
    """Add Gemini usage dictionaries without assuming every key exists."""
    keys = set(ZERO_USAGE)
    return {
        key: sum(int((value or {}).get(key, 0) or 0) for value in values)
        for key in keys
    }


def response_usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    candidates = int(getattr(metadata, "candidates_token_count", 0) or 0)
    thinking = int(getattr(metadata, "thoughts_token_count", 0) or 0)
    return {
        "input_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
        "output_tokens": candidates + thinking,
        "candidate_tokens": candidates,
        "thinking_tokens": thinking,
    }


def _is_english_period_boundary(text: str, index: int) -> bool:
    if index + 1 < len(text) and text[index + 1] == ".":
        return False
    prefix = text[: index + 1]
    token_match = re.search(r"(?:[A-Za-z](?:\.[A-Za-z])+\.|[A-Za-z]+\.)$", prefix)
    token = token_match.group(0).lower() if token_match else ""
    if token in _EN_ABBREVIATIONS:
        return False
    if re.search(r"\b[A-Z]\.$", prefix):
        return False
    if index and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    if (
        index
        and index + 1 < len(text)
        and text[index - 1].isalnum()
        and text[index + 1].isalnum()
        and text[index - 1].isascii()
        and text[index + 1].isascii()
    ):
        return False
    return True


def split_sentences(text: str, language: str, page_index: int) -> list[dict[str, Any]]:
    """Split source text in order and assign stable page/sentence IDs."""
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # OCR often inserts visual line and paragraph breaks inside one grammatical
    # sentence. Sentence boundaries are punctuation-driven, never layout-driven.
    source = re.sub(r"\s*\n+\s*", " ", source)
    source = re.sub(r"[ \t]+", " ", source).strip()
    if not source:
        return []
    sentences: list[str] = []
    buffer: list[str] = []
    stack: list[str] = []

    for index, char in enumerate(source):
        buffer.append(char)
        if char in _JA_OPEN:
            stack.append(_JA_OPEN[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char in _JA_CLOSE and char in stack:
            stack.remove(char)

        boundary = False
        if not stack:
            # Mixed OCR pages need punctuation from both writing systems.  A full
            # stop still goes through the abbreviation/decimal guard.
            if char in "。！？?!":
                boundary = True
            elif char in ".．":
                boundary = _is_english_period_boundary(source, index)

        if boundary:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)

    catalog = []
    for ordinal, original in enumerate(sentences, 1):
        detected_language, confidence, source_kind = detect_sentence_language(original, language)
        score, signals = score_complexity(original, detected_language)
        catalog.append(
            {
                "sentence_id": f"p{int(page_index)}-s{ordinal}",
                "ordinal": ordinal,
                "original": original,
                "detected_language": detected_language,
                "language_confidence": confidence,
                "language_source": source_kind,
                "complexity_score": score,
                "complexity_signals": signals,
                "eligible": is_complex_sentence(original, detected_language, score),
                "selected_auto": False,
                "analyzed": False,
                "analysis_origin": None,
            }
        )
    return catalog


def score_complexity(sentence: str, language: str) -> tuple[int, list[str]]:
    """Return a deterministic local complexity score and human-readable signals."""
    lang = _language(language)
    signals: list[str] = []
    if lang == "japanese":
        compact = re.sub(r"\s", "", sentence)
        length_points = min(4, len(compact) // 20)
        comma_count = sentence.count("、")
        comma_points = min(3, comma_count)
        markers = sum(len(re.findall(pattern, sentence)) for pattern in _JA_CLAUSE_PATTERNS)
        conjunctive_ga = len(_JA_CONJUNCTIVE_GA_RE.findall(sentence))
        markers += conjunctive_ga
        clause_points = min(6, markers * 2)
        noun_modifier = bool(re.search(r"(?:た|ている|ない|る|れる|られる|という)[^、。]{1,18}(?:こと|もの|人|時|場合|点|方法|理由)", sentence))
        condition = bool(re.search(r"(?:なら|たら|れば|ても|としても|にもかかわらず|ものの)", sentence))
        nested = any(opener in sentence for opener in _JA_OPEN) or sentence.count("（") > 0
        predicate_chain = bool(re.search(r"(?:て|で|ながら|つつ|たり|し)[ぁ-んァ-ン一-龯]{1,10}(?:て|た|る|ない|ます|です)", sentence))
        score = length_points + comma_points + clause_points + int(noun_modifier) * 2 + int(condition) * 2 + int(nested) * 2 + int(predicate_chain) * 2
        if len(compact) >= 35:
            signals.append(f"dài {len(compact)} ký tự")
        if comma_count:
            signals.append(f"{comma_count} dấu phẩy")
        if markers:
            signals.append(f"{markers} dấu hiệu mệnh đề/từ nối")
        if conjunctive_ga:
            signals.append("が nối mệnh đề")
        if noun_modifier:
            signals.append("bổ nghĩa danh từ")
        if condition:
            signals.append("điều kiện/nhượng bộ")
        if nested:
            signals.append("cấu trúc lồng")
        if predicate_chain:
            signals.append("chuỗi vị ngữ/dạng liên kết")
        return score, signals

    words = re.findall(r"\b[\w'-]+\b", sentence)
    punctuation = len(re.findall(r"[,;:]", sentence))
    markers = (
        len(_EN_CLAUSE_RE.findall(sentence))
        + len(_EN_NOUN_CLAUSE_RE.findall(sentence))
        + len(_EN_RELATIVE_THAT_RE.findall(sentence))
    )
    coordination = len(re.findall(r"\b(?:and|or)\b", sentence, re.I)) if re.search(r",\s*(?:and|or)\b|\b(?:not only|either)\b", sentence, re.I) else 0
    participle = bool(re.search(r"(?:^|[,;]\s+)(?:having|being|using|given|considering|despite)\b|\b\w+ing\s*,", sentence, re.I))
    parenthetical = bool(re.search(r"\([^)]{3,}\)|—[^—]+—", sentence))
    complex_forms = bool(_EN_COMPLEX_RE.search(sentence))
    score = min(4, len(words) // 12) + min(3, punctuation) + min(6, markers * 2) + min(2, coordination) + int(participle) * 2 + int(parenthetical) * 2 + int(complex_forms)
    if len(words) >= 20:
        signals.append(f"dài {len(words)} từ")
    if punctuation:
        signals.append(f"{punctuation} dấu ngắt")
    if markers:
        signals.append(f"{markers} mệnh đề/liên từ")
    if coordination:
        signals.append("liên kết song song")
    if participle:
        signals.append("cụm phân từ")
    if parenthetical:
        signals.append("phần chen giữa")
    if complex_forms:
        signals.append("dạng động từ/cấu trúc phức")
    return score, signals


def is_complex_sentence(sentence: str, language: str, score: int | None = None) -> bool:
    lang = _language(language)
    value = score if score is not None else score_complexity(sentence, lang)[0]
    if lang == "japanese":
        marker_count = sum(len(re.findall(pattern, sentence)) for pattern in _JA_CLAUSE_PATTERNS) + len(_JA_CONJUNCTIVE_GA_RE.findall(sentence))
        return value >= 5 and (len(re.sub(r"\s", "", sentence)) >= 35 or marker_count >= 2)
    words = re.findall(r"\b[\w'-]+\b", sentence)
    marker_count = (
        len(_EN_CLAUSE_RE.findall(sentence))
        + len(_EN_NOUN_CLAUSE_RE.findall(sentence))
        + len(_EN_RELATIVE_THAT_RE.findall(sentence))
    )
    return value >= 5 and (len(words) >= 20 or marker_count >= 2 or bool(_EN_COMPLEX_RE.search(sentence)))


def build_sentence_catalog(pages: list[dict[str, Any]], language: str) -> dict[int, list[dict[str, Any]]]:
    return {
        int(page["page_index"]): split_sentences(page.get("text", ""), language, int(page["page_index"]))
        for page in pages
    }


def deep_analysis_batches(sentences: list[dict[str, Any]], fallback_language: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group source-ordered sentences by detected language and output budget."""
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    current_language = ""
    current: list[dict[str, Any]] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            grouped.append((current_language, current))
        current, current_size = [], 0

    for sentence in sorted(sentences, key=lambda row: int(row.get("ordinal", 0) or 0)):
        language = _language(sentence.get("detected_language") or fallback_language)
        text = str(sentence.get("original") or "")
        size = len(re.sub(r"\s", "", text)) if language == "japanese" else len(re.findall(r"\b[\w'-]+\b", text))
        singleton = (
            int(sentence.get("complexity_score", 0) or 0) >= 12
            or (language == "japanese" and size >= 100)
            or (language == "english" and size >= 50)
        )
        limit = 240 if language == "japanese" else 120
        if singleton:
            flush()
            grouped.append((language, [sentence]))
            continue
        if current and (language != current_language or len(current) >= 3 or current_size + size > limit):
            flush()
        if not current:
            current_language = language
        current.append(sentence)
        current_size += size
    flush()
    return grouped


def select_auto_sentences(
    catalog_by_page: dict[int, list[dict[str, Any]]],
    per_page: int = 3,
    total: int = 15,
) -> dict[int, list[dict[str, Any]]]:
    """Select eligible sentences globally by score, preserving source order on ties."""
    ranked = [
        (int(page_index), sentence)
        for page_index, catalog in catalog_by_page.items()
        for sentence in catalog
        if sentence.get("eligible")
    ]
    ranked.sort(key=lambda item: (-int(item[1].get("complexity_score", 0)), item[0], int(item[1].get("ordinal", 0))))
    selected: dict[int, list[dict[str, Any]]] = {}
    page_counts: dict[int, int] = {}
    for page_index, sentence in ranked:
        if sum(page_counts.values()) >= total:
            break
        if page_counts.get(page_index, 0) >= per_page:
            continue
        sentence["selected_auto"] = True
        selected.setdefault(page_index, []).append(sentence)
        page_counts[page_index] = page_counts.get(page_index, 0) + 1
    for values in selected.values():
        values.sort(key=lambda sentence: int(sentence["ordinal"]))
    return selected


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _object_list(value: Any, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if isinstance(item, dict):
            output.append({field: _string(item.get(field)) for field in fields})
        elif item:
            output.append({fields[0]: _string(item), **{field: "" for field in fields[1:]}})
    return output


def normalize_breakdown(raw: dict[str, Any], requested: dict[str, Any], language: str, origin: str) -> dict[str, Any]:
    """Normalize model output into the backward-compatible V2 breakdown schema."""
    lang = _language(requested.get("detected_language") or language)
    translations = raw.get("translations") if isinstance(raw.get("translations"), dict) else {}
    skeleton = raw.get("sentence_skeleton") if isinstance(raw.get("sentence_skeleton"), dict) else {}
    questions = _object_list(raw.get("questions"), ("question", "answer", "explanation", "evidence"))
    result = {
        "sentence_breakdown_version": BREAKDOWN_VERSION,
        "sentence_id": requested["sentence_id"],
        "ordinal": int(requested.get("ordinal", 0) or 0),
        "original": requested.get("original", ""),
        "detected_language": lang,
        "language_confidence": requested.get("language_confidence", ""),
        "language_source": requested.get("language_source", ""),
        "reading": _string(raw.get("reading")) if lang == "japanese" else "",
        "segments": _object_list(
            raw.get("segments"),
            ("text", "reading", "role", "meaning_vi", "modifies", "base_form", "part_of_speech", "grammar_form", "particle_or_connector", "function_vi"),
        ),
        "clauses": _object_list(
            raw.get("clauses"),
            ("label", "text", "type", "role", "subject", "predicate", "object_or_complement", "connector", "relation_to_main"),
        ),
        "structure_summary": _string(raw.get("structure_summary")),
        "sentence_skeleton": {
            "pattern": _string(skeleton.get("pattern")),
            "topic": _string(skeleton.get("topic")),
            "subject": _string(skeleton.get("subject")),
            "predicate": _string(skeleton.get("predicate")),
            "object_or_complement": _string(skeleton.get("object_or_complement")),
            "adverbial": _string(skeleton.get("adverbial")),
            "tense_aspect": _string(skeleton.get("tense_aspect")),
            "voice_modality": _string(skeleton.get("voice_modality")),
            "polarity": _string(skeleton.get("polarity")),
        },
        "grammar_links": _object_list(raw.get("grammar_links"), ("source", "form", "function_vi", "nuance_vi", "scope")),
        "connectors": _object_list(raw.get("connectors"), ("source", "function_vi", "relation", "scope")),
        "translations": {
            "chunked": _string(translations.get("chunked") or raw.get("chunked_translation")),
            "literal": _string(translations.get("literal") or raw.get("literal_translation")),
            "natural": _string(translations.get("natural") or raw.get("natural_translation")),
        },
        "omitted_elements": _object_list(raw.get("omitted_elements"), ("element", "recovered", "reason")),
        "references": _object_list(raw.get("references"), ("expression", "referent", "reason")),
        "logic": _object_list(raw.get("logic"), ("marker", "relation", "scope", "evidence")),
        "translation_steps": _object_list(raw.get("translation_steps"), ("order", "source_chunk", "meaning_vi", "advice_vi")),
        "ambiguities": _object_list(raw.get("ambiguities"), ("source", "alternatives", "explanation_vi", "confidence")),
        "simplified_source": _string(raw.get("simplified_source")),
        "simplified_vi": _string(raw.get("simplified_vi")),
        "questions": questions,
        "analysis_origin": origin,
        "complexity_score": int(requested.get("complexity_score", 0) or 0),
    }
    missing = assess_breakdown_quality(result, lang)
    result["missing_fields"] = missing
    result["quality_score"] = max(0, 100 - len(missing) * 12)
    result["quality_status"] = "complete" if not missing else "partial"
    return result


def build_sentence_prompt(sentences: list[dict[str, Any]], page_text: str, language: str) -> str:
    lang = _language(language)
    requested = [
        {"sentence_id": item["sentence_id"], "ordinal": item.get("ordinal"), "original": item["original"]}
        for item in sentences
    ]
    language_note = (
        """TIẾNG NHẬT: reading là hiragana toàn câu; cụm có Kanji phải có reading. Nêu dạng từ điển, từ loại, trợ từ và chức năng, dạng chia/thể/phủ định/kính ngữ, chủ đề-chủ ngữ lược bỏ, vị ngữ chính, mệnh đề bổ nghĩa danh từ, danh từ hóa, trích dẫn và quan hệ giữa các vế."""
        if lang == "japanese"
        else """TIẾNG ANH: chỉ rõ S-V-O-C-A, động từ trung tâm, tense/aspect/modal/voice/polarity; phân biệt finite/non-finite, relative/noun/adverbial/reduced clause, participle, infinitive/gerund, phrasal verb, coordination, parallelism, inversion, dummy it/there và antecedent của đại từ."""
    )
    return f"""Bạn là giáo viên {('tiếng Nhật' if lang == 'japanese' else 'tiếng Anh')} chuyên giúp người Việt đọc câu dài.
Phân tích đúng các câu được yêu cầu theo ngữ cảnh. Toàn bộ giải thích và bản dịch đích phải bằng tiếng Việt. Giữ nguyên tuyệt đối original; không sửa OCR. Mọi cụm text phải là đoạn trích nguyên văn, theo đúng thứ tự; nêu rõ khi một nhận định chỉ là suy luận.
{language_note}

Trả về DUY NHẤT một JSON object hợp lệ, không Markdown, dạng:
{{"sentences":[{{
  "sentence_id":"p1-s1", "reading":"",
  "segments":[{{"text":"cụm nguyên văn", "reading":"hiragana hoặc rỗng", "role":"vai trò", "meaning_vi":"nghĩa trong câu", "modifies":"bổ nghĩa cho", "base_form":"dạng gốc", "part_of_speech":"từ loại", "grammar_form":"dạng chia/cấu trúc", "particle_or_connector":"trợ từ/từ nối", "function_vi":"chức năng"}}],
  "clauses":[{{"label":"Mệnh đề 1", "text":"nguyên văn", "type":"chính/phụ/quan hệ/rút gọn...", "role":"vai trò", "subject":"chủ ngữ hiện/ẩn", "predicate":"vị ngữ", "object_or_complement":"bổ ngữ", "connector":"từ nối", "relation_to_main":"quan hệ"}}],
  "structure_summary":"cấu trúc và quan hệ S-V-O-C/mệnh đề",
  "sentence_skeleton":{{"pattern":"khung câu", "topic":"chủ đề", "subject":"chủ ngữ", "predicate":"vị ngữ trung tâm", "object_or_complement":"tân ngữ/bổ ngữ", "adverbial":"trạng ngữ", "tense_aspect":"thì/thể", "voice_modality":"thái/thức", "polarity":"khẳng định/phủ định"}},
  "grammar_links":[{{"source":"đoạn nguyên văn", "form":"cấu trúc", "function_vi":"chức năng", "nuance_vi":"sắc thái", "scope":"phạm vi"}}],
  "connectors":[{{"source":"từ nối", "function_vi":"chức năng", "relation":"quan hệ logic", "scope":"nối phần nào"}}],
  "translations":{{"chunked":"dịch sát theo từng cụm có dấu phân cách", "literal":"dịch sát toàn câu", "natural":"dịch tự nhiên"}},
  "omitted_elements":[{{"element":"", "recovered":"", "reason":""}}],
  "references":[{{"expression":"", "referent":"", "reason":""}}],
  "logic":[{{"marker":"", "relation":"nguyên nhân/đối lập/điều kiện...", "scope":"hai phần được nối", "evidence":"căn cứ trong câu"}}],
  "translation_steps":[{{"order":"1", "source_chunk":"cụm cần xử lý", "meaning_vi":"nghĩa", "advice_vi":"thứ tự ghép khi dịch"}}],
  "ambiguities":[{{"source":"đoạn có thể hiểu nhiều cách", "alternatives":"các cách hiểu", "explanation_vi":"lý do", "confidence":"cao/trung bình/thấp"}}],
  "simplified_source":"viết lại đơn giản nhưng giữ nghĩa", "simplified_vi":"bản dịch tiếng Việt của câu đơn giản",
  "questions":[{{"question":"câu hỏi kiểm tra hiểu", "answer":"đáp án", "explanation":"giải thích", "evidence":"chi tiết chứng minh"}}]
}}]}}

Không bỏ qua trường nào; dùng [] hoặc "" chỉ khi thực sự không áp dụng. Có ít nhất một sentence_skeleton, một grammar_links, một translation_steps và một questions. Phân tích đủ 8 lớp, không chỉ dịch.

CÂU CẦN PHÂN TÍCH:
{json.dumps(requested, ensure_ascii=False, indent=2)}

NGỮ CẢNH TRANG:
{_context_for_sentences(page_text, sentences)}
"""


def _context_for_sentences(page_text: str, sentences: list[dict[str, Any]], max_chars: int = 2200) -> str:
    text = str(page_text or "")
    if len(text) <= max_chars:
        return text
    contexts = []
    for sentence in sentences:
        original = str(sentence.get("original") or "")
        position = text.find(original)
        if position < 0:
            continue
        start = max(0, position - 450)
        end = min(len(text), position + len(original) + 450)
        contexts.append(text[start:end])
    return "\n---\n".join(contexts)[:max_chars]


def _breakdown_response_schema() -> dict[str, Any]:
    """Small portable schema; prompt carries the detailed semantic contract."""
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sentence_id": {"type": "string"}},
                    "required": ["sentence_id"],
                },
            }
        },
        "required": ["sentences"],
    }


def assess_breakdown_quality(row: dict[str, Any], language: str) -> list[str]:
    """Return user-visible missing fields without discarding useful partial output."""
    missing = []
    translations = row.get("translations") or {}
    for key, label in (("literal", "dịch sát"), ("natural", "dịch tự nhiên")):
        if not str(translations.get(key) or "").strip():
            missing.append(label)
    if not row.get("segments"):
        missing.append("cụm từ")
    if not str(row.get("structure_summary") or "").strip():
        missing.append("tóm tắt cấu trúc")
    skeleton = row.get("sentence_skeleton") or {}
    if not str(skeleton.get("pattern") or "").strip() or not str(skeleton.get("predicate") or "").strip():
        missing.append("khung câu/vị ngữ trung tâm")
    if not row.get("grammar_links"):
        missing.append("chuỗi ngữ pháp")
    if not row.get("translation_steps"):
        missing.append("cách tháo câu")
    if not row.get("questions"):
        missing.append("câu hỏi hiểu bài")
    if _language(language) == "japanese" and not str(row.get("reading") or "").strip():
        missing.append("hiragana toàn câu")
    source = str(row.get("original") or "")
    cursor = 0
    for segment in row.get("segments") or []:
        segment_text = str(segment.get("text") or "")
        index = source.find(segment_text, cursor) if segment_text else -1
        if index < 0:
            missing.append("cụm từ không khớp OCR")
            break
        cursor = index + len(segment_text)
    return missing


def _repair_prompt(rows: list[dict[str, Any]], language: str) -> str:
    requested = [
        {"sentence_id": row["sentence_id"], "original": row["original"], "missing_fields": row.get("missing_fields", [])}
        for row in rows if row.get("missing_fields")
    ]
    return f"""Bạn đang hoàn thiện phân tích câu dài {'tiếng Nhật' if _language(language) == 'japanese' else 'tiếng Anh'} cho người Việt.
Chỉ bổ sung các trường thiếu dưới đây. Giữ nguyên sentence_id và original; trả về JSON object {{\"sentences\":[...]}} không Markdown. Mọi nội dung giải thích dùng tiếng Việt. Cụm text phải trích đúng OCR.
{json.dumps(requested, ensure_ascii=False, indent=2)}"""


def _merge_raw_breakdown(primary: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    """Merge a sparse repair without deleting useful fields from the first answer."""
    result = copy.deepcopy(primary or {})
    for key, value in (repair or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        elif value not in (None, "", [], {}):
            result[key] = value
    return result


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini không trả về JSON hợp lệ.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Phản hồi giải mã câu dài phải là JSON object.")
    return value


def analyze_sentence_batch(
    model: Any,
    sentences: list[dict[str, Any]],
    page_text: str,
    language: str,
    reasoning_effort: str = "standard",
    origin: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Analyze one language-homogeneous batch and repair important omissions once."""
    requested = list(sentences[:3])
    if not requested:
        return [], dict(ZERO_USAGE)
    detected = {_language(row.get("detected_language") or language) for row in requested}
    if len(detected) != 1:
        raise ValueError("Một batch giải mã câu dài chỉ được chứa một ngôn ngữ.")
    language = detected.pop()
    prompt = build_sentence_prompt(requested, page_text, language)
    config: dict[str, Any] = {
        "temperature": 0.1,
        "max_output_tokens": 16384 if len(requested) == 1 else 12288,
        "response_mime_type": "application/json",
        "response_json_schema": _breakdown_response_schema(),
    }
    if reasoning_effort == "deep":
        config["thinking_config"] = {"thinking_budget": 4096}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = _generate_structured(model, prompt, config)
            payload = _parse_json_response(getattr(response, "text", ""))
            rows = payload.get("sentences")
            if not isinstance(rows, list):
                raise ValueError("Phản hồi thiếu danh sách sentences.")
            by_id = {str(row.get("sentence_id")): row for row in rows if isinstance(row, dict)}
            normalized = [
                normalize_breakdown(by_id.get(item["sentence_id"], {}), item, language, origin)
                for item in requested
            ]
            primary_usage = response_usage(response)
            repair_usage = dict(ZERO_USAGE)
            usage = primary_usage
            missing = [row for row in normalized if row.get("missing_fields")]
            if missing:
                try:
                    repair_response = _generate_structured(
                        model,
                        _repair_prompt(missing, language),
                        {**config, "max_output_tokens": 8192},
                    )
                    repair_rows = _parse_json_response(getattr(repair_response, "text", "")).get("sentences")
                    if isinstance(repair_rows, list):
                        repair_by_id = {str(row.get("sentence_id")): row for row in repair_rows if isinstance(row, dict)}
                        normalized = [
                            normalize_breakdown(
                                _merge_raw_breakdown(
                                    by_id.get(item["sentence_id"], {}) or {},
                                    repair_by_id.get(item["sentence_id"], {}) or {},
                                ),
                                item,
                                language,
                                origin,
                            )
                            for item in requested
                        ]
                    repair_usage = response_usage(repair_response)
                    usage = merge_usage(primary_usage, repair_usage)
                except Exception as repair_error:
                    for row in missing:
                        row["quality_repair_error"] = str(repair_error)
            for row in normalized:
                row["analysis_usage_detail"] = {
                    "primary": merge_usage(primary_usage),
                    "repair": merge_usage(repair_usage),
                }
            return normalized, usage
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Giải mã câu dài thất bại sau 3 lần thử: {last_error}") from last_error


def _generate_structured(model: Any, prompt: str, config: dict[str, Any]) -> Any:
    """Use JSON schema when supported, then fall back safely for older models."""
    candidates = [config]
    without_schema = {key: value for key, value in config.items() if key != "response_json_schema"}
    candidates.append(without_schema)
    if "thinking_config" in without_schema:
        candidates.append({key: value for key, value in without_schema.items() if key != "thinking_config"})
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return model.generate_content(prompt, generation_config=candidate)
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("Không thể gọi model phân tích câu dài.")


def analyze_manual_sentence(
    sentence: dict[str, Any],
    page_text: str,
    language: str,
    model_name: str | None = None,
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    from modules.text_analyzer import _init_model

    model = _init_model(model_name) if model_name else _init_model()
    sentence_language = _language(sentence.get("detected_language") or language)
    rows, usage = analyze_sentence_batch(
        model, [sentence], page_text, sentence_language, reasoning_effort=reasoning_effort, origin="manual"
    )
    return {
        "job_kind": "sentence_deep_dive",
        "page_index": int(sentence["sentence_id"].split("-")[0][1:]),
        "sentence_id": sentence["sentence_id"],
        "breakdown": rows[0],
        "usage": usage,
        "model_used": getattr(model, "target_model_name", model_name or "gemini-3.5-flash"),
    }


def attach_sentence_data(
    page_analysis: dict[str, Any],
    catalog: list[dict[str, Any]],
    breakdowns: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    error: str | None = None,
    model_used: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(page_analysis)
    result["sentence_catalog"] = copy.deepcopy(catalog)
    result["sentence_breakdowns"] = sorted(
        copy.deepcopy(breakdowns or []), key=lambda item: int(item.get("ordinal", 0))
    )
    result["sentence_analysis_usage"] = merge_usage(usage)
    result["sentence_analysis_model"] = model_used or result.get("sentence_analysis_model")
    runs = copy.deepcopy(result.get("sentence_analysis_runs") or [])
    if usage and sum(merge_usage(usage).values()) > 0:
        runs = [run for run in runs if run.get("run_id") != "auto"]
        runs.append(
            {
                "run_id": "auto",
                "origin": "auto",
                "model_used": model_used,
                "usage": merge_usage(usage),
            }
        )
    result["sentence_analysis_runs"] = runs
    result["sentence_analysis_error"] = error
    analyzed = {item.get("sentence_id"): item.get("analysis_origin") for item in result["sentence_breakdowns"]}
    for item in result["sentence_catalog"]:
        if item["sentence_id"] in analyzed:
            item["analyzed"] = True
            item["analysis_origin"] = analyzed[item["sentence_id"]]
    return result


def aggregate_sentence_usage(pages: list[dict[str, Any]]) -> dict[str, int]:
    return merge_usage(*(page.get("sentence_analysis_usage") for page in pages))


def aggregate_sentence_runs(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [copy.deepcopy(run) for page in pages for run in page.get("sentence_analysis_runs", [])]


def merge_manual_breakdown(
    analysis: dict[str, Any] | None,
    envelope: dict[str, Any],
    job_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Idempotently merge one completed manual job into its owning page."""
    if not analysis or envelope.get("job_kind") != "sentence_deep_dive":
        return analysis, False
    applied = set(analysis.get("applied_sentence_job_ids") or [])
    if job_id in applied:
        return analysis, False
    updated = copy.deepcopy(analysis)
    page_index = int(envelope.get("page_index", 0) or 0)
    pages = updated.get("page_analyses") or [updated]
    target = next((page for page in pages if int(page.get("page_index", 0) or 0) == page_index), None)
    if target is None:
        return analysis, False
    sentence_id = str(envelope.get("sentence_id") or "")
    breakdowns = [item for item in target.get("sentence_breakdowns", []) if item.get("sentence_id") != sentence_id]
    breakdowns.append(copy.deepcopy(envelope.get("breakdown") or {}))
    target["sentence_breakdowns"] = sorted(breakdowns, key=lambda item: int(item.get("ordinal", 0) or 0))
    target["sentence_analysis_usage"] = merge_usage(target.get("sentence_analysis_usage"), envelope.get("usage"))
    target["sentence_analysis_model"] = envelope.get("model_used") or target.get("sentence_analysis_model")
    target.setdefault("sentence_analysis_runs", []).append(
        {
            "run_id": job_id,
            "origin": "manual",
            "model_used": envelope.get("model_used"),
            "usage": merge_usage(envelope.get("usage")),
        }
    )
    for item in target.get("sentence_catalog", []):
        if item.get("sentence_id") == sentence_id:
            item["analyzed"] = True
            item["analysis_origin"] = "manual"
    updated["sentence_analysis_usage"] = aggregate_sentence_usage(pages)
    updated["sentence_analysis_model"] = envelope.get("model_used") or updated.get("sentence_analysis_model")
    updated["sentence_analysis_runs"] = aggregate_sentence_runs(pages)
    updated["applied_sentence_job_ids"] = sorted(applied | {job_id})
    return updated, True


def sentence_breakdowns_markdown(page: dict[str, Any]) -> str:
    rows = sorted(page.get("sentence_breakdowns") or [], key=lambda item: int(item.get("ordinal", 0) or 0))
    if not rows:
        return ""
    lines = ["## Giải mã câu dài"]
    for row in rows:
        origin = "Tự động" if row.get("analysis_origin") == "auto" else "Phân tích thêm"
        lines.extend([
            "",
            f"### Câu {row.get('ordinal', '?')} - {origin}",
            f"**Nguyên văn:** {row.get('original', '')}",
        ])
        if row.get("detected_language"):
            lines.append(f"**Ngôn ngữ:** {'Tiếng Nhật' if row['detected_language'] == 'japanese' else 'Tiếng Anh'}")
        if row.get("quality_status") == "partial":
            lines.append(f"**Cần bổ sung:** {', '.join(row.get('missing_fields') or [])}")
        if row.get("reading"):
            lines.append(f"**Hiragana:** {row['reading']}")
        skeleton = row.get("sentence_skeleton") or {}
        if any(skeleton.values()):
            lines.extend(["", "**Khung câu trung tâm:**"])
            for label, key in (
                ("Mẫu", "pattern"), ("Chủ đề", "topic"), ("Chủ ngữ", "subject"),
                ("Vị ngữ", "predicate"), ("Tân ngữ/Bổ ngữ", "object_or_complement"),
                ("Trạng ngữ", "adverbial"), ("Thì/Thể", "tense_aspect"),
                ("Thái/Thức", "voice_modality"), ("Khẳng định/Phủ định", "polarity"),
            ):
                if skeleton.get(key):
                    lines.append(f"- **{label}:** {skeleton[key]}")
        segments = row.get("segments") or []
        if segments:
            lines.extend(["", "**Cụm từ và vai trò:**"])
            for segment in segments:
                reading = f" ({segment.get('reading')})" if segment.get("reading") else ""
                modifies = f"; bổ nghĩa: {segment.get('modifies')}" if segment.get("modifies") else ""
                details = "; ".join(
                    value for value in (
                        segment.get("base_form") and f"dạng gốc: {segment['base_form']}",
                        segment.get("grammar_form") and f"ngữ pháp: {segment['grammar_form']}",
                        segment.get("particle_or_connector") and f"trợ từ/từ nối: {segment['particle_or_connector']}",
                        segment.get("function_vi") and f"chức năng: {segment['function_vi']}",
                    ) if value
                )
                suffix = f"; {details}" if details else ""
                lines.append(f"- `{segment.get('text', '')}`{reading} [{segment.get('role', '')}]: {segment.get('meaning_vi', '')}{modifies}{suffix}")
        clauses = row.get("clauses") or []
        if clauses:
            lines.extend(["", "**Mệnh đề:**"])
            for clause in clauses:
                lines.append(f"- {clause.get('label', '')}: {clause.get('text', '')} - {clause.get('role', '')}; {clause.get('relation_to_main', '')}")
        if row.get("structure_summary"):
            lines.extend(["", f"**Cấu trúc:** {row['structure_summary']}"])
        for title, key, formatter in (
            ("Chuỗi ngữ pháp", "grammar_links", lambda x: f"{x.get('source', '')} [{x.get('form', '')}]: {x.get('function_vi', '')}; {x.get('nuance_vi', '')}; phạm vi: {x.get('scope', '')}"),
            ("Từ nối", "connectors", lambda x: f"{x.get('source', '')}: {x.get('function_vi', '')}; {x.get('relation', '')}; phạm vi: {x.get('scope', '')}"),
        ):
            if row.get(key):
                lines.extend(["", f"**{title}:**"])
                lines.extend(f"- {formatter(item)}" for item in row[key])
        translations = row.get("translations") or {}
        lines.extend([
            "",
            f"**Dịch theo cụm:** {translations.get('chunked', '')}",
            f"**Dịch sát:** {translations.get('literal', '')}",
            f"**Dịch tự nhiên:** {translations.get('natural', '')}",
        ])
        for title, key, formatter in (
            ("Thành phần lược bỏ", "omitted_elements", lambda x: f"{x.get('element', '')} → {x.get('recovered', '')}: {x.get('reason', '')}"),
            ("Từ quy chiếu", "references", lambda x: f"{x.get('expression', '')} → {x.get('referent', '')}: {x.get('reason', '')}"),
            ("Luồng logic", "logic", lambda x: f"{x.get('marker', '')} [{x.get('relation', '')}]: {x.get('scope', '')}"),
        ):
            if row.get(key):
                lines.extend(["", f"**{title}:**"])
                lines.extend(f"- {formatter(item)}" for item in row[key])
        if row.get("translation_steps"):
            lines.extend(["", "**Cách tháo câu từng bước:**"])
            for index, step in enumerate(row["translation_steps"], 1):
                lines.append(f"{step.get('order') or index}. `{step.get('source_chunk', '')}` → {step.get('meaning_vi', '')}: {step.get('advice_vi', '')}")
        if row.get("ambiguities"):
            lines.extend(["", "**Điểm dễ hiểu sai:**"])
            lines.extend(
                f"- `{item.get('source', '')}`: {item.get('alternatives', '')} - {item.get('explanation_vi', '')} ({item.get('confidence', '')})"
                for item in row["ambiguities"]
            )
        lines.extend([
            "",
            f"**Câu viết lại đơn giản:** {row.get('simplified_source', '')}",
            f"**Nghĩa tiếng Việt:** {row.get('simplified_vi', '')}",
        ])
        if row.get("questions"):
            lines.extend(["", "**Câu hỏi kiểm tra hiểu:**"])
            for question in row["questions"]:
                lines.append(f"- {question.get('question', '')}  ")
                lines.append(f"  Đáp án: {question.get('answer', '')}. {question.get('explanation', '')}")
    return "\n".join(lines).strip()


def analysis_markdown(analysis: dict[str, Any]) -> str:
    """Render exports dynamically so deep-dive sections are never duplicated."""
    from modules.translation_guidance import guidance_markdown

    pages = analysis.get("page_analyses")
    if not pages:
        base = str(analysis.get("full_markdown") or "").strip()
        guidance = guidance_markdown(analysis)
        deep = "" if guidance else sentence_breakdowns_markdown(analysis)
        return "\n\n".join(part for part in (base, guidance, deep) if part)
    rendered = []
    for page in pages:
        label = page.get("source_label") or page.get("page_name") or "Trang"
        base = str(page.get("full_markdown") or "").strip()
        guidance = guidance_markdown(page)
        deep = "" if guidance else sentence_breakdowns_markdown(page)
        content = "\n\n".join(part for part in (base, guidance, deep) if part)
        rendered.append(f"# {label}\n\n{content}".strip())
    return "\n\n---\n\n".join(rendered)
