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
    r"(?:ので|のに|ながら|けれども|けれど|が|ため(?:に)?|ところ|ものの|にもかかわらず)",
    r"(?:なら|たら|れば|と)(?:、|\s)",
    r"(?:しかし|そして|それで|そのため|一方で|つまり|したがって|ところが|また|なお)",
    r"(?:こと|もの|という|よう)(?:を|が|は|に|で|だ|です|になる)",
)
_EN_CLAUSE_RE = re.compile(
    r"\b(?:although|though|even though|because|since|while|whereas|if|unless|when|whenever|"
    r"before|after|until|once|so that|in order that|which|who|whom|whose|that|where|however|"
    r"therefore|moreover|nevertheless|yet|but|and|or|nor)\b",
    re.IGNORECASE,
)


def _language(value: str | None) -> str:
    return "japanese" if value == "japanese" else "english"


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
    lang = _language(language)
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
            if lang == "japanese" and char in "。！？":
                boundary = True
            elif lang == "japanese" and char in ".．":
                boundary = _is_english_period_boundary(source, index)
            elif lang == "english" and char in "?!":
                boundary = True
            elif lang == "english" and char == ".":
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
        score, signals = score_complexity(original, lang)
        catalog.append(
            {
                "sentence_id": f"p{int(page_index)}-s{ordinal}",
                "ordinal": ordinal,
                "original": original,
                "complexity_score": score,
                "complexity_signals": signals,
                "eligible": is_complex_sentence(original, lang, score),
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
        clause_points = min(6, markers * 2)
        noun_modifier = bool(re.search(r"(?:た|ている|ない|る|れる|られる|という)[^、。]{1,18}(?:こと|もの|人|時|場合|点|方法|理由)", sentence))
        condition = bool(re.search(r"(?:なら|たら|れば|ても|としても|にもかかわらず|ものの)", sentence))
        nested = any(opener in sentence for opener in _JA_OPEN) or sentence.count("（") > 0
        score = length_points + comma_points + clause_points + int(noun_modifier) * 2 + int(condition) * 2 + int(nested) * 2
        if len(compact) >= 35:
            signals.append(f"dài {len(compact)} ký tự")
        if comma_count:
            signals.append(f"{comma_count} dấu phẩy")
        if markers:
            signals.append(f"{markers} dấu hiệu mệnh đề/từ nối")
        if noun_modifier:
            signals.append("bổ nghĩa danh từ")
        if condition:
            signals.append("điều kiện/nhượng bộ")
        if nested:
            signals.append("cấu trúc lồng")
        return score, signals

    words = re.findall(r"\b[\w'-]+\b", sentence)
    punctuation = len(re.findall(r"[,;:]", sentence))
    markers = len(_EN_CLAUSE_RE.findall(sentence))
    participle = bool(re.search(r"(?:^|[,;]\s+)(?:having|being|using|given|considering|despite)\b|\b\w+ing\s*,", sentence, re.I))
    parenthetical = bool(re.search(r"\([^)]{3,}\)|—[^—]+—", sentence))
    score = min(4, len(words) // 12) + min(3, punctuation) + min(6, markers * 2) + int(participle) * 2 + int(parenthetical) * 2
    if len(words) >= 20:
        signals.append(f"dài {len(words)} từ")
    if punctuation:
        signals.append(f"{punctuation} dấu ngắt")
    if markers:
        signals.append(f"{markers} mệnh đề/liên từ")
    if participle:
        signals.append("cụm phân từ")
    if parenthetical:
        signals.append("phần chen giữa")
    return score, signals


def is_complex_sentence(sentence: str, language: str, score: int | None = None) -> bool:
    lang = _language(language)
    value = score if score is not None else score_complexity(sentence, lang)[0]
    if lang == "japanese":
        marker_count = sum(len(re.findall(pattern, sentence)) for pattern in _JA_CLAUSE_PATTERNS)
        return value >= 5 and (len(re.sub(r"\s", "", sentence)) >= 35 or marker_count >= 2)
    words = re.findall(r"\b[\w'-]+\b", sentence)
    return value >= 5 and (len(words) >= 20 or len(_EN_CLAUSE_RE.findall(sentence)) >= 2)


def build_sentence_catalog(pages: list[dict[str, Any]], language: str) -> dict[int, list[dict[str, Any]]]:
    return {
        int(page["page_index"]): split_sentences(page.get("text", ""), language, int(page["page_index"]))
        for page in pages
    }


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
    """Normalize incomplete model output into the stable eight-layer schema."""
    translations = raw.get("translations") if isinstance(raw.get("translations"), dict) else {}
    questions = _object_list(raw.get("questions"), ("question", "answer", "explanation"))
    result = {
        "sentence_id": requested["sentence_id"],
        "ordinal": int(requested.get("ordinal", 0) or 0),
        "original": requested.get("original", ""),
        "reading": _string(raw.get("reading")) if _language(language) == "japanese" else "",
        "segments": _object_list(raw.get("segments"), ("text", "reading", "role", "meaning_vi", "modifies")),
        "clauses": _object_list(raw.get("clauses"), ("label", "text", "role", "relation_to_main")),
        "structure_summary": _string(raw.get("structure_summary")),
        "translations": {
            "chunked": _string(translations.get("chunked") or raw.get("chunked_translation")),
            "literal": _string(translations.get("literal") or raw.get("literal_translation")),
            "natural": _string(translations.get("natural") or raw.get("natural_translation")),
        },
        "omitted_elements": _object_list(raw.get("omitted_elements"), ("element", "recovered", "reason")),
        "references": _object_list(raw.get("references"), ("expression", "referent", "reason")),
        "logic": _object_list(raw.get("logic"), ("marker", "relation", "scope")),
        "simplified_source": _string(raw.get("simplified_source")),
        "simplified_vi": _string(raw.get("simplified_vi")),
        "questions": questions,
        "analysis_origin": origin,
        "complexity_score": int(requested.get("complexity_score", 0) or 0),
    }
    return result


def build_sentence_prompt(sentences: list[dict[str, Any]], page_text: str, language: str) -> str:
    lang = _language(language)
    requested = [
        {"sentence_id": item["sentence_id"], "ordinal": item.get("ordinal"), "original": item["original"]}
        for item in sentences
    ]
    language_note = (
        "Với tiếng Nhật, reading phải là hiragana của toàn câu và mỗi segment có reading."
        if lang == "japanese"
        else "Với tiếng Anh, dùng segments/clauses để chỉ rõ S, V, O, C, modifiers và ranh giới mệnh đề."
    )
    return f"""Bạn là giáo viên {('tiếng Nhật' if lang == 'japanese' else 'tiếng Anh')} chuyên giúp người Việt đọc câu dài.
Phân tích đúng các câu được yêu cầu theo ngữ cảnh. Toàn bộ giải thích và bản dịch đích phải bằng tiếng Việt.
{language_note}

Trả về DUY NHẤT một JSON object hợp lệ, không Markdown, dạng:
{{"sentences":[{{
  "sentence_id":"p1-s1", "reading":"", 
  "segments":[{{"text":"", "reading":"", "role":"S/V/O/C/bổ ngữ/từ nối...", "meaning_vi":"", "modifies":""}}],
  "clauses":[{{"label":"Mệnh đề chính/phụ...", "text":"", "role":"", "relation_to_main":""}}],
  "structure_summary":"cấu trúc và quan hệ S-V-O-C/mệnh đề",
  "translations":{{"chunked":"dịch sát theo từng cụm có dấu phân cách", "literal":"dịch sát toàn câu", "natural":"dịch tự nhiên"}},
  "omitted_elements":[{{"element":"", "recovered":"", "reason":""}}],
  "references":[{{"expression":"", "referent":"", "reason":""}}],
  "logic":[{{"marker":"", "relation":"nguyên nhân/đối lập/điều kiện...", "scope":"hai phần được nối"}}],
  "simplified_source":"viết lại đơn giản nhưng giữ nghĩa", "simplified_vi":"bản dịch tiếng Việt của câu đơn giản",
  "questions":[{{"question":"câu hỏi kiểm tra hiểu", "answer":"đáp án", "explanation":"giải thích"}}]
}}]}}

Không bỏ qua trường nào; dùng [] hoặc "" nếu thực sự không áp dụng. Phân tích đủ 8 lớp, không chỉ dịch.

CÂU CẦN PHÂN TÍCH:
{json.dumps(requested, ensure_ascii=False, indent=2)}

NGỮ CẢNH TRANG:
{_context_for_sentences(page_text, sentences)}
"""


def _context_for_sentences(page_text: str, sentences: list[dict[str, Any]], max_chars: int = 6000) -> str:
    text = str(page_text or "")
    if len(text) <= max_chars:
        return text
    positions = [text.find(str(sentence.get("original") or "")) for sentence in sentences]
    valid = [position for position in positions if position >= 0]
    center = min(valid) if valid else 0
    start = max(0, center - max_chars // 3)
    return text[start : start + max_chars]


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
    """Analyze up to three sentences in one Gemini request."""
    requested = list(sentences[:3])
    if not requested:
        return [], dict(ZERO_USAGE)
    prompt = build_sentence_prompt(requested, page_text, language)
    config: dict[str, Any] = {
        "temperature": 0.1,
        "max_output_tokens": 12288,
        "response_mime_type": "application/json",
    }
    if reasoning_effort == "deep":
        config["thinking_config"] = {"thinking_budget": 4096}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            try:
                response = model.generate_content(prompt, generation_config=config)
            except Exception:
                if "thinking_config" not in config:
                    raise
                response = model.generate_content(
                    prompt,
                    generation_config={key: value for key, value in config.items() if key != "thinking_config"},
                )
            payload = _parse_json_response(getattr(response, "text", ""))
            rows = payload.get("sentences")
            if not isinstance(rows, list):
                raise ValueError("Phản hồi thiếu danh sách sentences.")
            by_id = {str(row.get("sentence_id")): row for row in rows if isinstance(row, dict)}
            normalized = [
                normalize_breakdown(by_id.get(item["sentence_id"], {}), item, language, origin)
                for item in requested
            ]
            return normalized, response_usage(response)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Giải mã câu dài thất bại sau 3 lần thử: {last_error}") from last_error


def analyze_manual_sentence(
    sentence: dict[str, Any],
    page_text: str,
    language: str,
    model_name: str | None = None,
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    from modules.text_analyzer import _init_model

    model = _init_model(model_name) if model_name else _init_model()
    rows, usage = analyze_sentence_batch(
        model, [sentence], page_text, language, reasoning_effort=reasoning_effort, origin="manual"
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
        if row.get("reading"):
            lines.append(f"**Hiragana:** {row['reading']}")
        segments = row.get("segments") or []
        if segments:
            lines.extend(["", "**Cụm từ và vai trò:**"])
            for segment in segments:
                reading = f" ({segment.get('reading')})" if segment.get("reading") else ""
                modifies = f"; bổ nghĩa: {segment.get('modifies')}" if segment.get("modifies") else ""
                lines.append(f"- `{segment.get('text', '')}`{reading} [{segment.get('role', '')}]: {segment.get('meaning_vi', '')}{modifies}")
        clauses = row.get("clauses") or []
        if clauses:
            lines.extend(["", "**Mệnh đề:**"])
            for clause in clauses:
                lines.append(f"- {clause.get('label', '')}: {clause.get('text', '')} - {clause.get('role', '')}; {clause.get('relation_to_main', '')}")
        if row.get("structure_summary"):
            lines.extend(["", f"**Cấu trúc:** {row['structure_summary']}"])
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
