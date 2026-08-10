"""Teacher-style sentence translation guidance and safe result merging."""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Callable, Iterator

from modules.sentence_analyzer import ZERO_USAGE, merge_usage, response_usage


MAX_BATCH_SENTENCES = 8
MAX_BATCH_CHARS = 5000


def _language(value: str | None) -> str:
    return "japanese" if value == "japanese" else "english"


def _sentence_language(sentence: dict[str, Any], fallback: str) -> str:
    return _language(sentence.get("detected_language") or fallback)


def guidance_batches(
    catalog: list[dict[str, Any]],
    max_sentences: int = MAX_BATCH_SENTENCES,
    max_chars: int = MAX_BATCH_CHARS,
) -> Iterator[list[dict[str, Any]]]:
    """Yield source-ordered batches bounded by sentence count and source size."""
    batch: list[dict[str, Any]] = []
    char_count = 0
    for sentence in sorted(catalog, key=lambda row: int(row.get("ordinal", 0) or 0)):
        length = len(str(sentence.get("original") or ""))
        if batch and (len(batch) >= max_sentences or char_count + length > max_chars):
            yield batch
            batch = []
            char_count = 0
        batch.append(sentence)
        char_count += length
    if batch:
        yield batch


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _object_list(value: Any, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, dict):
            rows.append({field: _string(item.get(field)) for field in fields})
        elif item:
            rows.append({fields[0]: _string(item), **{field: "" for field in fields[1:]}})
    return rows


def normalize_guidance(
    raw: dict[str, Any],
    requested: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    """Normalize partial Gemini JSON while preserving the exact OCR sentence."""
    lang = _sentence_language(requested, language)
    translations = raw.get("translations") if isinstance(raw.get("translations"), dict) else {}
    key_points = _object_list(raw.get("key_points"), ("label", "source", "explanation_vi"))[:3]
    return {
        "sentence_id": requested["sentence_id"],
        "ordinal": int(requested.get("ordinal", 0) or 0),
        "original": _string(requested.get("original")),
        "detected_language": lang,
        "language_confidence": _string(requested.get("language_confidence")),
        "language_source": _string(requested.get("language_source")),
        "reading": _string(raw.get("reading")) if lang == "japanese" else "",
        "translations": {
            "chunked": _string(translations.get("chunked") or raw.get("chunked_translation")),
            "literal": _string(translations.get("literal") or raw.get("literal_translation")),
            "natural": _string(translations.get("natural") or raw.get("natural_translation")),
        },
        "translation_steps": _object_list(
            raw.get("translation_steps"),
            ("order", "source_chunk", "meaning_vi", "advice_vi"),
        ),
        "key_points": key_points,
        "ocr_warning": _string(raw.get("ocr_warning")),
        "related_analysis": [],
    }


def _page_context(text: str, sentences: list[dict[str, Any]], max_chars: int = 2200) -> str:
    source = str(text or "")
    if len(source) <= max_chars:
        return source
    contexts = []
    for row in sentences:
        original = str(row.get("original") or "")
        position = source.find(original)
        if position >= 0:
            contexts.append(source[max(0, position - 450) : min(len(source), position + len(original) + 450)])
    return "\n---\n".join(contexts)[:max_chars]


def build_guidance_prompt(
    sentences: list[dict[str, Any]],
    page_text: str,
    language: str,
) -> str:
    lang = _language(language)
    requested = [
        {
            "sentence_id": row["sentence_id"],
            "ordinal": row.get("ordinal"),
            "original": row.get("original", ""),
        }
        for row in sentences
    ]
    reading_rule = (
        "reading phải là hiragana đầy đủ của nguyên văn."
        if lang == "japanese"
        else "reading phải để chuỗi rỗng."
    )
    return f"""Bạn là giáo viên {('tiếng Nhật' if lang == 'japanese' else 'tiếng Anh')} hướng dẫn người Việt dịch nhanh và chính xác.
Phân tích TẤT CẢ câu được yêu cầu theo đúng thứ tự. Toàn bộ lời giải thích và bản dịch đích phải bằng tiếng Việt.
Giữ nguyên tuyệt đối trường original. Không sửa hoặc thay câu OCR. Nếu nghi OCR sai, chỉ ghi đề xuất trong ocr_warning.
{reading_rule}

Trả về DUY NHẤT JSON object hợp lệ, không Markdown:
{{"sentences":[{{
  "sentence_id":"p1-s1",
  "reading":"",
  "translations":{{
    "chunked":"dịch theo cụm, dùng dấu / để đối chiếu",
    "literal":"dịch sát toàn câu",
    "natural":"bản dịch tiếng Việt tự nhiên"
  }},
  "translation_steps":[{{
    "order":"1", "source_chunk":"cụm nguồn", "meaning_vi":"nghĩa cụm", "advice_vi":"cách ghép khi dịch"
  }}],
  "key_points":[{{
    "label":"Chủ ngữ ẩn/Từ nối/Cấu trúc...", "source":"phần nguyên văn", "explanation_vi":"gợi ý ngắn"
  }}],
  "ocr_warning":"để trống nếu không có nghi vấn"
}}]}}

Mỗi câu cần 1-3 key_points hữu ích nhất. Dịch sát và dịch tự nhiên không được bỏ trống.

CÂU OCR CẦN HƯỚNG DẪN:
{json.dumps(requested, ensure_ascii=False, indent=2)}

NGỮ CẢNH TRANG:
{_page_context(page_text, sentences)}
"""


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini không trả về JSON hướng dẫn hợp lệ.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Phản hồi hướng dẫn phải là JSON object.")
    return value


def analyze_guidance_batch(
    model: Any,
    sentences: list[dict[str, Any]],
    page_text: str,
    language: str,
    reasoning_effort: str = "standard",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate compact teacher guidance for one bounded sentence batch."""
    requested = list(sentences)
    if not requested:
        return [], dict(ZERO_USAGE)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sentence in requested:
        grouped.setdefault(_sentence_language(sentence, language), []).append(sentence)
    if len(grouped) > 1:
        rows: list[dict[str, Any]] = []
        usages: list[dict[str, int]] = []
        for detected_language, subset in grouped.items():
            subset_rows, subset_usage = analyze_guidance_batch(
                model, subset, page_text, detected_language, reasoning_effort=reasoning_effort
            )
            rows.extend(subset_rows)
            usages.append(subset_usage)
        return sorted(rows, key=lambda row: int(row.get("ordinal", 0) or 0)), merge_usage(*usages)
    language = next(iter(grouped), _language(language))
    prompt = build_guidance_prompt(requested, page_text, language)
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
            payload = _parse_json(getattr(response, "text", ""))
            rows = payload.get("sentences")
            if not isinstance(rows, list):
                raise ValueError("Phản hồi thiếu danh sách sentences.")
            by_id = {str(row.get("sentence_id")): row for row in rows if isinstance(row, dict)}
            normalized = [
                normalize_guidance(by_id.get(row["sentence_id"], {}), row, language)
                for row in requested
            ]
            return normalized, response_usage(response)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Hướng dẫn dịch thất bại sau 3 lần thử: {last_error}") from last_error


def _clean_example(value: Any) -> str:
    text = str(value or "").strip().strip("`\"'“”")
    text = re.sub(r"^(?:Example from text|Ví dụ trong bài)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip("`\"'“”")


def _contains_term(sentence: str, term: str, language: str) -> bool:
    source = str(sentence or "")
    target = str(term or "").strip()
    if not target or target in {"—", "--", "N/A"}:
        return False
    if _language(language) == "japanese":
        return target in source
    return bool(re.search(rf"(?<![\w'-]){re.escape(target)}(?![\w'-])", source, re.I))


def related_analysis_for_sentence(
    sentence: str,
    page: dict[str, Any],
    language: str,
) -> list[dict[str, Any]]:
    """Link existing page analysis without fuzzy substring false positives."""
    refs: list[dict[str, Any]] = []
    groups = [
        ("vocabulary", "Từ vựng", page.get("vocabulary_all", []), "word", None),
        ("important_vocabulary", "Từ khó", page.get("vocabulary_important", []), "word", None),
        (
            "connector",
            "Từ nối",
            page.get("connectors", []) if _language(language) == "japanese" else page.get("discourse_markers", []),
            "phrase",
            None,
        ),
        (
            "phrase",
            "Cụm từ",
            page.get("phrasal_collocations", []) if _language(language) == "english" else [],
            "phrase",
            None,
        ),
        ("grammar", "Ngữ pháp", page.get("grammar_points", []), "name", "example"),
        ("pattern", "Mẫu câu", page.get("sentence_patterns", []), "pattern", "example"),
    ]
    seen: set[tuple[str, int]] = set()
    for category, category_label, rows, label_field, example_field in groups:
        for index, row in enumerate(rows):
            label = str(row.get(label_field) or "").strip()
            matched = _contains_term(sentence, label, language) if not example_field else False
            if example_field:
                example = _clean_example(row.get(example_field))
                matched = bool(example and (example in sentence or sentence in example))
            if not matched or (category, index) in seen:
                continue
            seen.add((category, index))
            summary = str(
                row.get("meaning")
                or row.get("vn_meaning")
                or row.get("explanation")
                or row.get("rule")
                or row.get("function")
                or ""
            ).strip()
            refs.append(
                {
                    "category": category,
                    "category_label": category_label,
                    "index": index,
                    "label": label,
                    "summary": summary,
                }
            )
    return refs


def add_related_analysis(
    rows: list[dict[str, Any]],
    page: dict[str, Any],
    language: str,
) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows)
    for row in output:
        row_language = _sentence_language(row, language)
        row["related_analysis"] = related_analysis_for_sentence(row.get("original", ""), page, row_language)
    return output


def apply_guidance_batch(
    page: dict[str, Any],
    rows: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    model_used: str | None,
    run_id: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Idempotently apply one completed or failed guidance batch to a page."""
    updated = copy.deepcopy(page)
    existing = {row.get("sentence_id"): row for row in updated.get("translation_guidance", [])}
    for row in rows:
        existing[row.get("sentence_id")] = copy.deepcopy(row)
    updated["translation_guidance"] = sorted(
        existing.values(), key=lambda row: int(row.get("ordinal", 0) or 0)
    )
    runs = copy.deepcopy(updated.get("translation_guidance_runs") or [])
    if usage and run_id not in {run.get("run_id") for run in runs}:
        runs.append(
            {
                "run_id": run_id,
                "model_used": model_used,
                "usage": merge_usage(usage),
            }
        )
    updated["translation_guidance_runs"] = runs
    updated["translation_guidance_usage"] = merge_usage(*(run.get("usage") for run in runs))
    updated["translation_guidance_model"] = model_used or updated.get("translation_guidance_model")
    errors = [item for item in updated.get("translation_guidance_errors", []) if item.get("run_id") != run_id]
    if error:
        errors.append({"run_id": run_id, "error": error})
    updated["translation_guidance_errors"] = errors
    return updated


def aggregate_guidance_fields(pages: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [copy.deepcopy(run) for page in pages for run in page.get("translation_guidance_runs", [])]
    errors = [
        {"page_index": page.get("page_index"), **copy.deepcopy(error)}
        for page in pages
        for error in page.get("translation_guidance_errors", [])
    ]
    models = [run.get("model_used") for run in runs if run.get("model_used")]
    return {
        "translation_guidance_usage": merge_usage(*(run.get("usage") for run in runs)),
        "translation_guidance_runs": runs,
        "translation_guidance_model": models[0] if models else None,
        "translation_guidance_errors": errors,
    }


def analyze_guidance_job(
    catalog: list[dict[str, Any]],
    page_text: str,
    language: str,
    page_index: int,
    model_name: str | None = None,
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    """Analyze missing legacy guidance in a standalone background job."""
    from modules.text_analyzer import _init_model

    model = _init_model(model_name) if model_name else _init_model()
    results = []
    for batch_index, batch in enumerate(guidance_batches(catalog), 1):
        try:
            rows, usage = analyze_guidance_batch(model, batch, page_text, language, reasoning_effort)
            results.append({"batch_index": batch_index, "rows": rows, "usage": usage, "error": None})
        except Exception as exc:
            results.append({"batch_index": batch_index, "rows": [], "usage": dict(ZERO_USAGE), "error": str(exc)})
    return {
        "job_kind": "translation_guidance",
        "page_index": int(page_index),
        "batch_results": results,
        "model_used": getattr(model, "target_model_name", model_name or "gemini-3.5-flash"),
        "analysis_language": _language(language),
    }


def merge_guidance_job(
    analysis: dict[str, Any] | None,
    envelope: dict[str, Any],
    job_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Merge a legacy/manual page-guidance job once into the current analysis."""
    if not analysis or envelope.get("job_kind") != "translation_guidance":
        return analysis, False
    applied = set(analysis.get("applied_guidance_job_ids") or [])
    if job_id in applied:
        return analysis, False
    updated = copy.deepcopy(analysis)
    pages = updated.get("page_analyses") or [updated]
    page_index = int(envelope.get("page_index", 0) or 0)
    target = next((page for page in pages if int(page.get("page_index", 0) or 0) == page_index), None)
    if target is None:
        return analysis, False
    language = envelope.get("analysis_language", updated.get("analysis_language", "english"))
    for batch in envelope.get("batch_results", []):
        rows = add_related_analysis(batch.get("rows") or [], target, language)
        target = apply_guidance_batch(
            target,
            rows,
            batch.get("usage"),
            envelope.get("model_used"),
            f"{job_id}-b{int(batch.get('batch_index', 0) or 0)}",
            batch.get("error"),
        )
    if updated.get("page_analyses"):
        updated["page_analyses"] = [
            target if int(page.get("page_index", 0) or 0) == page_index else page
            for page in pages
        ]
        pages = updated["page_analyses"]
    else:
        updated = target
        pages = [updated]
    updated.update(aggregate_guidance_fields(pages))
    updated["applied_guidance_job_ids"] = sorted(applied | {job_id})
    return updated, True


def guidance_markdown(page: dict[str, Any]) -> str:
    """Render the unified teacher guidance and deep-dive section for exports."""
    guidance = {row.get("sentence_id"): row for row in page.get("translation_guidance", [])}
    breakdowns = {row.get("sentence_id"): row for row in page.get("sentence_breakdowns", [])}
    catalog = sorted(page.get("sentence_catalog") or [], key=lambda row: int(row.get("ordinal", 0) or 0))
    if not guidance:
        return ""
    lines = ["## Đối chiếu OCR và giáo viên hướng dẫn dịch"]
    for sentence in catalog:
        row = guidance.get(sentence.get("sentence_id"))
        deep = breakdowns.get(sentence.get("sentence_id"))
        if not row and not deep:
            continue
        row = row or {
            "sentence_id": sentence.get("sentence_id"),
            "ordinal": sentence.get("ordinal"),
            "original": sentence.get("original"),
            "reading": (deep or {}).get("reading", ""),
            "translations": (deep or {}).get("translations", {}),
        }
        lines.extend(["", f"### Câu {row.get('ordinal', '?')}", f"**OCR:** {row.get('original', '')}"])
        if row.get("reading"):
            lines.append(f"**Hiragana:** {row['reading']}")
        translations = row.get("translations") or {}
        lines.extend([
            f"**Dịch tự nhiên:** {translations.get('natural', '')}",
            f"**Dịch theo cụm:** {translations.get('chunked', '')}",
            f"**Dịch sát:** {translations.get('literal', '')}",
        ])
        if row.get("key_points"):
            lines.append("**Điểm mấu chốt:**")
            for point in row["key_points"]:
                lines.append(f"- {point.get('label', '')} - `{point.get('source', '')}`: {point.get('explanation_vi', '')}")
        if row.get("translation_steps"):
            lines.append("**Thứ tự dịch đề xuất:**")
            for index, step in enumerate(row["translation_steps"], 1):
                order = step.get("order") or index
                lines.append(
                    f"{order}. `{step.get('source_chunk', '')}` → "
                    f"{step.get('meaning_vi', '')}: {step.get('advice_vi', '')}"
                )
        if row.get("ocr_warning"):
            lines.append(f"**Cảnh báo OCR:** {row['ocr_warning']}")
        if row.get("related_analysis"):
            lines.append("**Phân tích liên quan:**")
            for ref in row["related_analysis"]:
                lines.append(
                    f"- **{ref.get('category_label', '')}: {ref.get('label', '')}**: "
                    f"{ref.get('summary', '')}"
                )
        if deep:
            lines.extend(_deep_markdown_lines(deep))
    return "\n".join(lines).strip()


def _deep_markdown_lines(row: dict[str, Any]) -> list[str]:
    """Render a breakdown inline without repeating the sentence heading."""
    lines = ["", "#### Giải mã câu dài"]
    skeleton = row.get("sentence_skeleton") or {}
    if any(skeleton.values()):
        lines.append("**Khung câu trung tâm:**")
        for label, key in (
            ("Mẫu", "pattern"), ("Chủ đề", "topic"), ("Chủ ngữ", "subject"),
            ("Vị ngữ", "predicate"), ("Tân ngữ/Bổ ngữ", "object_or_complement"),
            ("Thì/Thể", "tense_aspect"), ("Thái/Thức", "voice_modality"),
        ):
            if skeleton.get(key):
                lines.append(f"- **{label}:** {skeleton[key]}")
    if row.get("segments"):
        lines.append("**Cụm từ và vai trò:**")
        for item in row["segments"]:
            detail = "; ".join(
                value for value in (
                    item.get("base_form") and f"dạng gốc: {item['base_form']}",
                    item.get("grammar_form") and f"ngữ pháp: {item['grammar_form']}",
                    item.get("particle_or_connector") and f"trợ từ/từ nối: {item['particle_or_connector']}",
                ) if value
            )
            lines.append(f"- `{item.get('text', '')}` [{item.get('role', '')}]: {item.get('meaning_vi', '')}{'; ' + detail if detail else ''}")
    if row.get("clauses"):
        lines.append("**Mệnh đề:**")
        for item in row["clauses"]:
            lines.append(f"- {item.get('label', '')}: {item.get('text', '')} - {item.get('relation_to_main', '')}")
    if row.get("structure_summary"):
        lines.append(f"**Cấu trúc:** {row['structure_summary']}")
    for title, key, formatter in (
        ("Chuỗi ngữ pháp", "grammar_links", lambda item: f"{item.get('source', '')} [{item.get('form', '')}]: {item.get('function_vi', '')}; {item.get('nuance_vi', '')}"),
        ("Từ nối", "connectors", lambda item: f"{item.get('source', '')}: {item.get('function_vi', '')}; {item.get('relation', '')}"),
    ):
        if row.get(key):
            lines.append(f"**{title}:**")
            lines.extend(f"- {formatter(item)}" for item in row[key])
    for title, key, formatter in (
        ("Thành phần lược bỏ", "omitted_elements", lambda item: f"{item.get('element', '')} → {item.get('recovered', '')}: {item.get('reason', '')}"),
        ("Từ quy chiếu", "references", lambda item: f"{item.get('expression', '')} → {item.get('referent', '')}: {item.get('reason', '')}"),
        ("Luồng logic", "logic", lambda item: f"{item.get('marker', '')} [{item.get('relation', '')}]: {item.get('scope', '')}"),
    ):
        if row.get(key):
            lines.append(f"**{title}:**")
            lines.extend(f"- {formatter(item)}" for item in row[key])
    if row.get("simplified_source"):
        lines.append(f"**Câu viết lại đơn giản:** {row['simplified_source']}")
    if row.get("simplified_vi"):
        lines.append(f"**Nghĩa tiếng Việt:** {row['simplified_vi']}")
    if row.get("translation_steps"):
        lines.append("**Cách tháo câu từng bước:**")
        for index, item in enumerate(row["translation_steps"], 1):
            lines.append(f"{item.get('order') or index}. `{item.get('source_chunk', '')}` → {item.get('meaning_vi', '')}: {item.get('advice_vi', '')}")
    if row.get("ambiguities"):
        lines.append("**Điểm dễ hiểu sai:**")
        lines.extend(f"- `{item.get('source', '')}`: {item.get('alternatives', '')} - {item.get('explanation_vi', '')}" for item in row["ambiguities"])
    if row.get("quality_status") == "partial":
        lines.append(f"**Cần bổ sung:** {', '.join(row.get('missing_fields') or [])}")
    if row.get("questions"):
        lines.append("**Câu hỏi kiểm tra hiểu:**")
        for question in row["questions"]:
            lines.append(f"- {question.get('question', '')}  ")
            lines.append(f"  Đáp án: {question.get('answer', '')}. {question.get('explanation', '')}")
    return lines
