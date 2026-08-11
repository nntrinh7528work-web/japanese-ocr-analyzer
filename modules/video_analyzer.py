"""YouTube/video transcript ingestion, segmentation, analysis, and costing."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import config as app_config
from modules.cost_estimator import estimate_video_plan_cost
from modules.gemini_client import create_gemini_model
from modules.sentence_analyzer import detect_sentence_language, split_sentences


# Keep the optional video subsystem bootable while a hosted deployment is
# upgrading an older config.py. Existing configuration still takes precedence.
GEMINI_API_KEY = getattr(app_config, "GEMINI_API_KEY", None)
GEMINI_MODEL_VIDEO = getattr(app_config, "GEMINI_MODEL_VIDEO", "gemini-3.6-flash")
GEMINI_MODEL_VIDEO_BATCH = getattr(app_config, "GEMINI_MODEL_VIDEO_BATCH", "gemini-2.5-flash-lite")
MAX_VIDEO_SIZE_MB = int(getattr(app_config, "MAX_VIDEO_SIZE_MB", 100))
SUPPORTED_VIDEO_FORMATS = list(
    getattr(app_config, "SUPPORTED_VIDEO_FORMATS", ["mp4", "mov", "webm", "mpeg", "mpg", "avi"])
)


_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
    "www.youtube-nocookie.com", "youtube-nocookie.com",
}
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_youtube_url(value: str) -> dict[str, str]:
    """Validate a public single-video URL without fetching arbitrary hosts."""
    raw = str(value or "").strip()
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise ValueError("Chỉ hỗ trợ URL YouTube hợp lệ.")
    if "list" in parse_qs(parsed.query):
        raise ValueError("Playlist chưa được hỗ trợ. Hãy nhập link của một video.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
        video_id = parsed.path.strip("/").split("/")[1]
    else:
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("Không tìm thấy video ID hợp lệ trong URL.")
    return {
        "video_id": video_id,
        "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
    }


def validate_video_upload(name: str, data: bytes, mime_type: str | None = None) -> dict[str, Any]:
    suffix = Path(str(name or "")).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError("Định dạng video chưa được hỗ trợ.")
    if not data:
        raise ValueError("File video trống.")
    if len(data) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Video vượt quá giới hạn {MAX_VIDEO_SIZE_MB} MB.")
    mime = mime_type or mimetypes.guess_type(name)[0] or "video/mp4"
    if not mime.startswith("video/"):
        raise ValueError("MIME type của file không phải video.")
    return {"suffix": suffix, "mime_type": mime, "size_bytes": len(data)}


def probe_video_duration(path: str) -> float:
    """Read duration locally without loading the video into SQLite."""
    try:
        import cv2

        capture = cv2.VideoCapture(path)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        duration = frames / fps if fps > 0 else 0
    except Exception:
        duration = 0
    return duration


def fetch_youtube_caption(video_id: str, preferred_language: str = "unknown") -> tuple[list[dict], str]:
    """Fetch the best original caption track, raising so callers can fall back."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    tracks = list(api.list(video_id))
    if not tracks:
        raise RuntimeError("Video không có caption công khai.")
    preferred_codes = ["ja", "en"]
    if preferred_language == "english":
        preferred_codes = ["en", "ja"]
    elif preferred_language == "japanese":
        preferred_codes = ["ja", "en"]

    def rank(track: Any) -> tuple[int, int]:
        code = str(getattr(track, "language_code", ""))
        preferred = preferred_codes.index(code) if code in preferred_codes else len(preferred_codes)
        generated = 1 if bool(getattr(track, "is_generated", False)) else 0
        return preferred, generated

    track = sorted(tracks, key=rank)[0]
    rows = []
    for snippet in track.fetch():
        text = str(getattr(snippet, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(snippet, "start", 0) or 0)
        duration = float(getattr(snippet, "duration", 0) or 0)
        rows.append({"start": start, "end": start + duration, "text": text, "speaker": ""})
    if not rows:
        raise RuntimeError("Caption không chứa nội dung đọc được.")
    provider = "youtube_caption_auto" if bool(getattr(track, "is_generated", False)) else "youtube_caption"
    return rows, provider


def _response_text(response: Any) -> str:
    direct = getattr(response, "text", None) or getattr(response, "output_text", None)
    if direct:
        return str(direct).strip()
    outputs = getattr(response, "outputs", None) or []
    fragments = []
    for output in outputs:
        value = getattr(output, "text", None)
        if value:
            fragments.append(str(value))
        for part in getattr(output, "content", None) or []:
            text = getattr(part, "text", None)
            if text:
                fragments.append(str(text))
    return "\n".join(fragments).strip()


def response_usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    def read(*names: str) -> int:
        for name in names:
            value = getattr(metadata, name, None) if metadata is not None else None
            if value is None and isinstance(metadata, dict):
                value = metadata.get(name)
            if value is not None:
                return int(value or 0)
        return 0
    candidates = read("candidates_token_count", "output_tokens")
    thinking = read("thoughts_token_count", "thinking_tokens")
    return {
        "input_tokens": read("prompt_token_count", "input_tokens"),
        "output_tokens": candidates + thinking,
        "candidate_tokens": candidates,
        "thinking_tokens": thinking,
        "cached_tokens": read("cached_content_token_count", "total_cached_tokens"),
    }


def _parse_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini không trả về JSON video hợp lệ.")
        parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Dữ liệu video phải là một JSON object.")
    return parsed


def transcribe_with_gemini(source: dict) -> tuple[list[dict], dict]:
    """Fallback transcript for a public URL or uploaded File API video."""
    model = create_gemini_model(GEMINI_MODEL_VIDEO, GEMINI_API_KEY)
    prompt = (
        "Transcribe the complete Japanese and/or English speech in this video. "
        "Return JSON only: {utterances:[{start,end,speaker,text}], duration_seconds:number, "
        "visual_context:[{start,end,description}], warnings:[string]}. Add visual_context only when "
        "the visible scene is needed to understand otherwise ambiguous speech. Use seconds, preserve source wording, "
        "never translate, and use an empty "
        "speaker when identity is uncertain."
    )
    metadata = source.get("metadata") or {}
    range_start = metadata.get("range_start")
    range_end = metadata.get("range_end")
    if range_start is not None and range_end is not None:
        prompt += (
            f" Only return speech from {float(range_start):.1f} to {float(range_end):.1f} seconds, "
            "while preserving timestamps relative to the original video."
        )
    inputs: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    uploaded_name = ""
    uploaded_uri = ""
    if source.get("source_kind") == "youtube":
        inputs.append({"type": "video", "uri": source["source_url"], "resolution": "low"})
    else:
        uploaded = model.upload_file(str(source["local_path"]), source.get("mime_type"))
        uploaded_name = str(getattr(uploaded, "name", "") or "")
        uploaded = model.wait_for_file(uploaded_name)
        uploaded_uri = str(getattr(uploaded, "uri", "") or "")
        inputs.append({
            "type": "video", "uri": uploaded_uri,
            "mime_type": source.get("mime_type") or "video/mp4", "resolution": "low",
        })
    response = model.create_interaction(inputs)
    data = _parse_json(_response_text(response))
    rows = normalize_transcript(data.get("utterances") or [])
    if not rows:
        raise ValueError("Gemini không nhận diện được lời nói trong video.")
    usage = response_usage(response)
    usage["model_used"] = GEMINI_MODEL_VIDEO
    usage["warnings"] = data.get("warnings") or []
    usage["visual_context"] = data.get("visual_context") or []
    usage["duration_seconds"] = float(data.get("duration_seconds") or rows[-1]["end"])
    if uploaded_uri:
        usage["gemini_file_uri"] = uploaded_uri
        usage["gemini_file_name"] = uploaded_name
    return rows, usage


def analyze_video_visual_segment(source: dict, segment: dict) -> tuple[dict, dict]:
    """On-demand visual explanation for one timestamp range."""
    model = create_gemini_model(GEMINI_MODEL_VIDEO, GEMINI_API_KEY)
    uri = source.get("source_url") or (source.get("ingest_usage") or {}).get("gemini_file_uri")
    if not uri:
        raise ValueError("Nguồn video tạm đã hết hạn; hãy tải lại file để phân tích hình ảnh.")
    prompt = (
        f"Analyze only the visual context from {float(segment.get('start_seconds', 0)):.1f} to "
        f"{float(segment.get('end_seconds', 0)):.1f} seconds. Explain in Vietnamese how the visible scene, "
        "onscreen text, gestures, or speaker changes help understand the transcript. Return JSON only with "
        "{summary:string, visual_cues:[{timestamp,description,learning_value}], onscreen_text:[string], warnings:[string]}."
    )
    response = model.create_interaction([
        {"type": "text", "text": prompt},
        {"type": "video", "uri": uri, "mime_type": source.get("mime_type") or "video/mp4", "resolution": "low"},
    ])
    result = _parse_json(_response_text(response))
    usage = response_usage(response)
    usage["model_used"] = GEMINI_MODEL_VIDEO
    return result, usage


def normalize_transcript(rows: list[Any]) -> list[dict]:
    result = []
    for row in rows:
        if not isinstance(row, dict):
            row = {
                "start": getattr(row, "start", 0), "duration": getattr(row, "duration", 0),
                "text": getattr(row, "text", ""),
            }
        text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
        if not text:
            continue
        start = max(0.0, float(row.get("start", 0) or 0))
        end = float(row.get("end", start + float(row.get("duration", 0) or 0)) or start)
        result.append({
            "start": start, "end": max(start, end), "text": text,
            "speaker": str(row.get("speaker") or ""),
        })
    return sorted(result, key=lambda row: (row["start"], row["end"]))


def clean_transcript(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Conservatively remove exact ASR repeats and standalone fillers without rewriting source words."""
    cleaned: list[dict] = []
    warnings: list[str] = []
    fillers = {"um", "uh", "erm", "えー", "ええと", "あのー"}
    for row in normalize_transcript(rows):
        comparable = re.sub(r"[\s,.!?。、！？]+", "", row["text"].lower())
        if comparable in fillers:
            warnings.append(f"Đã ẩn từ đệm tại {format_timestamp(row['start'])}: {row['text']}")
            continue
        if cleaned:
            previous = re.sub(r"[\s,.!?。、！？]+", "", cleaned[-1]["text"].lower())
            if comparable and comparable == previous and row["start"] <= cleaned[-1]["end"] + 2:
                warnings.append(f"Đã gộp đoạn ASR lặp tại {format_timestamp(row['start'])}.")
                cleaned[-1]["end"] = max(cleaned[-1]["end"], row["end"])
                continue
        cleaned.append(dict(row))
    return cleaned, warnings


def transcript_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_segments(rows: list[dict], window_seconds: int = 180) -> list[dict]:
    """Stable local chapter fallback; the model later supplies meaningful titles."""
    if not rows:
        return []
    buckets: list[list[dict]] = []
    current: list[dict] = []
    boundary = float(rows[0].get("start", 0) or 0) + window_seconds
    for row in rows:
        if current and row["start"] >= boundary:
            buckets.append(current)
            current = []
            boundary = int(row["start"] // window_seconds + 1) * window_seconds
        current.append(row)
    if current:
        buckets.append(current)
    segments = []
    for index, bucket in enumerate(buckets, 1):
        text = " ".join(row["text"] for row in bucket).strip()
        language, _, _ = detect_sentence_language(text, "english")
        speakers = sorted({row["speaker"] for row in bucket if row.get("speaker")})
        segments.append({
            "segment_id": str(hashlib.sha256(f"{index}|{bucket[0]['start']}|{text}".encode("utf-8")).hexdigest()[:24]),
            "start_seconds": bucket[0]["start"], "end_seconds": bucket[-1]["end"],
            "title": f"Đoạn {index}", "language": language,
            "original_text": text, "clean_text": text, "speakers": speakers,
        })
    return segments


def estimate_transcript_tokens(rows: list[dict]) -> int:
    # Conservative local fallback before count_tokens is available.
    return max(1, len(" ".join(str(row.get("text") or "") for row in rows)) // 3)


def build_cost_estimate(source: dict, segments: list[dict], billing_tier: str) -> dict:
    rows = source.get("clean_transcript") or source.get("raw_transcript") or []
    duration = float(source.get("duration_seconds") or (rows[-1].get("end", 0) if rows else 0))
    hard_count = min(len(segments), 15)
    return estimate_video_plan_cost(
        duration_seconds=duration,
        transcript_tokens=estimate_transcript_tokens(rows),
        segment_count=len(segments), hard_sentence_count=hard_count,
        transcript_provider=(
            "youtube_caption"
            if str(source.get("transcript_provider") or "").startswith("youtube_caption")
            else str(source.get("transcript_provider") or "gemini_video")
        ),
        billing_tier=billing_tier, media_resolution="low",
    )


def build_segment_batches(segments: list[dict], max_segments: int = 4, max_chars: int = 6000) -> list[list[dict]]:
    """Group source-ordered, language-homogeneous segments without resending the transcript."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    current_language = ""
    for segment in segments:
        language = "japanese" if segment.get("language") == "japanese" else "english"
        length = len(str(segment.get("clean_text") or segment.get("original_text") or ""))
        if current and (
            language != current_language or len(current) >= max_segments or current_chars + length > max_chars
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append(segment)
        current_chars += length
        current_language = language
        if length > max_chars:
            batches.append(current)
            current, current_chars, current_language = [], 0, ""
    if current:
        batches.append(current)
    return batches


def analyze_video_segment_batch(segments: list[dict]) -> tuple[list[dict], dict]:
    """Analyze up to four same-language chapters in one cost-efficient request."""
    if not segments:
        return [], {}
    languages = {"japanese" if row.get("language") == "japanese" else "english" for row in segments}
    if len(languages) != 1:
        raise ValueError("Một batch video chỉ được chứa một ngôn ngữ.")
    language = languages.pop()
    model = create_gemini_model(GEMINI_MODEL_VIDEO_BATCH, GEMINI_API_KEY)
    requested = [
        {
            "segment_id": row.get("segment_id"),
            "start_seconds": row.get("start_seconds"),
            "end_seconds": row.get("end_seconds"),
            "transcript": row.get("clean_text") or row.get("original_text") or "",
        }
        for row in segments
    ]
    prompt = f"""Bạn là giáo viên ngôn ngữ. Phân tích transcript {language} sau bằng tiếng Việt.
Giữ nguyên câu nguồn. Trả JSON duy nhất dạng {{"segments":[...]}}. Mỗi phần tử phải giữ đúng segment_id và có:
segment_id, title, summary, natural_translation, key_points (list), dialogue_turns (speaker,text,translation_vi),
vocabulary_all (word,reading,meaning,part_of_speech,example,example_translation),
kanji_analysis (kanji,onyomi,kunyomi,meaning,vocab),
connectors (connector,function,meaning), discourse_markers (marker,function,meaning),
grammar_points (pattern,meaning,formation,explanation,example,example_translation),
sentence_patterns (pattern,components,function,example,explanation),
hard_sentence_candidate. Với tiếng Anh để kanji_analysis rỗng; với tiếng Nhật để discourse_markers rỗng.
Không bỏ sót segment và không trộn nội dung giữa các segment.
{json.dumps(requested, ensure_ascii=False)}
"""
    input_tokens = model.count_tokens(prompt)
    response = model.generate_content(
        prompt,
        {"response_mime_type": "application/json", "max_output_tokens": 6000},
    )
    payload = _parse_json(_response_text(response))
    returned = payload.get("segments") or []
    by_id = {str(row.get("segment_id")): row for row in returned if isinstance(row, dict)}
    results = []
    for segment in segments:
        result = dict(by_id.get(str(segment.get("segment_id"))) or {})
        if not result:
            raise ValueError(f"Gemini bỏ sót segment {segment.get('segment_id')}.")
        result.update({
            "segment_id": segment.get("segment_id"), "ordinal": segment.get("ordinal"),
            "start_seconds": segment.get("start_seconds"), "end_seconds": segment.get("end_seconds"),
            "title": result.get("title") or segment.get("title"), "language": language,
            "source_text": segment.get("clean_text") or segment.get("original_text") or "",
        })
        results.append(result)
    usage = response_usage(response)
    usage["input_tokens"] = usage.get("input_tokens") or input_tokens
    usage["model_used"] = GEMINI_MODEL_VIDEO_BATCH
    return results, usage


def analyze_video_segment(segment: dict) -> tuple[dict, dict]:
    """Compatibility wrapper for callers that intentionally analyze one chapter."""
    rows, usage = analyze_video_segment_batch([segment])
    return rows[0], usage


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def video_analysis_markdown(source: dict, segments: list[dict]) -> str:
    metadata = source.get("metadata") or {}
    lines = [f"# {metadata.get('title') or source.get('file_name') or 'Phân tích video'}", ""]
    if source.get("source_url"):
        lines.append(f"Nguồn: {source['source_url']}")
    lines.extend(["", "## Mục lục video", ""])
    for segment in segments:
        lines.append(
            f"- {format_timestamp(segment.get('start_seconds', 0))}–{format_timestamp(segment.get('end_seconds', 0))}: "
            f"{segment.get('title') or 'Đoạn'}"
        )
    for segment in segments:
        analysis = segment.get("analysis") or {}
        lines.extend([
            "", f"## {format_timestamp(segment.get('start_seconds', 0))} · {analysis.get('title') or segment.get('title')}", "",
            str(segment.get("clean_text") or ""), "",
            f"**Tóm tắt:** {analysis.get('summary') or ''}", "",
            f"**Dịch tự nhiên:** {analysis.get('natural_translation') or ''}",
        ])
        points = analysis.get("key_points") or []
        if points:
            lines.extend(["", "### Điểm chính", *[f"- {value}" for value in points]])
        turns = analysis.get("dialogue_turns") or []
        if turns:
            lines.extend(["", "### Hội thoại"])
            for turn in turns:
                speaker = str(turn.get("speaker") or "Người nói")
                lines.append(f"- **{speaker}:** {turn.get('text') or ''}  ")
                lines.append(f"  {turn.get('translation_vi') or ''}")
        for label, key in (("Từ vựng", "vocabulary_all"), ("Kanji", "kanji_analysis"), ("Ngữ pháp", "grammar_points")):
            rows = analysis.get(key) or []
            if rows:
                lines.extend(["", f"### {label}"])
                for row in rows:
                    lines.append("- " + " | ".join(str(value) for value in row.values() if value not in (None, "", [])))
        for label, key in (("Từ nối", "connectors"), ("Discourse markers", "discourse_markers"), ("Mẫu câu", "sentence_patterns")):
            rows = analysis.get(key) or []
            if rows:
                lines.extend(["", f"### {label}"])
                for row in rows:
                    lines.append("- " + " | ".join(str(value) for value in row.values() if value not in (None, "", [])))
        breakdown = analysis.get("sentence_breakdown") or {}
        if breakdown:
            lines.extend(["", "### Giải mã câu dài", "", json.dumps(breakdown, ensure_ascii=False, indent=2)])
        visual = analysis.get("visual_context_detail") or {}
        if visual:
            lines.extend(["", "### Bối cảnh hình ảnh", "", str(visual.get("summary") or "")])
            for cue in visual.get("visual_cues") or []:
                lines.append("- " + " | ".join(str(value) for value in cue.values() if value not in (None, "", [])))
    return "\n".join(lines).strip()


def build_video_analysis(source: dict, segments: list[dict]) -> dict:
    completed = [segment for segment in segments if segment.get("analysis")]
    usage_runs = []
    ingest_usage = source.get("ingest_usage") or {}
    if ingest_usage and sum(int(ingest_usage.get(key, 0) or 0) for key in ("input_tokens", "output_tokens")):
        usage_runs.append({
            "run_id": f"{source.get('source_id')}:ingest",
            "model_used": ingest_usage.get("model_used") or GEMINI_MODEL_VIDEO,
            "usage": ingest_usage,
            "stage": "video_ingest",
        })
    for segment in completed:
        usage = segment.get("usage") or {}
        bulk_usage = {key: value for key, value in usage.items() if key != "deep_sentence_usage"}
        usage_runs.append({
            "run_id": usage.get("run_id") or f"{segment['segment_id']}:bulk",
            "model_used": bulk_usage.get("model_used") or GEMINI_MODEL_VIDEO_BATCH,
            "usage": bulk_usage,
            "stage": "segment_analysis",
        })
        analysis_row = segment.get("analysis") or {}
        deep_usage = usage.get("deep_sentence_usage") or analysis_row.get("sentence_analysis_usage") or {}
        if deep_usage:
            usage_runs.append({
                "run_id": f"{segment['segment_id']}:deep",
                "model_used": analysis_row.get("sentence_analysis_model") or GEMINI_MODEL_VIDEO,
                "usage": deep_usage,
                "stage": "hard_sentence",
            })
        visual_usage = usage.get("visual_context_usage") or {}
        if visual_usage:
            usage_runs.append({
                "run_id": f"{segment['segment_id']}:visual",
                "model_used": visual_usage.get("model_used") or GEMINI_MODEL_VIDEO,
                "usage": visual_usage,
                "stage": "visual_context",
            })
    page_analyses = []
    for fallback_index, segment in enumerate(completed, 1):
        row = dict(segment.get("analysis") or {})
        catalog = split_sentences(
            str(segment.get("clean_text") or ""), str(segment.get("language") or "english"), fallback_index
        )
        timestamp_url = ""
        if source.get("source_url"):
            timestamp_url = f"{source['source_url']}&t={int(segment.get('start_seconds', 0) or 0)}s"
        turns = row.get("dialogue_turns") or []
        guidance = []
        for sentence in catalog:
            original = str(sentence.get("original") or "")
            matched_turn = next(
                (
                    turn for turn in turns
                    if str(turn.get("text") or "").strip()
                    and (
                        str(turn.get("text") or "").strip() in original
                        or original in str(turn.get("text") or "").strip()
                    )
                ),
                {},
            )
            sentence["timestamp_url"] = timestamp_url
            sentence["video_start_seconds"] = segment.get("start_seconds", 0)
            guidance.append({
                "sentence_id": sentence.get("sentence_id"),
                "original": original,
                "detected_language": sentence.get("detected_language") or segment.get("language"),
                "translations": {
                    "natural": matched_turn.get("translation_vi") or (
                        row.get("natural_translation") if len(catalog) == 1 else ""
                    )
                },
                "key_points": row.get("key_points") or [],
                "timestamp_url": timestamp_url,
            })
        breakdown = dict(row.get("sentence_breakdown") or {})
        if breakdown:
            deep_original = str(breakdown.get("original") or "").strip()
            matched_sentence = next(
                (
                    sentence for sentence in catalog
                    if deep_original and (
                        deep_original in str(sentence.get("original") or "")
                        or str(sentence.get("original") or "") in deep_original
                    )
                ),
                None,
            )
            if matched_sentence:
                breakdown["sentence_id"] = matched_sentence.get("sentence_id")
                breakdown["ordinal"] = matched_sentence.get("ordinal")
            breakdown["timestamp_url"] = timestamp_url
            row["sentence_breakdown"] = breakdown
        row.update({
            "page_index": fallback_index,
            "page_name": f"{format_timestamp(segment.get('start_seconds', 0))} · {row.get('title') or segment.get('title')}",
            "source_label": f"Đoạn {fallback_index}",
            "source_text": segment.get("clean_text") or "",
            "sentence_catalog": catalog,
            "translation_guidance": guidance,
            "sentence_breakdowns": [breakdown] if breakdown else [],
            "usage": segment.get("usage") or {},
        })
        page_analyses.append(row)
    analysis = {
        "analysis_type": "video", "analysis_mode": "video_balanced",
        "analysis_language": "mixed" if len({row.get('language') for row in completed}) > 1 else (completed[0].get("language") if completed else "unknown"),
        "summary": " ".join(str((row.get("analysis") or {}).get("summary") or "") for row in completed).strip(),
        "video_source": {key: source.get(key) for key in ("source_id", "source_kind", "source_url", "video_id", "file_name", "duration_seconds", "transcript_provider", "transcript_hash")},
        "video_metadata": source.get("metadata") or {},
        "transcript_clean": source.get("clean_transcript") or [],
        "transcript_warnings": source.get("transcript_warnings") or [],
        "video_segments": completed,
        "page_analyses": page_analyses,
        "video_ingest_usage": source.get("ingest_usage") or {},
        "video_analysis_runs": usage_runs,
        "model_used": GEMINI_MODEL_VIDEO_BATCH,
    }
    analysis["full_markdown"] = video_analysis_markdown(source, completed)
    return analysis
