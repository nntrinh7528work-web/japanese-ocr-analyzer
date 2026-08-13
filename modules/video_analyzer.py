"""YouTube/video transcript ingestion, segmentation, analysis, and costing."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from difflib import SequenceMatcher
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.parse import parse_qs, urlparse

import config as app_config
from modules.cost_estimator import (
    estimate_cost, estimate_run_costs, estimate_video_plan_cost, PRICING_EFFECTIVE_DATE, sum_costs,
)
from modules.gemini_client import create_gemini_model
from modules.sentence_analyzer import detect_sentence_language, split_sentences


# Keep the optional video subsystem bootable while a hosted deployment is
# upgrading an older config.py. Existing configuration still takes precedence.
GEMINI_API_KEY = getattr(app_config, "GEMINI_API_KEY", None)
GEMINI_MODEL_VIDEO = getattr(app_config, "GEMINI_MODEL_VIDEO", "gemini-3.6-flash")
_DEPRECATED_VIDEO_MODELS = {
    # Google returns 404 for this model on new API projects as of August 2026.
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
}


def supported_video_model(model_name: str | None, fallback: str) -> str:
    value = str(model_name or fallback).strip() or fallback
    return _DEPRECATED_VIDEO_MODELS.get(value, value)


GEMINI_MODEL_VIDEO_BATCH = supported_video_model(
    getattr(app_config, "GEMINI_MODEL_VIDEO_BATCH", None), "gemini-3.5-flash-lite"
)
GEMINI_MODEL_AUDIO = supported_video_model(
    getattr(app_config, "GEMINI_MODEL_AUDIO", None), "gemini-3.6-flash"
)
TRANSCRIPT_PIPELINE_VERSION = 2
TRANSCRIPT_WINDOW_SECONDS = 90
TRANSCRIPT_OVERLAP_SECONDS = 5
MAX_VIDEO_SIZE_MB = int(getattr(app_config, "MAX_VIDEO_SIZE_MB", 100))
MAX_VIDEO_DURATION_SECONDS = int(getattr(app_config, "MAX_VIDEO_DURATION_SECONDS", 30 * 60))
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


def probe_video_metadata(path: str) -> dict[str, Any]:
    """Inspect media with ffprobe, with OpenCV as a duration-only fallback."""
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,codec_name", "-of", "json", path,
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams") or []
        duration = float((payload.get("format") or {}).get("duration") or 0)
        return {
            "duration_seconds": duration,
            "has_audio": any(str(row.get("codec_type")) == "audio" for row in streams if isinstance(row, dict)),
            "video_codec": next((str(row.get("codec_name") or "") for row in streams if row.get("codec_type") == "video"), ""),
            "audio_codec": next((str(row.get("codec_name") or "") for row in streams if row.get("codec_type") == "audio"), ""),
        }
    except Exception:
        duration = 0.0
        try:
            import cv2

            capture = cv2.VideoCapture(path)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frames / fps if fps > 0 else 0
            capture.release()
        except Exception:
            pass
        return {"duration_seconds": duration, "has_audio": None, "video_codec": "", "audio_codec": ""}


def probe_video_duration(path: str) -> float:
    """Read duration locally without loading the video into SQLite."""
    return float(probe_video_metadata(path).get("duration_seconds") or 0)


def validate_video_duration(metadata: dict[str, Any]) -> None:
    duration = float(metadata.get("duration_seconds") or 0)
    if duration <= 0:
        raise ValueError("Không đọc được thời lượng video.")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError("Video vượt quá giới hạn 30 phút.")
    if metadata.get("has_audio") is False:
        raise ValueError("Video không có track âm thanh để tạo script.")


def build_audio_windows(
    duration_seconds: float, window_seconds: float = TRANSCRIPT_WINDOW_SECONDS,
    overlap_seconds: float = TRANSCRIPT_OVERLAP_SECONDS,
) -> list[dict]:
    """Return stable 90-second windows while retaining a five-second overlap."""
    duration = max(0.0, float(duration_seconds or 0))
    if not duration:
        return []
    step = max(1.0, window_seconds - overlap_seconds)
    windows = []
    start = 0.0
    index = 1
    while start < duration:
        end = min(duration, start + window_seconds)
        windows.append({"index": index, "start": start, "end": end})
        if end >= duration:
            break
        start += step
        index += 1
    return windows


def extract_audio_window(video_path: str, output_path: str, start: float, end: float) -> None:
    """Create a compact mono speech track without modifying the uploaded video."""
    duration = max(0.1, float(end) - float(start))
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{float(start):.3f}", "-i", video_path, "-t", f"{duration:.3f}",
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", output_path,
            ],
            check=True, capture_output=True, timeout=max(60, int(duration * 2)),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Máy chủ chưa cài FFmpeg để tách audio.") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Không thể tách audio khỏi video: {message or 'FFmpeg thất bại'}") from exc


def estimate_audio_transcription_cost(duration_seconds: float, billing_tier: str = "paid") -> dict[str, Any]:
    """Estimate primary ASR and the bounded smart-verification allowance."""
    duration = max(0.0, float(duration_seconds or 0))
    input_tokens = int(duration * 32)
    expected_output = max(500, int(duration * 5))
    maximum_output = max(1000, int(duration * 8))
    primary_expected = estimate_cost(
        {"input_tokens": input_tokens, "output_tokens": expected_output},
        GEMINI_MODEL_AUDIO, billing_tier, modality="audio",
    )
    primary_maximum = estimate_cost(
        {"input_tokens": input_tokens, "output_tokens": maximum_output},
        GEMINI_MODEL_AUDIO, billing_tier, modality="audio",
    )
    # Smart mode rechecks only suspicious 20-30 second clips, at most eight.
    expected_recheck_seconds = min(duration * 0.10, 25 * 4)
    maximum_recheck_seconds = min(duration * 0.20, 25 * 8)
    verification_expected = estimate_cost(
        {
            "input_tokens": int(expected_recheck_seconds * 32),
            "output_tokens": max(0, int(expected_recheck_seconds * 5)),
        },
        GEMINI_MODEL_AUDIO, billing_tier, modality="audio",
    )
    verification_maximum = estimate_cost(
        {
            "input_tokens": int(maximum_recheck_seconds * 32),
            "output_tokens": max(0, int(maximum_recheck_seconds * 8)),
        },
        GEMINI_MODEL_AUDIO, billing_tier, modality="audio",
    )
    return {
        "model": GEMINI_MODEL_AUDIO,
        "input_tokens": input_tokens,
        "window_count": len(build_audio_windows(duration)),
        "primary_expected": primary_expected,
        "primary_maximum": primary_maximum,
        "verification_expected": verification_expected,
        "verification_maximum": verification_maximum,
        "expected": sum_costs([primary_expected, verification_expected]),
        "maximum": sum_costs([primary_maximum, verification_maximum]),
        "pricing_effective_date": PRICING_EFFECTIVE_DATE,
    }


def build_video_usage_cost_breakdown(
    source: dict, analysis: dict | None = None, billing_tier: str = "paid",
) -> dict[str, dict]:
    """Price transcript, verification, translation and learning analysis separately."""
    analysis_runs = [
        run for run in (analysis or {}).get("video_analysis_runs") or []
        if isinstance(run, dict)
    ]
    if analysis_runs:
        all_runs = analysis_runs
    else:
        ingest = source.get("ingest_usage") if isinstance(source.get("ingest_usage"), dict) else {}
        all_runs = [run for run in ingest.get("runs") or [] if isinstance(run, dict)]
        all_runs.extend(
            run for run in source.get("translation_runs") or [] if isinstance(run, dict)
        )

    def stage(run: dict) -> str:
        return str(run.get("stage") or "")

    verification = [run for run in all_runs if "verification" in stage(run)]
    translation = [run for run in all_runs if "translation" in stage(run)]
    primary = [
        run for run in all_runs
        if run not in verification and run not in translation
        and stage(run) in {"", "video_transcription", "transcript_primary", "video_ingest"}
    ]
    deep = [
        run for run in all_runs
        if run not in verification and run not in translation and run not in primary
    ]
    ingest_usage = source.get("ingest_usage") if isinstance(source.get("ingest_usage"), dict) else {}
    primary_fallback = ingest_usage if not all_runs else {}
    return {
        "transcript_primary": estimate_run_costs(
            primary, primary_fallback, GEMINI_MODEL_AUDIO, billing_tier,
        ),
        "transcript_verification": estimate_run_costs(
            verification, {}, GEMINI_MODEL_AUDIO, billing_tier,
        ),
        "translation_vi": estimate_run_costs(
            translation, {}, GEMINI_MODEL_VIDEO_BATCH, billing_tier,
        ),
        "deep_analysis": estimate_run_costs(
            deep, {}, GEMINI_MODEL_VIDEO_BATCH, billing_tier,
        ),
    }


def fetch_youtube_caption(video_id: str, preferred_language: str = "unknown") -> tuple[list[dict], str]:
    """Fetch source captions and attach a free Vietnamese track when available."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    tracks = list(api.list(video_id))
    if not tracks:
        raise RuntimeError("Video không có caption công khai.")
    # With no explicit lesson language, preserve YouTube's source-track order.
    # Hard-coding Japanese first can select a translated Japanese subtitle for
    # an English video even when the original English caption is available.
    preferred_codes: list[str] = []
    if preferred_language == "english":
        preferred_codes = ["en", "ja"]
    elif preferred_language == "japanese":
        preferred_codes = ["ja", "en"]

    def language_rank(code: str) -> int:
        normalized = code.lower().split("-", 1)[0]
        if not preferred_codes:
            return 0
        return preferred_codes.index(normalized) if normalized in preferred_codes else len(preferred_codes)

    def rank(track: Any) -> tuple[int, int]:
        code = str(getattr(track, "language_code", ""))
        preferred = language_rank(code)
        generated = 1 if bool(getattr(track, "is_generated", False)) else 0
        return preferred, generated

    source_tracks = [track for track in tracks if str(getattr(track, "language_code", "")).lower().split("-", 1)[0] in {"ja", "en"}]
    if not source_tracks:
        raise RuntimeError("Video không có caption tiếng Nhật hoặc tiếng Anh công khai.")
    track = sorted(source_tracks, key=rank)[0]
    source_code = str(getattr(track, "language_code", "")).lower().split("-", 1)[0]
    source_language = "japanese" if source_code == "ja" else "english"
    provider = "youtube_caption_auto" if bool(getattr(track, "is_generated", False)) else "youtube_caption"

    rows = []
    for snippet in track.fetch():
        text = str(getattr(snippet, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(snippet, "start", 0) or 0)
        duration = float(getattr(snippet, "duration", 0) or 0)
        rows.append({
            "start": start, "end": start + duration, "text": text, "speaker": "",
            "language": source_language, "transcript_provider": provider,
        })
    if not rows:
        raise RuntimeError("Caption không chứa nội dung đọc được.")

    translated_track = next(
        (
            candidate for candidate in tracks
            if str(getattr(candidate, "language_code", "")).lower().split("-", 1)[0] == "vi"
        ),
        None,
    )
    translation_provider = "youtube_caption_vi" if translated_track is not None else ""
    if translated_track is None and bool(getattr(track, "is_translatable", False)):
        available = {
            str(row.get("language_code") if isinstance(row, dict) else getattr(row, "language_code", ""))
            for row in (getattr(track, "translation_languages", None) or [])
        }
        if not available or "vi" in available:
            try:
                translated_track = track.translate("vi")
                translation_provider = "youtube_auto_translate_vi"
            except Exception:
                translated_track = None

    if translated_track is not None:
        try:
            translated = []
            for snippet in translated_track.fetch():
                text = str(getattr(snippet, "text", "") or "").strip()
                if not text:
                    continue
                start = float(getattr(snippet, "start", 0) or 0)
                translated.append({"start": start, "text": text})
            for index, row in enumerate(rows):
                nearest = min(translated, key=lambda item: abs(item["start"] - row["start"])) if translated else None
                if nearest and abs(nearest["start"] - row["start"]) <= 2:
                    row["translation_vi"] = nearest["text"]
                    row["translation_provider"] = translation_provider
                elif index < len(translated):
                    row["translation_vi"] = translated[index]["text"]
                    row["translation_provider"] = translation_provider
        except Exception:
            pass
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


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_record_list(value: Any, text_key: str = "text") -> list[dict[str, Any]]:
    """Accept imperfect model lists without letting one string crash the whole video."""
    if value is None or value == "":
        return []
    values = value if isinstance(value, list) else [value]
    records: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, dict):
            records.append(dict(item))
        elif item is not None:
            records.append({text_key: str(item)})
    return records


def _as_text_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if item is not None and str(item).strip()]


def normalize_video_segment_result(value: Any) -> dict[str, Any]:
    """Normalize lenient Gemini JSON before persistence and before rendering old results."""
    result = _as_mapping(value)
    result["title"] = str(result.get("title") or "").strip()
    result["summary"] = str(result.get("summary") or "").strip()
    result["natural_translation"] = str(result.get("natural_translation") or "").strip()
    result["key_points"] = _as_text_list(result.get("key_points"))
    result["dialogue_turns"] = _as_record_list(result.get("dialogue_turns"), "text")
    for key in (
        "vocabulary_all", "kanji_analysis", "connectors", "discourse_markers",
        "grammar_points", "sentence_patterns",
    ):
        result[key] = _as_record_list(result.get(key), "value")
    if not isinstance(result.get("sentence_breakdown"), dict):
        result.pop("sentence_breakdown", None)
    if not isinstance(result.get("visual_context_detail"), dict):
        result.pop("visual_context_detail", None)
    return result


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


def _audio_transcript_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "speaker": {"type": "string"},
                        "language": {"type": "string", "enum": ["japanese", "english"]},
                        "text": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "uncertainty_reason": {"type": "string"},
                    },
                    "required": ["start", "end", "speaker", "language", "text", "confidence", "uncertainty_reason"],
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["cues", "warnings"],
    }


def detect_non_silent_intervals(audio_path: str, duration_seconds: float) -> list[tuple[float, float]]:
    """Detect probable speech/audio regions locally; failure must not block ASR."""
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
                "-af", "silencedetect=n=-35dB:d=0.5", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=max(30, int(duration_seconds) + 15),
        )
        diagnostic = completed.stderr or ""
        starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", diagnostic)]
        ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", diagnostic)]
        silence: list[tuple[float, float]] = []
        for index, silence_start in enumerate(starts):
            silence_end = ends[index] if index < len(ends) else float(duration_seconds)
            silence.append((max(0.0, silence_start), min(float(duration_seconds), silence_end)))
        audible: list[tuple[float, float]] = []
        cursor = 0.0
        for silence_start, silence_end in sorted(silence):
            if silence_start - cursor >= 0.6:
                audible.append((cursor, silence_start))
            cursor = max(cursor, silence_end)
        if float(duration_seconds) - cursor >= 0.6:
            audible.append((cursor, float(duration_seconds)))
        return audible
    except Exception:
        return []


def _coverage_gaps(rows: list[dict], audible: list[tuple[float, float]], window_start: float) -> list[dict]:
    """Subtract recognized cue intervals from audible regions."""
    gaps = []
    cue_intervals = sorted(
        (
            float(row.get("start", 0) or 0),
            float(row.get("end", row.get("start", 0)) or row.get("start", 0) or 0),
        )
        for row in rows
    )
    for local_start, local_end in audible:
        absolute_start = window_start + local_start
        absolute_end = window_start + local_end
        cursor = absolute_start
        for cue_start, cue_end in cue_intervals:
            if cue_end <= cursor or cue_start >= absolute_end:
                continue
            if cue_start - cursor >= 1.5:
                gaps.append({
                    "start": cursor, "end": min(cue_start, absolute_end),
                    "reason": "Có âm thanh nhưng chưa có transcript.",
                })
            cursor = max(cursor, min(cue_end, absolute_end))
            if cursor >= absolute_end:
                break
        if absolute_end - cursor >= 1.5:
            gaps.append({
                "start": cursor, "end": absolute_end,
                "reason": "Có âm thanh nhưng chưa có transcript.",
            })
    return gaps


def transcribe_audio_window(source: dict, window: dict, temp_dir: str | None = None) -> tuple[list[dict], dict]:
    """Transcribe one short upload window; translation intentionally runs later."""
    local_path = str(source.get("local_path") or "")
    if not local_path or not Path(local_path).exists():
        raise ValueError("File video tạm không còn tồn tại. Hãy tải lại video.")
    start = float(window.get("start", 0) or 0)
    end = float(window.get("end", start) or start)
    temp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False, dir=temp_dir)
    audio_path = temp.name
    temp.close()
    model = create_gemini_model(GEMINI_MODEL_AUDIO, GEMINI_API_KEY)
    uploaded_name = ""
    try:
        extract_audio_window(local_path, audio_path, start, end)
        audible = detect_non_silent_intervals(audio_path, max(0.0, end - start))
        uploaded = model.upload_file(audio_path, "audio/flac")
        uploaded_name = str(getattr(uploaded, "name", "") or "")
        uploaded = model.wait_for_file(uploaded_name)
        uploaded_uri = str(getattr(uploaded, "uri", "") or "")
        prompt = """Bạn là hệ thống chép lời chính xác cho audio tiếng Nhật và/hoặc tiếng Anh.
Chỉ chép nguyên văn lời thực sự nghe thấy. Không dịch, không sửa văn phong, không đoán thêm từ trong khoảng im lặng.
Chia cue theo lượt nói hoặc câu ngắn. Timestamp start/end tính bằng giây từ đầu clip audio này.
Nếu không chắc một từ, vẫn ghi cách nghe hợp lý nhất nhưng đặt confidence=low và giải thích ngắn trong uncertainty_reason.
Dùng confidence=medium cho tên riêng, tiếng nói bị che hoặc phát âm khó; high chỉ khi nghe rõ.
Speaker để rỗng nếu không phân biệt chắc chắn. Trả duy nhất JSON đúng schema."""
        response = model.create_interaction(
            [
                {"type": "audio", "uri": uploaded_uri, "mime_type": "audio/flac"},
                {"type": "text", "text": prompt},
            ],
            response_mime_type="application/json",
            response_format=_audio_transcript_schema(),
        )
        payload = _parse_json(_response_text(response))
        rows = []
        for raw in _as_record_list(payload.get("cues"), "text"):
            text = str(raw.get("text") or raw.get("source_text") or "").strip()
            if not text:
                continue
            relative_start = max(0.0, min(end - start, float(raw.get("start", 0) or 0)))
            relative_end = max(relative_start, min(end - start, float(raw.get("end", relative_start) or relative_start)))
            language = str(raw.get("language") or "unknown").lower()
            if language not in {"japanese", "english"}:
                language, _, _ = detect_sentence_language(text, "english")
            confidence = str(raw.get("confidence") or "unknown").lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "unknown"
            uncertainty = str(raw.get("uncertainty_reason") or "").strip()
            cue_duration = relative_end - relative_start
            if cue_duration <= 0 or cue_duration > 30:
                uncertainty = "; ".join(
                    value for value in (uncertainty, "Timestamp cue bất thường.") if value
                )
            needs_review = confidence in {"low", "unknown"} or bool(uncertainty)
            rows.append({
                "start": start + relative_start, "end": start + relative_end,
                "speaker": str(raw.get("speaker") or ""), "language": language,
                "text": text, "translation_vi": "",
                "transcript_provider": "gemini_audio_v2", "translation_provider": "",
                "confidence": confidence,
                "verification_status": "needs_review" if needs_review else "primary",
                "uncertainty_reason": uncertainty,
                "source_window_index": int(window.get("index", 0) or 0),
            })
        normalized = normalize_transcript(rows)
        if not normalized:
            raise ValueError("Gemini không nhận diện được lời nói trong cửa sổ audio này.")
        coverage_gaps = _coverage_gaps(normalized, audible, start)
        usage = response_usage(response)
        usage.update({
            "model_used": GEMINI_MODEL_AUDIO,
            "window_index": int(window.get("index", 0) or 0),
            "window_start": start, "window_end": end,
            "warnings": _as_text_list(payload.get("warnings")),
            "coverage_gaps": coverage_gaps,
            "pipeline_version": TRANSCRIPT_PIPELINE_VERSION,
        })
        return normalized, usage
    finally:
        if uploaded_name:
            try:
                model.delete_file(uploaded_name)
            except Exception:
                pass
        Path(audio_path).unlink(missing_ok=True)

def build_cue_translation_batches(
    cues: list[dict], max_cues: int = 80, max_chars: int = 5000,
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    language = ""
    for cue in cues:
        cue_language = "japanese" if cue.get("language") == "japanese" else "english"
        length = len(str(cue.get("source_text") or ""))
        if current and (cue_language != language or len(current) >= max_cues or chars + length > max_chars):
            batches.append(current)
            current, chars = [], 0
        current.append(cue)
        chars += length
        language = cue_language
    if current:
        batches.append(current)
    return batches


def estimate_cue_translation_cost(cues: list[dict], billing_tier: str = "paid") -> dict[str, Any]:
    chars = sum(len(str(cue.get("source_text") or "")) for cue in cues)
    input_tokens = max(1, chars // 3) + len(cues) * 12
    output_tokens = max(1, chars // 2) + len(cues) * 8
    return {
        "model": GEMINI_MODEL_VIDEO_BATCH,
        "batch_count": len(build_cue_translation_batches(cues)),
        "expected": estimate_cost(
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
            GEMINI_MODEL_VIDEO_BATCH, billing_tier,
        ),
        "pricing_effective_date": PRICING_EFFECTIVE_DATE,
    }


def translate_video_cue_batch(cues: list[dict]) -> tuple[dict[str, str], dict]:
    """Translate source-aligned caption rows without changing their boundaries."""
    if not cues:
        return {}, {}
    requested = [
        {
            "cue_id": cue.get("cue_id"), "speaker": cue.get("speaker", ""),
            "language": cue.get("language", "unknown"), "text": cue.get("source_text", ""),
        }
        for cue in cues
    ]
    model = create_gemini_model(GEMINI_MODEL_VIDEO_BATCH, GEMINI_API_KEY)
    prompt = f"""Dịch từng caption Nhật hoặc Anh sau sang tiếng Việt tự nhiên.
Các caption có thể là mảnh câu; dùng toàn bộ batch làm ngữ cảnh nhưng không gộp, tách, sửa hay đổi thứ tự cue.
Giữ đúng cue_id. Trả duy nhất JSON object dạng {{"translations":[{{"cue_id":"...","translation_vi":"..."}}]}}.
{json.dumps(requested, ensure_ascii=False)}
"""
    input_tokens = model.count_tokens(prompt)
    response = model.generate_content(
        prompt, {"response_mime_type": "application/json", "max_output_tokens": 6000}
    )
    payload = _parse_json(_response_text(response))
    translated = {
        str(row.get("cue_id")): str(row.get("translation_vi") or row.get("translation") or "").strip()
        for row in _as_record_list(payload.get("translations"), "translation_vi")
        if row.get("cue_id") and str(row.get("translation_vi") or row.get("translation") or "").strip()
    }
    missing = [str(cue.get("cue_id")) for cue in cues if str(cue.get("cue_id")) not in translated]
    if missing:
        raise ValueError(f"Gemini bỏ sót {len(missing)} dòng caption khi dịch.")
    usage = response_usage(response)
    usage["input_tokens"] = usage.get("input_tokens") or input_tokens
    usage["model_used"] = GEMINI_MODEL_VIDEO_BATCH
    return translated, usage


def analyze_video_visual_segment(source: dict, segment: dict) -> tuple[dict, dict]:
    """On-demand visual explanation for one timestamp range."""
    model = create_gemini_model(GEMINI_MODEL_VIDEO, GEMINI_API_KEY)
    uri = source.get("source_url") or (source.get("ingest_usage") or {}).get("gemini_file_uri")
    uploaded_name = ""
    if not uri and source.get("local_path") and Path(str(source["local_path"])).exists():
        uploaded = model.upload_file(str(source["local_path"]), source.get("mime_type") or "video/mp4")
        uploaded_name = str(getattr(uploaded, "name", "") or "")
        uploaded = model.wait_for_file(uploaded_name)
        uri = str(getattr(uploaded, "uri", "") or "")
    if not uri:
        raise ValueError("Nguồn video tạm đã hết hạn; hãy tải lại file để phân tích hình ảnh.")
    prompt = (
        f"Analyze only the visual context from {float(segment.get('start_seconds', 0)):.1f} to "
        f"{float(segment.get('end_seconds', 0)):.1f} seconds. Explain in Vietnamese how the visible scene, "
        "onscreen text, gestures, or speaker changes help understand the transcript. Return JSON only with "
        "{summary:string, visual_cues:[{timestamp,description,learning_value}], onscreen_text:[string], warnings:[string]}."
    )
    try:
        response = model.create_interaction([
            {"type": "text", "text": prompt},
            {"type": "video", "uri": uri, "mime_type": source.get("mime_type") or "video/mp4", "resolution": "low"},
        ])
        result = _parse_json(_response_text(response))
        usage = response_usage(response)
        usage["model_used"] = GEMINI_MODEL_VIDEO
        return result, usage
    finally:
        if uploaded_name:
            try:
                model.delete_file(uploaded_name)
            except Exception:
                pass


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
        language = str(row.get("language") or "unknown")
        if language not in {"japanese", "english"}:
            language, _, _ = detect_sentence_language(text, "english")
        result.append({
            "start": start, "end": max(start, end), "text": text,
            "speaker": str(row.get("speaker") or ""), "language": language,
            "translation_vi": str(row.get("translation_vi") or "").strip(),
            "transcript_provider": str(row.get("transcript_provider") or ""),
            "translation_provider": str(row.get("translation_provider") or ""),
            "warning": str(row.get("warning") or ""),
            "original_source_text": str(row.get("original_source_text") or text),
            "confidence": str(row.get("confidence") or "unknown"),
            "verification_status": str(row.get("verification_status") or "unverified"),
            "uncertainty_reason": str(row.get("uncertainty_reason") or ""),
            "revision": int(row.get("revision", 0) or 0),
            "source_window_index": int(row.get("source_window_index", 0) or 0),
            "recheck": row.get("recheck") if isinstance(row.get("recheck"), dict) else {},
        })
    return sorted(result, key=lambda row: (row["start"], row["end"]))


def transcript_rows_to_cues(rows: list[dict], source_id: str, provider: str = "") -> list[dict]:
    """Convert external caption/ASR rows into stable UI and persistence records."""
    cues = []
    for ordinal, row in enumerate(normalize_transcript(rows), 1):
        value = "|".join((
            source_id, str(ordinal), f"{row['start']:.3f}", f"{row['end']:.3f}", row["text"],
        ))
        translation = str(row.get("translation_vi") or "")
        cues.append({
            "cue_id": hashlib.sha256(value.encode("utf-8")).hexdigest()[:24],
            "ordinal": ordinal, "start_seconds": row["start"], "end_seconds": row["end"],
            "speaker": row.get("speaker") or "", "language": row.get("language") or "unknown",
            "source_text": row["text"], "translation_vi": translation,
            "transcript_provider": row.get("transcript_provider") or provider,
            "translation_provider": row.get("translation_provider") or "",
            "status": "translated" if translation else "translation_pending",
            "warning": row.get("warning") or "",
            "original_source_text": row.get("original_source_text") or row["text"],
            "confidence": row.get("confidence") or "unknown",
            "verification_status": row.get("verification_status") or "unverified",
            "uncertainty_reason": row.get("uncertainty_reason") or "",
            "revision": int(row.get("revision", 0) or 0),
            "source_window_index": int(row.get("source_window_index", 0) or 0),
            "recheck": row.get("recheck") if isinstance(row.get("recheck"), dict) else {},
        })
    return cues


def cues_to_transcript_rows(cues: list[dict]) -> list[dict]:
    return normalize_transcript([
        {
            "start": cue.get("start_seconds", 0), "end": cue.get("end_seconds", 0),
            "text": cue.get("source_text", ""), "speaker": cue.get("speaker", ""),
            "language": cue.get("language", "unknown"),
            "translation_vi": cue.get("translation_vi", ""),
            "transcript_provider": cue.get("transcript_provider", ""),
            "translation_provider": cue.get("translation_provider", ""),
            "warning": cue.get("warning", ""),
            "original_source_text": cue.get("original_source_text", ""),
            "confidence": cue.get("confidence", "unknown"),
            "verification_status": cue.get("verification_status", "unverified"),
            "uncertainty_reason": cue.get("uncertainty_reason", ""),
            "revision": cue.get("revision", 0),
            "source_window_index": cue.get("source_window_index", 0),
            "recheck": cue.get("recheck") if isinstance(cue.get("recheck"), dict) else {},
        }
        for cue in cues if isinstance(cue, dict)
    ])


def _normalized_cue_text(value: object) -> str:
    return re.sub(r"[^0-9a-zぁ-んァ-ン一-龯々]+", "", str(value or "").lower())


def _cue_similarity(left: object, right: object) -> float:
    first, second = _normalized_cue_text(left), _normalized_cue_text(right)
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def merge_transcript_cues(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Fuzzy-align overlapping ASR windows while surfacing real conflicts."""
    merged = [dict(row) for row in existing if isinstance(row, dict)]
    confidence_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    for candidate in incoming:
        if not isinstance(candidate, dict) or not str(candidate.get("source_text") or "").strip():
            continue
        candidate_start = float(candidate.get("start_seconds", 0) or 0)
        nearby = [
            row for row in merged
            if abs(float(row.get("start_seconds", 0) or 0) - candidate_start) <= TRANSCRIPT_OVERLAP_SECONDS + 1
        ]
        best = max(
            nearby,
            key=lambda row: _cue_similarity(row.get("source_text"), candidate.get("source_text")),
            default=None,
        )
        similarity = _cue_similarity(best.get("source_text"), candidate.get("source_text")) if best else 0.0
        if best is not None and similarity >= 0.78:
            best_rank = confidence_rank.get(str(best.get("confidence") or "unknown"), 0)
            candidate_rank = confidence_rank.get(str(candidate.get("confidence") or "unknown"), 0)
            if candidate_rank > best_rank or (
                candidate_rank == best_rank
                and len(str(candidate.get("source_text") or "")) > len(str(best.get("source_text") or ""))
            ):
                preserved_translation = best.get("translation_vi") or candidate.get("translation_vi") or ""
                original = best.get("original_source_text") or best.get("source_text") or ""
                best.update(candidate)
                best["translation_vi"] = preserved_translation
                best["original_source_text"] = original
            if not best.get("translation_vi") and candidate.get("translation_vi"):
                best["translation_vi"] = candidate["translation_vi"]
                best["translation_provider"] = candidate.get("translation_provider") or "gemini_audio"
            best["start_seconds"] = min(float(best.get("start_seconds", 0)), candidate_start)
            best["end_seconds"] = max(
                float(best.get("end_seconds", 0)), float(candidate.get("end_seconds", 0))
            )
            if similarity < 0.94:
                best["verification_status"] = "needs_review"
                best["uncertainty_reason"] = "Hai cửa sổ audio nhận dạng hơi khác nhau."
            continue
        conflict = next(
            (
                row for row in nearby
                if 0.35 <= _cue_similarity(row.get("source_text"), candidate.get("source_text")) < 0.78
                and abs(float(row.get("start_seconds", 0)) - candidate_start) <= 2
            ),
            None,
        )
        if conflict:
            conflict["verification_status"] = "needs_review"
            conflict["uncertainty_reason"] = "Hai cửa sổ audio cho lời thoại xung đột."
            candidate = {
                **candidate,
                "verification_status": "needs_review",
                "uncertainty_reason": "Hai cửa sổ audio cho lời thoại xung đột.",
            }
        merged.append(dict(candidate))
    merged.sort(key=lambda row: (float(row.get("start_seconds", 0)), float(row.get("end_seconds", 0))))
    for ordinal, row in enumerate(merged, 1):
        row["ordinal"] = ordinal
    return merged

def clean_transcript(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Conservatively remove exact ASR repeats and standalone fillers without rewriting source words."""
    cleaned: list[dict] = []
    warnings: list[str] = []
    fillers = {"um", "uh", "erm", "えー", "ええと", "あのー"}
    for row in normalize_transcript(rows):
        comparable = re.sub(r"[\s,.!?。、！？]+", "", row["text"].lower())
        if comparable in fillers:
            warnings.append(f"Từ đệm tại {format_timestamp(row['start'])}: {row['text']}")
        if cleaned:
            previous = re.sub(r"[\s,.!?。、！？]+", "", cleaned[-1]["text"].lower())
            if comparable and comparable == previous and row["start"] <= cleaned[-1]["end"] + 2:
                warnings.append(f"Đã gộp đoạn ASR lặp tại {format_timestamp(row['start'])}.")
                cleaned[-1]["end"] = max(cleaned[-1]["end"], row["end"])
                continue
        cleaned.append(dict(row))
    return cleaned, warnings


def transcript_hash(rows: list[dict]) -> str:
    """Hash only accepted source speech, independent of translation and review metadata."""
    canonical = [
        {
            "start": round(float(row.get("start", row.get("start_seconds", 0)) or 0), 3),
            "end": round(float(row.get("end", row.get("end_seconds", 0)) or 0), 3),
            "speaker": str(row.get("speaker") or ""),
            "language": str(row.get("language") or "unknown"),
            "text": str(row.get("text") or row.get("source_text") or "").strip(),
        }
        for row in rows if isinstance(row, dict)
        and str(row.get("text") or row.get("source_text") or "").strip()
    ]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_segments(
    rows: list[dict], window_seconds: int = 180, *, namespace: str = "",
) -> list[dict]:
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
            "segment_id": str(hashlib.sha256(
                f"{namespace}|{index}|{bucket[0]['start']}|{text}".encode("utf-8")
            ).hexdigest()[:24]),
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
        result = normalize_video_segment_result(by_id.get(str(segment.get("segment_id"))) or {})
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
    cues = source.get("video_cues") or transcript_rows_to_cues(
        source.get("clean_transcript") or [], str(source.get("source_id") or "video"),
        str(source.get("transcript_provider") or ""),
    )
    if cues:
        lines.extend([
            "", "## Script và bản dịch đồng bộ", "",
            "| Thời gian | Nguyên văn | Bản dịch tiếng Việt |",
            "|---|---|---|",
        ])
        for cue in cues:
            original = str(cue.get("source_text") or "").replace("|", "\\|").replace("\n", " ")
            translation = str(cue.get("translation_vi") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {format_timestamp(cue.get('start_seconds', 0))} | {original} | {translation} |")
    lines.extend(["", "## Mục lục video", ""])
    for segment in segments:
        lines.append(
            f"- {format_timestamp(segment.get('start_seconds', 0))}–{format_timestamp(segment.get('end_seconds', 0))}: "
            f"{segment.get('title') or 'Đoạn'}"
        )
    for segment in segments:
        analysis = normalize_video_segment_result(segment.get("analysis"))
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
                for row in _as_record_list(rows, "value"):
                    lines.append("- " + " | ".join(str(value) for value in row.values() if value not in (None, "", [])))
        for label, key in (("Từ nối", "connectors"), ("Discourse markers", "discourse_markers"), ("Mẫu câu", "sentence_patterns")):
            rows = analysis.get(key) or []
            if rows:
                lines.extend(["", f"### {label}"])
                for row in _as_record_list(rows, "value"):
                    lines.append("- " + " | ".join(str(value) for value in row.values() if value not in (None, "", [])))
        breakdown = analysis.get("sentence_breakdown") or {}
        if breakdown:
            lines.extend(["", "### Giải mã câu dài", "", json.dumps(breakdown, ensure_ascii=False, indent=2)])
        visual = analysis.get("visual_context_detail") or {}
        if visual:
            lines.extend(["", "### Bối cảnh hình ảnh", "", str(visual.get("summary") or "")])
            for cue in _as_record_list(visual.get("visual_cues"), "description"):
                lines.append("- " + " | ".join(str(value) for value in cue.values() if value not in (None, "", [])))
    return "\n".join(lines).strip()


def build_video_analysis(source: dict, segments: list[dict]) -> dict:
    completed = [
        {**segment, "analysis": normalize_video_segment_result(segment.get("analysis"))}
        for segment in segments if _as_mapping(segment.get("analysis"))
    ]
    video_cues = source.get("video_cues") or transcript_rows_to_cues(
        source.get("clean_transcript") or [], str(source.get("source_id") or "video"),
        str(source.get("transcript_provider") or ""),
    )
    usage_runs = []
    ingest_usage = _as_mapping(source.get("ingest_usage"))
    ingest_runs = [run for run in ingest_usage.get("runs") or [] if isinstance(run, dict)]
    if ingest_runs:
        for run in ingest_runs:
            usage_runs.append({**run, "stage": run.get("stage") or "video_transcription"})
    elif ingest_usage and sum(int(ingest_usage.get(key, 0) or 0) for key in ("input_tokens", "output_tokens")):
        usage_runs.append({
            "run_id": f"{source.get('source_id')}:ingest",
            "model_used": ingest_usage.get("model_used") or GEMINI_MODEL_VIDEO,
            "usage": ingest_usage,
            "stage": "video_ingest",
        })
    for run in source.get("translation_runs") or []:
        if isinstance(run, dict):
            usage_runs.append({**run, "stage": "caption_translation"})
    for segment in completed:
        usage = _as_mapping(segment.get("usage"))
        bulk_usage = {key: value for key, value in usage.items() if key != "deep_sentence_usage"}
        usage_runs.append({
            "run_id": usage.get("run_id") or f"{segment['segment_id']}:bulk",
            "model_used": bulk_usage.get("model_used") or GEMINI_MODEL_VIDEO_BATCH,
            "usage": bulk_usage,
            "stage": "segment_analysis",
        })
        analysis_row = normalize_video_segment_result(segment.get("analysis"))
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
        row = normalize_video_segment_result(segment.get("analysis"))
        segment_cues = [
            cue for cue in video_cues
            if float(cue.get("end_seconds", 0) or 0) >= float(segment.get("start_seconds", 0) or 0)
            and float(cue.get("start_seconds", 0) or 0) <= float(segment.get("end_seconds", 0) or 0)
        ]
        catalog = [
            {
                "sentence_id": cue.get("cue_id") or f"p{fallback_index}-s{index}",
                "ordinal": index, "original": cue.get("source_text") or "",
                "detected_language": cue.get("language") or segment.get("language") or "unknown",
                "language_confidence": 1.0, "language_source": "video_cue",
                "video_start_seconds": cue.get("start_seconds", 0),
            }
            for index, cue in enumerate(segment_cues, 1)
        ]
        if not catalog:
            catalog = split_sentences(
                str(segment.get("clean_text") or ""), str(segment.get("language") or "english"), fallback_index
            )
        timestamp_url = ""
        if source.get("source_url"):
            timestamp_url = f"{source['source_url']}&t={int(segment.get('start_seconds', 0) or 0)}s"
        turns = _as_record_list(row.get("dialogue_turns"), "text")
        guidance = []
        for sentence in catalog:
            original = str(sentence.get("original") or "")
            matched_cue = next(
                (cue for cue in segment_cues if str(cue.get("cue_id")) == str(sentence.get("sentence_id"))),
                {},
            )
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
            sentence_start = float(matched_cue.get("start_seconds", segment.get("start_seconds", 0)) or 0)
            sentence_url = (
                f"{source['source_url']}&t={int(sentence_start)}s" if source.get("source_url") else ""
            )
            sentence["timestamp_url"] = sentence_url
            sentence["video_start_seconds"] = sentence_start
            guidance.append({
                "sentence_id": sentence.get("sentence_id"),
                "original": original,
                "detected_language": sentence.get("detected_language") or segment.get("language"),
                "translations": {
                    "natural": matched_cue.get("translation_vi") or matched_turn.get("translation_vi") or (
                        row.get("natural_translation") if len(catalog) == 1 else ""
                    )
                },
                "key_points": row.get("key_points") or [],
                "timestamp_url": sentence_url,
                "video_start_seconds": sentence_start,
            })
        breakdown = _as_mapping(row.get("sentence_breakdown"))
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
            "usage": _as_mapping(segment.get("usage")),
        })
        page_analyses.append(row)
    analysis = {
        "analysis_type": "video", "analysis_mode": "video_balanced",
        "analysis_language": "mixed" if len({row.get('language') for row in completed}) > 1 else (completed[0].get("language") if completed else "unknown"),
        "summary": " ".join(
            str(normalize_video_segment_result(row.get("analysis")).get("summary") or "")
            for row in completed
        ).strip(),
        "video_source": {key: source.get(key) for key in ("source_id", "source_kind", "source_url", "video_id", "file_name", "duration_seconds", "transcript_provider", "transcript_hash")},
        "video_metadata": source.get("metadata") or {},
        "transcript_clean": source.get("clean_transcript") or [],
        "video_cues": video_cues,
        "transcript_warnings": source.get("transcript_warnings") or [],
        "video_segments": completed,
        "page_analyses": page_analyses,
        "video_ingest_usage": source.get("ingest_usage") or {},
        "video_analysis_runs": usage_runs,
        "model_used": GEMINI_MODEL_VIDEO_BATCH,
    }
    analysis["full_markdown"] = video_analysis_markdown({**source, "video_cues": video_cues}, completed)
    return analysis
