"""Text-to-Speech engine using edge-tts (expressive Azure neural voices).

Provides audio generation for Japanese, English, and Vietnamese learning text.
Includes caching to avoid regenerating audio for the same text.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)
_LAST_TTS_ERROR: ContextVar[str] = ContextVar("last_tts_error", default="")
TTS_TIMEOUT_SECONDS = 30

# Highly natural Microsoft Edge Neural Voices
VOICES = {
    "ja": {
        "A": "ja-JP-NanamiNeural",   # Natural Female
        "B": "ja-JP-KeitaNeural",    # Natural Male
        "default": "ja-JP-NanamiNeural"
    },
    "en": {
        "A": "en-US-EmmaNeural",     # Natural Female (US)
        "B": "en-US-BrianNeural",    # Natural Male (US)
        "default": "en-US-EmmaNeural"
    },
    "vi": {
        "A": "vi-VN-HoaiMyNeural",
        "B": "vi-VN-NamMinhNeural",
        "default": "vi-VN-HoaiMyNeural"
    }
}


async def _edge_tts_async(text: str, voice: str, rate: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    return data


def text_to_speech(
    text: str,
    lang: str = "ja",
    slow: bool = False,
    speaker: str = "default",
) -> Optional[bytes]:
    """Convert text to speech audio bytes (MP3 format) using natural Edge neural voices.

    Args:
        text: The text to convert to speech.
        lang: Language code ('ja' or 'en').
        slow: If True, slow down the voice rate.
        speaker: Speaker identifier ('A', 'B', or 'default').

    Returns:
        MP3 audio bytes, or None if generation fails.
    """
    if not text or not text.strip():
        _LAST_TTS_ERROR.set("Văn bản đọc đang rỗng.")
        return None
    _LAST_TTS_ERROR.set("")

    # Normalise language code
    normalized_lang = str(lang or "ja").lower()
    lang_key = "vi" if "vi" in normalized_lang else ("en" if "en" in normalized_lang else "ja")
    
    # Select voice
    voice_map = VOICES.get(lang_key, VOICES["ja"])
    voice = voice_map.get(speaker, voice_map["default"])
    
    # Speech rate
    rate = "-20%" if slow else "+0%"

    try:
        # Run async communicate in synchronous wrapper
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio = loop.run_until_complete(
                asyncio.wait_for(
                    _edge_tts_async(text.strip(), voice, rate),
                    timeout=TTS_TIMEOUT_SECONDS,
                )
            )
            if not audio:
                raise RuntimeError("Edge TTS không trả về dữ liệu audio.")
            return audio
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        logger.error("Edge TTS generation failed: %s", e)
        # Fallback to gTTS if edge-tts fails for some reason
        try:
            from gtts import gTTS
            gtts_lang = lang_key
            tts = gTTS(
                text=text.strip(),
                lang=gtts_lang,
                slow=slow,
                timeout=TTS_TIMEOUT_SECONDS,
            )
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            audio = buffer.read()
            if not audio:
                raise RuntimeError("gTTS không trả về dữ liệu audio.")
            return audio
        except Exception as fallback_err:
            logger.error("Fallback gTTS also failed: %s", fallback_err)
            _LAST_TTS_ERROR.set(
                f"Edge TTS: {e}. Dự phòng gTTS: {fallback_err}."
            )
            return None


def get_last_tts_error() -> str:
    """Return a user-safe diagnostic for the latest TTS call in this context."""
    return _LAST_TTS_ERROR.get()


def generate_dialogue_audio(
    dialogue: list[dict],
    lang: str = "ja",
    slow: bool = False,
) -> list[Optional[bytes]]:
    """Generate audio for each turn in a dialogue.

    Args:
        dialogue: List of dialogue turn dicts with 'text' key.
        lang: Language code.
        slow: Whether to use slow speech.

    Returns:
        List of audio bytes (one per turn), None for failed turns.
    """
    results = []
    for turn in dialogue:
        text = turn.get("text", "")
        speaker = turn.get("speaker", "default")
        audio = text_to_speech(text, lang=lang, slow=slow, speaker=speaker)
        results.append(audio)
    return results


def generate_full_dialogue_audio(
    dialogue: list[dict],
    lang: str = "ja",
    slow: bool = False,
) -> Optional[bytes]:
    """Generate one valid MP3 for the full dialogue, with short turn pauses.

    MP3 files cannot safely be joined with ``b"".join``: browsers, especially on
    mobile, may stop at the first stream.  Decode and re-export the segments when
    ffmpeg is present; a single-voice Edge/gTTS fallback still gives every user a
    playable full-dialogue control.
    """
    audios = []
    full_text = []
    for turn in dialogue:
        text = turn.get("text", "")
        if text.strip():
            speaker = turn.get("speaker", "default")
            audio = text_to_speech(text, lang=lang, slow=slow, speaker=speaker)
            if audio:
                audios.append(audio)
            full_text.append(text.strip())
    if audios:
        try:
            from pydub import AudioSegment

            merged = AudioSegment.empty()
            pause = AudioSegment.silent(duration=350)
            for audio in audios:
                merged += AudioSegment.from_file(io.BytesIO(audio), format="mp3") + pause
            output = io.BytesIO()
            merged.export(output, format="mp3", bitrate="128k")
            data = output.getvalue()
            if data:
                return data
        except Exception as exc:
            logger.warning("Could not merge dialogue MP3 segments: %s", exc)
            _LAST_TTS_ERROR.set("Không ghép được hai giọng; đang dùng bản đọc một giọng để bảo đảm audio phát được.")
            fallback = text_to_speech(" ".join(full_text), lang=lang, slow=slow, speaker="default")
            if fallback:
                return fallback
            return None
    return None


def get_audio_cache_key(text: str, lang: str, slow: bool, speaker: str = "default") -> str:
    """Generate a cache key for a TTS request."""
    raw = f"{text}|{lang}|{slow}|{speaker}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
