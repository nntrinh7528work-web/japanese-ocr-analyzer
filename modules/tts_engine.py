"""Text-to-Speech engine using gTTS (free Google Translate TTS).

Provides audio generation for Japanese dialogue turns.
Includes caching to avoid regenerating audio for the same text.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def text_to_speech(text: str, lang: str = "ja", slow: bool = False) -> Optional[bytes]:
    """Convert text to speech audio bytes (MP3 format).

    Args:
        text: The text to convert to speech.
        lang: Language code (default: 'ja' for Japanese).
        slow: If True, speak slowly (useful for learners).

    Returns:
        MP3 audio bytes, or None if generation fails.
    """
    try:
        from gtts import gTTS
    except ImportError:
        logger.warning("gTTS is not installed. Run: pip install gTTS")
        return None

    if not text or not text.strip():
        return None

    try:
        tts = gTTS(text=text.strip(), lang=lang, slow=slow)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logger.error("TTS generation failed: %s", e)
        return None


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
        audio = text_to_speech(text, lang=lang, slow=slow)
        results.append(audio)
    return results


def get_audio_cache_key(text: str, lang: str, slow: bool) -> str:
    """Generate a cache key for a TTS request."""
    raw = f"{text}|{lang}|{slow}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
