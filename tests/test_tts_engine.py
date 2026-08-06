"""Tests for modules/tts_engine.py."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch, MagicMock


class TestTextToSpeech(unittest.TestCase):
    """Tests for the text_to_speech function."""

    def _setup_edge_tts_mock(self):
        """Create a fake edge_tts module with mock stream & Communicate class."""
        mock_instance = MagicMock()
        
        async def mock_stream():
            yield {"type": "audio", "data": b"fake_audio_data"}
            
        mock_instance.stream = mock_stream
        mock_comm_cls = MagicMock(return_value=mock_instance)

        fake_module = types.ModuleType("edge_tts")
        fake_module.Communicate = mock_comm_cls
        return fake_module, mock_comm_cls

    def test_basic_generation(self):
        """Test that audio bytes are returned for valid text."""
        from modules.tts_engine import text_to_speech

        fake_module, mock_cls = self._setup_edge_tts_mock()
        with patch.dict(sys.modules, {"edge_tts": fake_module}):
            result = text_to_speech("こんにちは", lang="ja", slow=False)

        self.assertIsNotNone(result)
        self.assertEqual(result, b"fake_audio_data")
        mock_cls.assert_called_once_with("こんにちは", "ja-JP-NanamiNeural", rate="+0%")

    def test_empty_text_returns_none(self):
        """Test that empty or whitespace text returns None."""
        from modules.tts_engine import text_to_speech

        self.assertIsNone(text_to_speech(""))
        self.assertIsNone(text_to_speech("   "))
        self.assertIsNone(text_to_speech(None))

    def test_slow_mode(self):
        """Test that slow mode parameter is passed correctly."""
        from modules.tts_engine import text_to_speech

        fake_module, mock_cls = self._setup_edge_tts_mock()
        with patch.dict(sys.modules, {"edge_tts": fake_module}):
            text_to_speech("テスト", lang="ja", slow=True)
            mock_cls.assert_called_once_with("テスト", "ja-JP-NanamiNeural", rate="-20%")

    def test_vietnamese_voice(self):
        from modules.tts_engine import text_to_speech

        fake_module, mock_cls = self._setup_edge_tts_mock()
        with patch.dict(sys.modules, {"edge_tts": fake_module}):
            result = text_to_speech("Đây là bản dịch.", lang="vi")

        self.assertEqual(result, b"fake_audio_data")
        mock_cls.assert_called_once_with(
            "Đây là bản dịch.", "vi-VN-HoaiMyNeural", rate="+0%"
        )


class TestGenerateDialogueAudio(unittest.TestCase):
    """Tests for generate_dialogue_audio function."""

    @patch("modules.tts_engine.text_to_speech")
    def test_generates_audio_per_turn(self, mock_tts):
        """Test that audio is generated for each turn."""
        from modules.tts_engine import generate_dialogue_audio

        mock_tts.side_effect = [b"audio1", b"audio2"]
        dialogue = [
            {"speaker": "A", "text": "こんにちは"},
            {"speaker": "B", "text": "こんにちは、元気ですか"},
        ]
        result = generate_dialogue_audio(dialogue, lang="ja", slow=False)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], b"audio1")
        self.assertEqual(result[1], b"audio2")
        self.assertEqual(mock_tts.call_count, 2)

    @patch("modules.tts_engine.text_to_speech")
    def test_handles_empty_text(self, mock_tts):
        """Test turns with missing text."""
        from modules.tts_engine import generate_dialogue_audio

        mock_tts.return_value = None
        dialogue = [{"speaker": "A", "text": ""}]
        result = generate_dialogue_audio(dialogue, lang="ja", slow=False)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0])


class TestGetAudioCacheKey(unittest.TestCase):
    """Tests for get_audio_cache_key function."""

    def test_different_inputs_different_keys(self):
        from modules.tts_engine import get_audio_cache_key

        key1 = get_audio_cache_key("hello", "ja", False)
        key2 = get_audio_cache_key("hello", "ja", True)
        key3 = get_audio_cache_key("world", "ja", False)
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_same_inputs_same_key(self):
        from modules.tts_engine import get_audio_cache_key

        key1 = get_audio_cache_key("test", "ja", False)
        key2 = get_audio_cache_key("test", "ja", False)
        self.assertEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
