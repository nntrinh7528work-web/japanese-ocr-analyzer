import io
from types import SimpleNamespace

from PIL import Image

from modules import ocr_engine


RESPONSE = """TEXT_DIRECTION: vertical
TEXT_REGIONS: 2
HAS_FURIGANA: yes
CONFIDENCE: high

---OCR_START---
日本語《にほんご》です。【要確認: 語→語 | rõ】
---OCR_END---
---NOTES_START---
- Một ký tự cần xác nhận
---NOTES_END---
"""


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, "PNG")
    return buffer.getvalue()


def test_build_and_parse():
    prompt = ocr_engine.build_ocr_prompt({"quality_level": "good", "rotation_detected": 90, "issues": []})
    result = ocr_engine.parse_ocr_response(RESPONSE)

    assert "good" in prompt and "90°" in prompt
    assert result["text_direction"] == "vertical"
    assert result["text_regions"] == 2
    assert result["has_furigana"] is True
    assert "要確認" not in result["clean_text"]
    assert result["ocr_notes"] == ["Một ký tự cần xác nhận"]


def test_run_ocr_retries_and_usage(monkeypatch):
    calls = {"count": 0}

    class Model:
        def generate_content(self, _content):
            calls["count"] += 1
            if calls["count"] < 2:
                raise TimeoutError("temporary")
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
            )

    monkeypatch.setattr(ocr_engine, "init_gemini", lambda: Model())
    result = ocr_engine.run_ocr(png_bytes(), {"quality_level": "good"})

    assert calls["count"] == 2
    assert result["raw_text"]
    assert result["usage"]["input_tokens"] == 12
    assert result["usage"]["output_tokens"] == 34
    assert result["usage"]["candidate_tokens"] == 34
    assert result["usage"]["thinking_tokens"] == 0
