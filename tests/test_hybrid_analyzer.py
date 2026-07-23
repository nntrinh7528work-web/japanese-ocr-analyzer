"""Unit tests for modules/hybrid_analyzer.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import modules.hybrid_analyzer as hybrid_analyzer


@patch("modules.hybrid_analyzer.analyze_text_with_himotoki")
@patch("modules.hybrid_analyzer._init_model")
def test_hybrid_analysis(mock_init_model, mock_analyze_himotoki):
    # Mock Himotoki output
    mock_analyze_himotoki.return_value = {
        "vocabulary_all": [{"word": "テスト", "reading": "てすと", "dictionary_form": "テスト", "part_of_speech": "noun", "meaning": "test"}],
        "kanji_analysis": [],
        "grammar_points": []
    }
    
    # Mock Gemini model generate_content
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
## 1. Tóm tắt nội dung
Tóm tắt: Đây là một bài test.

## 2. Từ vựng
### 2.1. Bảng từ vựng đầy đủ
| # | Từ gốc | Phiên âm | Loại từ | Nghĩa tiếng Việt | Cấp JLPT |
|---|---|---|---|---|---|
| 1 | テスト | てすと | noun | kiểm tra | N5 |

### 2.2. Từ vựng khó — Giải thích chi tiết
**[[テスト]]**
- Loại từ: Noun
- Ý nghĩa: bài kiểm tra
- Cấu trúc/Cách dùng: Sử dụng trong thi cử
- Ví dụ trong bài: テスト
- Phân tích ví dụ: Câu test
- Trình độ: N5

## 3. PHÂN TÍCH KANJI
| Kanji | Onyomi | Kunyomi | Nghĩa cơ bản | JLPT | Từ vựng trong bài | Câu ví dụ trong bài | Vai trò trong từ |

## 4. TỪ NỐI CÂU & LIÊN TỪ
| Từ/Cụm | Phiên âm | Loại | Nghĩa tiếng Việt | Câu ví dụ trong bài | Vai trò ngữ nghĩa | Mức độ khó |

## 5. PHÂN TÍCH NGỮ PHÁP

## 6. MẪU CÂU ĐẶC TRƯNG

## 7. Trạng thái phân tích
Phân tích hoàn tất.
"""
    mock_model.generate_content.return_value = mock_response
    mock_init_model.return_value = mock_model
    
    result = hybrid_analyzer.run_hybrid_analysis("テスト")
    
    # Check we got enriched results
    assert result["confirmed_text"] == "テスト"
    assert "Đây là một bài test" in result["summary"]
    assert len(result["vocabulary_all"]) == 1
    assert result["vocabulary_all"][0]["meaning"] == "kiểm tra"
    assert result["vocabulary_all"][0]["jlpt"] == "N5"
    assert len(result["vocabulary_important"]) == 1
