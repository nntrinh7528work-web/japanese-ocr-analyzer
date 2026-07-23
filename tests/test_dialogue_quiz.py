import pytest
from modules.dialogue_quiz import generate_pure_quiz

def test_generate_pure_quiz():
    result = {
        "topic": "Nhà hàng",
        "dialogue": [
            {
                "speaker": "A",
                "text": "予約をしたいのですが。",
                "text_vi": "Tôi muốn đặt bàn.",
                "highlights": ["予約"]
            },
            {
                "speaker": "B",
                "text": "かしこまりました。",
                "text_vi": "Vâng ạ.",
                "highlights": []
            }
        ],
        "summary": "予約 (yoyaku): đặt chỗ"
    }

    quiz = generate_pure_quiz(result)
    assert "cloze" in quiz
    assert "mcq" in quiz
    assert "reorder" in quiz
    assert "translate" in quiz
    assert len(quiz["cloze"]) > 0
    assert quiz["cloze"][0]["target_word"] == "予約"
