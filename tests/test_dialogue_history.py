import uuid
import pytest
from modules.dialogue_history import (
    save_dialogue_session,
    get_practice_history,
    get_due_sm2_cards,
    update_sm2_card,
    get_streak_days,
)

def test_dialogue_history_crud():
    unique_word = f"test_word_{uuid.uuid4().hex[:6]}"
    sample = {
        "topic": "Mua sắm",
        "language": "Tiếng Nhật",
        "level": "Sơ cấp",
        "situation": "Siêu thị",
        "politeness_level": "Lịch sự (です/ます)",
        "dialogue": [{"speaker": "A", "text": "いくらですか。"}],
        "coverage_check": {unique_word: True},
        "summary": f"{unique_word}: bao nhiêu tiền",
    }

    session_id = save_dialogue_session(sample, quiz_score=100)
    assert session_id > 0

    history = get_practice_history(limit=5)
    assert len(history) > 0
    assert history[0]["topic"] == "Mua sắm"

    cards = get_due_sm2_cards()
    assert len(cards) > 0
    target_card = [c for c in cards if c["word"] == unique_word][0]

    updated = update_sm2_card(target_card["id"], quality_rating=4)
    assert updated["new_interval"] == 1
    assert updated["new_repetition"] == 1

    streak = get_streak_days()
    assert streak >= 1

