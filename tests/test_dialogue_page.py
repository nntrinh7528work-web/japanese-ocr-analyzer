from streamlit.testing.v1 import AppTest


def test_pending_dialogue_suggestions_are_applied_before_widgets_exist():
    """Suggested values must not mutate an already-instantiated Streamlit widget."""
    app = AppTest.from_file("app.py")
    app.session_state["dialogue_pending_topic"] = "Đổi lịch hẹn"
    app.session_state["dialogue_pending_targets"] = {"vocab": "予約", "grammar": "〜てもらう"}
    app.run(timeout=30)

    assert not app.exception
    topic = next(item for item in app.text_input if item.label == "Chủ đề")
    vocab = next(item for item in app.text_area if item.label.startswith("Từ vựng muốn luyện"))
    grammar = next(item for item in app.text_area if item.label.startswith("Ngữ pháp/cấu trúc"))
    assert topic.value == "Đổi lịch hẹn"
    assert vocab.value == "予約"
    assert grammar.value == "〜てもらう"
