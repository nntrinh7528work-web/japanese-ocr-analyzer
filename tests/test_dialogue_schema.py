from __future__ import annotations

from modules.dialogue_schema import normalize_dialogue_payload


def _turns(language: str) -> list[dict]:
    rows = []
    for index in range(8):
        speaker = "A" if index % 2 == 0 else "B"
        text = "取ってもらえますか。" if language == "ja" and index == 0 else ("Hello, I can help you." if language == "en" else "はい、承知しました。")
        rows.append(
            {
                "id": f"turn-{index + 1:02d}", "speaker_id": speaker, "text": text,
                "reading": "とってもらえますか。" if language == "ja" and index == 0 else "はい、しょうちしました。",
                "translation_vi": "Bản dịch tiếng Việt.", "speech_intent": "trao đổi", "register": "lịch sự",
                "target_ids": ["grammar-1"] if index == 0 else [], "practice_chunks": ["取って", "もらえ", "ますか。"],
            }
        )
    return rows


def test_japanese_inflected_grammar_is_verified_from_realized_form():
    payload = {
        "roles": [{"id": "A", "name": "Khách"}, {"id": "B", "name": "Nhân viên"}],
        "scenario": {}, "turns": _turns("ja"),
        "learning_targets": [{"id": "grammar-1", "realized_form": "取ってもらえますか", "turn_ids": ["turn-01"], "explanation_vi": "nhờ ai làm giúp"}],
        "summary": "", "notes": "", "quality": {},
    }
    result = normalize_dialogue_payload(payload, topic="Nhà hàng", language="Tiếng Nhật", level="Sơ cấp", situation="Nhà hàng", politeness_level="Lịch sự", scenario_description="", vocab=[], grammar=["〜てもらう"])
    assert result["coverage_check"]["〜てもらう"] is True
    assert result["quality"]["quality_status"] == "complete"


def test_japanese_grammar_falls_back_to_inflected_stem_when_model_omits_form():
    payload = {
        "roles": [], "scenario": {}, "turns": _turns("ja"),
        "learning_targets": [{"id": "grammar-1", "turn_ids": ["turn-01"], "explanation_vi": "nhờ ai làm giúp"}],
        "summary": "", "notes": "", "quality": {},
    }
    result = normalize_dialogue_payload(payload, topic="x", language="Tiếng Nhật", level="x", situation="x", politeness_level="x", scenario_description="", vocab=[], grammar=["〜てもらう"])
    assert result["coverage_check"]["〜てもらう"] is True
    assert result["learning_targets"][0]["coverage_warning"]


def test_english_coverage_uses_word_boundaries_not_substrings():
    payload = {
        "roles": [], "scenario": {}, "turns": _turns("en"),
        "learning_targets": [{"id": "vocabulary-1", "realized_form": "he", "turn_ids": ["turn-01"], "explanation_vi": "anh ấy"}],
        "summary": "", "notes": "", "quality": {},
    }
    result = normalize_dialogue_payload(payload, topic="Help", language="Tiếng Anh", level="Sơ cấp", situation="", politeness_level="", scenario_description="", vocab=["he"], grammar=[])
    assert result["coverage_check"]["he"] is False
    assert "Chưa xác minh" in result["quality"]["issues"][-1]


def test_missing_reading_is_reported_for_japanese_but_not_english():
    japanese = {"roles": [], "scenario": {}, "turns": _turns("ja"), "learning_targets": [], "summary": "", "notes": ""}
    japanese["turns"][2]["reading"] = ""
    jp = normalize_dialogue_payload(japanese, topic="x", language="Tiếng Nhật", level="x", situation="x", politeness_level="x", scenario_description="", vocab=[], grammar=[])
    assert any("Thiếu cách đọc" in issue for issue in jp["quality"]["issues"])

    english = {"roles": [], "scenario": {}, "turns": _turns("en"), "learning_targets": [], "summary": "", "notes": ""}
    en = normalize_dialogue_payload(english, topic="x", language="Tiếng Anh", level="x", situation="x", politeness_level="x", scenario_description="", vocab=[], grammar=[])
    assert not any("Thiếu cách đọc" in issue for issue in en["quality"]["issues"])
