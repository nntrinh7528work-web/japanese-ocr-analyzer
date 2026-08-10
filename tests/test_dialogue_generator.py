from __future__ import annotations

import json

from modules import dialogue_generator


class _Response:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload, ensure_ascii=False)
        self.usage_metadata = type("Usage", (), {"prompt_token_count": 10, "candidates_token_count": 20, "thoughts_token_count": 0})()


class _Model:
    target_model_name = "test-model"

    def __init__(self, responses):
        self.responses = list(responses)

    def generate_content(self, prompt, generation_config=None):
        return _Response(self.responses.pop(0))


def _payload(turn_count=8):
    turns = []
    for index in range(turn_count):
        turns.append({"id": f"turn-{index + 1:02d}", "speaker_id": "A" if index % 2 == 0 else "B", "text": "Please confirm the booking." if index == 0 else "I understand and will help.", "reading": "", "translation_vi": "Dịch Việt", "speech_intent": "xác nhận", "register": "polite", "target_ids": ["vocabulary-1"] if index == 0 else [], "practice_chunks": ["Please", "confirm", "the booking."]})
    return {"roles": [{"id": "A", "name": "Customer"}, {"id": "B", "name": "Staff"}], "scenario": {}, "turns": turns, "learning_targets": [{"id": "vocabulary-1", "realized_form": "confirm", "turn_ids": ["turn-01"], "explanation_vi": "xác nhận"}], "summary": "", "notes": "", "quality": {}}


def test_generator_repairs_invalid_first_response(monkeypatch):
    model = _Model([_payload(turn_count=2), _payload(turn_count=8)])
    monkeypatch.setattr(dialogue_generator, "_init_model", lambda model_name=None: model)
    result = dialogue_generator.generate_dialogue("Booking", "Tiếng Anh", ["confirm"], [], "Sơ cấp")
    assert len(result["dialogue"]) == 8
    assert result["quality"]["quality_status"] == "complete"
    assert result["usage_detail"]["repair"]["input_tokens"] == 10


def test_variation_preserves_grammar_targets(monkeypatch):
    received = {}
    def fake_generate(**kwargs):
        received.update(kwargs)
        return kwargs
    monkeypatch.setattr(dialogue_generator, "generate_dialogue", fake_generate)
    dialogue_generator.generate_variation({"topic": "x", "language": "Tiếng Nhật", "learning_targets": [{"type": "vocabulary", "term": "予約"}, {"type": "grammar", "term": "〜てもらう"}], "dialogue": []})
    assert received["vocab"] == ["予約"]
    assert received["grammar"] == ["〜てもらう"]
