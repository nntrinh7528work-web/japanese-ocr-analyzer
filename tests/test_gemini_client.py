from types import SimpleNamespace

from modules.gemini_client import GeminiModel


def test_adapter_removes_deprecated_sampling_for_flash_lite():
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="ok")

    model = GeminiModel.__new__(GeminiModel)
    model.model_name = "gemini-3.5-flash-lite"
    model._client = SimpleNamespace(models=Models())

    response = model.generate_content(
        "hello",
        {"temperature": 0.7, "thinking_config": {"thinking_budget": 4096}},
    )

    assert response.text == "ok"
    assert captured["config"].temperature is None
    assert captured["config"].thinking_config.thinking_level.value == "HIGH"


def test_adapter_keeps_supported_generation_config():
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="ok")

    model = GeminiModel.__new__(GeminiModel)
    model.model_name = "gemini-2.5-flash"
    model._client = SimpleNamespace(models=Models())

    model.generate_content("hello", {"temperature": 0.1, "max_output_tokens": 8192})

    assert captured["config"].temperature == 0.1
    assert captured["config"].max_output_tokens == 8192
