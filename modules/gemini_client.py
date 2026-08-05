"""Small compatibility adapter around Google's supported Gen AI SDK."""

from __future__ import annotations

from typing import Any


class GeminiModel:
    """Expose a minimal ``generate_content`` API used by existing modules."""

    def __init__(self, model_name: str, api_key: str) -> None:
        from google import genai

        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

    def generate_content(
        self,
        contents: Any,
        generation_config: dict[str, Any] | None = None,
    ) -> Any:
        from google.genai import types

        raw_config = dict(generation_config or {})
        thinking = raw_config.pop("thinking_config", None)

        # New Gemini generations reject deprecated sampling parameters.
        if self.model_name.startswith(("gemini-3.5-flash-lite", "gemini-3.6-")):
            raw_config.pop("temperature", None)
            raw_config.pop("top_p", None)
            raw_config.pop("top_k", None)

        if thinking:
            if self.model_name.startswith("gemini-3"):
                raw_config["thinking_config"] = types.ThinkingConfig(thinking_level="high")
            else:
                raw_config["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=int(thinking.get("thinking_budget", 0))
                )

        config = types.GenerateContentConfig(**raw_config) if raw_config else None
        return self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )


def create_gemini_model(model_name: str, api_key: str | None) -> GeminiModel:
    """Create a configured model or fail with a user-facing message."""
    if not api_key:
        raise ValueError("Thiếu GEMINI_API_KEY. Hãy cấu hình key trong Streamlit secrets.")
    return GeminiModel(model_name, api_key)
