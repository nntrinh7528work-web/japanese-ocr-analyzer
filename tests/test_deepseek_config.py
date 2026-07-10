"""Tests for DeepSeek / AI pipeline configuration."""

from __future__ import annotations

import importlib


def test_deepseek_api_key_from_env(monkeypatch):
    """DEEPSEEK_API_KEY is read from environment."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    import config

    importlib.reload(config)
    assert config.DEEPSEEK_API_KEY == "test-key"


def test_deepseek_model_default(monkeypatch):
    """DEEPSEEK_MODEL defaults to deepseek-v4-pro."""
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    import config

    importlib.reload(config)
    assert config.DEEPSEEK_MODEL == "deepseek-v4-pro"


def test_deepseek_api_key_empty_when_missing(monkeypatch):
    """DEEPSEEK_API_KEY defaults to empty string when not set."""
    from unittest.mock import patch

    import config

    # Patch at the source (dotenv module) so that when reload re-executes
    # ``from dotenv import load_dotenv``, it picks up the mock.
    with patch("dotenv.load_dotenv"):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        importlib.reload(config)
    assert config.DEEPSEEK_API_KEY == ""


def test_config_imports_without_key(monkeypatch):
    """Config module loads without errors when DEEPSEEK_API_KEY is absent."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    import config

    importlib.reload(config)
    # Module should be importable; basic attributes exist.
    assert hasattr(config, "DEEPSEEK_API_KEY")
    assert hasattr(config, "DEEPSEEK_MODEL")
    assert hasattr(config, "GEMINI_REVIEW_MODEL")
    assert hasattr(config, "AI_PIPELINE_ENABLED")


def test_ai_pipeline_enabled_default(monkeypatch):
    """AI_PIPELINE_ENABLED defaults to True."""
    monkeypatch.delenv("AI_PIPELINE_ENABLED", raising=False)
    import config

    importlib.reload(config)
    assert config.AI_PIPELINE_ENABLED is True


def test_ai_pipeline_disabled(monkeypatch):
    """AI_PIPELINE_ENABLED can be set to false."""
    monkeypatch.setenv("AI_PIPELINE_ENABLED", "false")
    import config

    importlib.reload(config)
    assert config.AI_PIPELINE_ENABLED is False
