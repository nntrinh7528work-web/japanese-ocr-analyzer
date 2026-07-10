"""Tests for DeepSeek client — all API calls are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Import the module so patch() can resolve attributes on it.
import modules.deepseek_client as ds_client


# ---------------------------------------------------------------------------
# A. No API key -> ValueError
# ---------------------------------------------------------------------------
def test_get_client_raises_without_key():
    """get_deepseek_client raises ValueError when DEEPSEEK_API_KEY is empty."""
    with patch.object(ds_client, "DEEPSEEK_API_KEY", ""):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            ds_client.get_deepseek_client()


# ---------------------------------------------------------------------------
# B. With key -> OpenAI called with correct params
# ---------------------------------------------------------------------------
def test_get_client_creates_openai_with_correct_params():
    """OpenAI client is instantiated with api_key and base_url."""
    with patch.object(ds_client, "DEEPSEEK_API_KEY", "test-key"):
        with patch.object(ds_client, "OpenAI") as mock_openai:
            ds_client.get_deepseek_client()
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.deepseek.com",
            )


# ---------------------------------------------------------------------------
# C. Mock chat completion returns PONG
# ---------------------------------------------------------------------------
def test_ping_deepseek_returns_pong():
    """ping_deepseek returns 'PONG' from mocked completion."""
    mock_message = SimpleNamespace(content="PONG")
    mock_choice = SimpleNamespace(message=mock_message)
    mock_response = SimpleNamespace(choices=[mock_choice])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(ds_client, "DEEPSEEK_API_KEY", "test-key"):
        with patch.object(ds_client, "OpenAI", return_value=mock_client):
            result = ds_client.ping_deepseek()
            assert result == "PONG"


# ---------------------------------------------------------------------------
# D. Assert model matches DEEPSEEK_MODEL
# ---------------------------------------------------------------------------
def test_ping_uses_correct_model():
    """ping_deepseek passes DEEPSEEK_MODEL to the API call."""
    mock_message = SimpleNamespace(content="PONG")
    mock_choice = SimpleNamespace(message=mock_message)
    mock_response = SimpleNamespace(choices=[mock_choice])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    test_model = "deepseek-v4-pro"

    with patch.object(ds_client, "DEEPSEEK_API_KEY", "test-key"):
        with patch.object(ds_client, "DEEPSEEK_MODEL", test_model):
            with patch.object(ds_client, "OpenAI", return_value=mock_client):
                ds_client.ping_deepseek()

                call_kwargs = mock_client.chat.completions.create.call_args
                assert call_kwargs.kwargs["model"] == test_model
