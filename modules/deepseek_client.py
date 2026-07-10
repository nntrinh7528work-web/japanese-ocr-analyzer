"""DeepSeek API client using OpenAI-compatible SDK."""

from __future__ import annotations

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL


def get_deepseek_client() -> OpenAI:
    """Create an OpenAI client configured for DeepSeek API."""
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "Thiếu DEEPSEEK_API_KEY. "
            "Hãy thêm key vào .env hoặc Streamlit secrets."
        )
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def ping_deepseek() -> str:
    """Send a simple ping to verify DeepSeek API connectivity."""
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Reply with exactly PONG.",
            },
            {
                "role": "user",
                "content": "Ping",
            },
        ],
        temperature=0,
        max_tokens=10,
    )
    return (response.choices[0].message.content or "").strip()
