"""
OpenRouter LLM client.
"""

import httpx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def chat(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Send chat request to OpenRouter.

    Args:
        system_prompt: System message setting the persona
        user_prompt: User message with context and question
        model: Model to use (default from config)
        temperature: Sampling temperature
        max_tokens: Max response tokens

    Returns:
        Assistant's response text
    """
    model = model or LLM_MODEL

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/podcast-persona-bot",
        "X-Title": "Podcast Persona Bot"
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=data,
        timeout=60.0
    )

    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.status_code} - {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


def chat_with_history(
    system_prompt: str,
    history: list[dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Chat with conversation history.

    Args:
        system_prompt: System message
        history: List of {"role": "user"|"assistant", "content": "..."}
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Max response tokens

    Returns:
        Assistant's response
    """
    model = model or LLM_MODEL

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/podcast-persona-bot",
        "X-Title": "Podcast Persona Bot"
    }

    messages = [{"role": "system", "content": system_prompt}] + history

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=data,
        timeout=60.0
    )

    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.status_code} - {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


# Test
if __name__ == "__main__":
    print("Testing LLM client...")

    response = chat(
        system_prompt="Ты — дружелюбный помощник.",
        user_prompt="Привет! Как дела?"
    )

    print(f"Response: {response}")
