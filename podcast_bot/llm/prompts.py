"""
Prompt templates for persona-based chat.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPEAKERS_DIR


def load_speaker_profile(speaker_name: str) -> dict:
    """Load speaker profile from JSON."""
    # Try exact match first
    profile_path = SPEAKERS_DIR / f"{speaker_name.lower()}.json"

    if not profile_path.exists():
        # Try to find by name field
        for filepath in SPEAKERS_DIR.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                profile = json.load(f)
                if profile.get("name", "").lower() == speaker_name.lower():
                    return profile

        raise FileNotFoundError(f"Speaker profile not found: {speaker_name}")

    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_speakers() -> list[str]:
    """Get list of available speaker names."""
    speakers = []
    for filepath in SPEAKERS_DIR.glob("*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            profile = json.load(f)
            speakers.append(profile.get("name", filepath.stem))
    return speakers


SYSTEM_PROMPT_TEMPLATE = """Ты — {name}, участник подкаста Drops Capital о криптовалютах и финансовых рынках.

## Твой характер и манера речи:
{personality}

## Твои характерные речевые обороты (используй их естественно):
{speech_patterns}

## ВАЖНЫЕ ПРАВИЛА:
1. Отвечай В СВОЁМ СТИЛЕ — используй свои обороты, манеру речи, жаргон
2. Опирайся на СВОИ ЗНАНИЯ из подкастов (см. контекст ниже)
3. Не говори "в подкасте мы обсуждали" или "как я говорил в подкасте" — просто отвечай как будто это твоё текущее мнение
4. Если в контексте нет информации по теме — можешь порассуждать в своём стиле, но НЕ выдумывай конкретные факты и цифры
5. Отвечай на русском языке
6. Будь живым и естественным, как в реальном разговоре

## О ВРЕМЕННОМ КОНТЕКСТЕ:
- Подкасты помечены датами (формат YYYY-MM-DD)
- 🔥 означает "свежее мнение" (последние 2 недели) - приоритизируй эти данные
- Если видишь разные мнения по времени, ОБЯЗАТЕЛЬНО укажи:
  * Актуальное мнение (свежие подкасты с 🔥)
  * Как оно изменилось со временем (если есть старые подкасты)
- Пример: "Сейчас (январь 2026) я считаю X, но раньше (декабрь) думал Y"

## Примеры твоих реплик (для понимания стиля):
{style_examples}
"""


USER_PROMPT_TEMPLATE = """## Релевантный контекст из подкастов:

{rag_context}

## Вопрос:
{question}

Ответь в своём стиле, опираясь на контекст."""


def build_system_prompt(speaker_name: str) -> str:
    """Build system prompt for a speaker."""
    profile = load_speaker_profile(speaker_name)

    speech_patterns = "\n".join(f"- {p}" for p in profile.get("speech_patterns", []))
    style_examples = "\n\n".join(f'"{ex}"' for ex in profile.get("style_examples", []))

    return SYSTEM_PROMPT_TEMPLATE.format(
        name=profile["name"],
        personality=profile.get("personality", ""),
        speech_patterns=speech_patterns,
        style_examples=style_examples
    )


def build_user_prompt(rag_context: str, question: str, history: list[dict] = None) -> str:
    """Build user prompt with RAG context and optional history."""
    prompt_parts = []

    # Add conversation history if present
    if history:
        history_text = "## Недавний контекст диалога:\n"
        for msg in history[-50:]:  # Last 24-25 exchanges
            role = "Собеседник" if msg["role"] == "user" else "Ты"
            history_text += f"{role}: {msg['content']}\n"
        prompt_parts.append(history_text)

    # Main prompt
    prompt_parts.append(USER_PROMPT_TEMPLATE.format(
        rag_context=rag_context if rag_context else "(контекст не найден)",
        question=question
    ))

    return "\n".join(prompt_parts)


# Test
if __name__ == "__main__":
    print("Available speakers:", get_available_speakers())

    for speaker in get_available_speakers():
        print(f"\n--- {speaker} ---")
        prompt = build_system_prompt(speaker)
        print(prompt[:500] + "...")
