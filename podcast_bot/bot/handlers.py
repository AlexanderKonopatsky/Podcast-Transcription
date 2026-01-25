"""
Telegram bot handlers.
"""

import json
import asyncio
import time
from pathlib import Path
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SESSIONS_DIR, DATA_DIR, LLM_MODEL, AVAILABLE_MODELS
from rag.search import search, format_search_results
from llm.prompts import build_system_prompt, build_user_prompt, get_available_speakers
from llm.client import chat
from bot.keyboards import get_speaker_keyboard, get_menu_keyboard
from logging_system import get_interaction_logger

router = Router()


# Session management
def load_session(user_id: int) -> dict:
    """Load user session from file."""
    session_path = SESSIONS_DIR / f"{user_id}.json"
    if session_path.exists():
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"speaker": None, "model": None, "history": [], "waiting_for_model": False}


def save_session(user_id: int, session: dict):
    """Save user session to file."""
    session_path = SESSIONS_DIR / f"{user_id}.json"
    with open(session_path, 'w', encoding='utf-8') as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def check_index_exists() -> bool:
    """Check if RAG index exists."""
    return (DATA_DIR / "embeddings.npy").exists() and (DATA_DIR / "chunks.json").exists()


# Command handlers
@router.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    if not check_index_exists():
        await message.answer(
            "Индекс не найден. Запустите сначала:\n"
            "`python scripts/index_podcasts.py`",
            parse_mode="Markdown"
        )
        return

    speakers = get_available_speakers()
    await message.answer(
        "Привет! Я бот для общения с персонажами подкаста Drops Capital.\n\n"
        "Выберите, с кем хотите пообщаться:",
        reply_markup=get_speaker_keyboard(speakers)
    )


@router.message(Command("switch"))
async def cmd_switch(message: Message):
    """Handle /switch command - change speaker."""
    speakers = get_available_speakers()
    await message.answer(
        "Выберите нового персонажа:",
        reply_markup=get_speaker_keyboard(speakers)
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    """Handle /clear command - clear history."""
    session = load_session(message.from_user.id)
    session["history"] = []
    save_session(message.from_user.id, session)

    speaker = session.get("speaker")
    if speaker:
        await message.answer(f"История очищена. Продолжаем общение с {speaker}.")
    else:
        await message.answer("История очищена.")


@router.message(Command("menu"))
async def cmd_menu(message: Message):
    """Show menu with actions."""
    await message.answer(
        "Меню:",
        reply_markup=get_menu_keyboard()
    )


@router.message(Command("model"))
async def cmd_model(message: Message):
    """Handle /model command - show available models."""
    # Build model list message
    model_list = "Доступные модели LLM:\n\n"
    models = list(AVAILABLE_MODELS.items())
    for i, (model_id, description) in enumerate(models, 1):
        model_list += f"{i}. {description}\n"

    model_list += "\nОтправьте номер модели для выбора (1-4)"

    # Set waiting flag
    session = load_session(message.from_user.id)
    session["waiting_for_model"] = True
    save_session(message.from_user.id, session)

    await message.answer(model_list)


# Callback handlers
@router.callback_query(F.data.startswith("speaker:"))
async def callback_speaker(callback: CallbackQuery):
    """Handle speaker selection."""
    speaker = callback.data.split(":", 1)[1]

    # Save to session
    session = load_session(callback.from_user.id)
    session["speaker"] = speaker
    session["history"] = []  # Clear history on speaker change
    save_session(callback.from_user.id, session)

    await callback.message.edit_text(
        f"Вы выбрали: {speaker}\n\n"
        f"Теперь можете задавать вопросы. Я буду отвечать в стиле {speaker}.\n\n"
        f"Команды:\n"
        f"/switch - сменить персонажа\n"
        f"/clear - очистить историю диалога\n"
        f"/model - выбрать модель (llm)"
    )
    await callback.answer()


@router.callback_query(F.data == "action:switch")
async def callback_switch(callback: CallbackQuery):
    """Handle switch action from menu."""
    speakers = get_available_speakers()
    await callback.message.edit_text(
        "Выберите нового персонажа:",
        reply_markup=get_speaker_keyboard(speakers)
    )
    await callback.answer()


@router.callback_query(F.data == "action:clear")
async def callback_clear(callback: CallbackQuery):
    """Handle clear action from menu."""
    session = load_session(callback.from_user.id)
    session["history"] = []
    save_session(callback.from_user.id, session)

    await callback.message.edit_text("История диалога очищена.")
    await callback.answer("Очищено")


# Message handler
@router.message(F.text)
async def handle_message(message: Message):
    """Handle regular text messages."""
    # Start timing
    start_time = time.time()

    session = load_session(message.from_user.id)

    # Check if waiting for model selection
    if session.get("waiting_for_model", False):
        user_input = message.text.strip()

        # Validate input is a number
        if not user_input.isdigit():
            await message.answer("Пожалуйста, отправьте номер модели (1-4)")
            return

        choice = int(user_input)
        models = list(AVAILABLE_MODELS.keys())

        # Validate choice range
        if choice < 1 or choice > len(models):
            await message.answer(f"Неверный номер. Выберите от 1 до {len(models)}")
            return

        # Save selected model
        selected_model = models[choice - 1]
        session["model"] = selected_model
        session["waiting_for_model"] = False
        save_session(message.from_user.id, session)

        model_name = AVAILABLE_MODELS[selected_model]
        await message.answer(f"✓ Модель изменена на: {model_name}")
        return

    speaker = session.get("speaker")

    if not speaker:
        speakers = get_available_speakers()
        await message.answer(
            "Сначала выберите персонажа:",
            reply_markup=get_speaker_keyboard(speakers)
        )
        return

    # Show typing indicator
    await message.bot.send_chat_action(message.chat.id, "typing")

    user_input = message.text.strip()
    history = session.get("history", [])

    # Variables for logging
    rag_results = []
    response_text = ""
    error_msg = None

    # Search for context
    try:
        rag_results = search(user_input, speaker=speaker, top_k=25)
        rag_context = format_search_results(rag_results, speaker=speaker)
    except Exception as e:
        print(f"Search error: {e}")
        rag_context = ""
        error_msg = f"RAG error: {str(e)}"

    # Build prompts
    system_prompt = build_system_prompt(speaker)
    user_prompt = build_user_prompt(rag_context, user_input, history)

    # Get LLM response
    try:
        # Get selected model from session (None = use default)
        model = session.get("model")
        # Run sync function in executor to not block
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None,
            lambda: chat(system_prompt, user_prompt, model=model)
        )
    except Exception as e:
        await message.answer(f"Ошибка генерации ответа: {e}")
        error_msg = f"LLM error: {str(e)}"

        # Log error
        processing_time = (time.time() - start_time) * 1000
        logger = get_interaction_logger()
        logger.log_interaction(
            user_id=message.from_user.id,
            username=message.from_user.username,
            speaker=speaker,
            query=user_input,
            response="",
            rag_results=rag_results,
            llm_metadata={
                "model": session.get("model") or LLM_MODEL,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "history_messages_count": len(history)
            },
            processing_time_ms=processing_time,
            error=error_msg
        )
        return

    # Update history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response_text})

    # Keep last 10 messages
    if len(history) > 10:
        history = history[-10:]

    session["history"] = history
    save_session(message.from_user.id, session)

    # Log successful interaction
    processing_time = (time.time() - start_time) * 1000
    logger = get_interaction_logger()
    logger.log_interaction(
        user_id=message.from_user.id,
        username=message.from_user.username,
        speaker=speaker,
        query=user_input,
        response=response_text,
        rag_results=rag_results,
        llm_metadata={
            "model": session.get("model") or LLM_MODEL,
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
            "history_messages_count": len(history)
        },
        processing_time_ms=processing_time,
        error=None
    )

    await message.answer(response_text)
