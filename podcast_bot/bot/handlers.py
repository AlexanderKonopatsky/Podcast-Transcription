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

from config import SESSIONS_DIR, DATA_DIR, LLM_MODEL, AVAILABLE_MODELS, SENTIMENT_TOP_K, SENTIMENT_MAX_CONCURRENT_REQUESTS
from rag.search import search, format_search_results
from llm.prompts import build_system_prompt, build_user_prompt, get_available_speakers
from llm.client import chat
from bot.keyboards import get_speaker_keyboard, get_menu_keyboard
from logging_system import get_interaction_logger
from sentiment import SentimentAnalyzer, create_sentiment_chart

router = Router()


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """
    Split long message into chunks respecting Telegram's 4096 character limit.

    Args:
        text: Text to split
        max_length: Maximum length per chunk (default 4096 for Telegram)

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    for line in text.split('\n'):
        # If adding this line would exceed limit, save current chunk and start new one
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.rstrip())
                current_chunk = ""

        # If single line is too long, split it by words
        if len(line) > max_length:
            words = line.split(' ')
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.rstrip())
                    current_chunk = word + " "
                else:
                    current_chunk += word + " "
        else:
            current_chunk += line + "\n"

    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.rstrip())

    return chunks


# Session management
def load_session(user_id: int) -> dict:
    """Load user session from file."""
    session_path = SESSIONS_DIR / f"{user_id}.json"
    if session_path.exists():
        with open(session_path, 'r', encoding='utf-8') as f:
            session = json.load(f)
            # Add time_filter if not present (backward compatibility)
            if "time_filter" not in session:
                session["time_filter"] = "balanced"
            return session
    return {
        "speaker": None,
        "model": None,
        "history": [],
        "waiting_for_model": False,
        "waiting_for_dates": False,
        "time_filter": "balanced"  # Default: temporal decay mode
    }


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


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Show help message with all commands."""
    help_text = (
        "<b>📚 Справка по командам</b>\n\n"
        "<b>Основные команды:</b>\n"
        "/start - начать работу с ботом\n"
        "/switch - сменить персонажа\n"
        "/clear - очистить историю диалога\n"
        "/model - выбрать модель LLM\n"
        "/menu - показать меню\n"
        "/help - показать эту справку\n\n"
        "<b>⏱ Управление временным контекстом:</b>\n"
        "/time_mode - показать текущий режим\n"
        "/recent - только свежие (2 недели) 🔥\n"
        "/balanced - балансированный (по умолчанию) ⚖️\n"
        "/historical - вся история 📚\n"
        "/date_range - выбрать период 📅\n\n"
        "<b>📊 Анализ изменения мнений:</b>\n"
        "/sentiment <тема> - анализ изменения мнений о теме\n"
        "Пример: /sentiment биткоин\n"
        "Пример: /sentiment eigen layer\n\n"
        "<b>Режимы временного контекста:</b>\n"
        "• <b>Recent</b> - показывает только подкасты за последние 2 недели\n"
        "• <b>Balanced</b> - приоритизирует свежие мнения (~70%), но учитывает и историю (~30%)\n"
        "• <b>Historical</b> - все подкасты имеют равный вес, хорошо для анализа изменений мнений\n"
        "• <b>Date Range</b> - выбрать конкретный период для поиска\n"
    )
    await message.answer(help_text, parse_mode="HTML")


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

    model_list += "\nОтправьте номер модели для выбора (1-2)"

    # Set waiting flag
    session = load_session(message.from_user.id)
    session["waiting_for_model"] = True
    save_session(message.from_user.id, session)

    await message.answer(model_list)


@router.message(Command("recent"))
async def cmd_recent(message: Message):
    """Switch to recent mode (last 2 weeks only)."""
    user_id = message.from_user.id
    session = load_session(user_id)

    session["time_filter"] = "recent"
    save_session(user_id, session)

    await message.answer(
        "✅ Режим: <b>Свежие подкасты</b>\n"
        "Показываю только последние 2 недели.\n\n"
        "Другие режимы:\n"
        "/balanced - балансированный (default)\n"
        "/historical - вся история\n"
        "/date_range - выбрать период",
        parse_mode="HTML"
    )


@router.message(Command("balanced"))
async def cmd_balanced(message: Message):
    """Switch to balanced mode (temporal decay)."""
    user_id = message.from_user.id
    session = load_session(user_id)

    session["time_filter"] = "balanced"
    save_session(user_id, session)

    await message.answer(
        "✅ Режим: <b>Балансированный</b> (по умолчанию)\n"
        "~70% свежие мнения (2 недели) + ~30% история.\n"
        "Старые мнения не исчезают, но понижаются.",
        parse_mode="HTML"
    )


@router.message(Command("historical"))
async def cmd_historical(message: Message):
    """Switch to historical mode (all history, no decay)."""
    user_id = message.from_user.id
    session = load_session(user_id)

    session["time_filter"] = "historical"
    save_session(user_id, session)

    await message.answer(
        "✅ Режим: <b>Вся история</b>\n"
        "Все подкасты имеют равный вес.\n"
        "Подходит для вопросов типа:\n"
        "\"Как менялось мнение о BTC за год?\"",
        parse_mode="HTML"
    )


@router.message(Command("date_range"))
async def cmd_date_range(message: Message):
    """Initiate custom date range selection."""
    user_id = message.from_user.id
    session = load_session(user_id)

    session["waiting_for_dates"] = True
    save_session(user_id, session)

    await message.answer(
        "📅 Отправьте диапазон дат в формате:\n"
        "<code>YYYY-MM-DD YYYY-MM-DD</code>\n\n"
        "Пример: <code>2025-12-01 2026-01-15</code>\n\n"
        "Или /cancel для отмены",
        parse_mode="HTML"
    )


@router.message(Command("time_mode"))
async def cmd_time_mode(message: Message):
    """Show current time filter mode."""
    user_id = message.from_user.id
    session = load_session(user_id)

    time_filter = session.get("time_filter", "balanced")

    # Format mode description
    mode_descriptions = {
        "recent": "🔥 Свежие подкасты (последние 2 недели)",
        "balanced": "⚖️ Балансированный (70% свежие / 30% история)",
        "historical": "📚 Вся история (без фильтра)"
    }

    if isinstance(time_filter, dict):
        description = f"📅 Кастомный период: {time_filter['start']} — {time_filter['end']}"
    else:
        description = mode_descriptions.get(time_filter, "⚖️ Балансированный")

    await message.answer(
        f"<b>Текущий режим:</b>\n{description}\n\n"
        "Изменить: /recent | /balanced | /historical | /date_range",
        parse_mode="HTML"
    )


@router.message(Command("sentiment"))
async def cmd_sentiment(message: Message):
    """Analyze sentiment trends for an entity."""
    from aiogram.types import BufferedInputFile
    from sentiment.exporter import export_sentiment_to_txt
    from datetime import datetime

    # Parse command arguments
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "📊 <b>Анализ изменения мнений</b>\n\n"
            "Использование:\n"
            "<code>/sentiment &lt;тема&gt;</code>\n\n"
            "Примеры:\n"
            "• <code>/sentiment биткоин</code>\n"
            "• <code>/sentiment eigen layer</code>\n"
            "• <code>/sentiment альткоины</code>\n\n"
            "Бот найдет все упоминания темы в подкастах, "
            "проанализирует мнения спикеров и покажет график изменений во времени.",
            parse_mode="HTML"
        )
        return

    entity = args[1].strip()

    # Show processing message
    processing_msg = await message.answer(
        f"📊 Анализирую мнения о '{entity}'...\n"
        f"⚡ Параллельная обработка: до {SENTIMENT_MAX_CONCURRENT_REQUESTS} запросов одновременно\n"
        "Это займет ~30-60 секунд...",
        parse_mode="HTML"
    )

    try:
        # Run async analysis (no executor needed)
        analyzer = SentimentAnalyzer()
        sentiments = await analyzer.analyze_entity_async(
            entity=entity,
            top_k=SENTIMENT_TOP_K,  # Configurable via SENTIMENT_TOP_K env variable
            max_concurrent=SENTIMENT_MAX_CONCURRENT_REQUESTS
        )

        if not sentiments:
            await processing_msg.edit_text(
                f"❌ Не найдено упоминаний о '{entity}' в подкастах.\n\n"
                "Попробуйте другую формулировку или тему.",
                parse_mode="HTML"
            )
            return

        # Generate text summary
        summary = analyzer.summarize_trend(sentiments)

        # Generate chart and TXT export
        try:
            # Create chart
            chart_buffer = create_sentiment_chart(sentiments, entity)
            photo = BufferedInputFile(chart_buffer.getvalue(), filename=f"sentiment_{entity}.png")

            # Create TXT report
            txt_buffer = export_sentiment_to_txt(sentiments, entity)
            txt_filename = f"sentiment_report_{entity}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            txt_file = BufferedInputFile(txt_buffer.getvalue(), filename=txt_filename)

            # Send chart
            await message.answer_photo(photo=photo, caption=f"📊 График анализа мнений: {entity}")

            # Send TXT report
            await message.answer_document(document=txt_file, caption="📄 Подробный отчет с полными текстами")

            # Send summary (split if too long)
            summary_chunks = split_message(summary, max_length=4096)
            for chunk in summary_chunks:
                await message.answer(chunk, parse_mode="HTML")

            # Delete processing message
            await processing_msg.delete()

        except Exception as export_error:
            # Fallback: send at least chart and summary if TXT export fails
            print(f"TXT export failed: {export_error}")
            try:
                chart_buffer = create_sentiment_chart(sentiments, entity)
                photo = BufferedInputFile(chart_buffer.getvalue(), filename=f"sentiment_{entity}.png")
                await message.answer_photo(photo=photo)

                # Send summary (split if too long)
                summary_chunks = split_message(summary, max_length=4096)
                for chunk in summary_chunks:
                    await message.answer(chunk, parse_mode="HTML")

                await processing_msg.delete()
            except Exception as chart_error:
                # If chart also fails, still send text summary (split if needed)
                error_msg = f"⚠️ Не удалось создать график: {chart_error}"
                summary_with_error = f"{summary}\n\n{error_msg}"

                # Split and send
                chunks = split_message(summary_with_error, max_length=4096)
                await processing_msg.edit_text(chunks[0], parse_mode="HTML")

                # Send remaining chunks as separate messages
                for chunk in chunks[1:]:
                    await message.answer(chunk, parse_mode="HTML")

    except Exception as e:
        await processing_msg.edit_text(
            f"❌ Ошибка при анализе: {e}\n\n"
            "Попробуйте еще раз или выберите другую тему.",
            parse_mode="HTML"
        )
        import traceback
        print(f"Sentiment analysis error: {traceback.format_exc()}")


@router.message(Command("cancel"))
async def cmd_cancel(message: Message):
    """Cancel current operation."""
    user_id = message.from_user.id
    session = load_session(user_id)

    if session.get("waiting_for_dates"):
        session["waiting_for_dates"] = False
        save_session(user_id, session)
        await message.answer("Отменено.")
    elif session.get("waiting_for_model"):
        session["waiting_for_model"] = False
        save_session(user_id, session)
        await message.answer("Отменено.")
    else:
        await message.answer("Нечего отменять.")


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
        f"<b>Основные команды:</b>\n"
        f"/switch - сменить персонажа\n"
        f"/clear - очистить историю диалога\n"
        f"/model - выбрать модель (llm)\n\n"
        f"<b>⏱ Управление временным контекстом:</b>\n"
        f"/time_mode - показать текущий режим\n"
        f"/recent - только свежие (2 недели) 🔥\n"
        f"/balanced - балансированный (по умолчанию) ⚖️\n"
        f"/historical - вся история 📚\n"
        f"/date_range - выбрать период 📅",
        parse_mode="HTML"
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

    # Check if waiting for date range input
    if session.get("waiting_for_dates", False):
        from datetime import datetime
        try:
            # Parse format "YYYY-MM-DD YYYY-MM-DD"
            dates = message.text.strip().split()
            if len(dates) != 2:
                raise ValueError("Нужно 2 даты")

            start_date = datetime.fromisoformat(dates[0]).date()
            end_date = datetime.fromisoformat(dates[1]).date()

            if start_date > end_date:
                raise ValueError("Начальная дата позже конечной")

            # Save to session
            session["time_filter"] = {
                "start": dates[0],
                "end": dates[1]
            }
            session["waiting_for_dates"] = False
            save_session(message.from_user.id, session)

            await message.answer(
                f"✅ Установлен диапазон:\n"
                f"📅 {dates[0]} — {dates[1]}",
                parse_mode="HTML"
            )
            return

        except (ValueError, IndexError) as e:
            await message.answer(
                "❌ Неверный формат. Используйте:\n"
                "<code>YYYY-MM-DD YYYY-MM-DD</code>\n\n"
                "Или /cancel для отмены",
                parse_mode="HTML"
            )
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
        time_filter = session.get("time_filter", "balanced")
        rag_results = search(user_input, speaker=speaker, top_k=25, time_filter=time_filter)
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

    # No limit on session storage - keep all history
    # if len(history) > 10:
    #     history = history[-10:]

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
