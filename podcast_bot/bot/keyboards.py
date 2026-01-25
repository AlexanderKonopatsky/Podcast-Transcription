"""
Telegram keyboard builders.
"""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


def get_speaker_keyboard(speakers: list[str]) -> InlineKeyboardMarkup:
    """Build inline keyboard for speaker selection."""
    builder = InlineKeyboardBuilder()

    for speaker in speakers:
        builder.add(InlineKeyboardButton(
            text=speaker,
            callback_data=f"speaker:{speaker}"
        ))

    builder.adjust(1)  # One button per row
    return builder.as_markup()


def get_menu_keyboard() -> InlineKeyboardMarkup:
    """Build menu keyboard with common actions."""
    builder = InlineKeyboardBuilder()

    builder.add(InlineKeyboardButton(
        text="Сменить персонажа",
        callback_data="action:switch"
    ))
    builder.add(InlineKeyboardButton(
        text="Очистить историю",
        callback_data="action:clear"
    ))

    builder.adjust(2)
    return builder.as_markup()
