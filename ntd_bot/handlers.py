
from __future__ import annotations

import asyncio
import logging
import tempfile
from collections import defaultdict, deque
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)

from ntd_bot import asr
from ntd_bot.config import Settings
from ntd_bot.rag import answer_question
from ntd_bot.user_store import add_user, remove_user, list_users

logger = logging.getLogger(__name__)

router = Router(name="user")

def _is_admin(user_id: int, settings: Settings) -> bool:
    return user_id in settings.admin_ids

def _user_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📋 Помощь")],
            [KeyboardButton(text="🗑 Очистить историю")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Введите вопрос по НТД...",
    )

def _admin_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📋 Помощь")],
            [KeyboardButton(text="🗑 Очистить историю")],
            [KeyboardButton(text="👥 Список пользователей")],
            [KeyboardButton(text="➕ Добавить пользователя"), KeyboardButton(text="➖ Удалить пользователя")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Введите вопрос по НТД или команду...",
    )


def _get_keyboard(user_id: int, settings: Settings) -> ReplyKeyboardMarkup:
    return _admin_keyboard() if _is_admin(user_id, settings) else _user_keyboard()

_dialog_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=6))


def _get_history(user_id: int) -> list[dict]:
    return [{"role": role, "content": text} for role, text in _dialog_history[user_id]]


def _add_to_history(user_id: int, role: str, text: str) -> None:
    _dialog_history[user_id].append((role, text))

_waiting_add: set[int] = set()
_waiting_remove: set[int] = set()

def _split_message(text: str, max_len: int = 3900) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        cut = -1
        for sep in ("\n\n", "\n", " "):
            pos = text.rfind(sep, start, end)
            if pos > start:
                cut = pos
                break
        if cut == -1:
            cut = end
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(cut, start + 1)
    return chunks

async def _reply_long(message: Message, text: str, keyboard=None) -> None:
    parts = _split_message(text)
    for i, part in enumerate(parts):
        kb = keyboard if i == len(parts) - 1 else None
        try:
            await message.answer(part, parse_mode="Markdown", reply_markup=kb)
        except Exception:
            await message.answer(part, parse_mode=None, reply_markup=kb)


async def _thinking_reply(
    message: Message,
    settings: Settings,
    text: str,
    history: list[dict] | None = None,
) -> str:
    steps = [
        "🔍 Ищу в базе документов...",
        "📄 Анализирую фрагменты НТД...",
        "🧠 Формирую ответ...",
    ]

    status_msg = await message.answer(steps[0])

    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, answer_question, settings, text, history or [])

    step_index = 1
    while not task.done():
        await asyncio.sleep(4)
        if task.done():
            break
        try:
            await status_msg.edit_text(steps[min(step_index, len(steps) - 1)])
        except Exception:
            pass
        step_index += 1

    try:
        ans = await task
    finally:
        try:
            await status_msg.delete()
        except Exception:
            pass

    return ans

@router.message(CommandStart())
async def cmd_start(message: Message, settings: Settings) -> None:
    user_id = message.from_user.id
    kb = _get_keyboard(user_id, settings)

    if _is_admin(user_id, settings):
        text = (
            "✅ Ассистент по нормативно-технической документации\n\n"
            "Помогает оперативному персоналу быстро находить информацию "
            "в регламентах, инструкциях и НТД объектов атомной энергетики.\n\n"
            "Как пользоваться:\n"
            "— Отправьте вопрос текстом или голосовым сообщением\n"
            "— Ответ строится только на основе документов из базы НТД\n"
            "— Каждый ответ содержит указание источника\n\n"
            "⚙️ Вы вошли как администратор.\n"
            "Используйте кнопки ниже для управления пользователями."
        )
    else:
        text = (
            "✅ Ассистент по нормативно-технической документации\n\n"
            "Помогает оперативному персоналу быстро находить информацию "
            "в регламентах, инструкциях и НТД объектов атомной энергетики.\n\n"
            "Как пользоваться:\n"
            "— Отправьте вопрос текстом или голосовым сообщением\n"
            "— Ответ строится только на основе документов из базы НТД\n"
            "— Каждый ответ содержит указание источника\n\n"
            "Используйте кнопки ниже для быстрого доступа к функциям."
        )

    await message.answer(text, reply_markup=kb)


@router.message(Command("help"))
@router.message(F.text == "📋 Помощь")
async def cmd_help(message: Message, settings: Settings) -> None:
    user_id = message.from_user.id
    kb = _get_keyboard(user_id, settings)

    if _is_admin(user_id, settings):
        text = (
            "🤖 Ассистент по нормативно-технической документации\n\n"
            "Как пользоваться:\n"
            "— Отправьте вопрос текстом или голосовым сообщением\n"
            "— Ответ строится только на основе документов из базы НТД\n"
            "— Каждый ответ содержит указание источника\n\n"
            "Контекст диалога:\n"
            "— Бот помнит последние 3 пары вопрос-ответ\n"
            "— Можно задавать уточняющие вопросы\n"
            "— Кнопка «🗑 Очистить историю» сбрасывает память\n\n"
            "Команды администратора:\n"
            "/adduser [id] [имя] — добавить пользователя\n"
            "/removeuser [id] — удалить пользователя\n"
            "/listusers — список разрешённых пользователей\n\n"
        )
    else:
        text = (
            "🤖 Ассистент по нормативно-технической документации\n\n"
            "Как пользоваться:\n"
            "— Отправьте вопрос текстом или голосовым сообщением\n"
            "— Ответ строится только на основе документов из базы НТД\n"
            "— Каждый ответ содержит указание источника\n\n"
            "Контекст диалога:\n"
            "— Бот помнит последние 3 пары вопрос-ответ\n"
            "— Можно задавать уточняющие вопросы\n"
            "— Кнопка «🗑 Очистить историю» сбрасывает память"
        )

    await message.answer(text, reply_markup=kb)


@router.message(Command("clear"))
@router.message(F.text == "🗑 Очистить историю")
async def cmd_clear(message: Message, settings: Settings) -> None:
    _dialog_history[message.from_user.id].clear()
    kb = _get_keyboard(message.from_user.id, settings)
    await message.answer("🗑 История диалога очищена.", reply_markup=kb)


@router.message(Command("listusers"))
@router.message(F.text == "👥 Список пользователей")
async def cmd_listusers(message: Message, settings: Settings) -> None:
    if not _is_admin(message.from_user.id, settings):
        await message.answer("Нет прав.")
        return
    users = await list_users()
    kb = _get_keyboard(message.from_user.id, settings)
    if not users:
        await message.answer("Список пуст.", reply_markup=kb)
        return
    text = "\n".join(f"{u['name']} — {u['id']}" for u in users)
    await message.answer(f"Разрешённые пользователи:\n\n{text}", reply_markup=kb)


@router.message(F.text == "➕ Добавить пользователя")
async def btn_adduser(message: Message, settings: Settings) -> None:
    if not _is_admin(message.from_user.id, settings):
        await message.answer("Нет прав.")
        return
    _waiting_add.add(message.from_user.id)
    await message.answer(
        "Введите Telegram ID и имя сотрудника через пробел:\n"
        "(например: 123456789 Иванов И.И.)",
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(F.text == "➖ Удалить пользователя")
async def btn_removeuser(message: Message, settings: Settings) -> None:
    if not _is_admin(message.from_user.id, settings):
        await message.answer("Нет прав.")
        return
    _waiting_remove.add(message.from_user.id)
    users = await list_users()
    users_text = "\n".join(f"{u['name']} — {u['id']}" for u in users) if users else "Список пуст."
    await message.answer(
        f"Текущие пользователи:\n{users_text}\n\n"
        "Введите Telegram ID пользователя которого хотите удалить:",
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(Command("adduser"))
async def cmd_adduser(message: Message, settings: Settings) -> None:
    if not _is_admin(message.from_user.id, settings):
        await message.answer("Нет прав.")
        return
    parts = (message.text or "").split(maxsplit=2)
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer("Использование: /adduser 123456789 Иванов И.И.")
        return
    uid = int(parts[1])
    name = parts[2] if len(parts) > 2 else "Без имени"
    kb = _get_keyboard(message.from_user.id, settings)
    if await add_user(uid, name):
        await message.answer(f"✅ Пользователь {name} ({uid}) добавлен.", reply_markup=kb)
    else:
        await message.answer(f"Пользователь {uid} уже в списке.", reply_markup=kb)


@router.message(Command("removeuser"))
async def cmd_removeuser(message: Message, settings: Settings) -> None:
    if not _is_admin(message.from_user.id, settings):
        await message.answer("Нет прав.")
        return
    parts = (message.text or "").split()
    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer("Использование: /removeuser 123456789")
        return
    uid = int(parts[1])
    kb = _get_keyboard(message.from_user.id, settings)
    if await remove_user(uid):
        await message.answer(f"✅ Пользователь {uid} удалён.", reply_markup=kb)
    else:
        await message.answer(f"Пользователь {uid} не найден.", reply_markup=kb)

@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, settings: Settings) -> None:
    user_id = message.from_user.id
    text = (message.text or "").strip()

    if not text:
        return

    if user_id in _waiting_add:
        _waiting_add.discard(user_id)
        kb = _get_keyboard(user_id, settings)
        parts = text.split(maxsplit=1)
        if not parts[0].isdigit():
            await message.answer("Ошибка: первым введите числовой ID.", reply_markup=kb)
            return
        uid = int(parts[0])
        name = parts[1] if len(parts) > 1 else "Без имени"
        if await add_user(uid, name):
            await message.answer(f"✅ Пользователь {name} ({uid}) добавлен.", reply_markup=kb)
        else:
            await message.answer(f"Пользователь {uid} уже в списке.", reply_markup=kb)
        return

    if user_id in _waiting_remove:
        _waiting_remove.discard(user_id)
        kb = _get_keyboard(user_id, settings)
        if not text.isdigit():
            await message.answer("Ошибка: введите числовой ID.", reply_markup=kb)
            return
        uid = int(text)
        if await remove_user(uid):
            await message.answer(f"✅ Пользователь {uid} удалён.", reply_markup=kb)
        else:
            await message.answer(f"Пользователь {uid} не найден.", reply_markup=kb)
        return
    await message.bot.send_chat_action(message.chat.id, "typing")
    _add_to_history(user_id, "user", text)

    try:
        history = _get_history(user_id)
        ans = await _thinking_reply(message, settings, text, history)
    except Exception:
        logger.exception("RAG error on text")
        kb = _get_keyboard(user_id, settings)
        await message.answer(
            "Ошибка при поиске по документации. Попробуйте ещё раз.",
            reply_markup=kb,
        )
        return

    if not ans:
        kb = _get_keyboard(user_id, settings)
        await message.answer("Не удалось получить ответ. Попробуйте ещё раз.", reply_markup=kb)
        return

    _add_to_history(user_id, "assistant", ans)
    kb = _get_keyboard(user_id, settings)
    await _reply_long(message, ans, keyboard=kb)


@router.message(F.voice)
async def on_voice(message: Message, settings: Settings, bot: Bot) -> None:
    if not asr.asr_available():
        await message.answer(
            "Распознавание речи недоступно.\n"
            "Установите faster-whisper и перезапустите бота."
        )
        return

    await message.bot.send_chat_action(message.chat.id, "typing")

    vf = await bot.get_file(message.voice.file_id)
    suffix = Path(vf.file_path or "voice.ogg").suffix or ".ogg"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await bot.download_file(vf.file_path, destination=tmp_path)
        text = asr.transcribe_audio_file(tmp_path)

    except Exception:
        logger.exception("ASR error")
        await message.answer(
            "Не удалось распознать голосовое сообщение. Попробуйте отправить текст."
        )
        return
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    if not text or not text.strip():
        await message.answer("Речь не распознана. Повторите или отправьте текстом.")
        return

    await message.answer(f"🎤 Распознано:\n{text}")
    await message.bot.send_chat_action(message.chat.id, "typing")

    user_id = message.from_user.id
    _add_to_history(user_id, "user", text)

    try:
        history = _get_history(user_id)
        ans = await _thinking_reply(message, settings, text, history)
    except Exception:
        logger.exception("RAG error after voice")
        kb = _get_keyboard(user_id, settings)
        await message.answer("Ошибка при поиске по документации.", reply_markup=kb)
        return

    _add_to_history(user_id, "assistant", ans)
    kb = _get_keyboard(user_id, settings)
    await _reply_long(message, ans, keyboard=kb)