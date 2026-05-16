# ntd_bot/auth.py
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from ntd_bot.user_store import is_allowed


class AllowlistMiddleware(BaseMiddleware):

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        if user is None:
            return await handler(event, data)

        # is_allowed теперь async (aiosqlite-совместимый user_store)
        if not await is_allowed(user.id):
            if isinstance(event, Message):
                await event.answer(
                    "Доступ запрещён. Обратитесь к администратору для добавления "
                    "вашего Telegram ID в список разрешённых."
                )
            return None

        return await handler(event, data)