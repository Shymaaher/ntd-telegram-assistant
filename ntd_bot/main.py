from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from ntd_bot.auth import AllowlistMiddleware
from ntd_bot.config import Settings, load_settings
from ntd_bot.deps import SettingsMiddleware
from ntd_bot.handlers import router


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def _run_bot(settings: Settings) -> None:
    if not settings.bot_token:
        raise SystemExit("Укажите BOT_TOKEN в .env")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    dp = Dispatcher()
    dp.message.middleware(AllowlistMiddleware())
    dp.message.middleware(SettingsMiddleware(settings))
    dp.include_router(router)

    logging.info("Бот запущен")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


def main() -> None:
    _setup_logging()
    settings = load_settings()
    logging.info("ADMIN_IDS: %s", settings.admin_ids)  # <- внутри функции
    asyncio.run(_run_bot(settings))


if __name__ == "__main__":
    main()
