from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

JSON_PATH = Path(__file__).parent / "users.json"

_lock = asyncio.Lock()


def _load() -> dict:
    if not JSON_PATH.exists():
        return {"allowed_ids": []}
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _is_allowed_sync(user_id: int) -> bool:
    data = _load()
    return any(u["id"] == user_id for u in data["allowed_ids"])


def _add_user_sync(user_id: int, name: str) -> bool:
    data = _load()
    if any(u["id"] == user_id for u in data["allowed_ids"]):
        return False
    data["allowed_ids"].append({"id": user_id, "name": name})
    _save(data)
    return True


def _remove_user_sync(user_id: int) -> bool:
    data = _load()
    before = len(data["allowed_ids"])
    data["allowed_ids"] = [u for u in data["allowed_ids"] if u["id"] != user_id]
    if len(data["allowed_ids"]) == before:
        return False
    _save(data)
    return True


def _list_users_sync() -> list[dict]:
    data = _load()
    return sorted(data["allowed_ids"], key=lambda u: u["name"])


async def is_allowed(user_id: int) -> bool:
    async with _lock:
        return _is_allowed_sync(user_id)


async def add_user(user_id: int, name: str = "Без имени") -> bool:
    async with _lock:
        result = _add_user_sync(user_id, name)
        if result:
            logger.info("Добавлен пользователь: %s (%d)", name, user_id)
        return result


async def remove_user(user_id: int) -> bool:
    async with _lock:
        result = _remove_user_sync(user_id)
        if result:
            logger.info("Удалён пользователь: %d", user_id)
        return result


async def list_users() -> list[dict]:
    async with _lock:
        return _list_users_sync()
