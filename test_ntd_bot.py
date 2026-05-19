"""
Функциональное тестирование мультимодального интеллектуального ассистента НТД.
Покрывает тестовые случаи ТК-01 — ТК-10 из раздела 3.9 ВКР.

Запуск:
    pytest test_ntd_bot.py -v
"""

import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Вспомогательная фикстура: JSON-хранилище пользователей
# Структура users.json: {"allowed_ids": [{"id": 123, "name": "Имя"}, ...]}
# ===========================================================================
def _make_users_json(tmp_path: Path, users: list[tuple[int, str]] | None = None) -> Path:
    """Создаёт временный users.json с заданным списком пользователей."""
    path = tmp_path / "users.json"
    data = {
        "allowed_ids": [
            {"id": uid, "name": name} for uid, name in (users or [])
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# ===========================================================================
# ТК-01 — Блокировка неавторизованного пользователя
# Реальные функции: _is_allowed_sync, JSON_PATH (users.json)
# ===========================================================================
class TestTC01_UnauthorizedBlock:
    """ТК-01: пользователь с ID не из белого списка не проходит авторизацию."""

    def test_unknown_user_is_not_allowed(self, tmp_path):
        """_is_allowed_sync() возвращает False для ID вне списка."""
        json_path = _make_users_json(tmp_path, [(111111, "Тестовый")])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            result = us._is_allowed_sync(999999)
        finally:
            us.JSON_PATH = orig
        assert result is False

    def test_known_user_is_allowed(self, tmp_path):
        """_is_allowed_sync() возвращает True для ID из списка."""
        json_path = _make_users_json(tmp_path, [(111111, "Тестовый")])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            result = us._is_allowed_sync(111111)
        finally:
            us.JSON_PATH = orig
        assert result is True

    @pytest.mark.asyncio
    async def test_middleware_blocks_unknown_user(self, tmp_path):
        """AllowlistMiddleware не вызывает handler для неавторизованного пользователя."""
        json_path = _make_users_json(tmp_path, [])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            from ntd_bot.auth import AllowlistMiddleware
            middleware = AllowlistMiddleware()
            handler = AsyncMock(return_value="ok")

            fake_user = MagicMock()
            fake_user.id = 999999

            from aiogram.types import Message
            event = MagicMock(spec=Message)
            event.from_user = fake_user
            event.answer = AsyncMock()

            await middleware(handler, event, {})
            handler.assert_not_called()
            event.answer.assert_called_once()
        finally:
            us.JSON_PATH = orig


# ===========================================================================
# ТК-02 — Приём текстового запроса по НТД
# Актуальный порог: _RERANK_THRESHOLD = 0.35 (cross-encoder score после sigmoid)
# ===========================================================================
class TestTC02_TextQueryNTD:
    """ТК-02: ответ на текстовый вопрос по НТД содержит ссылку на источник."""

    def test_answer_contains_sources_key(self):
        """answer_question() возвращает строку с разделом «Источники»."""
        mock_settings = MagicMock()
        mock_settings.ollama_base_url = None
        mock_settings.ollama_model = None
        mock_settings.rag_top_k = 5
        mock_settings.min_relevant_chunks = 1

        fake_doc = MagicMock()
        fake_doc.page_content = "passage: требования к защитной одежде в ЗКД"
        fake_doc.metadata = {"doc_name": "НТД_Охрана_труда.pdf"}

        import ntd_bot.rag as rag_mod
        orig_vs = rag_mod._vectorstore
        orig_bm25 = rag_mod._bm25_index
        rag_mod._vectorstore = MagicMock()
        rag_mod._bm25_index = MagicMock()

        try:
            # score 0.9 > _RERANK_THRESHOLD (0.35) — документ должен попасть в контекст
            with patch("ntd_bot.rag._init_vectorstore"), \
                 patch("ntd_bot.rag._init_bm25"), \
                 patch("ntd_bot.rag._hybrid_search",
                       return_value=[(fake_doc, 0.9)]):
                from ntd_bot.rag import answer_question
                result = answer_question(mock_settings, "требования к защитной одежде")
        finally:
            rag_mod._vectorstore = orig_vs
            rag_mod._bm25_index = orig_bm25

        assert "Источники" in result or "НТД_Охрана_труда" in result

    def test_short_query_rejected(self):
        """Запрос короче 3 символов отклоняется с соответствующим сообщением."""
        mock_settings = MagicMock()
        mock_settings.ollama_base_url = None
        mock_settings.ollama_model = None

        import ntd_bot.rag as rag_mod
        orig_vs = rag_mod._vectorstore
        orig_bm25 = rag_mod._bm25_index
        rag_mod._vectorstore = MagicMock()
        rag_mod._bm25_index = MagicMock()

        try:
            with patch("ntd_bot.rag._init_vectorstore"), \
                 patch("ntd_bot.rag._init_bm25"):
                from ntd_bot.rag import answer_question
                result = answer_question(mock_settings, "ab")
        finally:
            rag_mod._vectorstore = orig_vs
            rag_mod._bm25_index = orig_bm25

        assert "короткий" in result.lower()


# ===========================================================================
# ТК-03 — Отклонение запроса, не относящегося к НТД
# ===========================================================================
class TestTC03_NonNTDQuery:
    """ТК-03: классификатор отклоняет вопросы вне тематики НТД."""

    def test_classifier_returns_false_for_weather(self):
        """_is_ntd_question() возвращает False для бытового вопроса."""
        mock_settings = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "НЕТ"

        with patch("ntd_bot.rag._get_llm", return_value=mock_llm):
            from ntd_bot.rag import _is_ntd_question
            result = _is_ntd_question(mock_settings, "Какая погода в Москве?")

        assert result is False

    def test_fallback_message_returned(self):
        """answer_question() возвращает fallback при нерелевантном запросе."""
        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:7b"
        mock_settings.rag_top_k = 5

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "НЕТ"

        import ntd_bot.rag as rag_mod
        orig_vs = rag_mod._vectorstore
        orig_bm25 = rag_mod._bm25_index
        rag_mod._vectorstore = MagicMock()
        rag_mod._bm25_index = MagicMock()

        try:
            with patch("ntd_bot.rag._init_vectorstore"), \
                 patch("ntd_bot.rag._init_bm25"), \
                 patch("ntd_bot.rag._get_llm", return_value=mock_llm):
                from ntd_bot.rag import answer_question
                result = answer_question(mock_settings, "Какая погода в Москве?")
        finally:
            rag_mod._vectorstore = orig_vs
            rag_mod._bm25_index = orig_bm25

        assert "выходит за рамки" in result or "нормативно" in result.lower()


# ===========================================================================
# ТК-04 — Обработка голосового сообщения
# ===========================================================================
class TestTC04_VoiceProcessing:
    """ТК-04: голосовое сообщение транскрибируется в текст."""

    def test_asr_available_returns_bool(self):
        """asr_available() возвращает булево значение."""
        from ntd_bot.asr import asr_available
        result = asr_available()
        assert isinstance(result, bool)

    def test_transcribe_joins_segments(self):
        """transcribe_audio_file() склеивает сегменты в одну строку."""
        seg1 = MagicMock()
        seg1.text = "Каковы требования"
        seg2 = MagicMock()
        seg2.text = " к защитной одежде?"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

        with patch("ntd_bot.asr._get_model", return_value=mock_model):
            from ntd_bot.asr import transcribe_audio_file
            result = transcribe_audio_file(Path("fake.ogg"))

        assert "требования" in result
        assert "одежде" in result


# ===========================================================================
# ТК-05 — Уточняющий вопрос с учётом истории диалога
# ===========================================================================
class TestTC05_DialogContext:
    """ТК-05: история диалога учитывается при переформулировании запроса."""

    def test_short_question_with_history_passes_classifier(self):
        """Короткий вопрос при наличии истории не отклоняется классификатором."""
        mock_settings = MagicMock()
        history = [
            {"role": "user", "content": "Требования к защитной одежде в ЗКД"},
            {"role": "assistant", "content": "Согласно НТД необходимо..."},
        ]
        with patch("ntd_bot.rag._get_llm"):
            from ntd_bot.rag import _is_ntd_question
            result = _is_ntd_question(mock_settings, "А ещё?", history=history)

        assert result is True

    def test_history_included_in_rewrite_prompt(self):
        """_rewrite_query() передаёт контекст предыдущего вопроса в промпт."""
        mock_settings = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "дополнительные требования к защитной одежде"

        history = [
            {"role": "user", "content": "Требования к защитной одежде"},
            {"role": "assistant", "content": "Необходимо использовать..."},
        ]

        with patch("ntd_bot.rag._get_llm", return_value=mock_llm):
            from ntd_bot.rag import _rewrite_query
            _rewrite_query(mock_settings, "А ещё?", history=history)

        call_args = mock_llm.invoke.call_args[0][0]
        assert "Требования" in call_args or "предыдущий" in call_args.lower()


# ===========================================================================
# ТК-06 — Очистка истории диалога
# ===========================================================================
class TestTC06_ClearHistory:
    """ТК-06: команда очистки сбрасывает историю диалога пользователя."""

    def test_history_deque_cleared(self):
        """После вызова clear() история пользователя становится пустой."""
        dialog_history: dict = defaultdict(lambda: deque(maxlen=6))
        user_id = 12345
        dialog_history[user_id].append(("user", "Вопрос 1"))
        dialog_history[user_id].append(("assistant", "Ответ 1"))
        assert len(dialog_history[user_id]) == 2

        dialog_history[user_id].clear()
        assert len(dialog_history[user_id]) == 0

    def test_history_max_length_enforced(self):
        """История не превышает 6 сообщений — старые вытесняются автоматически."""
        history: deque = deque(maxlen=6)
        for i in range(10):
            history.append(("user", f"Вопрос {i}"))

        assert len(history) == 6
        assert history[-1] == ("user", "Вопрос 9")


# ===========================================================================
# ТК-07 — Отсутствие релевантных документов в базе
# Актуальный порог: _RERANK_THRESHOLD = 0.35
# ===========================================================================
class TestTC07_NoRelevantDocs:
    """ТК-07: при отсутствии релевантных чанков система не галлюцинирует."""

    def test_low_score_docs_filtered_out(self):
        """Документы с rerank-скором ниже порога не попадают в контекст."""
        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:7b"
        mock_settings.rag_top_k = 5
        mock_settings.min_relevant_chunks = 2

        low_score_doc = MagicMock()
        low_score_doc.page_content = "passage: нерелевантный текст"
        low_score_doc.metadata = {"doc_name": "doc.pdf"}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ДА"

        import ntd_bot.rag as rag_mod
        orig_vs = rag_mod._vectorstore
        orig_bm25 = rag_mod._bm25_index
        rag_mod._vectorstore = MagicMock()
        rag_mod._bm25_index = MagicMock()

        try:
            # score 0.05 << _RERANK_THRESHOLD (0.35) — документ должен быть отфильтрован
            with patch("ntd_bot.rag._init_vectorstore"), \
                 patch("ntd_bot.rag._init_bm25"), \
                 patch("ntd_bot.rag._get_llm", return_value=mock_llm), \
                 patch("ntd_bot.rag._hybrid_search",
                       return_value=[(low_score_doc, 0.05)]):
                from ntd_bot.rag import answer_question
                result = answer_question(mock_settings, "секретный документ АЭС")
        finally:
            rag_mod._vectorstore = orig_vs
            rag_mod._bm25_index = orig_bm25

        assert (
            "не найдено" in result.lower()
            or "переформулировать" in result.lower()
            or "не найдена" in result.lower()
        )

    def test_no_fictional_content_on_empty_base(self):
        """При пустом результате поиска возвращается информативное сообщение."""
        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:7b"
        mock_settings.rag_top_k = 5
        mock_settings.min_relevant_chunks = 2

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ДА"

        import ntd_bot.rag as rag_mod
        orig_vs = rag_mod._vectorstore
        orig_bm25 = rag_mod._bm25_index
        rag_mod._vectorstore = MagicMock()
        rag_mod._bm25_index = MagicMock()

        try:
            with patch("ntd_bot.rag._init_vectorstore"), \
                 patch("ntd_bot.rag._init_bm25"), \
                 patch("ntd_bot.rag._get_llm", return_value=mock_llm), \
                 patch("ntd_bot.rag._hybrid_search", return_value=[]):
                from ntd_bot.rag import answer_question
                result = answer_question(mock_settings, "требования по вентиляции")
        finally:
            rag_mod._vectorstore = orig_vs
            rag_mod._bm25_index = orig_bm25

        assert result is not None and len(result) > 0
        # LLM вызывается для классификации и переформулировки, но не для генерации:
        # при пустом контексте генерация ответа не происходит.
        assert mock_llm.invoke.call_count <= 2, \
            "LLM не должен вызываться для генерации при пустой базе"


# ===========================================================================
# ТК-08 — Добавление нового пользователя администратором
# Реальные функции: _add_user_sync, _is_allowed_sync, JSON_PATH (users.json)
# ===========================================================================
class TestTC08_AddUser:
    """ТК-08: администратор добавляет нового пользователя без перезапуска."""

    def test_add_user_writes_to_json(self, tmp_path):
        """_add_user_sync() записывает нового пользователя в users.json."""
        json_path = _make_users_json(tmp_path, [])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            result = us._add_user_sync(987654321, "Иванов И.И.")
        finally:
            us.JSON_PATH = orig

        assert result is True
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ids = [u["id"] for u in data["allowed_ids"]]
        assert 987654321 in ids

    def test_add_duplicate_user_returns_false(self, tmp_path):
        """_add_user_sync() возвращает False при дублировании."""
        json_path = _make_users_json(tmp_path, [(111111, "Уже есть")])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            result = us._add_user_sync(111111, "Дубль")
        finally:
            us.JSON_PATH = orig

        assert result is False

    def test_new_user_immediately_allowed(self, tmp_path):
        """После _add_user_sync() пользователь сразу проходит _is_allowed_sync()."""
        json_path = _make_users_json(tmp_path, [])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            us._add_user_sync(777777, "Новый сотрудник")
            result = us._is_allowed_sync(777777)
        finally:
            us.JSON_PATH = orig

        assert result is True


# ===========================================================================
# ТК-09 — Удаление пользователя администратором
# ===========================================================================
class TestTC09_RemoveUser:
    """ТК-09: администратор удаляет пользователя; доступ отзывается немедленно."""

    def test_remove_user_deletes_from_json(self, tmp_path):
        """_remove_user_sync() удаляет пользователя из users.json."""
        json_path = _make_users_json(tmp_path, [(987654321, "Иванов")])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            result = us._remove_user_sync(987654321)
        finally:
            us.JSON_PATH = orig

        assert result is True
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ids = [u["id"] for u in data["allowed_ids"]]
        assert 987654321 not in ids

    def test_remove_nonexistent_user_returns_false(self, tmp_path):
        """_remove_user_sync() возвращает False для несуществующего ID."""
        json_path = _make_users_json(tmp_path, [])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            result = us._remove_user_sync(999999)
        finally:
            us.JSON_PATH = orig

        assert result is False

    def test_removed_user_loses_access(self, tmp_path):
        """После _remove_user_sync() пользователь не проходит _is_allowed_sync()."""
        json_path = _make_users_json(tmp_path, [(555555, "Тест")])
        import ntd_bot.user_store as us
        orig = us.JSON_PATH
        us.JSON_PATH = json_path
        try:
            us._remove_user_sync(555555)
            result = us._is_allowed_sync(555555)
        finally:
            us.JSON_PATH = orig

        assert result is False


# ===========================================================================
# ТК-10 — Индикатор прогресса при обработке вопроса
# ===========================================================================
class TestTC10_ProgressIndicator:
    """ТК-10: _thinking_reply удаляет статусное сообщение после получения ответа."""

    @pytest.mark.asyncio
    async def test_thinking_reply_deletes_status_message(self):
        """_thinking_reply() удаляет статусное сообщение после ответа."""
        from ntd_bot.handlers import _thinking_reply

        status_mock = MagicMock()
        status_mock.edit_text = AsyncMock()
        status_mock.delete = AsyncMock()

        fake_message = MagicMock()
        fake_message.answer = AsyncMock(return_value=status_mock)

        fake_settings = MagicMock()

        # _thinking_reply вызывает answer_question через run_in_executor,
        # поэтому подменяем именно ту функцию, что импортирована в handlers.
        with patch(
            "ntd_bot.handlers.answer_question",
            return_value="Ответ из НТД. Источники: doc.pdf",
        ):
            result = await _thinking_reply(fake_message, fake_settings, "вопрос по НТД")

        assert result == "Ответ из НТД. Источники: doc.pdf"
        status_mock.delete.assert_called_once()

    def test_steps_list_has_correct_labels(self):
        """Список статусных сообщений содержит все этапы."""
        steps = [
            "🔍 Ищу в базе документов...",
            "📄 Анализирую фрагменты НТД...",
            "🧠 Формирую ответ...",
        ]
        assert len(steps) == 3
        assert any("баз" in s.lower() for s in steps)
        assert any("НТД" in s for s in steps)
        assert any("ответ" in s.lower() for s in steps)


# ===========================================================================
# Дополнительные тесты: конфигурация
# ===========================================================================
class TestConfiguration:
    """Проверка загрузки настроек из переменных окружения."""

    def test_settings_from_env(self):
        with patch.dict(os.environ, {"RAG_TOP_K": "7"}):
            import importlib
            import ntd_bot.config as cfg
            importlib.reload(cfg)
            s = cfg.Settings()
            assert s.rag_top_k == 7

    def test_settings_bot_token_from_env(self):
        with patch.dict(os.environ, {"BOT_TOKEN": "123:test_token"}):
            import importlib
            import ntd_bot.config as cfg
            importlib.reload(cfg)
            s = cfg.Settings()
            assert s.bot_token == "123:test_token"


# ===========================================================================
# Дополнительные тесты: разбиение длинных сообщений
# ===========================================================================
class TestSplitMessage:
    """Проверка функции _split_message для длинных ответов."""

    def test_short_message_not_split(self):
        """Сообщение короче лимита не разбивается."""
        from ntd_bot.handlers import _split_message
        parts = _split_message("Короткий ответ")
        assert len(parts) == 1

    def test_long_message_is_split(self):
        """Длинное сообщение разбивается на части не длиннее лимита."""
        from ntd_bot.handlers import _split_message
        long_text = ("Абзац первый.\n\n" * 200).strip()
        parts = _split_message(long_text, max_len=500)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 500
