# NTD Telegram Assistant

Telegram-бот — ассистент оперативного персонала для поиска информации в нормативно-технической документации (НТД) объектов атомной энергетики. Работает локально, без отправки данных во внешние LLM.

## Возможности

* **Гибридный RAG-поиск** по базе документов: векторный (Chroma + multilingual-e5-large) + BM25, объединение через Reciprocal Rank Fusion.
* **Локальная LLM через Ollama** — данные не покидают контур.
* **Распознавание голосовых сообщений** через faster-whisper.
* **Контекст диалога**: помнит последние 3 пары вопрос–ответ для уточняющих вопросов.
* **Allowlist пользователей**: доступ только по Telegram ID.
* **Админ-панель**: добавление/удаление пользователей прямо из чата.
* **Указание источников** в каждом ответе.

## Стек

|Слой|Технология|
|-|-|
|Bot framework|aiogram 3|
|Эмбеддинги|`intfloat/multilingual-e5-large` (HuggingFace)|
|Векторная БД|Chroma|
|Лексический поиск|rank-bm25|
|LLM|Ollama (по умолчанию `qwen2.5:14b`)|
|ASR|faster-whisper (small, int8)|
|Конфиг|pydantic-settings|

## Архитектура

```
Пользователь (Telegram)
       │
       ▼
   aiogram router ── AllowlistMiddleware ── SettingsMiddleware
       │
       ▼
   handlers.py ── ASR (если голосовое)
       │
       ▼
   rag.py
       ├── классификатор НТД-вопроса (LLM)
       ├── переформулировка запроса (LLM)
       ├── гибридный поиск (Chroma + BM25 → RRF)
       └── генерация ответа (LLM)
```

## Установка

### 1\. Клонировать репозиторий

```bash
git clone https://github.com/<your-user>/ntd-telegram-assistant.git
cd ntd-telegram-assistant
```

### 2\. Виртуальное окружение и зависимости

```bash
python -m venv .venv
# Windows:
.venv\\\\Scripts\\\\activate
# Linux / macOS:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 3\. Установить Ollama и модель

Скачайте Ollama: [https://ollama.com](https://ollama.com)

```bash
ollama pull qwen2.5:14b
ollama serve  # обычно запускается автоматически
```

### 4\. Конфигурация

```bash
cp .env.example .env
```

Откройте `.env` и пропишите:

* `BOT\\\_TOKEN` — токен от [@BotFather](https://t.me/BotFather)
* `ADMIN\\\_IDS` — ваш Telegram ID (узнать у [@userinfobot](https://t.me/userinfobot))
* При необходимости измените пути к документам и параметры RAG.

### 5\. Положить документы и проиндексировать

Сложите файлы `.pdf` и `.txt` в `data/documents/` (или папку, указанную в `DOCUMENTS\\\_DIR`):

```bash
python -m ntd\\\_bot.ingest
```

Индексация инкрементная: уже добавленные файлы пропускаются. Для полной переиндексации откройте `ntd\\\_bot/ingest.py` и запустите с `reset=True` (последняя строка модуля).

### 6\. Запустить бота

```bash
python -m ntd\\\_bot
```

## Использование

В чате с ботом:

* **Текст** или **голосовое сообщение** с вопросом по НТД.
* `📋 Помощь` — справка.
* `🗑 Очистить историю` — сброс контекста диалога.

**Команды администратора:**

* `/adduser <id> <имя>` — добавить пользователя.
* `/removeuser <id>` — удалить пользователя.
* `/listusers` — список допущенных пользователей.

Те же действия доступны через кнопки в админ-клавиатуре.

## Структура проекта

```
ntd\\\_bot/
├── \\\_\\\_init\\\_\\\_.py
├── \\\_\\\_main\\\_\\\_.py        # python -m ntd\\\_bot
├── main.py            # точка входа, поднимает aiogram
├── config.py          # pydantic-settings, .env
├── auth.py            # middleware allowlist
├── deps.py            # middleware: проброс settings в handlers
├── handlers.py        # роутер aiogram: команды, текст, голос
├── rag.py             # гибридный поиск + LLM
├── ingest.py          # индексация документов в Chroma
├── embeddings.py      # singleton HF-эмбеддингов
├── asr.py             # faster-whisper
└── user\\\_store.py      # allowlist в users.json
```

