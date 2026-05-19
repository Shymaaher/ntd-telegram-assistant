
import logging
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from .config import Settings
from .embeddings import get_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def ingest_documents(reset: bool = False) -> None:
    settings = Settings()
    t_start = time.perf_counter()
    logger.info(" Запуск индексации НТД")
    embeddings = get_embeddings(settings)

    if reset:
        tmp = Chroma(
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
            collection_name="ntd_docs",
        )
        tmp.delete_collection()
        logger.info("🗑  Старая база полностью удалена")

    vectorstore = Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
        collection_name="ntd_docs",
        collection_metadata={"hnsw:space": "cosine"},
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n",  # крупные блоки
            "\n\n",  # абзацы
            "\n",  # строки
            ". ",  # предложения
            "; ",  # части сложных предложений
            ", ",  # перечисления
            " ",  # слова
            "",  # символы (последний резерв)
        ],
        keep_separator=True,
        length_function=len,
    )

    doc_dir = Path(settings.documents_dir)
    if not doc_dir.exists():
        logger.error("Папка с документами не найдена: %s", doc_dir)
        return

    total_chunks = 0
    skipped = 0
    errors = 0

    files = sorted(f for f in doc_dir.iterdir() if f.is_file())
    logger.info("Найдено файлов: %d", len(files))

    for file_path in files:
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Пропущен (не поддерживается): %s", file_path.name)
            continue

        logger.info("📄 Обрабатываю: %s", file_path.name)

        try:
            existing = vectorstore.get(where={"doc_name": file_path.name})
            if existing["ids"]:
                logger.info("   ✅ Уже в базе (%d чанков), пропускаем", len(existing["ids"]))
                skipped += 1
                continue
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")

            docs = loader.load()
            if not docs:
                logger.warning("   ⚠️  Файл пустой или не читается: %s", file_path.name)
                continue

            split_docs = text_splitter.split_documents(docs)

            for doc in split_docs:
                doc.page_content = f"passage: {doc.page_content}"
                doc.metadata["doc_name"] = file_path.name

            if not split_docs:
                logger.warning("     Нет чанков после разбивки: %s", file_path.name)
                continue

            t_doc = time.perf_counter()
            for i in range(0, len(split_docs), BATCH_SIZE):
                vectorstore.add_documents(split_docs[i:i + BATCH_SIZE])
            elapsed = time.perf_counter() - t_doc

            total_chunks += len(split_docs)
            logger.info("    %s → %d чанков (%.1fs)", file_path.name, len(split_docs), elapsed)

        except Exception as e:
            logger.error("    Ошибка при обработке %s: %s", file_path.name, e, exc_info=True)
            errors += 1

    elapsed_total = time.perf_counter() - t_start
    logger.info(
        "\nИндексация завершена за %.1fs\n"
        "   Добавлено чанков : %d\n"
        "   Пропущено файлов : %d (уже в базе)\n"
        "   Ошибок           : %d",
        elapsed_total, total_chunks, skipped, errors,
    )


if __name__ == "__main__":
    ingest_documents(reset=True)