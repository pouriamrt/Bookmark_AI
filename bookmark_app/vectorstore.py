"""FAISS vector store management."""

import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from . import config

logger = logging.getLogger(__name__)

_INDEXED_URLS_FILE = "indexed_urls.json"


def get_embeddings() -> OpenAIEmbeddings:
    """Create an ``OpenAIEmbeddings`` instance with the configured model."""
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def bookmarks_to_documents(bookmarks: list[dict]) -> list[Document]:
    """Convert bookmark dicts into LangChain ``Document`` objects."""
    docs: list[Document] = []
    for bm in bookmarks:
        folder = bm.get("folder", "")
        page_content = f"{bm['name']}\nFolder: {folder}\n\n{bm['description']}"
        metadata = {"source": bm["url"], "folder": folder}
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def _load_indexed_urls(store_path: Path) -> set[str]:
    """Load the set of indexed URLs from the sidecar file."""
    sidecar = store_path / _INDEXED_URLS_FILE
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_indexed_urls(store_path: Path, urls: set[str]) -> None:
    """Persist the set of indexed URLs to the sidecar file."""
    sidecar = store_path / _INDEXED_URLS_FILE
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(sorted(urls), f, ensure_ascii=False, indent=2)


def load_or_create_vectorstore(
    documents: list[Document],
    store_dir: str | None = None,
) -> FAISS:
    """Load an existing FAISS index or create one from *documents*.

    New documents are added and stale documents (no longer in *documents*)
    trigger a full rebuild to keep the index in sync with Chrome bookmarks.
    """
    store_dir = store_dir or config.VECTOR_STORE_DIR
    store_path = Path(store_dir)
    embeddings = get_embeddings()

    current_urls = {d.metadata["source"] for d in documents}

    if store_path.exists():
        indexed_urls = _load_indexed_urls(store_path)
        stale_urls = indexed_urls - current_urls
        new_docs = [d for d in documents if d.metadata["source"] not in indexed_urls]

        if stale_urls:
            logger.info(
                "Rebuilding vector store: %d stale documents removed", len(stale_urls),
            )
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(store_path)
            _save_indexed_urls(store_path, current_urls)
        elif new_docs:
            logger.info("Adding %d new documents to vector store", len(new_docs))
            vector_store = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True,
            )
            vector_store.add_documents(new_docs)
            vector_store.save_local(store_path)
            _save_indexed_urls(store_path, indexed_urls | {d.metadata["source"] for d in new_docs})
        else:
            logger.info("Vector store is up to date")
            vector_store = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True,
            )
    else:
        logger.info("Creating new vector store with %d documents", len(documents))
        store_path.mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(store_path)
        _save_indexed_urls(store_path, current_urls)

    return vector_store
