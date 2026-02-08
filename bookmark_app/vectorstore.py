"""FAISS vector store management."""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import EMBEDDING_MODEL, VECTOR_STORE_DIR

logger = logging.getLogger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """Create an ``OpenAIEmbeddings`` instance with the configured model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def bookmarks_to_documents(bookmarks: list[dict]) -> list[Document]:
    """Convert bookmark dicts into LangChain ``Document`` objects."""
    docs: list[Document] = []
    for bm in bookmarks:
        page_content = f"{bm['name']}\n\n{bm['description']}"
        metadata = {"source": bm["url"]}
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def load_or_create_vectorstore(
    documents: list[Document],
    store_dir: str | None = None,
) -> FAISS:
    """Load an existing FAISS index or create one from *documents*.

    Only documents whose ``source`` URL is not already in the store are added
    on subsequent runs.
    """
    store_dir = store_dir or VECTOR_STORE_DIR
    store_path = Path(store_dir)
    embeddings = get_embeddings()

    if store_path.exists():
        logger.info("Loading existing vector store from %s", store_path)
        vector_store = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True,
        )
        existing_sources = {
            doc.metadata["source"]
            for doc in vector_store.docstore._dict.values()
        }
        new_docs = [
            d for d in documents if d.metadata["source"] not in existing_sources
        ]
        if new_docs:
            logger.info("Adding %d new documents to vector store", len(new_docs))
            vector_store.add_documents(new_docs)
            vector_store.save_local(store_path)
        else:
            logger.info("Vector store is up to date")
    else:
        logger.info("Creating new vector store with %d documents", len(documents))
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(store_path)

    return vector_store
