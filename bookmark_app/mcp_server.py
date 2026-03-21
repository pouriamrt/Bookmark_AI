"""MCP server exposing Bookmark AI tools and resources."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from mcp.server.fastmcp import Context, FastMCP

from . import config
from .bookmarks import (
    load_cache,
    load_chrome_bookmarks,
    merge_bookmarks,
    save_cache,
)
from .descriptions import generate_all_descriptions_sync
from .vectorstore import (
    bookmarks_to_documents,
    load_or_create_vectorstore,
)

logger = logging.getLogger(__name__)

MAX_LIST_LIMIT = 100


@dataclass
class AppContext:
    """Shared state initialized at startup."""

    bookmarks: list[dict] = field(default_factory=list)
    vector_store: FAISS | None = None
    folders: list[str] = field(default_factory=list)


def _build_app_state() -> tuple[list[dict], FAISS, list[str]]:
    """Load bookmarks, generate descriptions, build vector store.

    Extracted so both the lifespan and refresh_bookmarks share one pipeline.
    Saves cache incrementally every 10 descriptions via the on_progress callback.
    """
    fresh = load_chrome_bookmarks()
    cached = load_cache(config.BOOKMARKS_CACHE_PATH)
    bookmarks = merge_bookmarks(fresh, cached)

    def _save_progress():
        save_cache(bookmarks, config.BOOKMARKS_CACHE_PATH)

    bookmarks = generate_all_descriptions_sync(
        bookmarks, on_progress=_save_progress,
    )
    save_cache(bookmarks, config.BOOKMARKS_CACHE_PATH)

    documents = bookmarks_to_documents(bookmarks)
    vector_store = load_or_create_vectorstore(documents)

    folders = sorted(
        {bm.get("folder", "") for bm in bookmarks if bm.get("folder")}
    )
    return bookmarks, vector_store, folders


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize bookmarks and vector store once at startup."""
    config.load_env()
    config.setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    config.validate_config()

    # asyncio.to_thread avoids "asyncio.run() inside running loop" crash
    bookmarks, vector_store, folders = await asyncio.to_thread(
        _build_app_state,
    )

    logger.info(
        "MCP server ready: %d bookmarks, %d folders",
        len(bookmarks),
        len(folders),
    )

    yield AppContext(
        bookmarks=bookmarks, vector_store=vector_store, folders=folders,
    )


mcp = FastMCP(
    "Bookmark AI",
    instructions=(
        "Search and explore the user's Chrome bookmarks using semantic search. "
        "Use search_bookmarks to find relevant bookmarks by meaning, "
        "list_bookmarks to browse by folder or keyword, "
        "and get_bookmark_stats for a high-level summary."
    ),
    lifespan=app_lifespan,
)


# -- Pure logic (testable without MCP runtime) ---------------------------


def _search_bookmarks_logic(app: AppContext, query: str, k: int = 10) -> str:
    """Search bookmarks by semantic similarity (pure logic)."""
    k = max(1, min(30, k))
    docs = app.vector_store.similarity_search(query, k=k)
    if not docs:
        return "No bookmarks found matching your query."
    lines = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get("source", "")
        folder = doc.metadata.get("folder", "")
        name = doc.page_content.split("\n")[0]
        description = "\n".join(doc.page_content.split("\n")[2:]).strip()
        lines.append(
            f"{i}. [{name}]({url})\n   Folder: {folder}\n   {description}"
        )
    return "\n\n".join(lines)


def _list_bookmarks_logic(
    app: AppContext,
    folder: str = "",
    keyword: str = "",
    limit: int = 20,
) -> str:
    """List bookmarks with optional filters (pure logic)."""
    limit = max(1, min(MAX_LIST_LIMIT, limit))
    results = app.bookmarks
    if folder:
        results = [
            bm for bm in results
            if folder.lower() in bm.get("folder", "").lower()
        ]
    if keyword:
        kw = keyword.lower()
        results = [
            bm for bm in results
            if kw in bm.get("name", "").lower()
            or kw in bm.get("description", "").lower()
        ]
    results = results[:limit]
    if not results:
        return "No bookmarks match the given filters."
    lines = []
    for i, bm in enumerate(results, 1):
        desc = bm.get("description", "")
        lines.append(
            f"{i}. [{bm['name']}]({bm['url']})\n"
            f"   Folder: {bm.get('folder', '')}\n   {desc}"
        )
    return "\n\n".join(lines)


def _get_bookmark_stats_logic(app: AppContext) -> str:
    """Get summary statistics (pure logic)."""
    bookmarks = app.bookmarks
    total = len(bookmarks)
    with_desc = sum(1 for bm in bookmarks if bm.get("description"))
    folders: dict[str, int] = {}
    for bm in bookmarks:
        f = bm.get("folder", "(no folder)")
        folders[f] = folders.get(f, 0) + 1
    top_folders = sorted(folders.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    lines = [
        f"Total bookmarks: {total}",
        f"With descriptions: {with_desc}/{total}"
        f" ({100 * with_desc // max(total, 1)}%)",
        f"Unique folders: {len(folders)}",
        "",
        "Top folders:",
    ]
    for f, count in top_folders:
        lines.append(f"  - {f}: {count} bookmarks")
    return "\n".join(lines)


# -- MCP Tools ------------------------------------------------------------


@mcp.tool()
def search_bookmarks(query: str, k: int = 10, ctx: Context = None) -> str:
    """Search bookmarks by semantic similarity.

    Args:
        query: Natural language search query (e.g. "machine learning tutorials")
        k: Number of results to return (1-30, default 10)
    """
    app: AppContext = ctx.request_context.lifespan_context
    return _search_bookmarks_logic(app, query, k)


@mcp.tool()
def list_bookmarks(
    folder: str = "",
    keyword: str = "",
    limit: int = 20,
    ctx: Context = None,
) -> str:
    """List bookmarks, optionally filtered by folder path or keyword.

    Args:
        folder: Filter by folder path (substring match, e.g. "/Tools")
        keyword: Filter by name or description keyword (case-insensitive)
        limit: Maximum number of results (1-100, default 20)
    """
    app: AppContext = ctx.request_context.lifespan_context
    return _list_bookmarks_logic(app, folder, keyword, limit)


@mcp.tool()
def get_bookmark_stats(ctx: Context = None) -> str:
    """Get summary statistics about the bookmark collection.

    Returns total count, folder count, top folders, and description coverage.
    """
    app: AppContext = ctx.request_context.lifespan_context
    return _get_bookmark_stats_logic(app)


@mcp.tool()
async def refresh_bookmarks(ctx: Context = None) -> str:
    """Re-extract bookmarks from Chrome and update the vector store.

    Call this after adding or removing Chrome bookmarks to sync the MCP server.
    """
    app: AppContext = ctx.request_context.lifespan_context

    # Run in thread to avoid asyncio.run() crash inside running loop
    bookmarks, vector_store, folders = await asyncio.to_thread(
        _build_app_state,
    )

    # Update shared state
    app.bookmarks = bookmarks
    app.vector_store = vector_store
    app.folders = folders

    return f"Refreshed: {len(bookmarks)} bookmarks across {len(folders)} folders."


# -- MCP Resources ---------------------------------------------------------


@mcp.resource("bookmarks://folders")
def list_folders(ctx: Context = None) -> str:
    """List all bookmark folder paths."""
    app: AppContext = ctx.request_context.lifespan_context
    if not app.folders:
        return "No folders found."
    return "\n".join(app.folders)


# -- MCP Prompts -----------------------------------------------------------


@mcp.prompt()
def find_bookmarks(topic: str) -> str:
    """Generate a prompt to search bookmarks about a topic."""
    topic = topic.strip()[:200]
    return (
        f"Search the user's Chrome bookmarks for anything related to: {topic}\n\n"
        "Use the search_bookmarks tool, then present results as a numbered list "
        "with clickable links and brief descriptions."
    )
