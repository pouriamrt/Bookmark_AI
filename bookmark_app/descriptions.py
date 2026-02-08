"""Async LLM-powered bookmark description generation."""

import asyncio
import logging

from langchain_openai import ChatOpenAI

from .config import LLM_MODEL

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 5

PROMPT_TEMPLATE = (
    "Generate a concise description (2-3 sentences) for the following bookmark.\n"
    "Folder: {folder}\n"
    "Name: {name}\n"
    "URL: {url}\n\n"
    "If you cannot access the URL, infer the description from the name and folder "
    "context. Do NOT mention that you cannot access the URL."
)


class _ProgressCounter:
    """Simple atomic counter for tracking completed tasks."""

    def __init__(self, total: int):
        self.completed = 0
        self.total = total

    def increment(self) -> None:
        self.completed += 1
        if self.completed % 10 == 0 or self.completed == self.total:
            logger.info(
                "Descriptions: %d/%d complete", self.completed, self.total,
            )


async def _generate_one(
    bookmark: dict,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    progress: _ProgressCounter,
) -> str:
    """Generate a description for a single bookmark."""
    prompt = PROMPT_TEMPLATE.format(
        folder=bookmark.get("folder", ""),
        name=bookmark["name"],
        url=bookmark["url"],
    )
    async with semaphore:
        try:
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception:
            logger.exception("Failed to generate description for %s", bookmark["url"])
            return f"Bookmark: {bookmark['name']}"
        finally:
            progress.increment()


async def _generate_all(bookmarks: list[dict], llm: ChatOpenAI) -> list[dict]:
    """Generate missing descriptions concurrently."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    indices: list[int] = [
        i for i, bm in enumerate(bookmarks) if "description" not in bm
    ]

    if not indices:
        logger.info("All bookmarks already have descriptions")
        return bookmarks

    total = len(indices)
    logger.info("Generating descriptions for %d bookmarks ...", total)
    progress = _ProgressCounter(total)

    tasks = [
        asyncio.create_task(
            _generate_one(bookmarks[idx], llm, semaphore, progress)
        )
        for idx in indices
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, result in zip(indices, results):
        if isinstance(result, Exception):
            logger.error("Description failed for index %d: %s", idx, result)
            bookmarks[idx]["description"] = f"Bookmark: {bookmarks[idx]['name']}"
        else:
            bookmarks[idx]["description"] = result

    logger.info("Description generation complete")
    return bookmarks


def generate_all_descriptions_sync(
    bookmarks: list[dict],
    llm: ChatOpenAI | None = None,
) -> list[dict]:
    """Synchronous wrapper -- generates missing descriptions in parallel.

    If *llm* is not provided, a new ``ChatOpenAI`` instance is created using
    the configured model.
    """
    if llm is None:
        llm = ChatOpenAI(model=LLM_MODEL)
    return asyncio.run(_generate_all(bookmarks, llm))
