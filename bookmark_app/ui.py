"""Gradio UI with streaming and application orchestrator."""

import logging

import gradio as gr
from gradio.themes.utils import colors
from langchain_core.messages import AIMessageChunk

from .agent import create_agent, get_llm, set_retrieval_k
from .bookmarks import load_cache, load_chrome_bookmarks, merge_bookmarks, save_cache
from .config import (
    BOOKMARKS_CACHE_PATH,
    RETRIEVAL_K,
    load_env,
    setup_logging,
    validate_config,
)
from .descriptions import generate_all_descriptions_sync
from .vectorstore import bookmarks_to_documents, load_or_create_vectorstore

logger = logging.getLogger(__name__)

AGENT_CONFIG = {"configurable": {"thread_id": "default"}}


def _build_bot_response(agent):
    """Return a generator-based bot_response function bound to *agent*."""

    def bot_response(message: str, history: list, k: int) -> str:
        set_retrieval_k(int(k))
        accumulated = ""
        for chunk_event in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=AGENT_CONFIG,
            stream_mode="messages",
        ):
            chunk, metadata = chunk_event
            # Only yield AI text content (skip tool calls / tool results)
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                accumulated += chunk.content
                yield accumulated

        # If nothing was streamed (edge case), fall back to empty string
        if not accumulated:
            yield ""

    return bot_response


def main() -> None:
    """Full initialization pipeline -> launch Gradio UI."""

    # -- 1. Configuration -------------------------------------------------
    load_env()
    setup_logging()

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger.info("Starting Bookmark AI ...")
    validate_config()

    # -- 2. Bookmarks -----------------------------------------------------
    fresh_bookmarks = load_chrome_bookmarks()
    cached_bookmarks = load_cache(BOOKMARKS_CACHE_PATH)
    bookmarks = merge_bookmarks(fresh_bookmarks, cached_bookmarks)

    # -- 3. Descriptions --------------------------------------------------
    bookmarks = generate_all_descriptions_sync(bookmarks)
    save_cache(bookmarks, BOOKMARKS_CACHE_PATH)

    # -- 4. Vector store --------------------------------------------------
    documents = bookmarks_to_documents(bookmarks)
    vector_store = load_or_create_vectorstore(documents)

    # -- 5. Agent ---------------------------------------------------------
    llm = get_llm()
    agent = create_agent(llm, vector_store)
    bot_response = _build_bot_response(agent)

    # -- 6. Gradio UI -----------------------------------------------------
    logger.info("Launching Gradio UI ...")

    theme = gr.themes.Soft(
        primary_hue=colors.blue,
        secondary_hue=colors.indigo,
    )

    examples = [
        ["Find my bookmarks about machine learning"],
        ["Do I have any bookmarks related to Python tutorials?"],
        ["Show me bookmarks about web development"],
        ["Find bookmarks related to productivity tools"],
        ["What cooking or recipe bookmarks do I have?"],
    ]

    k_slider = gr.Slider(
        minimum=1,
        maximum=30,
        value=RETRIEVAL_K,
        step=1,
        label="Number of results to retrieve",
    )

    demo = gr.ChatInterface(
        fn=bot_response,
        title="Bookmark AI",
        description=(
            "Search your Chrome bookmarks using natural language. "
            "Ask anything and I'll find the most relevant bookmarks for you."
        ),
        examples=examples,
        save_history=True,
        textbox=gr.Textbox(
            placeholder="Ask me about your bookmarks ...",
            container=False,
            scale=7,
        ),
        additional_inputs=[k_slider],
        additional_inputs_accordion=gr.Accordion("Settings", open=False),
    )

    demo.launch(theme=theme)
