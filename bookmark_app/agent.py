"""LangGraph ReAct agent with system prompt and streaming."""

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .config import LLM_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an AI bookmark assistant. You help users find relevant bookmarks from \
their personal Chrome bookmark collection.

**Instructions:**
- ALWAYS use the `retrieve` tool to search the bookmark database before answering.
- Present results as a numbered markdown list with clickable links: \
`[Name](url)` followed by a brief description.
- If the retrieve tool returns no relevant results, say so honestly — do not \
make up bookmarks.
- You can handle follow-up questions and refine searches based on conversation \
context.
- Be concise and helpful.\
"""

# Mutable default so the UI slider can update it at runtime.
_retrieval_k: dict[str, int] = {"value": RETRIEVAL_K}


def set_retrieval_k(k: int) -> None:
    """Update the number of results returned by the retrieve tool."""
    _retrieval_k["value"] = k


def create_retrieve_tool(vector_store: FAISS):
    """Build a retrieval tool bound to *vector_store*."""

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve bookmarks related to a query."""
        retrieved_docs = vector_store.similarity_search(
            query, k=_retrieval_k["value"],
        )
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve


def get_llm() -> ChatOpenAI:
    """Create a streaming ``ChatOpenAI`` instance."""
    return ChatOpenAI(model=LLM_MODEL, streaming=True)


def create_agent(llm: ChatOpenAI, vector_store: FAISS):
    """Create the ReAct agent with memory and system prompt."""
    retrieve = create_retrieve_tool(vector_store)
    memory = MemorySaver()
    agent = create_react_agent(
        llm,
        [retrieve],
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    logger.info("Agent created with model=%s, k=%d", LLM_MODEL, _retrieval_k["value"])
    return agent
