# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Powered Bookmark Assistant — a Python package (`bookmark_app/`) that reads Chrome bookmarks, generates LLM-powered descriptions, stores them in a FAISS vector database, and provides a conversational Gradio chat UI with streaming for semantic search over bookmarks.

## Commands

```bash
# Install dependencies (use a venv)
pip install -r requirements.txt

# Run the app (launches Gradio web UI)
python -m bookmark_app
# or
python run.py

# Run the MCP server (stdio transport, for Claude Code / Cursor / etc.)
python run_mcp.py
```

## Environment

- Requires an `OPENAI_API_KEY` environment variable (or `.env` file) for the OpenAI API.
- Copy `.env.example` to `.env` and fill in your key.
- Optionally set `BOOKMARKS_PATH`, `LLM_MODEL`, `EMBEDDING_MODEL`, `RETRIEVAL_K`, `LOG_LEVEL` in `.env`.

## Architecture

The application is structured as a Python package with clear module boundaries:

```
bookmark_app/
├── __init__.py       # Package marker + version
├── __main__.py       # python -m bookmark_app support
├── config.py         # .env loading, settings, path detection, logging
├── bookmarks.py      # Chrome extraction, JSON cache, pickle migration
├── descriptions.py   # Async LLM description generation with batching
├── vectorstore.py    # FAISS vector store management
├── agent.py          # LangGraph ReAct agent with system prompt
├── ui.py             # Gradio 5 UI with streaming + main() orchestrator
└── mcp_server.py     # MCP server: tools, resources, prompt for AI clients
```

### Pipeline (executed by `ui.main()`):

1. **Config** — Load `.env`, set up logging, validate API key and bookmarks path.
2. **Bookmark extraction** — Reads Chrome's `Bookmarks` JSON, recursively flattens into `{folder, name, url}` dicts.
3. **Merge with cache** — Compares fresh bookmarks against `all_bookmarks.json` cache. Auto-migrates legacy `all_bookmarks.pkl`.
4. **Description generation** — Async parallel (semaphore-limited) calls to gpt-4.1 for bookmarks missing descriptions.
5. **Vector store** — Embeds bookmarks with `text-embedding-3-large` into FAISS. Only new docs added on subsequent runs.
6. **ReAct agent** — LangGraph agent with system prompt, retrieve tool, and memory checkpointing.
7. **Gradio UI** — Themed chat interface with token-level streaming, example queries, and settings accordion.

## Key Data Files

- `all_bookmarks.json` — JSON cache of enriched bookmarks (gitignored).
- `vector_store/` — FAISS index directory (gitignored).
- `.env` — Environment variables (gitignored). Copy from `.env.example`.

## Key Dependencies

- **LangChain / LangGraph** — Agent framework, document model, FAISS integration
- **OpenAI** — LLM (gpt-4.1) and embeddings (text-embedding-3-large)
- **FAISS (faiss-cpu)** — Local vector similarity search
- **Gradio** — Streaming chat web UI with theming
- **python-dotenv** — Environment variable management
- **MCP (mcp[cli])** — Model Context Protocol server SDK

## MCP Server

The project includes an MCP server (`bookmark_app/mcp_server.py`) that exposes bookmark operations to any MCP-compatible AI client (Claude Code, Cursor, etc.).

### Tools
- **search_bookmarks(query, k=10)** — Semantic similarity search over bookmarks
- **list_bookmarks(folder, keyword, limit=20)** — Filter bookmarks by folder/keyword
- **get_bookmark_stats()** — Summary statistics (counts, folders, coverage)
- **refresh_bookmarks()** — Re-extract from Chrome and rebuild the vector store

### Resources
- **bookmarks://folders** — List of all bookmark folder paths

### Prompts
- **find_bookmarks(topic)** — Pre-built prompt template for bookmark search

### Usage with Claude Code

Add to your MCP config (e.g. `~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "bookmark-ai": {
      "command": "python",
      "args": ["run_mcp.py"],
      "cwd": "/path/to/Bookmark_AI"
    }
  }
}
```
