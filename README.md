
# 📚 AI-Powered Bookmark Assistant

An intelligent assistant that reads your Chrome bookmarks, generates short descriptions for them, stores them in a persistent FAISS vector database, and lets you query them conversationally using **gpt-4.1** and a streaming **Gradio** chat UI.

---

## 🚀 Project Overview

This project creates a smart, searchable chatbot based on your Chrome bookmarks. It:

- Extracts all your bookmarks from Chrome automatically.
- Generates meaningful descriptions in parallel using a powerful LLM.
- Stores bookmark content and metadata in a local persistent FAISS vector database.
- Retrieves relevant bookmarks based on user queries using semantic search.
- Offers a themed chat interface with real-time streaming built with **Gradio 5**.

This allows you to *ask questions* like:

> _"Show me resources about machine learning fundamentals."_

> _"Find articles related to personal finance advice."_

And the assistant will retrieve the best-matching bookmarks instantly.

---

## 🛠 Features

- 📥 **Bookmark Extraction:** Recursively reads and flattens Chrome's hierarchical bookmark structure.
- ✍️ **Async Description Generation:** Uses **gpt-4.1** with parallel async calls (semaphore-limited) to generate concise summaries for each bookmark.
- 🗂 **Persistent Vector Search Engine:** Embeds and indexes bookmarks using **OpenAI's text-embedding-3-large** model and stores them with **FAISS** for fast semantic search.
- 🤖 **Conversational Agent:** A ReAct-style agent using **LangGraph** with a comprehensive system prompt to handle complex queries.
- 🧠 **In-Memory Checkpointing:** Maintains state between interactions for a smoother chat experience.
- 💬 **Streaming Chat UI:** Built with **Gradio 5** featuring a themed interface, example queries, and adjustable settings.
- ⚙️ **Configurable:** All settings (model, retrieval count, paths) configurable via `.env` file.
- 🔄 **Auto-Migration:** Automatically migrates legacy pickle caches to JSON format.

---

## 📂 Project Structure

```
Bookmark_AI/
├── .env.example              # Environment variable template
├── .gitignore
├── CLAUDE.md                 # Claude Code instructions
├── README.md
├── LICENSE
├── run.py                    # Entry point (python run.py)
├── requirements.txt          # Direct dependencies only
├── bookmark_app/
│   ├── __init__.py           # Package marker + version
│   ├── __main__.py           # python -m bookmark_app support
│   ├── config.py             # .env loading, settings, path detection, logging
│   ├── bookmarks.py          # Chrome extraction, JSON cache, pickle migration
│   ├── descriptions.py       # Async LLM description generation with batching
│   ├── vectorstore.py        # FAISS vector store management
│   ├── agent.py              # LangGraph ReAct agent with system prompt
│   └── ui.py                 # Gradio 5 UI with streaming + main() orchestrator
```

| File / Folder | Purpose |
|:--------------|:--------|
| `all_bookmarks.json` | JSON cache of enriched bookmarks with descriptions (auto-generated) |
| `vector_store/` | Persistent FAISS vector store with embedded documents |
| `.env` | Your local environment variables (copy from `.env.example`) |

---

## 📋 How It Works

1. **Load Configuration:**
   Reads `.env` file, sets up logging, and validates that the API key and bookmarks file exist.

2. **Extract Bookmarks:**
   Parses Chrome's JSON-formatted bookmark file into a flat list of `{folder, name, url}` entries.

3. **Merge with Cache:**
   Compares freshly extracted bookmarks against the JSON cache (URL-based matching). Preserves existing descriptions, adds new bookmarks, drops removed ones.

4. **Generate Descriptions:**
   For bookmarks without a description, makes parallel async calls to **gpt-4.1** (up to 5 concurrent) to generate concise summaries. Failures produce graceful fallbacks.

5. **Embed and Store:**
   Converts bookmark content into embeddings and stores them using a **FAISS** vector database. Only new bookmarks are embedded on subsequent runs.

6. **Setup Retrieval Agent:**
   Creates a ReAct agent with a system prompt that instructs it to always search bookmarks and format results as clickable markdown links.

7. **Launch Streaming Chat UI:**
   Runs a themed Gradio interface with token-by-token streaming, example queries, and a settings panel for adjusting the number of results retrieved.

---

## 🧩 Dependencies

- `langchain` / `langchain-community` / `langchain-openai`
- `langgraph`
- `openai`
- `faiss-cpu`
- `gradio`
- `python-dotenv`

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

1. **Clone the repository and create a virtual environment:**

```bash
git clone https://github.com/pouriamrt/Bookmark_AI.git
cd Bookmark_AI
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

2. **Set up your environment:**

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Run the app:**

```bash
python -m bookmark_app
# or
python run.py
```

4. **The Gradio interface will launch automatically!**
   Open the provided URL in your browser.

The app auto-detects your Chrome bookmarks file. If it can't find it, set `BOOKMARKS_PATH` in your `.env` file.

---

## ⚙️ Configuration

All settings are optional and configurable via `.env`:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `BOOKMARKS_PATH` | Auto-detected | Path to Chrome's Bookmarks file |
| `LLM_MODEL` | `gpt-4.1` | LLM model for descriptions and agent |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model for vector search |
| `RETRIEVAL_K` | `10` | Number of results per search query |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## 📝 Example Usage

User Input:
> _"What are some useful resources on AI ethics?"_

Bot Response:
>
> 1. [AI and Ethics Blog](https://example.com/ai-ethics) — A curated collection of articles exploring the ethical challenges in AI development and deployment.
> 2. [Responsible AI Toolkit](https://example.com/responsible-ai) — Tools and frameworks for building fair and transparent AI systems.

---

## 📢 Notes

- If new bookmarks are added to Chrome, the assistant will automatically detect and describe them on the next run.
- Bookmark data and embeddings are stored **locally** — no cloud storage or external data saving.
- Legacy `all_bookmarks.pkl` files are automatically migrated to JSON format on first run.
- The project uses **gpt-4.1** and **text-embedding-3-large** models via the **OpenAI** API. Make sure you have access.

---

## ⚡ Future Improvements

- Improve description generation with website scraping.
- Support Firefox, Edge, and other browsers.
- Add bookmark categorization and tagging.

---

## 📜 License

This project is open-source and available for use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

Thanks to the open-source community around **LangChain**, **LangGraph**, **OpenAI**, and **Gradio** for making rapid prototyping like this possible.

## 📝 How to Cite

If you use the **AI-Powered Bookmark Assistant** in your research, projects, or publications, please cite:

**Software**
> Mortezaagha, P. (2025). *AI-Powered Bookmark Assistant (v2.0.0)*. GitHub. https://github.com/pouriamrt/Bookmark_AI

### BibTeX
```bibtex
@software{mortezaagha_ai_bookmark_assistant_2025,
  author  = {Mortezaagha, Pouria},
  title   = {AI-Powered Bookmark Assistant},
  version = {2.0.0},
  year    = {2025},
  url     = {https://github.com/pouriamrt/Bookmark_AI},
  license = {MIT},
  note    = {Gradio + LangGraph chatbot for semantic search over Chrome bookmarks with persistent FAISS vector database}
}
```
