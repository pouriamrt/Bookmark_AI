
# 📚 AI-Powered Bookmark Assistant

An intelligent assistant that reads your Chrome bookmarks, generates short descriptions for them, stores them in a persistent FAISS vector database, and lets you query them conversationally using **gpt-4.1** and **Gradio**.

---

## 🚀 Project Overview

This project creates a smart, searchable chatbot based on your Chrome bookmarks. It:

- Extracts all your bookmarks from Chrome.
- Automatically generates meaningful descriptions for bookmarks using a powerful LLM.
- Stores bookmark content and metadata in a local persistent FAISS vector database.
- Retrieves relevant bookmarks based on user queries using semantic search.
- Offers a simple and elegant chat interface built with **Gradio**.

This allows you to *ask questions* like:

> _"Show me resources about machine learning fundamentals."_

> _"Find articles related to personal finance advice."_

And the assistant will retrieve the best-matching bookmarks instantly.

---

## 🛠 Features

- 📥 **Bookmark Extraction:** Recursively reads and flattens Chrome's hierarchical bookmark structure.
- ✍️ **Automatic Description Generation:** Uses **gpt-4.1** to generate concise, human-readable summaries for each bookmark.
- 🗂 **Persistent Vector Search Engine:** Embeds and indexes bookmarks using **OpenAI's text-embedding-3-large** model and stores them with **FAISS** for fast semantic search.
- 🤖 **Conversational Agent:** A ReAct-style agent using **LangGraph** to handle complex queries.
- 🧠 **In-Memory Checkpointing:** Maintains state between interactions for a smoother chat experience.
- 💬 **Interactive Chat UI:** Built with **Gradio** for easy local deployment.

---

## 📂 Project Structure

| File / Folder | Purpose |
|:--------------|:--------|
| `Bookmarks.json` | Chrome bookmarks file automatically loaded |
| `all_bookmarks.pkl` | Pickled enriched bookmarks with descriptions |
| `vector_store` | Persistent FAISS vector store with embedded documents |
| `gradio_app` | Gradio-based chat interface |

---

## 📋 How It Works

1. **Extract Bookmarks:**  
   Parses Chrome's JSON-formatted bookmark file into a flat list of bookmark entries.

2. **Compare and Merge:**  
   Merges newly extracted bookmarks with previously saved ones to avoid duplicates.

3. **Generate Descriptions:**  
   For bookmarks without a description, sends prompts to the **gpt-4.1** model to generate concise summaries.

4. **Embed and Store:**  
   Converts bookmark content into embeddings and stores them using a **FAISS** vector database persisted to disk.

5. **Setup Retrieval Agent:**  
   Creates a retrieval-augmented chatbot using LangGraph's prebuilt ReAct agent.

6. **Launch Gradio Chatbot:**  
   Runs a Gradio interface where users can chat and receive bookmark recommendations.

---

## 🧩 Dependencies

- `langchain`
- `langchain_openai`
- `langgraph`
- `tqdm`
- `pickle`
- `gradio`
- `openai`
- `typing_extensions`
- `faiss-cpu`

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

1. **Export your Chrome bookmarks** if not already available.
2. The app tries to locate Chrome's bookmarks file automatically. If it can't find it, set the `BOOKMARKS_PATH` environment variable to the file's location.
3. **Run the script:**

```bash
python bookmark_app.py
```

4. **The Gradio interface will launch automatically!**  
   Open the provided URL in your browser.

---

## 📝 Example Usage

User Input:
> _"What are some useful resources on AI ethics?"_

Bot Response:
>  
> Source: [AI and Ethics Blog](https://example.com/ai-ethics)  
> Content: A curated collection of articles exploring the ethical challenges in AI development and deployment.  

---

## 📢 Notes

- If new bookmarks are added, the assistant will automatically update and describe them.
- Bookmark data and embeddings are stored **locally** (no cloud storage or external API data saving).
- The project currently uses **gpt-4.1** and **text-embedding-3-large** models via **OpenAI** API. Make sure you have access.

---

## ⚡ Future Improvements

- Add multi-turn conversation memory.
- Improve description generation with website scraping.
- Support Firefox, Edge, and other browsers.

---

## 📜 License

This project is open-source and available for use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

Thanks to the open-source community around **LangChain**, **LangGraph**, **OpenAI**, and **Gradio** for making rapid prototyping like this possible.

## 📝 How to Cite

If you use the **AI-Powered Bookmark Assistant** in your research, projects, or publications, please cite:

**Software**
> Mortezaagha, P. (2025). *AI-Powered Bookmark Assistant (v1.0.0)*. GitHub. https://github.com/your-username/ai-bookmark-assistant

### BibTeX
```bibtex
@software{mortezaagha_ai_bookmark_assistant_2025,
  author  = {Mortezaagha, Pouria},
  title   = {AI-Powered Bookmark Assistant},
  version = {1.0.0},
  year    = {2025},
  url     = {https://github.com/your-username/ai-bookmark-assistant},
  license = {MIT},
  note    = {Gradio + LangGraph chatbot for semantic search over Chrome bookmarks with persistent FAISS vector database}
}
