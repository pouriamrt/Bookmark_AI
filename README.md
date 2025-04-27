
# 📚 AI-Powered Bookmark Assistant

An intelligent assistant that reads your Chrome bookmarks, generates short descriptions for them, stores them in an in-memory vector database, and lets you query them conversationally using **GPT-4o-mini** and **Gradio**.

---

## 🚀 Project Overview

This project creates a smart, searchable chatbot based on your Chrome bookmarks. It:

- Extracts all your bookmarks from Chrome.
- Automatically generates meaningful descriptions for bookmarks using a powerful LLM.
- Stores bookmark content and metadata in a local in-memory vector database.
- Retrieves relevant bookmarks based on user queries using semantic search.
- Offers a simple and elegant chat interface built with **Gradio**.

This allows you to *ask questions* like:

> _"Show me resources about machine learning fundamentals."_

> _"Find articles related to personal finance advice."_

And the assistant will retrieve the best-matching bookmarks instantly.

---

## 🛠 Features

- 📥 **Bookmark Extraction:** Recursively reads and flattens Chrome's hierarchical bookmark structure.
- ✍️ **Automatic Description Generation:** Uses **GPT-4o-mini** to generate concise, human-readable summaries for each bookmark.
- 🗂 **Vector Search Engine:** Embeds and indexes bookmarks using **OpenAI's text-embedding-3-large** model for semantic search.
- 🤖 **Conversational Agent:** A ReAct-style agent using **LangGraph** to handle complex queries.
- 🧠 **In-Memory Checkpointing:** Maintains state between interactions for a smoother chat experience.
- 💬 **Interactive Chat UI:** Built with **Gradio** for easy local deployment.

---

## 📂 Project Structure

| File / Folder | Purpose |
|:--------------|:--------|
| `Bookmarks.json` | Chrome bookmarks file automatically loaded |
| `all_bookmarks.pkl` | Pickled enriched bookmarks with descriptions |
| `vector_store` | In-memory vector store with embedded documents |
| `gradio_app` | Gradio-based chat interface |

---

## 📋 How It Works

1. **Extract Bookmarks:**  
   Parses Chrome's JSON-formatted bookmark file into a flat list of bookmark entries.

2. **Compare and Merge:**  
   Merges newly extracted bookmarks with previously saved ones to avoid duplicates.

3. **Generate Descriptions:**  
   For bookmarks without a description, sends prompts to the **GPT-4o-mini** model to generate concise summaries.

4. **Embed and Store:**  
   Converts bookmark content into embeddings and stores them using an **InMemoryVectorStore**.

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

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

1. **Export your Chrome bookmarks** if not already available.
2. **Adjust the `bookmarks_path`** in the code to point to your Chrome `Bookmarks` file.
3. **Run the script:**

```bash
python bookmark_assistant.py
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
- The project currently uses **GPT-4o-mini** and **text-embedding-3-large** models via **OpenAI** API. Make sure you have access.

---

## ⚡ Future Improvements

- Save embeddings persistently to disk (instead of memory) for faster restarts.
- Add multi-turn conversation memory.
- Improve description generation with website scraping.
- Support Firefox, Edge, and other browsers.

---

## 📜 License

This project is open-source and available for use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

Thanks to the open-source community around **LangChain**, **LangGraph**, **OpenAI**, and **Gradio** for making rapid prototyping like this possible.
