# RAG Customer Support Assistant

A web app where you upload PDFs (policies, manuals, FAQs) and chat with them. Answers are generated with Retrieval-Augmented Generation (RAG), cite the file and page they came from, and are escalated to a human agent when the documents don't contain a confident match.

Everything runs locally with open-source models, so no API key is needed and documents never leave the machine.

## Features

- **Upload one or more PDFs** from the sidebar. They are indexed as soon as they are added, and removing a file from the uploader also removes it from the index.
- **Chat interface** with a conversation history for the session.
- **Source citations** for every answer (file name and page), plus an expander that shows the exact passages that were retrieved and their scores.
- **Human escalation.** If even the best matching passage is too far from the question, the assistant hands the question to a human agent instead of guessing. The threshold can be tuned with a slider.
- **Per-session knowledge base.** Each browser session gets its own vector store, so users don't see each other's documents.

## How it works

```
PDF upload -> pypdf (text per page) -> chunks (1000 chars, 200 overlap)
           -> all-MiniLM-L6-v2 embeddings -> ChromaDB (in memory, one collection per session)

Question -> LangGraph workflow:
              retrieve top 3 chunks
                 |
                 +-- best distance <= threshold --> flan-t5-base answers from the chunks
                 +-- best distance >  threshold --> escalate to a human agent
```

**Tools:** Streamlit, LangChain, LangGraph, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Hugging Face Transformers (flan-t5-base), pypdf.

## Project structure

```
app.py                  Streamlit UI: upload sidebar, chat, sources, escalation
rag/
  config.py             Settings (models, chunk size, top-k, threshold), overridable by env vars
  ingest.py             PDF bytes -> page Documents -> chunks
  store.py              KnowledgeBase: per-session Chroma collection (add / replace / remove / search)
  llm.py                flan-t5 text generator
  graph.py              LangGraph workflow: retrieve -> generate | escalate
tests/                  Unit tests (no model downloads needed)
RAG_Customer_Support_Assistant.ipynb   The original notebook prototype
```

## Run it locally

Requires Python 3.10+.

```bash
git clone https://github.com/sreshtanaik/RAG_Customer_Support_Assistant.git
cd RAG_Customer_Support_Assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, upload a PDF and ask a question. The first start downloads the two models (about 1 GB) from Hugging Face. After that they are cached.

### With Docker

```bash
docker build -t rag-support .
docker run -p 8501:8501 rag-support
```

### Configuration

These environment variables override the defaults in `rag/config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `google/flan-t5-base` | Seq2seq model that writes answers (e.g. `google/flan-t5-large` for better answers) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking |
| `TOP_K` | `3` | Chunks retrieved per question |
| `THRESHOLD` | `1.3` | Default escalation threshold (the slider starts here) |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake search and a fake language model, so they run in about a second without downloading anything. They cover the answer and escalation routing, the per-question threshold, PDF loading with file and page metadata, and replacing and removing documents.

## From notebook to application

The project started as a Colab notebook ([`RAG_Customer_Support_Assistant.ipynb`](RAG_Customer_Support_Assistant.ipynb)) that ran the pipeline one cell at a time against one hard-coded PDF. The application keeps the same retrieval and escalation logic but:

- splits the code into a reusable `rag` package with a UI on top,
- accepts any number of PDFs uploaded by the user instead of a fixed file,
- loads the models once and shares them, while each user gets their own document store,
- cites the file name as well as the page, because answers can now come from several documents,
- adds unit tests, a requirements file and a Dockerfile.

## Known limitations

- **Keyword-like matches.** Some questions match a word but not its meaning. For example, "leave policy" can return a rule about employees *leaving* the company. A reranker did not fix this in the notebook. A possible next step is an LLM check on whether the retrieved text really answers the question.
- **Small answer model.** flan-t5-base is fast and runs on a CPU, but its answers are short and it only reads about 1500 characters of context. A larger model (set `LLM_MODEL`) or a hosted LLM would give fuller answers.
- **Scanned PDFs** have no text layer and are skipped. OCR would be needed to support them.
- **Escalation is simulated.** The question is flagged in the UI and not sent to a real ticketing system.
- Each question is answered on its own: earlier chat turns are not used as context.
