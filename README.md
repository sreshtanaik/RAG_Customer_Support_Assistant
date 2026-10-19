# RAG Customer Support Assistant

A simple Streamlit app that lets you upload PDFs and ask questions about them. It finds the most relevant passages, stores vector embeddings in Supabase Postgres with pgvector, and answers using a local RAG pipeline.

## What it does

- Upload PDF files in VS Code using Streamlit
- Extract text from each page
- Split the text into chunks
- Convert chunks into embeddings and save them in Supabase Postgres with pgvector
- Search the uploaded documents for the best match
- Answer the question from the retrieved text
- Escalate to a human if the match is weak

## Tech stack

- Python
- Streamlit
- VS Code
- Supabase Postgres
- pgvector
- LangChain / LangGraph
- Hugging Face Transformers
- sentence-transformers
- pypdf

## Setup

1. Open the project in VS Code.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Supabase values:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key
```

Do not commit real keys. You must log in to Supabase before using the project.

5. In Supabase SQL Editor, run this SQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    page_number INT,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_count INT
)
RETURNS TABLE (
    id BIGINT,
    file_name TEXT,
    page_number INT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE SQL
AS $$
    SELECT
        id,
        file_name,
        page_number,
        content,
        1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

## Run it

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser, upload a PDF, and ask a question.

## Notes

- This project uses Supabase Postgres instead of ChromaDB for embeddings storage.
- The app is meant to run locally in VS Code with a Supabase-backed vector store.
