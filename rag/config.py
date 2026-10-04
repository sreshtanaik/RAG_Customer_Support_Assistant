"""Settings for the RAG pipeline. Each one can be overridden with an environment variable."""

import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # max characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # characters repeated between neighbouring chunks

TOP_K = int(os.getenv("TOP_K", "3"))                     # chunks retrieved per question

# Chroma returns a distance: lower means closer. If even the best chunk is
# further away than this, the question is escalated to a human agent.
THRESHOLD = float(os.getenv("THRESHOLD", "1.3"))

MAX_CONTEXT_CHARS = 1500   # flan-t5 only reads 512 tokens, so keep the prompt short
MAX_NEW_TOKENS = 150
