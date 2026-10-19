"""Stores the chunks of uploaded PDFs in Postgres (Supabase) using pgvector."""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag import config
from rag.ingest import load_pdf, split_documents


def load_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


class KnowledgeBase:
    """The documents one logged-in user has uploaded, stored in Postgres."""

    def __init__(self, embeddings: Embeddings, client, user_id: str):
        self.embeddings = embeddings
        self.client = client      # Supabase client, already signed in as this user
        self.user_id = user_id
        self.files = self._load_files()   # file name -> number of chunks

    def _load_files(self) -> dict:
        rows = self.client.table("documents").select("file_name").execute().data
        files = {}
        for row in rows:
            files[row["file_name"]] = files.get(row["file_name"], 0) + 1
        return files

    def add_pdf(self, file_bytes: bytes, file_name: str) -> int:
        """Index a PDF and return how many chunks were stored."""
        if file_name in self.files:
            self.remove(file_name)
        chunks = split_documents(load_pdf(file_bytes, file_name))
        if chunks:
            vectors = self.embeddings.embed_documents([c.page_content for c in chunks])
            rows = [
                {
                    "user_id": self.user_id,
                    "file_name": file_name,
                    "page_label": str(c.metadata.get("page_label", "?")),
                    "content": c.page_content,
                    "embedding": v,
                }
                for c, v in zip(chunks, vectors)
            ]
            for i in range(0, len(rows), 100):
                self.client.table("documents").insert(rows[i:i + 100]).execute()
        self.files[file_name] = len(chunks)
        return len(chunks)

    def remove(self, file_name: str) -> None:
        self.client.table("documents").delete().eq("file_name", file_name).execute()
        self.files.pop(file_name, None)

    def search(self, question: str, k: int = config.TOP_K) -> List[Tuple[Document, float]]:
        if self.is_empty():
            return []
        vector = self.embeddings.embed_query(question)
        rows = self.client.rpc(
            "match_documents", {"query_embedding": vector, "match_count": k}
        ).execute().data
        return [
            (
                Document(
                    page_content=r["content"],
                    metadata={"source": r["file_name"], "page_label": r["page_label"]},
                ),
                float(r["distance"]),
            )
            for r in rows
        ]

    def is_empty(self) -> bool:
        return sum(self.files.values()) == 0