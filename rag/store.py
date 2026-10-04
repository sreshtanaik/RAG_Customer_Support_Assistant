"""A small wrapper around ChromaDB that holds the chunks of the uploaded PDFs."""

import uuid
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag import config
from rag.ingest import load_pdf, split_documents


def load_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


class KnowledgeBase:
    """The documents one user has uploaded.

    Every KnowledgeBase gets its own in-memory Chroma collection, so two people
    using the app at the same time never see each other's PDFs.
    """

    def __init__(self, embeddings: Embeddings):
        self.vectorstore = Chroma(
            collection_name=f"kb_{uuid.uuid4().hex}",
            embedding_function=embeddings,
        )
        self.files = {}   # file name -> number of chunks

    def add_pdf(self, file_bytes: bytes, file_name: str) -> int:
        """Index a PDF and return how many chunks were stored."""
        if file_name in self.files:
            self.remove(file_name)   # re-uploading a file replaces it, so no duplicates
        chunks = split_documents(load_pdf(file_bytes, file_name))
        if chunks:
            self.vectorstore.add_documents(chunks)
        self.files[file_name] = len(chunks)
        return len(chunks)

    def remove(self, file_name: str) -> None:
        self.vectorstore._collection.delete(where={"source": file_name})
        self.files.pop(file_name, None)

    def search(self, question: str, k: int = config.TOP_K) -> List[Tuple[Document, float]]:
        if self.is_empty():
            return []
        return self.vectorstore.similarity_search_with_score(question, k=k)

    def is_empty(self) -> bool:
        return sum(self.files.values()) == 0
