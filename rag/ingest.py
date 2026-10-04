"""Turn an uploaded PDF into chunks ready for the vector store."""

import io
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config


def load_pdf(file_bytes: bytes, file_name: str) -> List[Document]:
    """Read a PDF (given as bytes, e.g. from an upload) into one Document per page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Scanned pages have no text layer, so there is nothing to search.
        if not text.strip():
            continue
        pages.append(Document(
            page_content=text,
            # page_label is the number printed on the page (e.g. "iv" or "12"),
            # which is what a user would look for when checking a source.
            metadata={"source": file_name, "page": i, "page_label": reader.page_labels[i]},
        ))
    return pages


def split_documents(pages: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)
