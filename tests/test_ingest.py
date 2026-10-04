"""Tests for PDF loading and the knowledge base, using a tiny PDF built in memory."""

import io

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.ingest import load_pdf, split_documents
from rag.store import KnowledgeBase

reportlab = pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas  # noqa: E402


def make_pdf(pages):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_load_pdf_keeps_file_name_and_page():
    pages = load_pdf(make_pdf(["First page", "Second page"]), "handbook.pdf")
    assert len(pages) == 2
    assert pages[1].metadata["source"] == "handbook.pdf"
    assert pages[1].metadata["page_label"] == "2"
    assert len(split_documents(pages)) == 2


def test_knowledge_base_add_replace_remove():
    kb = KnowledgeBase(DeterministicFakeEmbedding(size=32))
    assert kb.is_empty()

    kb.add_pdf(make_pdf(["Passwords must be 12 characters."]), "policy.pdf")
    kb.add_pdf(make_pdf(["Passwords must be 12 characters."]), "policy.pdf")   # re-upload
    assert kb.files == {"policy.pdf": 1}
    assert len(kb.search("password", k=5)) == 1   # no duplicate chunks

    kb.remove("policy.pdf")
    assert kb.is_empty()
    assert kb.search("password") == []
