"""Tests for the routing logic. They use a fake search and a fake LLM, so no models are downloaded."""

from langchain_core.documents import Document

from rag.graph import ESCALATION_MESSAGE, build_graph


class FakeKnowledgeBase:
    def __init__(self, score):
        self.score = score

    def search(self, question):
        doc = Document(page_content="Passwords must be 12 characters.",
                       metadata={"source": "policy.pdf", "page_label": "4"})
        return [(doc, self.score)]


def fake_llm(prompt):
    assert "Passwords must be 12 characters." in prompt
    return "At least 12 characters."


def test_close_match_is_answered_with_sources():
    result = build_graph(FakeKnowledgeBase(score=0.6), fake_llm).invoke({"question": "Password length?"})
    assert result["escalated"] is False
    assert result["answer"] == "At least 12 characters."
    assert result["sources"][0]["file"] == "policy.pdf"
    assert result["sources"][0]["page"] == "4"


def test_far_match_is_escalated():
    result = build_graph(FakeKnowledgeBase(score=1.8), fake_llm).invoke({"question": "Capital of France?"})
    assert result["escalated"] is True
    assert result["answer"] == ESCALATION_MESSAGE


def test_threshold_can_be_changed_per_question():
    workflow = build_graph(FakeKnowledgeBase(score=1.0), fake_llm)
    assert workflow.invoke({"question": "q", "threshold": 1.3})["escalated"] is False
    assert workflow.invoke({"question": "q", "threshold": 0.8})["escalated"] is True


def test_empty_knowledge_base_escalates():
    class Empty:
        def search(self, question):
            return []

    result = build_graph(Empty(), fake_llm).invoke({"question": "anything"})
    assert result["escalated"] is True
