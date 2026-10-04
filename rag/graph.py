"""The LangGraph workflow: retrieve -> (answer | escalate to a human)."""

from typing import Callable, List, TypedDict

from langgraph.graph import END, StateGraph

from rag import config

ESCALATION_MESSAGE = (
    "I'm not confident I can answer this from the uploaded documents. "
    "I have forwarded your question to a human agent."
)


class Source(TypedDict):
    file: str
    page: str
    score: float
    text: str


class RAGState(TypedDict, total=False):
    question: str
    threshold: float
    context: str
    sources: List[Source]
    best_score: float
    answer: str
    escalated: bool


def build_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context[:config.MAX_CONTEXT_CHARS]}\n\n"
        "Answer:"
    )


def build_graph(knowledge_base, generate_text: Callable[[str], str]):
    """Wire the workflow to a KnowledgeBase (anything with .search) and a text generator."""

    def retrieve(state: RAGState):
        results = knowledge_base.search(state["question"])
        sources = [
            Source(
                file=doc.metadata.get("source", "unknown"),
                page=str(doc.metadata.get("page_label", "?")),
                score=float(score),
                text=doc.page_content,
            )
            for doc, score in results
        ]
        context = "\n\n".join(s["text"] for s in sources)
        best = min((s["score"] for s in sources), default=float("inf"))
        return {"context": context, "sources": sources, "best_score": best}

    def generate(state: RAGState):
        answer = generate_text(build_prompt(state["question"], state["context"]))
        return {"answer": answer, "escalated": False}

    def escalate(state: RAGState):
        return {"answer": ESCALATION_MESSAGE, "escalated": True}

    def route(state: RAGState):
        threshold = state.get("threshold", config.THRESHOLD)
        return "escalate" if state["best_score"] > threshold else "generate"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("escalate", escalate)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route, {"generate": "generate", "escalate": "escalate"})
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()
