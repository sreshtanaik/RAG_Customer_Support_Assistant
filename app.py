"""RAG Customer Support Assistant - Streamlit app.

Run with:  streamlit run app.py
"""

import streamlit as st

from rag import config
from rag.graph import build_graph
from rag.llm import Flant5Generator
from rag.store import KnowledgeBase, load_embeddings

st.set_page_config(page_title="RAG Support Assistant", page_icon="💬", layout="wide")


# ---------- Models: loaded once and shared by every user session ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return load_embeddings()


@st.cache_resource(show_spinner="Loading language model...")
def get_generator():
    return Flant5Generator()


# ---------- Per-session state ----------

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase(get_embeddings())
    st.session_state.workflow = build_graph(st.session_state.kb, get_generator())
    st.session_state.messages = []

kb: KnowledgeBase = st.session_state.kb


# ---------- Sidebar: documents and settings ----------

with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    # The uploader is the source of truth: index new files, drop removed ones.
    uploaded = {f.name: f for f in uploads or []}
    for name in [n for n in kb.files if n not in uploaded]:
        kb.remove(name)
    for name, f in uploaded.items():
        if name in kb.files:
            continue
        with st.spinner(f"Indexing {name}..."):
            n_chunks = kb.add_pdf(f.getvalue(), name)
        if n_chunks == 0:
            st.warning(f"No text found in {name}. Is it a scanned PDF?")
        else:
            st.toast(f"Indexed {name} ({n_chunks} chunks)")

    if not kb.is_empty():
        st.caption("Indexed documents")
        for name, n_chunks in kb.files.items():
            if n_chunks:
                st.markdown(f"- **{name}** · {n_chunks} chunks")
    else:
        st.info("Upload one or more PDFs to start asking questions.")

    st.divider()
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Escalation threshold",
        min_value=0.5, max_value=2.0, value=config.THRESHOLD, step=0.05,
        help="Distance of the best matching chunk (lower = closer). "
             "Questions whose best match is further than this go to a human agent.",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------- Chat ----------

st.title("💬 RAG Customer Support Assistant")
st.caption("Ask questions about your uploaded PDFs. Answers cite their source pages; "
           "questions the documents can't answer are escalated to a human agent.")


def show_details(message):
    if message.get("escalated"):
        st.warning("Escalated to a human agent", icon="🙋")
    sources = message.get("sources") or []
    if sources:
        pages = sorted({f"{s['file']} p.{s['page']}" for s in sources})
        st.caption("Sources: " + ", ".join(pages) + f"  ·  best score {message['best_score']:.3f}")
        with st.expander("Show retrieved passages"):
            for s in sources:
                st.markdown(f"**{s['file']}, page {s['page']}** (score {s['score']:.3f})")
                st.text(s["text"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_details(message)

question = st.chat_input("Ask a question about your documents", disabled=kb.is_empty())

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            result = st.session_state.workflow.invoke({"question": question, "threshold": threshold})
        reply = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "best_score": result["best_score"],
            "escalated": result["escalated"],
        }
        st.markdown(reply["content"])
        show_details(reply)
    st.session_state.messages.append(reply)
