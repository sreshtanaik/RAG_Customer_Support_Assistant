# RAG Customer Support Assistant

A RAG assistant that answers questions from a company IT policy PDF, shows the pages it used, and escalates to a human agent when it is not confident.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreshtanaik/rag-customer-support-assistant/blob/main/RAG_Customer_Support.ipynb)

## How it works

PDF -> chunks -> embeddings -> ChromaDB -> retrieve top 3 chunks. A LangGraph workflow then checks the retrieval score: if it is close enough, the model answers with source pages, otherwise the question is escalated to a human agent.

**Tools:** LangChain, LangGraph, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), flan-t5-base.

## How to run

1. Open the notebook in Colab using the badge above.
2. Upload your own policy PDF and name it `IT-Policy.pdf`. The PDF I used is confidential, so it is not included.
3. Choose Runtime -> Run all.

## Known limitation

Some questions match a word but not its meaning. For example, "leave policy" returns a rule about employees leaving the company. A reranker did not fix this, so a possible next step is an LLM check on whether the retrieved text really answers the question.
