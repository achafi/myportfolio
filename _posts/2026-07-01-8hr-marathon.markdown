---
title : "Enterprise Agentic RAG pipeline"
date : 2026-07-01
tags : [GenAI, RAG, LangGraph, Python, FastAPI]
header :
  image : ""
excerpt : "GenAI, RAG, LangGraph, Guardrails"
---
[Source code](https://github.com/d-hackmt/8hr-MARATHON)

# Enterprise Agentic RAG (scalable pipeline)
*The objective of this project is to build a production-grade, enterprise-level RAG system that goes beyond simple "retrieve and stuff" prompting. The system answers questions over a corpus of technical documentation while distinguishing between authoritative "True Data" and random "Noisy Data", using agentic reasoning, semantic re-ranking, history-aware planning and guardrails that block unsafe or off-topic inputs before any retrieval happens.*

## Introduction

Most RAG demos stop at: embed the documents, retrieve the top-k chunks, paste them into a prompt. That works for a notebook, but it breaks down in an enterprise setting where the corpus contains noise, users try jailbreaks, API keys fail, and nobody can explain what the system did five minutes ago.

This project was built to close exactly that gap. It is a complete RAG platform with four layers that are usually missing from a weekend demo:

- an **agentic orchestration layer** (LangGraph) that plans, retrieves, re-ranks and responds with conversation memory,
- a **safety layer** (NeMo Guardrails) that inspects every input and output before and after retrieval,
- a **resilience layer** (Portkey LLM gateway) that routes and falls back between LLM keys automatically,
- and an **evaluation layer** (RAGAS + a Streamlit demo) that measures the whole thing instead of trusting vibes.

## Key features

- **Agentic intelligence** — LangGraph for cyclic reasoning, multi-step planning and conversation memory.
- **Guardrails** — NeMo Guardrails blocks off-topic, jailbreak and injection inputs before any retrieval happens.
- **LLM gateway** — Portkey routes all LLM calls with automatic fallback between primary and backup Groq keys.
- **Enterprise search** — Qdrant Cloud for high-performance vector search, plus FlashRank for local semantic re-ranking.
- **Gemini embeddings** — Google `gemini-embedding-2-preview` (3072-dim) through `langchain-google-genai`.
- **Local document parsing** — PDF, HTML, TXT, DOCX and PPTX parsed entirely on-device, no external OCR service.
- **Observability** — full trace nesting with Pydantic Logfire and LangSmith across every agent node.
- **Evaluation suite** — RAGAS-powered eval pipeline (6 metrics) with a dedicated Streamlit demo app.

## Agent intelligence flow

The request travels through the system in a single, traceable path:

1. The user asks a question in the **Streamlit UI**, which calls the **FastAPI `/query` endpoint**.
2. The input passes through the **NeMo Guardrails gate**. Blocked inputs (jailbreak, injection, off-topic) are rejected and never reach retrieval.
3. The **Planner node** classifies the intent: a conversational follow-up goes straight to the Responder, a technical question goes to retrieval.
4. The **Retriever node** embeds the query with Gemini, searches **Qdrant Cloud**, and passes the candidates to the **FlashRank local re-ranker** for semantic re-ordering with zero added latency.
5. The **Responder node** composes the final answer and writes the exchange into **LangGraph MemorySaver**, so the next turn has context.

Every node emits a nested trace, so the full decision path is visible in Logfire and LangSmith rather than hidden inside a single opaque LLM call.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Groq (Llama 3.3 70B) via Portkey gateway |
| Guardrails | NeMo Guardrails |
| Vector DB | Qdrant Cloud |
| Reranking | FlashRank (local, zero-latency) |
| Embeddings | Gemini `gemini-embedding-2-preview` (3072-dim) |
| Document parsing | pypdf + pdfplumber (local, no OCR service) |
| Observability | Pydantic Logfire + LangSmith |
| Evaluation | RAGAS + custom Tool Correctness (Jaccard) |
| Interfaces | FastAPI backend, Streamlit chat UI |

## Ingestion engine

The ingestion pipeline is deliberately separate from the query path. `python -m app.ingestion.processor DATA --wipe` parses every document in the corpus, chunks it with a paragraph-based splitter (1500 characters max), writes the parsed and chunked JSON to `processed_data/`, and indexes the vectors into Qdrant. Passing `--wipe` drops and recreates the collection; omitting it appends.

Parsing happens locally with pypdf and pdfplumber, which keeps sensitive technical documentation inside the machine instead of sending it to an external OCR service.

## Guardrails and gateway

Two failure modes kill RAG systems in production: unsafe inputs and dead API keys.

- **NeMo Guardrails** runs before retrieval, so a prompt injection never gets a chance to contaminate what the system reads back. Output filtering catches unsafe responses on the way out.
- **Portkey** sits between the application and Groq, routes every LLM call, and automatically falls back to a backup Groq key when the primary one fails or is rate-limited. The eval judge uses a separate key to avoid the eval pipeline starving the live app.

## Observability and evaluation

Every agent node is traced with **Pydantic Logfire** and **LangSmith**, with full trace nesting — you can see the planner's decision, the retrieved chunks, the re-ranking order and the final composition as child spans of one request.

The evaluation suite runs **RAGAS** over six metrics plus a custom Tool Correctness score (Jaccard overlap), driven by a golden dataset, with a dedicated Streamlit three-tab demo app for inspecting results.

## How to run

```bash
python -m venv tenvv
.\tenvv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` with the Groq (primary + fallback), Portkey, Qdrant, Logfire, LangSmith, Gemini and judge keys, then:

```bash
# index the corpus
python -m app.ingestion.processor DATA --wipe

# Terminal 1 — FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run ui/app.py

# Optional — evaluation suite (needs the backend running)
streamlit run evals/app.py
```

The repository ships with eleven architecture and operations guides covering the system overview, ingestion, node intelligence, tracing, environment variables, known gotchas, FlashRank re-ranking, guardrails, the LLM gateway and both eval documents.

## Conclusion

This project is the closest thing I have built to a real RAG platform: guardrails in front, a gateway behind, traces everywhere, and an evaluation suite that keeps everyone honest. The main lessons were architectural rather than algorithmic — most of the hard work was in failure handling (key fallback, blocked inputs, trace nesting) rather than in the retrieval itself, which is exactly what separates a demo from a system you can operate.

[Source code](https://github.com/d-hackmt/8hr-MARATHON)
