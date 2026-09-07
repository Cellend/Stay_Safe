"""The RAG chat function itself - retrieval (Elasticsearch or Supabase,
picked via rag/retriever.py's dispatcher) + generation (NVIDIA NIM).
Equivalent to the notebook's `chat_with_ready`, but swapping Chroma for a
real retrieval backend and Ollama for hosted NVIDIA, and returning
structured data (not just printing) so the Streamlit app and the metrics
logger can both use it.
"""
import time

from rag.retriever import retrieve, RetrievalMode
from rag.llm_client import generate, strip_think

SYSTEM_PROMPT_TEMPLATE = """You are a helpful and scientific AI assistant for Hurricane/Typhoon/Cyclone Disaster Management Guidance.
Answer the user's question as follows:

Instructions:
1. Primary Source: Use the provided CONTEXT to answer the user's question first.
2. Secondary Source: If the CONTEXT is missing details needed to give a complete answer, you may use your general scientific knowledge to fill in the gaps.
3. Attribution Rule: You MUST clearly distinguish between database sources and general knowledge in your response using this format:
   - Use "[Ready.gov Database]" when citing information from the context.
   - Use "[General Knowledge]" when filling in missing scientific background.

CONTEXT:
{context}
"""


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "No relevant documents found."
    parts = []
    for hit in hits:
        parts.append(f"\n--- Source: {hit.get('page_title', 'Unknown')} ---\n{hit['chunk_text']}\n")
    return "".join(parts)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=format_context(hits))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def ask(question: str, k: int = 3, mode: RetrievalMode = "hybrid") -> dict:
    """Non-streaming: retrieve -> generate -> return everything needed for
    display and for metrics logging (see metrics/db.py).
    """
    start = time.monotonic()
    hits = retrieve(question, k=k, mode=mode)
    messages = build_messages(question, hits)

    completion = generate(messages, stream=False)
    answer = strip_think(completion.choices[0].message.content)
    latency_ms = int((time.monotonic() - start) * 1000)

    return {
        "question": question,
        "answer": answer,
        "retrieved": hits,
        "retrieval_mode": mode,
        "latency_ms": latency_ms,
    }


def ask_stream(question: str, k: int = 3, mode: RetrievalMode = "hybrid"):
    """Generator of text pieces, for a CLI demo or `st.write_stream`. Note:
    since <think> stripping needs to see the closing tag, this yields raw
    text as it arrives - fine for Mistral (no thinking traces), but if you
    swap GENERATOR_MODEL for a reasoning model, prefer `ask()` instead so
    the think-block can be stripped before display.
    """
    hits = retrieve(question, k=k, mode=mode)
    messages = build_messages(question, hits)

    stream = generate(messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
