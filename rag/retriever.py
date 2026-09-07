"""Picks the retrieval backend (Elasticsearch or Supabase) based on
RETRIEVAL_BACKEND in .env, and re-exports a single `retrieve()` with an
identical signature either way. Everything else in the project
(rag/chat.py, eval/retrieval_eval.py) imports from HERE, not from
rag/es_retriever.py or rag/supabase_retriever.py directly, so switching
backends is a one-line .env change, not a code change.
"""
from common.config import RETRIEVAL_BACKEND

if RETRIEVAL_BACKEND == "supabase":
    from rag.supabase_retriever import retrieve, get_embedder, RetrievalMode
elif RETRIEVAL_BACKEND == "elasticsearch":
    from rag.es_retriever import retrieve, get_embedder, RetrievalMode
else:
    raise ValueError(f"Unknown RETRIEVAL_BACKEND: {RETRIEVAL_BACKEND!r} (expected 'elasticsearch' or 'supabase')")

__all__ = ["retrieve", "get_embedder", "RetrievalMode"]
