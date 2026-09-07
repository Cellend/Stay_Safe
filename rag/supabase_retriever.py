"""Retrieval against Supabase (hosted Postgres + pgvector): keyword
(Postgres full-text search), vector (pgvector cosine), and hybrid (client-
side RRF fusion of both) - the free-hosting alternative to
rag/es_retriever.py, with the SAME function signatures and return shape,
so rag/chat.py and the eval scripts work against either backend unchanged
(see rag/retriever.py, the dispatcher that picks one via .env).

Why Supabase for this role: Elasticsearch has no genuinely-free-forever
hosted tier (self-host, or a paid/trial managed cluster), while Supabase's
free Postgres project is free indefinitely - the tradeoff for a corpus
this small (~300 chunks) is worth it. See README.md's deployment section
for the schema setup this module expects.
"""
from typing import Literal

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from common.config import SUPABASE_DB_URL, EMBED_MODEL, require_supabase_url

_embedder = None  # lazy-loaded singleton - loading SentenceTransformer is slow


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_connection():
    """A fresh connection per call, matching this project's SQLite modules'
    style - simple and safe at this project's traffic scale. Registers the
    pgvector type adapter so a plain Python list can be bound directly to a
    `vector` column/parameter instead of hand-formatting SQL strings.
    """
    require_supabase_url()
    conn = psycopg2.connect(SUPABASE_DB_URL)
    register_vector(conn)
    return conn


def _format_rows(rows) -> list[dict]:
    return [
        {
            "score": float(row["score"]) if row["score"] is not None else 0.0,
            "chunk_text": row["chunk_text"],
            "page_title": row["page_title"] or "Unknown",
            "url": row["url"] or "",
            "sqlite_id": row["sqlite_id"],
        }
        for row in rows
    ]


def keyword_search(query: str, k: int = 3, conn=None) -> list[dict]:
    """Postgres full-text search (tsvector/ts_rank) - the BM25-ish equivalent."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT sqlite_id, page_title, url, chunk_text,
                       ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) AS score
                FROM chunks
                WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, k),
            )
            rows = cur.fetchall()
        return _format_rows(rows)
    finally:
        if own_conn:
            conn.close()


def vector_search(query: str, k: int = 3, conn=None) -> list[dict]:
    """pgvector cosine search - the direct equivalent of the Elasticsearch kNN query."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        query_vector = get_embedder().encode([query])[0]
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT sqlite_id, page_title, url, chunk_text,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vector, query_vector, k),
            )
            rows = cur.fetchall()
        return _format_rows(rows)
    finally:
        if own_conn:
            conn.close()


RRF_K = 60  # same damping constant as rag/es_retriever.py, for a like-for-like comparison


def hybrid_search(query: str, k: int = 3, conn=None) -> list[dict]:
    """Keyword + vector combined via client-side Reciprocal Rank Fusion -
    the identical approach used in rag/es_retriever.py's hybrid_search(),
    just fusing Postgres full-text + pgvector results instead of
    Elasticsearch's. Kept as its own implementation (not shared code, since
    the two retrievers' row shapes differ before formatting) but the fusion
    math is deliberately the same for a fair comparison between backends.
    """
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        fetch_k = max(20, k * 5)
        keyword_hits = keyword_search(query, k=fetch_k, conn=conn)
        vector_hits = vector_search(query, k=fetch_k, conn=conn)

        rrf_scores: dict = {}
        docs_by_id: dict = {}
        for rank_list in (keyword_hits, vector_hits):
            for rank, hit in enumerate(rank_list, start=1):
                doc_id = hit["sqlite_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
                docs_by_id.setdefault(doc_id, hit)

        ranked_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:k]
        results = []
        for doc_id in ranked_ids:
            hit = dict(docs_by_id[doc_id])
            hit["score"] = rrf_scores[doc_id]
            results.append(hit)
        return results
    finally:
        if own_conn:
            conn.close()


RetrievalMode = Literal["keyword", "vector", "hybrid"]

_MODE_FUNCS = {"keyword": keyword_search, "vector": vector_search, "hybrid": hybrid_search}


def retrieve(query: str, k: int = 3, mode: RetrievalMode = "hybrid", **kwargs) -> list[dict]:
    """Same entry point signature as rag/es_retriever.py's retrieve() -
    that's what lets rag/retriever.py dispatch to either backend transparently.
    """
    return _MODE_FUNCS[mode](query, k=k, **kwargs)
