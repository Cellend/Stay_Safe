"""Retrieval against Elasticsearch: keyword (BM25), vector (kNN), and hybrid
(RRF fusion of both). Keeping all three as separate functions - rather than
only shipping "hybrid" - is deliberate: eval/retrieval_eval.py compares them
against the ground-truth set so the choice of retrieval mode is a measured
decision, not a guess.

Also exposes a LangChain-compatible retriever via
`langchain_elasticsearch.ElasticsearchRetriever`, for anyone who wants to
plug this into a LangChain chain instead of calling the raw functions.
"""
from typing import Literal

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from common.config import ELASTICSEARCH_URL, ES_INDEX_NAME, EMBED_MODEL

_embedder = None  # lazy-loaded singleton - loading SentenceTransformer is slow


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_client() -> Elasticsearch:
    return Elasticsearch(ELASTICSEARCH_URL)


def _format_hits(response) -> list[dict]:
    hits = response["hits"]["hits"]
    return [
        {
            "score": hit["_score"],
            "chunk_text": hit["_source"]["chunk_text"],
            "page_title": hit["_source"].get("page_title", "Unknown"),
            "url": hit["_source"].get("url", ""),
            "sqlite_id": hit["_source"].get("sqlite_id"),
        }
        for hit in hits
    ]


def keyword_search(query: str, k: int = 3, index_name: str = ES_INDEX_NAME, es: Elasticsearch = None) -> list[dict]:
    """Pure BM25 search over chunk_text."""
    es = es or get_client()
    response = es.search(
        index=index_name,
        query={"match": {"chunk_text": query}},
        size=k,
    )
    return _format_hits(response)


def vector_search(query: str, k: int = 3, index_name: str = ES_INDEX_NAME, es: Elasticsearch = None) -> list[dict]:
    """Pure kNN vector search - the direct equivalent of the old Chroma query."""
    es = es or get_client()
    query_vector = get_embedder().encode([query])[0].tolist()
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(50, k * 10),
        },
        size=k,
    )
    return _format_hits(response)


RRF_K = 60  # standard RRF damping constant (see Cormack et al. 2009)


def hybrid_search(query: str, k: int = 3, index_name: str = ES_INDEX_NAME, es: Elasticsearch = None) -> list[dict]:
    """BM25 + kNN combined via client-side Reciprocal Rank Fusion.

    Deliberately NOT using Elasticsearch's built-in `retriever: {rrf: ...}`
    API: that endpoint is a Platinum-licensed feature and returns a 403
    (`current license is non-compliant for RRF`) on the free Basic license
    this project's docker-compose.yml runs - confirmed by actually running
    it. A 30-day trial license would unlock it, but that's a ticking clock
    for a "no paid service" demo, so this fuses the two result sets in
    Python instead. Works on any ES tier, forever.
    """
    es = es or get_client()
    fetch_k = max(20, k * 5)  # pull a wider candidate pool from each retriever before fusing

    keyword_hits = keyword_search(query, k=fetch_k, index_name=index_name, es=es)
    vector_hits = vector_search(query, k=fetch_k, index_name=index_name, es=es)

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


RetrievalMode = Literal["keyword", "vector", "hybrid"]

_MODE_FUNCS = {"keyword": keyword_search, "vector": vector_search, "hybrid": hybrid_search}


def retrieve(query: str, k: int = 3, mode: RetrievalMode = "hybrid", **kwargs) -> list[dict]:
    """Single entry point used by rag/chat.py and the eval scripts."""
    return _MODE_FUNCS[mode](query, k=k, **kwargs)


def get_langchain_retriever(mode: Literal["keyword", "vector"] = "vector", k: int = 3):
    """Optional LangChain-compatible wrapper, for anyone wiring this into a
    LangChain chain instead of calling `retrieve()` directly.

    Only "keyword" and "vector" are offered here, not "hybrid": LangChain's
    ElasticsearchRetriever issues one query per call via `body_func`, but
    this project's hybrid mode fuses two separate result sets in Python
    (see hybrid_search() above, and why - the native ES RRF `retriever`
    query is a Platinum-licensed feature that 403s on a free license). Use
    `retrieve(query, mode="hybrid")` directly for hybrid search.
    """
    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        if mode == "keyword":
            return {"query": {"match": {"chunk_text": query}}, "size": k}
        query_vector = get_embedder().encode([query])[0].tolist()
        return {
            "knn": {"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": max(50, k * 10)},
            "size": k,
        }

    return ElasticsearchRetriever.from_es_params(
        url=ELASTICSEARCH_URL,
        index_name=ES_INDEX_NAME,
        body_func=body_func,
        content_field="chunk_text",
    )
