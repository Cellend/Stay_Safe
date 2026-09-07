"""Embed chunks and index them into Elasticsearch.

This replaces the notebook's ChromaDB cell. Same source data (the `chunks`
table in SQLite) and the same embedding model (all-MiniLM-L6-v2), but the
destination is now an ES index with BOTH:
  - a `chunk_text` field (mapped as `text`) for BM25 keyword search
  - an `embedding` field (mapped as `dense_vector`) for kNN vector search

That's what makes hybrid retrieval (see rag/es_retriever.py) possible -
Chroma only ever gave you the vector half.
"""
import argparse
import sqlite3

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from common.config import ELASTICSEARCH_URL, ES_INDEX_NAME, EMBED_MODEL, EMBED_DIMS, SQLITE_DB_PATH

BATCH_SIZE = 64

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBED_DIMS,
                "index": True,
                "similarity": "cosine",
            },
            "sqlite_id": {"type": "integer"},
            "page_id": {"type": "integer"},
            "url": {"type": "keyword"},
            "page_title": {"type": "text"},
            "headers": {"type": "text"},
        }
    }
}


def get_client() -> Elasticsearch:
    return Elasticsearch(ELASTICSEARCH_URL)


def create_index(es: Elasticsearch, index_name: str, recreate: bool = True):
    if recreate and es.indices.exists(index=index_name):
        print(f"Deleting existing index '{index_name}' (rerun-safe, like the old Chroma 'DELETE FROM chunks').")
        es.indices.delete(index=index_name)
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=INDEX_MAPPING)
        print(f"Created index '{index_name}'.")


def embed_and_index(db_path: str, index_name: str, recreate: bool = True) -> int:
    es = get_client()
    if not es.ping():
        raise ConnectionError(
            f"Could not reach Elasticsearch at {ELASTICSEARCH_URL}. "
            "Start it with `docker compose up -d` first."
        )

    create_index(es, index_name, recreate=recreate)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, page_id, url, page_title, headers, chunk_text FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No chunks found. Run ingest/chunk.py first.")
        return 0

    print(f"Fetched {len(rows)} chunks from SQLite.")
    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    rows = [r for r in rows if r[5] and r[5].strip()]
    total_indexed = 0

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [r[5] for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        actions = []
        for (chunk_id, page_id, url, page_title, headers, chunk_text), embedding in zip(batch, embeddings):
            actions.append(
                {
                    "_index": index_name,
                    "_id": f"chunk_{chunk_id}",
                    "_source": {
                        "chunk_text": chunk_text,
                        "embedding": embedding,
                        "sqlite_id": chunk_id,
                        "page_id": page_id or 0,
                        "url": url or "",
                        "page_title": page_title or "",
                        "headers": headers or "",
                    },
                }
            )

        helpers.bulk(es, actions)
        total_indexed += len(actions)
        total_batches = (len(rows) - 1) // BATCH_SIZE + 1
        print(f"  Indexed batch {i // BATCH_SIZE + 1}/{total_batches} ({len(actions)} items)")

    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)["count"]
    print(f"\nSuccess! Total documents indexed in Elasticsearch: {count}")
    return total_indexed


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and index them into Elasticsearch.")
    parser.add_argument("--db-path", default=SQLITE_DB_PATH)
    parser.add_argument("--index-name", default=ES_INDEX_NAME)
    parser.add_argument("--no-recreate", action="store_true", help="Append instead of rebuilding the index.")
    args = parser.parse_args()
    embed_and_index(args.db_path, args.index_name, recreate=not args.no_recreate)


if __name__ == "__main__":
    main()
