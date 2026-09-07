"""Embed chunks and index them into Supabase (hosted Postgres + pgvector) -
the free-hosting alternative to ingest/index_es.py, for a deployed copy
where Elasticsearch has nowhere free to live. Same source data (the
`chunks` table in SQLite) and the same embedding model as index_es.py.

One-time setup this script assumes has been done in your Supabase project
(SQL editor in the dashboard):
    create extension if not exists vector;
(Everything else - the table and its indexes - this script creates itself,
idempotently, so you don't need to run any other SQL by hand.)
"""
import argparse
import sqlite3

import psycopg2
from pgvector.psycopg2 import register_vector

from common.config import SUPABASE_DB_URL, EMBED_MODEL, EMBED_DIMS, SQLITE_DB_PATH, require_supabase_url

BATCH_SIZE = 64


def get_connection(with_vector_type: bool = True):
    """`with_vector_type=False` for the very first connection of a fresh
    database: pgvector's register_vector() looks up the `vector` type's
    OID in pg_type, which doesn't exist until `CREATE EXTENSION vector` has
    run at least once - calling it too early raises "vector type not found
    in the database" (hit this live on a brand-new Supabase project).
    ensure_schema() creates the extension on a plain connection first; every
    connection after that can safely request the type adapter.
    """
    require_supabase_url()
    conn = psycopg2.connect(SUPABASE_DB_URL)
    if with_vector_type:
        register_vector(conn)
    return conn


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                sqlite_id INTEGER UNIQUE,
                page_id INTEGER,
                url TEXT,
                page_title TEXT,
                headers TEXT,
                chunk_text TEXT,
                embedding vector({EMBED_DIMS})
            )
            """
        )
        # HNSW for cosine similarity (pgvector's approximate-nearest-neighbor index)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        # GIN index over the full-text vector, for keyword_search's ts_rank query
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin (to_tsvector('english', chunk_text))"
        )
    conn.commit()


def embed_and_index(sqlite_path: str, truncate: bool = True) -> int:
    conn = get_connection(with_vector_type=False)
    ensure_schema(conn)  # creates the `vector` extension if this is a fresh database
    register_vector(conn)  # now safe: the type exists, whether it was just created or already there

    if truncate:
        # Rerun-safe, same idea as index_es.py deleting and recreating its
        # index - clears old data without dropping the (possibly slow to
        # rebuild) HNSW index itself.
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks")
        conn.commit()
        print("Cleared existing rows from 'chunks' (schema/indexes kept).")

    sqlite_conn = sqlite3.connect(sqlite_path)
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, page_id, url, page_title, headers, chunk_text FROM chunks")
    rows = cursor.fetchall()
    sqlite_conn.close()

    rows = [r for r in rows if r[5] and r[5].strip()]
    if not rows:
        print("No chunks found. Run ingest/chunk.py first.")
        conn.close()
        return 0

    print(f"Fetched {len(rows)} chunks from SQLite.")
    print(f"Loading embedding model ({EMBED_MODEL})...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)

    total_indexed = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            texts = [r[5] for r in batch]
            embeddings = model.encode(texts, show_progress_bar=False)

            for (chunk_id, page_id, url, page_title, headers, chunk_text), embedding in zip(batch, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (sqlite_id, page_id, url, page_title, headers, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sqlite_id) DO UPDATE SET
                        page_id = EXCLUDED.page_id, url = EXCLUDED.url, page_title = EXCLUDED.page_title,
                        headers = EXCLUDED.headers, chunk_text = EXCLUDED.chunk_text, embedding = EXCLUDED.embedding
                    """,
                    (chunk_id, page_id or 0, url or "", page_title or "", headers or "", chunk_text, embedding),
                )
            conn.commit()
            total_indexed += len(batch)
            total_batches = (len(rows) - 1) // BATCH_SIZE + 1
            print(f"  Indexed batch {i // BATCH_SIZE + 1}/{total_batches} ({len(batch)} items)")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
    conn.close()

    print(f"\nSuccess! Total rows in Supabase 'chunks' table: {count}")
    return total_indexed


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and index them into Supabase (Postgres + pgvector).")
    parser.add_argument("--db-path", default=SQLITE_DB_PATH)
    parser.add_argument("--no-truncate", action="store_true", help="Upsert without clearing existing rows first.")
    args = parser.parse_args()
    embed_and_index(args.db_path, truncate=not args.no_truncate)


if __name__ == "__main__":
    main()
