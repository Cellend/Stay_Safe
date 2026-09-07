"""Chunk pages into token-sized pieces. Same logic as the original notebook
cell (Markdown-header split -> tiktoken-based recursive split), just made
importable/runnable as a script.
"""
import argparse
import sqlite3

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from common.config import SQLITE_DB_PATH

CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 150  # tokens
TOKENIZER_MODEL_NAME = "gpt-4"  # cl100k_base encoding, used only to size chunks


def setup_chunk_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER,
            url TEXT,
            page_title TEXT,
            headers TEXT,
            chunk_text TEXT,
            FOREIGN KEY(page_id) REFERENCES pages(id)
        )
        """
    )
    cursor.execute("DELETE FROM chunks")  # rerun-safe: clear old chunks
    conn.commit()


def process_chunks(db_path: str) -> int:
    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    setup_chunk_table(conn)
    cursor = conn.cursor()

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=TOKENIZER_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    cursor.execute("SELECT id, url, title, markdown_content FROM pages")
    pages = cursor.fetchall()

    total_chunks_created = 0
    print(f"Processing {len(pages)} pages...")

    for page_id, url, title, markdown_content in pages:
        if not markdown_content:
            continue

        md_docs = md_splitter.split_text(markdown_content)
        for doc in md_docs:
            doc.metadata["url"] = url
            doc.metadata["page_title"] = title

        final_chunks = token_splitter.split_documents(md_docs)

        for chunk in final_chunks:
            headers_dict = {k: v for k, v in chunk.metadata.items() if k.startswith("Header")}
            headers_str = str(headers_dict) if headers_dict else "No Subheaders"

            cursor.execute(
                """
                INSERT INTO chunks (page_id, url, page_title, headers, chunk_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (page_id, chunk.metadata["url"], chunk.metadata["page_title"], headers_str, chunk.page_content),
            )
            total_chunks_created += 1

    conn.commit()
    conn.close()
    print(f"Success! Created {total_chunks_created} optimized token chunks.")
    return total_chunks_created


def main():
    parser = argparse.ArgumentParser(description="Chunk crawled pages into token-sized pieces.")
    parser.add_argument("--db-path", default=SQLITE_DB_PATH)
    args = parser.parse_args()
    process_chunks(args.db_path)


if __name__ == "__main__":
    main()
