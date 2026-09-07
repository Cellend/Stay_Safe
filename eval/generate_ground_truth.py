"""Generate a ground-truth eval set: sample chunks from SQLite, ask the
judge model (Nemotron) to write one realistic question each chunk should
answer. Output is `(question -> expected_chunk_id)` pairs, which is exactly
what retrieval_eval.py needs to compute Hit Rate@k / MRR.

Target size defaults to ~30 questions (fast to generate and to sanity-check
by hand for a 50-page demo corpus) - override with --num-questions.
"""
import argparse
import json
import random
import sqlite3
import time

from openai import RateLimitError

from common.config import SQLITE_DB_PATH, NVIDIA_REQUEST_DELAY_SECONDS
from rag.llm_client import judge

OUTPUT_PATH = "eval/ground_truth.json"

QUESTION_PROMPT = """Below is a passage from a disaster-preparedness knowledge base (ready.gov).

Write ONE specific, realistic question a person might type into a chat assistant that this passage - and only this passage - directly and fully answers. The question should be answerable without needing information from elsewhere in the corpus.

Return ONLY the question text. No preamble, no quotes, no numbering.

PASSAGE:
{passage}
"""


def sample_chunks(db_path: str, num_questions: int) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, url, page_title, chunk_text FROM chunks WHERE LENGTH(chunk_text) > 200"
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No chunks found - run ingest/chunk.py first.")

    sample_size = min(num_questions, len(rows))
    return [dict(r) for r in random.sample(rows, sample_size)]


def generate_ground_truth(db_path: str, num_questions: int, output_path: str) -> list[dict]:
    chunks = sample_chunks(db_path, num_questions)
    ground_truth = []

    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] Generating question for chunk {chunk['id']}...")
        try:
            question = judge(
                [{"role": "user", "content": QUESTION_PROMPT.format(passage=chunk["chunk_text"])}]
            ).strip()
        except RateLimitError:
            # openai-python already retried this internally and still got
            # 429'd - back off harder and skip rather than losing the whole
            # batch's progress to one exhausted item.
            print("  -> Rate limited by NVIDIA even after retries. Skipping this item; "
                  "consider raising NVIDIA_REQUEST_DELAY_SECONDS in .env if this repeats.")
            time.sleep(NVIDIA_REQUEST_DELAY_SECONDS * 5)
            continue
        except Exception as e:
            print(f"  -> {type(e).__name__}: {e}. Skipping this item.")
            continue

        if not question:
            print("  -> Empty response, skipping.")
            continue

        ground_truth.append(
            {
                "question": question,
                "expected_chunk_id": chunk["id"],
                "expected_url": chunk["url"],
                "expected_page_title": chunk["page_title"],
            }
        )
        time.sleep(NVIDIA_REQUEST_DELAY_SECONDS)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(ground_truth)} ground-truth question/chunk pairs to {output_path}.")
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Generate a ground-truth eval set from SQLite chunks.")
    parser.add_argument("--db-path", default=SQLITE_DB_PATH)
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    generate_ground_truth(args.db_path, args.num_questions, args.output)


if __name__ == "__main__":
    main()
