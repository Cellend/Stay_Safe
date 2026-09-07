"""Metrics store backed by Supabase (hosted Postgres) - the free-hosting
alternative to metrics/db.py's SQLite file, for a deployed copy where the
filesystem is ephemeral (e.g. Streamlit Community Cloud resets local files
on redeploy/sleep-wake, so SQLite there doesn't durably persist). Same
function signatures as metrics/db.py so app/streamlit_app.py and
eval/llm_judge.py work against either one unchanged (see metrics/store.py,
the dispatcher that picks one via .env).
"""
import json
from datetime import datetime

import psycopg2
import psycopg2.extras

from common.config import SUPABASE_DB_URL, require_supabase_url


def get_conn():
    require_supabase_url()
    return psycopg2.connect(SUPABASE_DB_URL)


def init_db():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    retrieval_mode TEXT,
                    retrieved_ids JSONB,
                    latency_ms INTEGER,
                    judge_verdict TEXT,
                    judge_reason TEXT,
                    feedback TEXT,
                    feedback_ts TIMESTAMPTZ
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def log_interaction(result: dict) -> int:
    """`result` is the dict returned by rag.chat.ask()."""
    retrieved_ids = json.dumps([h.get("sqlite_id") for h in result.get("retrieved", [])])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interactions (question, answer, retrieval_mode, retrieved_ids, latency_ms)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (result["question"], result["answer"], result.get("retrieval_mode"), retrieved_ids, result.get("latency_ms")),
            )
            interaction_id = cur.fetchone()[0]
        conn.commit()
        return interaction_id
    finally:
        conn.close()


def log_judge_result(interaction_id: int, verdict: str, reason: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE interactions SET judge_verdict = %s, judge_reason = %s WHERE id = %s",
                (verdict, reason, interaction_id),
            )
        conn.commit()
    finally:
        conn.close()


def log_feedback(interaction_id: int, vote: str):
    """`vote` should be 'up' or 'down'."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE interactions SET feedback = %s, feedback_ts = %s WHERE id = %s",
                (vote, datetime.now(), interaction_id),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_recent(limit: int = 100) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def fetch_unjudged(limit: int = 50) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM interactions WHERE judge_verdict IS NULL ORDER BY id DESC LIMIT %s", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def aggregate_stats() -> dict:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS c FROM interactions")
            total = cur.fetchone()["c"]

            cur.execute("SELECT AVG(latency_ms) AS a FROM interactions")
            avg_latency = cur.fetchone()["a"]

            cur.execute(
                "SELECT feedback, COUNT(*) AS c FROM interactions WHERE feedback IS NOT NULL GROUP BY feedback"
            )
            feedback_counts = {row["feedback"]: row["c"] for row in cur.fetchall()}

            cur.execute(
                "SELECT judge_verdict, COUNT(*) AS c FROM interactions WHERE judge_verdict IS NOT NULL GROUP BY judge_verdict"
            )
            judge_counts = {row["judge_verdict"]: row["c"] for row in cur.fetchall()}
    finally:
        conn.close()

    return {
        "total_interactions": total,
        "avg_latency_ms": round(float(avg_latency), 1) if avg_latency else None,
        "feedback_counts": feedback_counts,
        "judge_verdict_counts": judge_counts,
    }
