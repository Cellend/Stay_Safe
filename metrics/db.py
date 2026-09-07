"""Metrics store for the live app: one SQLite file, one table, no extra
infra. Logs every chat interaction (question/answer/latency/retrieval),
the LLM-judge verdict (filled in later, possibly by a batch job), and the
user's thumbs up/down. The Streamlit "Monitoring" tab reads straight from
this file.
"""
import json
import sqlite3
from datetime import datetime

from common.config import METRICS_DB_PATH


def get_conn(db_path: str = METRICS_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = METRICS_DB_PATH):
    conn = get_conn(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            retrieval_mode TEXT,
            retrieved_ids TEXT,
            latency_ms INTEGER,
            judge_verdict TEXT,
            judge_reason TEXT,
            feedback TEXT,
            feedback_ts TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_interaction(result: dict, db_path: str = METRICS_DB_PATH) -> int:
    """`result` is the dict returned by rag.chat.ask()."""
    retrieved_ids = json.dumps([h.get("sqlite_id") for h in result.get("retrieved", [])])
    conn = get_conn(db_path)
    cursor = conn.execute(
        """
        INSERT INTO interactions (ts, question, answer, retrieval_mode, retrieved_ids, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            result["question"],
            result["answer"],
            result.get("retrieval_mode"),
            retrieved_ids,
            result.get("latency_ms"),
        ),
    )
    conn.commit()
    interaction_id = cursor.lastrowid
    conn.close()
    return interaction_id


def log_judge_result(interaction_id: int, verdict: str, reason: str, db_path: str = METRICS_DB_PATH):
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE interactions SET judge_verdict = ?, judge_reason = ? WHERE id = ?",
        (verdict, reason, interaction_id),
    )
    conn.commit()
    conn.close()


def log_feedback(interaction_id: int, vote: str, db_path: str = METRICS_DB_PATH):
    """`vote` should be 'up' or 'down'."""
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE interactions SET feedback = ?, feedback_ts = ? WHERE id = ?",
        (vote, datetime.now().isoformat(), interaction_id),
    )
    conn.commit()
    conn.close()


def fetch_recent(limit: int = 100, db_path: str = METRICS_DB_PATH) -> list[dict]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_unjudged(limit: int = 50, db_path: str = METRICS_DB_PATH) -> list[dict]:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM interactions WHERE judge_verdict IS NULL ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def aggregate_stats(db_path: str = METRICS_DB_PATH) -> dict:
    conn = get_conn(db_path)
    total = conn.execute("SELECT COUNT(*) AS c FROM interactions").fetchone()["c"]
    avg_latency = conn.execute("SELECT AVG(latency_ms) AS a FROM interactions").fetchone()["a"]
    feedback_counts = dict(
        conn.execute(
            "SELECT feedback, COUNT(*) AS c FROM interactions WHERE feedback IS NOT NULL GROUP BY feedback"
        ).fetchall()
    )
    judge_counts = dict(
        conn.execute(
            "SELECT judge_verdict, COUNT(*) AS c FROM interactions WHERE judge_verdict IS NOT NULL GROUP BY judge_verdict"
        ).fetchall()
    )
    conn.close()
    return {
        "total_interactions": total,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
        "feedback_counts": feedback_counts,
        "judge_verdict_counts": judge_counts,
    }
