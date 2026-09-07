"""Picks the metrics backend (SQLite or Supabase) based on METRICS_BACKEND
in .env, and re-exports the same functions either way. app/streamlit_app.py
and eval/llm_judge.py import from HERE, not from metrics/db.py or
metrics/supabase_db.py directly, so switching backends is a one-line .env
change, not a code change. (Independent of RETRIEVAL_BACKEND - you could,
for instance, keep Elasticsearch for retrieval but move metrics to
Supabase for durability on a host with an ephemeral filesystem.)
"""
from common.config import METRICS_BACKEND

if METRICS_BACKEND == "supabase":
    from metrics.supabase_db import (
        init_db,
        log_interaction,
        log_judge_result,
        log_feedback,
        fetch_recent,
        fetch_unjudged,
        aggregate_stats,
    )
elif METRICS_BACKEND == "sqlite":
    from metrics.db import (
        init_db,
        log_interaction,
        log_judge_result,
        log_feedback,
        fetch_recent,
        fetch_unjudged,
        aggregate_stats,
    )
else:
    raise ValueError(f"Unknown METRICS_BACKEND: {METRICS_BACKEND!r} (expected 'sqlite' or 'supabase')")

__all__ = [
    "init_db",
    "log_interaction",
    "log_judge_result",
    "log_feedback",
    "fetch_recent",
    "fetch_unjudged",
    "aggregate_stats",
]
