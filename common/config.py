"""Central place every module reads its configuration from.

Loads `.env` (see `.env.example`) so nothing sensitive - the NVIDIA API key -
ever needs to be hardcoded or pasted into a chat/notebook cell.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "ready_raw.db")
METRICS_DB_PATH = os.getenv("METRICS_DB_PATH", "metrics.db")

# --- Elasticsearch ---
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "ready_chunks")

# --- Backend selection ---
# Local dev defaults to Elasticsearch + SQLite (what's been built and tested
# throughout this project). Supabase (a free-forever hosted Postgres +
# pgvector) is the alternative for a deployed copy that needs to stay free
# and online indefinitely - see rag/supabase_retriever.py, metrics/supabase_db.py,
# and README.md's deployment section. Switch via .env, not by editing code.
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "elasticsearch")  # "elasticsearch" | "supabase"
METRICS_BACKEND = os.getenv("METRICS_BACKEND", "sqlite")  # "sqlite" | "supabase"

# Postgres connection string from Supabase (Project Settings -> Database ->
# Connection string -> URI). Only needed if either backend above is "supabase".
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")


def require_supabase_url():
    if not SUPABASE_DB_URL:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set, but RETRIEVAL_BACKEND or METRICS_BACKEND is "
            "'supabase'. Add it to .env - get it from your Supabase project's "
            "Settings -> Database -> Connection string (URI format)."
        )

# --- Embeddings (unchanged from the original Chroma pipeline) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIMS = 384  # all-MiniLM-L6-v2 output size - update if EMBED_MODEL changes

# sentence-transformers phones home to Hugging Face on every load to check
# for model updates; on machines with a broken/incomplete local CA bundle
# that HEAD request fails and retries with backoff (harmless but slow -
# ~30s of noise per process). Once EMBED_MODEL is cached locally (it is,
# after the first successful run), set HF_HUB_OFFLINE=1 in .env (picked up
# automatically by load_dotenv() above) to skip the network check entirely.
# Leave unset on a fresh machine's first run so the model can download.

# --- NVIDIA NIM (OpenAI-compatible hosted LLMs) ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Free-tier accounts are rate-limited (exact numbers aren't published and
# seem to vary). openai-python already retries a 429 automatically with
# backoff, but the batch eval scripts (generate_ground_truth.py,
# llm_judge.py) fire dozens of sequential calls - pacing them proactively
# means fewer 429s to retry from in the first place. Bump this via .env if
# you still see rate-limit errors in eval/*.py output.
NVIDIA_REQUEST_DELAY_SECONDS = float(os.getenv("NVIDIA_REQUEST_DELAY_SECONDS", "1.0"))
# mistralai/mistral-nemotron was the original pick (per the "Mistral generator
# + Nemotron judge" family-separation decision) but proved unreliable in live
# testing - see rag/llm_client.py's module docstring for the full story.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")


def require_nvidia_key():
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://build.nvidia.com"
        )
