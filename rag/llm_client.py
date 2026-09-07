"""Thin wrapper around NVIDIA's hosted, OpenAI-compatible NIM endpoint.

Two roles, two models: a fast model answers live chat, a different model
judges/generates ground truth offline. Originally this was meant to be
"Mistral generator + Nemotron judge" for family separation, but live
testing found mistralai/mistral-nemotron genuinely unreliable on the free
tier - two separate real calls both failed with a backend 500
("Inference connection error"), and a third succeeded but took 70s. So the
generator was switched to nvidia/nemotron-3.5-lightning-30b-a3b, which was
fast (2-19s) and error-free across every real call made while building
this. That does mean generator and judge are both "Nemotron"-branded now,
which weakens (doesn't eliminate - they're different sizes/generations)
the judge-independence rationale. If NVIDIA's mistral-nemotron endpoint
becomes reliable again later, it's worth switching back.

Model IDs are read from config (.env) rather than hardcoded here - verify
they still exist in the catalog at https://build.nvidia.com before relying
on this long-term; hosted catalogs change (and being listed by
client.models.list() does NOT mean a given account can actually call it -
several catalog-listed models 404'd with "Function ... Not found for
account" in testing).
"""
from openai import OpenAI

from common.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, require_nvidia_key

_live_client = None
_batch_client = None


def get_client() -> OpenAI:
    """Batch/offline client (ground-truth generation, judging): latency
    doesn't matter here, so it's tuned to absorb a slow cold start (one
    real call took 70s) - long timeout, several retries.
    """
    global _batch_client
    require_nvidia_key()
    if _batch_client is None:
        _batch_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=90.0, max_retries=3)
    return _batch_client


def get_live_client() -> OpenAI:
    """Live-chat client: a user is waiting, so this fails fast instead of
    the batch client's long timeout - which, with several retries, can
    otherwise compound to several minutes on a genuinely hung request
    (confirmed in testing: an unhandled APITimeoutError surfaced in the
    Streamlit app after ~100s with the batch-tuned settings applied here
    too). Short timeout, fewer retries, so a bad response comes back
    quickly enough to show the user an error instead of a frozen spinner.
    """
    global _live_client
    require_nvidia_key()
    if _live_client is None:
        _live_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=20.0, max_retries=2)
    return _live_client


def strip_think(text: str) -> str:
    """Some reasoning models (phi4-mini-reasoning, DeepSeek-reasoning, etc.
    via Ollama) inline a <think>...</think> block into the message content
    itself - strip it before showing text to a user or feeding it to the
    judge. NOTE: the NVIDIA-hosted judge model this project uses
    (nemotron-3-ultra) does NOT do this - it returns its reasoning in a
    separate `message.reasoning_content` field and leaves `content` clean
    already (confirmed by inspecting a real response). This function is
    therefore a harmless no-op for that model, kept here so swapping in an
    inline-<think> model (e.g. via Ollama) doesn't silently leak reasoning
    into logged answers.
    """
    if "<think>" not in text:
        return text.strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return ""  # thinking block never closed (e.g. truncated stream) - nothing usable yet


def generate(messages: list[dict], stream: bool = False, **kwargs):
    """Calls the live-chat generator model. Uses the fast-fail client (see
    get_live_client) since a real user is waiting on this.
    """
    client = get_live_client()
    kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
    return client.chat.completions.create(model=GENERATOR_MODEL, messages=messages, stream=stream, **kwargs)


def judge(messages: list[dict], **kwargs) -> str:
    """Calls the offline judge/ground-truth model (Nemotron), non-streaming,
    and returns the answer text with any <think> block already stripped.
    """
    client = get_client()
    kwargs.setdefault("temperature", 0.0)  # judging should be deterministic
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        **kwargs,
    )
    return strip_think(completion.choices[0].message.content)
