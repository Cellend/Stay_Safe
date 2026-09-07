"""Streamlit chat app for the ready.gov RAG assistant.

Two tabs:
  - Chat: ask a question, get an answer (Elasticsearch hybrid retrieval +
    Mistral via NVIDIA), thumbs up/down on each answer.
  - Monitoring: reads metrics.db (see metrics/db.py) and charts volume,
    latency, feedback ratio, and LLM-judge verdicts.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Allow running via `streamlit run app/streamlit_app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from common.config import require_nvidia_key
from eval.llm_judge import batch_judge_logged_interactions
from metrics.store import init_db, log_interaction, log_feedback, fetch_recent, fetch_unjudged, aggregate_stats
from rag.chat import ask

JUDGE_BATCH_SIZE = 10  # per button click - keeps a click's wait time predictable

st.set_page_config(page_title="Cyclone Disaster Assistant", page_icon="🌀", layout="wide")
init_db()

chat_tab, monitoring_tab = st.tabs(["💬 Chat", "📊 Monitoring"])

# --------------------------------------------------------------------------
# Chat tab
# --------------------------------------------------------------------------
with chat_tab:
    st.title("🌀 Cyclone Disaster Assistant")
    st.caption("Retrieval-augmented chat over ready.gov hurricane/typhoon/cyclone guidance.")

    try:
        require_nvidia_key()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    retrieval_mode = st.sidebar.selectbox("Retrieval mode", ["hybrid", "vector", "keyword"], index=0)
    top_k = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=3)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # each: {role, content, interaction_id?, feedback?}

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("interaction_id"):
                col1, col2, _ = st.columns([1, 1, 10])
                feedback = msg.get("feedback")
                if col1.button("👍", key=f"up_{i}", disabled=feedback is not None):
                    log_feedback(msg["interaction_id"], "up")
                    msg["feedback"] = "up"
                    st.rerun()
                if col2.button("👎", key=f"down_{i}", disabled=feedback is not None):
                    log_feedback(msg["interaction_id"], "down")
                    msg["feedback"] = "down"
                    st.rerun()
                if feedback:
                    st.caption(f"Feedback recorded: {'👍' if feedback == 'up' else '👎'}")

    if question := st.chat_input("Ask about hurricane, typhoon, or cyclone preparedness..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching ready.gov and generating an answer..."):
                    result = ask(question, k=top_k, mode=retrieval_mode)
            except Exception as e:
                # NVIDIA's free-tier endpoints occasionally 500 or time out
                # (observed directly while building this) - fail visibly
                # instead of crashing the whole app on one bad response.
                st.error(f"The generator model didn't respond ({type(e).__name__}). Try again in a moment.")
                st.stop()
            st.markdown(result["answer"])
            st.caption(f"Retrieved {len(result['retrieved'])} chunks ({retrieval_mode}) in {result['latency_ms']} ms")

            interaction_id = log_interaction(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "interaction_id": interaction_id, "feedback": None}
            )
        st.rerun()

# --------------------------------------------------------------------------
# Monitoring tab
# --------------------------------------------------------------------------
with monitoring_tab:
    st.title("📊 Monitoring")

    # st.rerun() below cuts the script off immediately, so a message shown
    # right before it would never actually render - stash it and show it
    # here instead, on the run right after the rerun.
    if "judge_result_message" in st.session_state:
        st.success(st.session_state.pop("judge_result_message"))

    stats = aggregate_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total interactions", stats["total_interactions"])
    col2.metric("Avg latency (ms)", stats["avg_latency_ms"] or "—")

    up = stats["feedback_counts"].get("up", 0)
    down = stats["feedback_counts"].get("down", 0)
    total_feedback = up + down
    col3.metric("👍 / 👎", f"{up} / {down}", f"{round(100*up/total_feedback)}% positive" if total_feedback else None)

    judged = sum(stats["judge_verdict_counts"].values())
    col4.metric("Judged interactions", judged)

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("User feedback")
        if total_feedback:
            st.bar_chart(pd.DataFrame({"count": [up, down]}, index=["👍 up", "👎 down"]))
        else:
            st.caption("No feedback recorded yet.")

    with right:
        st.subheader("LLM-judge verdicts")
        if judged:
            verdict_df = pd.DataFrame.from_dict(stats["judge_verdict_counts"], orient="index", columns=["count"])
            st.bar_chart(verdict_df)
        else:
            st.caption("No interactions judged yet.")

        pending = fetch_unjudged(limit=1000)
        button_label = f"⚖️ Judge up to {JUDGE_BATCH_SIZE} now" + (f" ({len(pending)} pending)" if pending else "")
        if st.button(button_label, disabled=not pending):
            with st.spinner(f"Scoring up to {JUDGE_BATCH_SIZE} interactions with the judge model..."):
                try:
                    judged_count, total = batch_judge_logged_interactions(limit=JUDGE_BATCH_SIZE)
                except Exception as e:
                    st.error(f"Judging failed ({type(e).__name__}). Try again in a moment.")
                    st.stop()
            st.session_state["judge_result_message"] = f"Judged {judged_count}/{total} interactions."
            st.rerun()

    st.divider()
    st.subheader("Recent interactions")
    recent = fetch_recent(limit=50)
    if recent:
        df = pd.DataFrame(recent)[
            ["id", "ts", "question", "answer", "retrieval_mode", "latency_ms", "judge_verdict", "feedback"]
        ]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No interactions logged yet - ask something in the Chat tab.")
