"""LLM-as-a-judge for automatic relevance scoring. Two entry points:

- judge_answer(): score a single (question, answer) pair - used both for
  the offline ground-truth answer-quality report and for scoring live
  interactions logged by the Streamlit app.
- batch_judge_ground_truth(): run the full RAG pipeline over the
  ground-truth questions and score the real answers (offline eval).
- batch_judge_logged_interactions(): score whatever's sitting unjudged in
  the metrics store (SQLite or Supabase, via metrics/store.py's dispatcher)
  - the "online"/batch-monitoring path - see Kestra note in the README for
  running this on a schedule.

Judge model is nemotron-3-ultra (see common/config.py); the generator is a
smaller, different Nemotron variant (nemotron-3.5-lightning) - not the
fully separate "Mistral generator + Nemotron judge" family split this
project originally aimed for, but the more reliable pairing found in live
testing (see rag/llm_client.py's module docstring for why).
"""
import argparse
import json
import time

from openai import RateLimitError

from common.config import JUDGE_MODEL, NVIDIA_REQUEST_DELAY_SECONDS
from metrics.store import fetch_unjudged, log_judge_result, init_db
from rag.chat import ask
from rag.llm_client import judge

VERDICTS = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]

JUDGE_PROMPT = """You are grading whether an AI assistant's answer is relevant to the user's question.

QUESTION:
{question}

ANSWER:
{answer}

Judge ONLY relevance - does the answer actually address what was asked? Do not judge factual correctness beyond that.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"verdict": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT", "reason": "one short sentence"}}
"""


def judge_answer(question: str, answer: str) -> dict:
    raw = judge([{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer)}])
    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "").upper()
        if verdict not in VERDICTS:
            verdict = "NON_RELEVANT"
        return {"verdict": verdict, "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, AttributeError):
        # Judge model didn't return clean JSON - don't crash the batch over it.
        return {"verdict": "NON_RELEVANT", "reason": f"Judge returned unparseable output: {raw[:200]}"}


def batch_judge_ground_truth(ground_truth_path: str, output_path: str, k: int = 3, mode: str = "hybrid") -> dict:
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = []
    for i, item in enumerate(ground_truth, 1):
        print(f"[{i}/{len(ground_truth)}] {item['question']}")
        try:
            rag_result = ask(item["question"], k=k, mode=mode)
            verdict = judge_answer(item["question"], rag_result["answer"])
        except RateLimitError:
            print("  -> Rate limited by NVIDIA even after retries. Skipping this item; "
                  "consider raising NVIDIA_REQUEST_DELAY_SECONDS in .env if this repeats.")
            time.sleep(NVIDIA_REQUEST_DELAY_SECONDS * 5)
            continue
        except Exception as e:
            # One bad item (transient 500, timeout, etc.) shouldn't cost the
            # whole batch's results - skip it and keep going.
            print(f"  -> {type(e).__name__}: {e}. Skipping this item.")
            continue

        results.append({**item, "answer": rag_result["answer"], **verdict})
        print(f"  -> {verdict['verdict']}: {verdict['reason']}")
        time.sleep(NVIDIA_REQUEST_DELAY_SECONDS)

    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICTS}
    skipped = len(ground_truth) - len(results)
    report = {
        "judge_model": JUDGE_MODEL,
        "retrieval_mode": mode,
        "verdict_counts": counts,
        "skipped_items": skipped,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nVerdict counts: {counts}" + (f" ({skipped} skipped due to errors)" if skipped else ""))
    print(f"Report saved to {output_path}")
    return report


def batch_judge_logged_interactions(limit: int = 50) -> tuple[int, int]:
    """Scores whatever's unjudged in metrics.db. Run this periodically
    (manually, via /loop, via the Streamlit Monitoring tab's button, or as
    a Kestra flow) rather than judging inline on every live chat turn -
    keeps user-facing latency down.

    Returns (judged_count, total_unjudged_found) so callers - the CLI and
    the Streamlit button - can report what actually happened.
    """
    init_db()
    unjudged = fetch_unjudged(limit=limit)
    if not unjudged:
        print("Nothing to judge - all logged interactions already have a verdict.")
        return 0, 0

    judged_count = 0
    for row in unjudged:
        try:
            verdict = judge_answer(row["question"], row["answer"])
        except RateLimitError:
            print(f"  interaction {row['id']}: rate limited by NVIDIA even after retries - skipping.")
            time.sleep(NVIDIA_REQUEST_DELAY_SECONDS * 5)
            continue
        except Exception as e:
            print(f"  interaction {row['id']}: {type(e).__name__}: {e} - skipping.")
            continue

        log_judge_result(row["id"], verdict["verdict"], verdict["reason"])
        print(f"  interaction {row['id']}: {verdict['verdict']} - {verdict['reason']}")
        judged_count += 1
        time.sleep(NVIDIA_REQUEST_DELAY_SECONDS)

    return judged_count, len(unjudged)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge relevance scoring.")
    parser.add_argument("--mode", choices=["ground-truth", "logged"], default="ground-truth")
    parser.add_argument("--ground-truth", default="eval/ground_truth.json")
    parser.add_argument("--output", default="eval/answer_quality_report.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--retrieval-mode", default="hybrid")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "ground-truth":
        batch_judge_ground_truth(args.ground_truth, args.output, k=args.k, mode=args.retrieval_mode)
    else:
        judged_count, total = batch_judge_logged_interactions(limit=args.limit)
        if total:
            print(f"\nJudged {judged_count}/{total} logged interactions.")


if __name__ == "__main__":
    main()
