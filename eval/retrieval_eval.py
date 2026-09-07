"""Retrieval evaluation: Hit Rate@k and MRR, computed separately for
keyword, vector, and hybrid (RRF) retrieval - so the choice of retrieval
mode is backed by a measurement, not a guess. Runs against whichever
backend RETRIEVAL_BACKEND in .env points at (rag/retriever.py dispatches).
Run this after generate_ground_truth.py.
"""
import argparse
import json

from rag.retriever import retrieve

GROUND_TRUTH_PATH = "eval/ground_truth.json"
MODES = ["keyword", "vector", "hybrid"]


def evaluate_mode(ground_truth: list[dict], mode: str, k: int) -> dict:
    hits, reciprocal_ranks = 0, []

    for item in ground_truth:
        results = retrieve(item["question"], k=k, mode=mode)
        retrieved_ids = [r["sqlite_id"] for r in results]

        if item["expected_chunk_id"] in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(item["expected_chunk_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth)
    return {
        "mode": mode,
        "k": k,
        "hit_rate": round(hits / n, 3) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0.0,
        "n_questions": n,
    }


def run_eval(ground_truth_path: str, k: int) -> list[dict]:
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = []
    for mode in MODES:
        print(f"Evaluating mode='{mode}' (k={k})...")
        result = evaluate_mode(ground_truth, mode, k)
        results.append(result)
        print(f"  Hit Rate@{k}: {result['hit_rate']}   MRR: {result['mrr']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare keyword/vector/hybrid retrieval on the ground-truth set.")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="eval/retrieval_eval_report.json")
    args = parser.parse_args()

    results = run_eval(args.ground_truth, args.k)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
