"""
Runs both agents over tests/test_conversations.py and writes:
  results/proposed_results.csv
  results/baseline_results.csv
  results/summary.json

Metrics computed (see Proposed Methodology, Section 5):
  - Intent / category accuracy vs. gold label
  - Task-completion rate
  - Human-in-the-loop intervention rate (proposed only)
  - Unsafe-immediate-action rate (baseline only - cancels with no confirmation)
  - Escalation rate
  - Average latency per conversation
"""

import csv
import json
import os
import time
import copy
import statistics

from graph_proposed import PROPOSED_GRAPH
from graph_baseline import BASELINE_GRAPH
from tests.test_conversations import TEST_CASES
from tools import backend

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")


def run_proposed(case):
    backend.reset()
    t0 = time.perf_counter()
    state = {"user_message": case["text"]}
    result = PROPOSED_GRAPH.invoke(state)

    human_in_loop = bool(result.get("needs_approval"))
    if human_in_loop:
        # Simulate the user confirming the sensitive action on the next turn.
        followup_state = copy.deepcopy(result)
        followup_state["user_approved"] = True
        result = PROPOSED_GRAPH.invoke(followup_state)

    latency = time.perf_counter() - t0
    return {
        "id": case["id"],
        "text": case["text"],
        "gold_intent": case["gold_intent"],
        "predicted_intent": result.get("intent"),
        "intent_correct": result.get("intent") == case["gold_intent"],
        "specialist": result.get("specialist"),
        "human_in_loop": human_in_loop,
        "resolved": bool(result.get("resolved")),
        "escalated": bool(result.get("escalated")),
        "response": result.get("response"),
        "latency_sec": round(latency, 4),
    }


def _baseline_expected_category(gold_intent):
    if gold_intent.startswith("cancel_"):
        return gold_intent
    if gold_intent in ("flight_booking", "hotel_booking", "car_rental"):
        return gold_intent
    return None  # baseline has no concept of policy_query / general_query


def run_baseline(case):
    backend.reset()
    t0 = time.perf_counter()
    state = {"user_message": case["text"]}
    result = BASELINE_GRAPH.invoke(state)
    latency = time.perf_counter() - t0

    predicted = result.get("intent")  # "cancel_generic" or "lookup_generic"
    expected = _baseline_expected_category(case["gold_intent"])
    # baseline is judged "correct" only if it at least distinguished
    # cancel vs lookup correctly AND a booking id was resolvable
    if expected is None:
        correct = False  # baseline structurally cannot handle policy/general queries
    else:
        is_cancel_gold = expected.startswith("cancel_")
        is_cancel_pred = predicted == "cancel_generic"
        correct = (is_cancel_gold == is_cancel_pred) and bool(result.get("resolved"))

    unsafe_immediate_cancel = (predicted == "cancel_generic") and bool(result.get("resolved"))

    return {
        "id": case["id"],
        "text": case["text"],
        "gold_intent": case["gold_intent"],
        "predicted_category": predicted,
        "category_correct": correct,
        "unsafe_immediate_cancel": unsafe_immediate_cancel,
        "resolved": bool(result.get("resolved")),
        "response": result.get("response"),
        "latency_sec": round(latency, 4),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    proposed_rows = [run_proposed(c) for c in TEST_CASES]
    baseline_rows = [run_baseline(c) for c in TEST_CASES]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_csv(os.path.join(RESULTS_DIR, "proposed_results.csv"), proposed_rows)
    write_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), baseline_rows)

    n = len(TEST_CASES)
    summary = {
        "n_test_cases": n,
        "proposed": {
            "intent_accuracy": round(sum(r["intent_correct"] for r in proposed_rows) / n, 3),
            "task_completion_rate": round(sum(r["resolved"] for r in proposed_rows) / n, 3),
            "human_in_loop_rate": round(sum(r["human_in_loop"] for r in proposed_rows) / n, 3),
            "escalation_rate": round(sum(r["escalated"] for r in proposed_rows) / n, 3),
            "avg_latency_sec": round(statistics.mean(r["latency_sec"] for r in proposed_rows), 4),
        },
        "baseline": {
            "category_accuracy": round(sum(r["category_correct"] for r in baseline_rows) / n, 3),
            "task_completion_rate": round(sum(r["resolved"] for r in baseline_rows) / n, 3),
            "unsafe_immediate_cancel_rate": round(sum(r["unsafe_immediate_cancel"] for r in baseline_rows) / n, 3),
            "avg_latency_sec": round(statistics.mean(r["latency_sec"] for r in baseline_rows), 4),
        },
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
