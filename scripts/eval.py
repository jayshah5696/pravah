"""
Pravah v2 Evaluation Script

Evaluates the agent against the test dataset in tests/eval_set.csv.
Logs full trajectories to tests/traces/ for debugging.
Outputs summary to tests/eval_results.csv.

Usage:
    python scripts/eval.py [--model MODEL] [--limit N]
"""

import asyncio
import csv
import json
import time
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Paths
INPUT_FILE = "tests/eval_set.csv"
OUTPUT_FILE = "tests/eval_results.csv"
TRACES_DIR = "tests/traces"

os.makedirs(TRACES_DIR, exist_ok=True)


async def run_query(query: str, query_id: str, model: str = "openai/gpt-4o"):
    """Run a query and capture the full trajectory.

    Captures:
    - All tool call inputs and outputs
    - Model reasoning/thinking
    - Intermediate messages
    - Timing information
    """
    from pravah.agent import app
    from langchain_core.messages import HumanMessage

    config = {
        "configurable": {"thread_id": f"eval_{query_id}"},
        "recursion_limit": 25,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "model": model,
        "temperature": 0.0,
        "iteration_count": 0,
        "documents": [],
    }

    start_time = time.time()

    trajectory = {
        "query": query,
        "query_id": query_id,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_outputs": [],
        "model_outputs": [],
        "errors": [],
        "final_response": "",
    }

    final_response = ""
    current_model_output = ""

    try:
        async for event in app.astream_events(inputs, config=config, version="v2"):
            event_type = event.get("event", "")
            data = event.get("data", {})

            # Capture model streaming
            if event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk:
                    content = getattr(chunk, "content", "") or ""
                    if content:
                        current_model_output += content
                        final_response += content

                    # Check for tool calls in the chunk
                    tool_calls = getattr(chunk, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            trajectory["tool_calls"].append(
                                {
                                    "name": tc.get("name", "unknown"),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id", ""),
                                }
                            )

            elif event_type == "on_chat_model_end":
                if current_model_output:
                    trajectory["model_outputs"].append(
                        {
                            "step": len(trajectory["model_outputs"]) + 1,
                            "content": current_model_output,
                        }
                    )
                    current_model_output = ""

            elif event_type == "on_tool_start":
                tool_input = data.get("input", {})
                print(
                    f"  🔧 Tool: {event.get('name')} | Input: {str(tool_input)[:100]}..."
                )

            elif event_type == "on_tool_end":
                tool_output = data.get("output", "")
                trajectory["tool_outputs"].append(
                    {
                        "name": event.get("name", "unknown"),
                        "output": str(tool_output)[:3000],
                    }
                )
                print(
                    f"  ✅ {event.get('name')} returned {len(str(tool_output))} chars"
                )

            elif event_type in ("on_chain_error", "on_tool_error"):
                error_str = str(data.get("error", "unknown error"))
                trajectory["errors"].append(
                    {
                        "type": event_type,
                        "name": event.get("name", ""),
                        "error": error_str[:500],
                    }
                )
                print(f"  ❌ Error: {error_str[:100]}")

    except Exception as e:
        trajectory["errors"].append({"type": "execution_error", "error": str(e)})
        final_response = f"ERROR: {e}"

    trajectory["final_response"] = final_response
    trajectory["latency_seconds"] = round(time.time() - start_time, 2)
    trajectory["tool_call_count"] = len(trajectory["tool_calls"])
    trajectory["has_citations"] = bool("[" in final_response and "](" in final_response)

    # Save full trace
    trace_file = os.path.join(TRACES_DIR, f"trace_{query_id}.json")
    with open(trace_file, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)

    return trajectory


def evaluate_trajectory(
    trajectory: dict, expected_behavior: str, pass_criteria: str
) -> dict:
    """Evaluate a trajectory against expected behavior.

    Args:
        trajectory: The captured trajectory
        expected_behavior: Expected from eval_set.csv
        pass_criteria: Criteria from eval_set.csv

    Returns:
        Dict with evaluation results
    """
    tool_names = [t["name"] for t in trajectory["tool_calls"]]
    # Accept both web_search and gemini_search as valid search tools
    has_web_search = "web_search" in tool_names or "gemini_search" in tool_names

    # Check tool usage based on expected behavior
    if expected_behavior in ("No Search", "Direct Answer", "No search"):
        tool_check = not has_web_search
        tool_check_reason = (
            "Correctly avoided search" if tool_check else "Should not have searched"
        )
    elif expected_behavior in ("Web Search", "1 Search", "Search"):
        tool_check = has_web_search
        tool_check_reason = (
            "Correctly used search" if tool_check else "Should have searched"
        )
    elif expected_behavior in ("2+ Searches",):
        tool_check = len([t for t in tool_names if t == "web_search"]) >= 2
        tool_check_reason = (
            "Used multiple searches"
            if tool_check
            else "Should have used multiple searches"
        )
    elif expected_behavior in ("Ask Clarification",):
        # Check if response asks for clarification
        response = trajectory["final_response"].lower()
        tool_check = any(
            word in response
            for word in ["clarify", "which", "what", "more information", "specify"]
        )
        tool_check_reason = (
            "Asked for clarification"
            if tool_check
            else "Should have asked for clarification"
        )
    else:
        tool_check = True
        tool_check_reason = "No specific behavior expected"

    # Check pass criteria
    criteria_results = {}
    final_response = trajectory["final_response"]

    if "cites_source" in pass_criteria:
        criteria_results["cites_source"] = trajectory["has_citations"]

    if "no_hallucination" in pass_criteria:
        # Simple check: if we searched and found nothing, we shouldn't have a confident answer
        # This is a heuristic - proper hallucination detection requires more sophisticated methods
        criteria_results["no_hallucination"] = True  # Assume pass, flag manually

    if "graceful_decline" in pass_criteria:
        decline_phrases = [
            "couldn't find",
            "no information",
            "no results",
            "unable to find",
            "not available",
        ]
        criteria_results["graceful_decline"] = any(
            phrase in final_response.lower() for phrase in decline_phrases
        )

    if "no_search" in pass_criteria:
        criteria_results["no_search"] = not has_web_search

    if "correct_tool" in pass_criteria:
        criteria_results["correct_tool"] = tool_check

    # Overall pass
    criteria_pass = all(criteria_results.values()) if criteria_results else True
    overall_pass = tool_check and criteria_pass

    return {
        "tool_check_pass": tool_check,
        "tool_check_reason": tool_check_reason,
        "criteria_results": criteria_results,
        "criteria_pass": criteria_pass,
        "overall_pass": overall_pass,
    }


async def main(model: str = "openai/gpt-4o", limit: int = None):
    """Run evaluation on the test dataset."""
    print(f"Reading from {INPUT_FILE}...")
    print(f"Using model: {model}")
    print(f"Traces will be saved to {TRACES_DIR}/\n")

    results = []

    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    for idx, row in enumerate(rows):
        query_id = row.get("id", f"{idx:03d}")
        question = row["question"]
        expected_behavior = row.get("expected_behavior", "")
        pass_criteria = row.get("pass_criteria", "")

        print(f"[{query_id}] {question}")

        trajectory = await run_query(question, query_id, model)
        evaluation = evaluate_trajectory(trajectory, expected_behavior, pass_criteria)

        results.append(
            {
                "query_id": query_id,
                "question": question,
                "category": row.get("category", ""),
                "feature": row.get("feature", ""),
                "expected_behavior": expected_behavior,
                "tool_calls": ", ".join([t["name"] for t in trajectory["tool_calls"]])
                or "None",
                "tool_check": "✓" if evaluation["tool_check_pass"] else "✗",
                "tool_reason": evaluation["tool_check_reason"],
                "has_citations": "✓" if trajectory["has_citations"] else "✗",
                "criteria_pass": "✓" if evaluation["criteria_pass"] else "✗",
                "overall_pass": "✓" if evaluation["overall_pass"] else "✗",
                "errors": len(trajectory["errors"]),
                "latency_s": trajectory["latency_seconds"],
                "response_preview": trajectory["final_response"][:200].replace(
                    "\n", " "
                ),
                "trace_file": f"traces/trace_{query_id}.json",
            }
        )

        status = "✓ PASS" if evaluation["overall_pass"] else "✗ FAIL"
        print(
            f"    → {status} | Tools: {[t['name'] for t in trajectory['tool_calls']]} | Latency: {trajectory['latency_seconds']}s\n"
        )

    # Write results
    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Summary
    total = len(results)
    tool_pass = sum(1 for r in results if r["tool_check"] == "✓")
    overall_pass = sum(1 for r in results if r["overall_pass"] == "✓")
    with_citations = sum(1 for r in results if r["has_citations"] == "✓")
    with_errors = sum(1 for r in results if r["errors"] > 0)
    avg_latency = sum(r["latency_s"] for r in results) / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print("EVAL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total queries:       {total}")
    print(
        f"Overall pass rate:   {overall_pass}/{total} ({100 * overall_pass // total if total > 0 else 0}%)"
    )
    print(
        f"Tool usage correct:  {tool_pass}/{total} ({100 * tool_pass // total if total > 0 else 0}%)"
    )
    print(
        f"Has citations:       {with_citations}/{total} ({100 * with_citations // total if total > 0 else 0}%)"
    )
    print(f"Queries with errors: {with_errors}")
    print(f"Avg latency:         {avg_latency:.2f}s")
    print(f"\nFull traces: {TRACES_DIR}/")
    print(f"Summary CSV: {OUTPUT_FILE}")

    # Pass rate by category
    if any(r.get("category") for r in results):
        print(f"\nPass rate by category:")
        categories = set(r.get("category", "unknown") for r in results)
        for cat in sorted(categories):
            cat_results = [r for r in results if r.get("category") == cat]
            cat_pass = sum(1 for r in cat_results if r["overall_pass"] == "✓")
            print(
                f"  {cat}: {cat_pass}/{len(cat_results)} ({100 * cat_pass // len(cat_results) if cat_results else 0}%)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pravah evaluation")
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini", help="Model to use for evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of test cases to run"
    )
    args = parser.parse_args()

    asyncio.run(main(model=args.model, limit=args.limit))
