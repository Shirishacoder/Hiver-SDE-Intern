"""
Single entry point: reads (customer_message, brand_reply) pairs, runs each
customer_message through the agent (and both baselines), prints a compact
comparison table, and writes full JSON results to eval/results/pipeline_run.json.

    python scripts/run_pipeline.py --input data/processed/sample_pairs.csv --mock
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from src.agent import SupportAgent  # noqa: E402
from src.baselines import run_trivial_baseline, run_simple_baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/sample_pairs.csv")
    ap.add_argument("--mock", action="store_true", help="skip real LLM calls")
    ap.add_argument("--limit", type=int, default=None, help="only run first N rows")
    ap.add_argument("--out", default="eval/results/pipeline_run.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    agent = SupportAgent(df, mock=args.mock)

    results = []
    print(f"{'thread_id':>10} | {'intent':<20} | {'esc?':<5} | reply (truncated)")
    print("-" * 100)
    for _, row in df.iterrows():
        msg = row["customer_message"]
        tid = row.get("thread_id")

        agent_out = agent.handle(msg, thread_id=tid)
        trivial_out = run_trivial_baseline(msg)
        simple_out = run_simple_baseline(msg)

        results.append({
            "thread_id": tid,
            "customer_message": msg,
            "agent": agent_out,
            "trivial_baseline": trivial_out,
            "simple_baseline": simple_out,
        })

        reply_preview = agent_out["reply_draft"][:60].replace("\n", " ")
        print(f"{str(tid):>10} | {agent_out['intent']:<20} | "
              f"{str(agent_out['escalate']):<5} | {reply_preview}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} full results to {args.out}")

    esc_rate = sum(r["agent"]["escalate"] for r in results) / len(results)
    print(f"Agent escalation rate: {esc_rate:.0%} "
          f"(trivial baseline: 100%, simple baseline: "
          f"{sum(r['simple_baseline']['escalate'] for r in results)/len(results):.0%})")


if __name__ == "__main__":
    main()
