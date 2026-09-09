"""
Runs the agent AND both baselines over the golden set, computes intent
accuracy, escalation P/R/F1, and (non-mock only) LLM-judge reply quality.
Writes eval/results/eval_report.json and prints a summary to stdout.

    python eval/run_eval.py --golden eval/golden_set.csv --pairs data/processed/sample_pairs.csv --mock
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from src.agent import SupportAgent  # noqa: E402
from src.baselines import run_trivial_baseline, run_simple_baseline  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from eval.metrics import intent_metrics, escalation_metrics, judge_reply  # noqa: E402


def run_system(name, predict_fn, golden_df):
    intents_pred, escalate_pred, replies = [], [], []
    for _, row in golden_df.iterrows():
        out = predict_fn(row["customer_message"], row.get("thread_id"))
        intents_pred.append(out["intent"])
        escalate_pred.append(bool(out["escalate"]))
        replies.append(out["reply_draft"])
    return intents_pred, escalate_pred, replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--pairs", default="data/processed/sample_pairs.csv",
                     help="historical pairs the agent retrieves grounding examples from")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--run-judge", action="store_true",
                     help="also run LLM-as-judge on the agent's replies (costs API calls)")
    ap.add_argument("--out", default="eval/results/eval_report.json")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    pairs = pd.read_csv(args.pairs)
    agent = SupportAgent(pairs, mock=args.mock)

    def agent_predict(msg, tid):
        return agent.handle(msg, thread_id=tid)

    def trivial_predict(msg, tid):
        return run_trivial_baseline(msg)

    def simple_predict(msg, tid):
        return run_simple_baseline(msg)

    report = {}
    for name, fn in [("agent", agent_predict), ("trivial_baseline", trivial_predict),
                      ("simple_baseline", simple_predict)]:
        intents_pred, escalate_pred, replies = run_system(name, fn, golden)
        im = intent_metrics(golden["true_intent"].tolist(), intents_pred)
        em = escalation_metrics(golden["gold_escalate"].astype(bool).tolist(), escalate_pred)
        report[name] = {"intent_metrics": im, "escalation_metrics": em}
        print(f"\n=== {name} ===")
        print(f"Intent accuracy: {im['accuracy']:.2%} | macro-F1: {im['macro_f1']:.2f}")
        print(f"Escalation precision: {em['precision']:.2f} | recall: {em['recall']:.2f} "
              f"| f1: {em['f1']:.2f}")

    if args.run_judge and not args.mock:
        judge_client = LLMClient(mock=False)
        judge_scores = []
        for _, row in golden.iterrows():
            out = agent.handle(row["customer_message"], thread_id=row.get("thread_id"))
            score = judge_reply(judge_client, row["customer_message"],
                                 row["reference_resolution_summary"], out["reply_draft"])
            score["thread_id"] = row["thread_id"]
            judge_scores.append(score)
        report["judge_scores"] = judge_scores
        os.makedirs("eval/results", exist_ok=True)
        pd.DataFrame(judge_scores).to_csv("eval/results/judge_scores.csv", index=False)
        print(f"\nWrote {len(judge_scores)} judge scores to eval/results/judge_scores.csv "
              f"-- now hand-score the same threads into eval/results/human_scores.csv and run "
              f"`python eval/metrics.py` for agreement.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
